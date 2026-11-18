#include <cuda/atomic>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int TileM = 16;
constexpr int TileN = 8;
constexpr int TileK = 32;
constexpr int Lanes = 32;
constexpr int Stages = 4;
constexpr unsigned Warp = 0xffffffffu;

struct alignas(16) Stage {
    half a[TileM][TileK];
    uint32_t packed_b[Lanes];
    half scales[Lanes];
};

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// 每个 lane 有两个 B 寄存器，每个寄存器放两个 FP16。一个打包字覆盖两个 K=16 的 MMA，故一个 lane 正好拿到八个 4-bit 码。
void repack(const std::vector<uint8_t>& codes, const std::vector<half>& scales,
            int n, int k_groups, std::vector<uint32_t>& packed,
            std::vector<half>& packed_scales) {
    for (int nt = 0; nt < n / TileN; ++nt) {
        for (int kt = 0; kt < k_groups; ++kt) {
            for (int lane = 0; lane < Lanes; ++lane) {
                const int col = nt * TileN + lane / 4;
                const int base_k = kt * TileK + 2 * (lane % 4);
                uint32_t word = 0;
                for (int mma_id = 0; mma_id < 2; ++mma_id) {
                    for (int reg = 0; reg < 2; ++reg) {
                        for (int half_lane = 0; half_lane < 2; ++half_lane) {
                            const int row = base_k + 16 * mma_id + 8 * reg + half_lane;
                            // 把 half2 的两个值分别安排在低、高半字的同一位置。
                            const int shift = 16 * half_lane + 8 * mma_id + 4 * reg;
                            word |= uint32_t{codes[row * n + col]} << shift;
                        }
                    }
                }
                const int index = (nt * k_groups + kt) * Lanes + lane;
                packed[index] = word;
                // 按消费 lane 保存 scale，固定零点 8 由反量化指令常数吸收。
                packed_scales[index] = scales[kt * n + col];
            }
        }
    }
}

__device__ void copy_16_async(void* dst, const void* src) {
    const auto addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 :: "r"(addr), "l"(src) : "memory");
}

// 每个 16B 向量保持连续且对齐。按行扰动向量位置，使 ldmatrix
// 读取同一逻辑列的连续八行时，从不同的 shared-memory bank 组起读。
__device__ int a_shared_col(int row, int vector_col) {
    return 8 * (vector_col ^ ((row / 2) % 4));
}

__device__ void issue_stage(Stage& stage, const half* a, const uint32_t* packed,
                            const half* scales, int lane, int mt, int nt, int kt,
                            int k, int k_groups) {
    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        const int vec = lane + pass * Lanes;
        const int row = vec / 4;
        const int col = vec % 4;
        copy_16_async(&stage.a[row][a_shared_col(row, col)],
                      a + (mt * TileM + row) * k + kt * TileK + col * 8);
    }
    const int offset = (nt * k_groups + kt) * Lanes;
    if (lane < 8) {
        copy_16_async(stage.packed_b + 4 * lane, packed + offset + 4 * lane);
    }
    if (lane < 4) {
        copy_16_async(stage.scales + 8 * lane, scales + offset + 8 * lane);
    }
    // 每个线程每个 stage 都提交一个 group，wait_group 才能按同一顺序推进。
    asm volatile("cp.async.commit_group;" ::: "memory");
}

__device__ void load_a(const Stage& stage, int lane, int mma_id, uint32_t (&a)[4]) {
    // lane 0–7 / 8–15 分别提供左上 / 左下的行地址，后 16 个 lane
    // 提供右上 / 右下。ldmatrix.x4 按 MMA 的 A fragment 组装寄存器。
    const int row = lane % 16;
    const int col = 2 * mma_id + lane / 16;
    const auto addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(&stage.a[row][a_shared_col(row, col)]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(addr) : "memory");
}

__device__ uint32_t half_mantissa(uint32_t word, uint32_t mask) {
    uint32_t result;
    // 0xea 是 (word & mask) | exponent_bits 的真值表。
    asm("lop3.b32 %0, %1, %2, %3, 0xea;"
        : "=r"(result) : "r"(word), "r"(mask), "r"(0x64006400u));
    return result;
}

__device__ half2 as_half2(uint32_t word) {
    __half2_raw raw;
    asm("mov.b32 {%0,%1}, %2;" : "=h"(raw.x), "=h"(raw.y) : "r"(word));
    return half2(raw);
}

__device__ uint32_t as_word(half2 pair) {
    const __half2_raw raw = pair;
    uint32_t word;
    asm("mov.b32 %0, {%1,%2};" : "=r"(word) : "h"(raw.x), "h"(raw.y));
    return word;
}

__device__ void decode_b(uint32_t word, half scale, uint32_t (&b)[2]) {
    // 低 nibble 构造 1024+q，减 1032 得 q-8。
    // 高 nibble 构造 1024+16q，经 /16-72 同样得到 q-8。
    const half2 low = as_half2(half_mantissa(word, 0x000f000fu));
    const half2 high = as_half2(half_mantissa(word, 0x00f000f0u));
    const half2 signed_lo = __hsub2(low, as_half2(0x64086408u));
    const half2 signed_hi = __hfma2(high, as_half2(0x2c002c00u), as_half2(0xd480d480u));
    const half2 pair_scale = __half2half2(scale);
    b[0] = as_word(__hmul2(signed_lo, pair_scale));
    b[1] = as_word(__hmul2(signed_hi, pair_scale));
}

__device__ void mma(const uint32_t (&a)[4], const uint32_t (&b)[2], float (&acc)[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

struct Schedule {
    int tasks_per_cta;
    int extra_ctas;

    __host__ __device__ int begin(int cta) const {
        return cta * tasks_per_cta + (cta < extra_ctas ? cta : extra_ctas);
    }

    __host__ __device__ int owner(int task) const {
        const int cut = extra_ctas * (tasks_per_cta + 1);
        return task < cut ? task / (tasks_per_cta + 1)
                          : extra_ctas + (task - cut) / tasks_per_cta;
    }
};

__global__ void marlin_w4a16(const half* a, const uint32_t* packed, const half* scales,
                           float* output, float* partials, int* arrivals,
                           int n, int k, Schedule schedule) {
    __shared__ Stage stages[Stages];
    const int lane = threadIdx.x;
    const int k_groups = k / TileK;
    const int n_tiles = n / TileN;
    const int end = schedule.begin(blockIdx.x + 1);

    // 将 [输出 tile, K group] 展平。每 CTA 处理一个连续任务区间，
    // 相邻 CTA 的任务数最多相差 1，一个区间可以跨越输出 tile 边界。
    for (int task = schedule.begin(blockIdx.x); task < end;) {
        const int tile = task / k_groups;
        const int tile_begin = tile * k_groups;
        const int tile_end = tile_begin + k_groups;
        const int segment_end = min(end, tile_end);
        const int groups = segment_end - task;
        const int first_group = task - tile_begin;
        const int mt = tile / n_tiles;
        const int nt = tile % n_tiles;
        float acc[4] = {};

        // 预发三个 group，稳态时再发第四个并只等最老的一个。
        // 到尾部时统一排空，保证复用 shared slot 前所有 lane 都读完。
        for (int i = 0; i < min(groups, Stages - 1); ++i) {
            issue_stage(stages[i], a, packed, scales, lane, mt, nt,
                        first_group + i, k, k_groups);
        }
        for (int i = 0; i < groups; ++i) {
            const int future = i + Stages - 1;
            if (future < groups) {
                issue_stage(stages[future % Stages], a, packed, scales, lane, mt, nt,
                            first_group + future, k, k_groups);
                asm volatile("cp.async.wait_group 3;" ::: "memory");
            } else {
                asm volatile("cp.async.wait_group 0;" ::: "memory");
            }
            __syncwarp();

            const Stage& stage = stages[i % Stages];
            const uint32_t word = stage.packed_b[lane];
            const half scale = stage.scales[lane];
            uint32_t a_frag[2][4], b_frag[2][2];
            // 两套寄存器各保存一个 K=16 fragment。预备下一片，
            // 使它的加载和解码与当前 MMA 的输入没有数据依赖。
            load_a(stage, lane, 0, a_frag[0]);
            decode_b(word, scale, b_frag[0]);
            load_a(stage, lane, 1, a_frag[1]);
            decode_b(word >> 8, scale, b_frag[1]);
            mma(a_frag[0], b_frag[0], acc);
            mma(a_frag[1], b_frag[1], acc);
            __syncwarp();
        }

        const int first_owner = schedule.owner(tile_begin);
        const int last_owner = schedule.owner(tile_end - 1);
        if (first_owner != last_owner) {
            // 每个片段写入自己的 FP32 slot，再发布到达计数。
            // 最后到达的 CTA 接手归约。其它 CTA 不自旋，调度顺序无需固定。
            float* slot = partials + (tile * k_groups + first_group) * TileM * TileN;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                slot[lane * 4 + i] = acc[i];
            }
            __syncwarp();
            int previous = 0;
            if (lane == 0) {
                cuda::atomic_ref<int, cuda::thread_scope_device> count(arrivals[tile]);
                previous = count.fetch_add(1, cuda::memory_order_acq_rel);
            }
            previous = __shfl_sync(Warp, previous, 0);
            __syncwarp();
            if (previous != last_owner - first_owner) {
                task = segment_end;
                continue;
            }

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[i] = 0;
            }
            for (int owner = first_owner; owner <= last_owner; ++owner) {
                const int start = max(tile_begin, schedule.begin(owner)) - tile_begin;
                const float* peer = partials + (tile * k_groups + start) * TileM * TileN;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    acc[i] += peer[lane * 4 + i];
                }
            }
        }

        const int row = mt * TileM + lane / 4;
        const int col = nt * TileN + 2 * (lane % 4);
        output[row * n + col] = acc[0];
        output[row * n + col + 1] = acc[1];
        output[(row + 8) * n + col] = acc[2];
        output[(row + 8) * n + col + 1] = acc[3];
        task = segment_end;
    }
}

template <typename T>
T* managed(size_t count) {
    T* ptr;
    check(cudaMallocManaged(&ptr, count * sizeof(T)));
    return ptr;
}

bool run_case(int m, int n, int k, int ctas, const char* output_path) {
    const int k_groups = k / TileK;
    const int tiles = (m / TileM) * (n / TileN);
    const int tasks = tiles * k_groups;
    const Schedule schedule{tasks / ctas, tasks % ctas};
    std::vector<uint8_t> codes(k * n);
    std::vector<half> scales(k_groups * n);
    std::vector<uint32_t> packed((n / TileN) * k_groups * Lanes);
    std::vector<half> packed_scales(packed.size());
    half* a_device = managed<half>(m * k);
    uint32_t* b_device = managed<uint32_t>(packed.size());
    half* scales_device = managed<half>(packed.size());
    float* output = managed<float>(m * n);
    float* partials = managed<float>(tiles * k_groups * TileM * TileN);
    int* arrivals = managed<int>(tiles);

    for (int i = 0; i < m * k; ++i) {
        a_device[i] = __float2half(float((i * 17 + i / k * 13) % 31 - 15) / 16);
    }
    for (int i = 0; i < k * n; ++i) {
        codes[i] = (i * 11 + i / n * 7 + i / 19) % 16;
    }
    for (int i = 0; i < k_groups * n; ++i) {
        scales[i] = __float2half(float(1 + (i * 7 + i / n) % 15) / 128);
    }
    repack(codes, scales, n, k_groups, packed, packed_scales);
    std::copy(packed.begin(), packed.end(), b_device);
    std::copy(packed_scales.begin(), packed_scales.end(), scales_device);
    std::fill_n(arrivals, tiles, 0);
    std::fill_n(output, m * n, NAN);
    marlin_w4a16<<<ctas, Lanes>>>(
        a_device, b_device, scales_device, output, partials, arrivals, n, k, schedule);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());

    float max_error = 0;
    bool valid = true;
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float ref = 0;
            for (int kk = 0; kk < k; ++kk) {
                const float scale = __half2float(scales[(kk / TileK) * n + col]);
                ref += __half2float(a_device[row * k + kk]) * (codes[kk * n + col] - 8) * scale;
            }
            const float actual = output[row * n + col];
            valid &= std::isfinite(actual);
            max_error = std::max(max_error, std::abs(actual - ref));
        }
    }

    int split_tiles = 0;
    int max_participants = 1;
    for (int tile = 0; tile < tiles; ++tile) {
        const int first = schedule.owner(tile * k_groups);
        const int last = schedule.owner((tile + 1) * k_groups - 1);
        const int expected = first == last ? 0 : last - first + 1;
        valid &= arrivals[tile] == expected;
        split_tiles += expected != 0;
        max_participants = std::max(max_participants, expected);
    }
    valid &= max_error <= 1e-5f;
    std::printf("M=%d N=%d K=%d CTAs=%d split_tiles=%d/%d max_participants=%d max_abs_error=%g %s\n",
                m, n, k, ctas, split_tiles, tiles, max_participants, max_error,
                valid ? "OK" : "FAIL");
    if (output_path != nullptr) {
        FILE* file = std::fopen(output_path, "wb");
        if (file == nullptr) {
            std::perror(output_path);
            valid = false;
        } else {
            valid &= std::fwrite(output, sizeof(float), m * n, file) == size_t(m * n);
            valid &= std::fclose(file) == 0;
        }
    }
    for (void* pointer : {static_cast<void*>(a_device), static_cast<void*>(b_device),
                          static_cast<void*>(scales_device), static_cast<void*>(output),
                          static_cast<void*>(partials), static_cast<void*>(arrivals)}) {
        check(cudaFree(pointer));
    }
    return valid;
}

}  // namespace

int main(int argc, char** argv) {
    // 一组只走尾部，另一组覆盖流水线绕回。最后两组分别触发 2 和多个 CTA
    // 共同处理一个 tile，并核对实际到达计数。
    bool valid = run_case(16, 8, 32, 1, nullptr);
    valid &= run_case(32, 24, 320, 6, nullptr);
    valid &= run_case(32, 24, 320, 7, nullptr);
    valid &= run_case(32, 24, 320, 26, argc > 1 ? argv[1] : nullptr);
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
