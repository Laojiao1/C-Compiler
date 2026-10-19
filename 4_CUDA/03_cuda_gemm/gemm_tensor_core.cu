/**
 * nvcc -std=c++17 -O3 -lineinfo -arch=sm_120 --ptxas-options=-v \
  gemm_tensor_core.cu -o gemm_tensor_core -lcublas

  nsys profile --trace=cuda --stats=true --force-overwrite=true \
  --output=nsys/gemm_tensor_core \
  -- ./gemm_tensor_core --benchmark-only

  ncu --set full --launch-skip 1 --launch-count 1 --force-overwrite \
  --export ncu/gemm_tensor_core.ncu-rep \
  -- ./gemm_tensor_core --benchmark-only
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>

using namespace nvcuda;

// 一个 Thread Block 计算 C 中 128x128 的 tile，并沿 K 轴每次推进 32。
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;

// 空间换时间：
// 通过 NCU 性能分析工具发现 B fragment 的 LDSM.16.MT88.4 有严重 Bank Conflict
// 将行跨度从 128 改为 136 half，使相邻行的 Bank 起点产生偏移
// 之所以+8，是因为 136 仍是 8 的倍数，满足API约束，也能完美错开 Bank 起点
constexpr int BN_PADDED = BN + 8;

// 当前 WMMA API 使用 m16n16k16：一个 warp 协作完成一个 16x16 输出 tile。
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t error = (call);                                       \
        if (error != cudaSuccess) {                                             \
            std::cerr << "CUDA error: " << cudaGetErrorString(error)            \
                      << " (" << __FILE__ << ':' << __LINE__ << ")\n";             \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        const cublasStatus_t status = (call);                                   \
        if (status != CUBLAS_STATUS_SUCCESS) {                                  \
            std::cerr << "cuBLAS error: status=" << static_cast<int>(status)    \
                      << " (" << __FILE__ << ':' << __LINE__ << ")\n";             \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// cp.async 绕过寄存器，实现全局显存到共享内存的硬件级直接异步搬运
// 与普通 LDG + STS 不同，发射后当前 warp 可以继续执行独立的 HMMA 指令，
// 直到显式 wait 时才要求拷贝完成，这正是计算与访存重叠的基础。
__device__ __forceinline__ void copy_global_to_shared_async_16(
    half* shared_dst, const half* global_src) {
    const unsigned shared_address =
        static_cast<unsigned>(__cvta_generic_to_shared(shared_dst));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :
        : "r"(shared_address), "l"(global_src)
        : "memory");
}

__device__ __forceinline__ void commit_async_copy_group() {
    // 当前线程此前发射的 cp.async 被归入同一个等待组。
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void wait_for_all_async_copies() {
    // wait_all 只保证本线程发射的异步拷贝完成；随后仍需 __syncthreads，
    // 才能保证整个 Block 都完成搬运并安全读取完整 tile。
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}

__device__ __forceinline__ void load_aligned_tile_async(
    half* As_stage,
    half* Bs_stage,
    const half* A,
    const half* B,
    int N,
    int K,
    int bk,
    int block_m,
    int block_n,
    int tid) {
    // 与同步版本保持相同的协作搬运映射：每个线程搬 A/B 各 16 个 half，
    // 每一侧拆为两条 16-byte cp.async。
    const int a_load_row = tid / 2;
    const int a_load_col = (tid % 2) * 16;
    const std::size_t a_index =
        static_cast<std::size_t>(block_m * BM + a_load_row) * K +
        bk + a_load_col;

    const int b_load_row = tid / 8;
    const int b_load_col = (tid % 8) * 16;
    const std::size_t b_index =
        static_cast<std::size_t>(bk + b_load_row) * N +
        block_n * BN + b_load_col;

    #pragma unroll
    for (int vec = 0; vec < 16; vec += 8) {
        copy_global_to_shared_async_16(
            &As_stage[a_load_row * BK + a_load_col + vec],
            &A[a_index + vec]);
        copy_global_to_shared_async_16(
            &Bs_stage[b_load_row * BN_PADDED + b_load_col + vec],
            &B[b_index + vec]);
    }
    commit_async_copy_group();
}

/**
 * C = A * B。A/B 为 row-major FP16，C 为 row-major FP32。
 *
 * Aligned=true 是规则尺寸快路径：编译期删除边界分支，使用 cp.async 双缓冲
 * 重叠下一 K tile 的搬运与当前 tile 的 HMMA，并让 WMMA 直接写 C。
 * Aligned=false 是通用路径：尾块逐元素补零；输出先经过 shared memory，
 * 再由各 lane 对合法位置做掩码写回，避免 WMMA 整块写导致越界。
 */
template <bool Aligned>
__global__ void gemm_tensor_core_kernel(const half* __restrict__ A,
                                        const half* __restrict__ B,
                                        float* __restrict__ C,
                                        int M, int N, int K) {
    // 快路径需要两个 stage 做 ping-pong；通用路径只实例化一个 stage，
    // 因此不会为不使用的双缓冲额外占用 Shared Memory。
    constexpr int Stages = Aligned ? 2 : 1;
    // 只让 Aligned 实例使用 padding；通用路径仍保持 BN=128，
    // 避免为不是 Benchmark 主路径的尾块 Kernel 增加 Shared Memory。
    constexpr int BSharedStride = Aligned ? BN_PADDED : BN;
    __shared__ half As[Stages][BM][BK];
    __shared__ half Bs[Stages][BK][BSharedStride];

    // Block 使用 16x16=256 个线程，即 8 个 warp。
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // 8 个 warp 排成 4x2：每个 warp 负责 C 中 32x64 的区域。
    // 32x64 又被拆成 2x4 个 16x16 WMMA tile。
    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;

    // 每个 warp 持有 8 个 FP32 累加 fragment。它们贯穿整个 K 循环，
    // 避免把中间结果写回 Shared/Global Memory，但也会带来较高寄存器压力。
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   half, wmma::row_major> b_frag[4];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                   float> c_frag[2][4];

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    // A/B tile 各有 4096 个 half，256 个线程平均每线程搬运 16 个 half。
    // 16 个 half 拆成两次 float4 搬运，每次传输 8 个 half = 128 bit。
    const int a_load_row = tid / 2;
    const int a_load_col = (tid % 2) * 16;
    const int b_load_row = tid / 8;
    const int b_load_col = (tid % 8) * 16;

    int read_stage = 0;
    int write_stage = 1;

    if constexpr (Aligned) {
        // Prologue：在进入主循环前先把 tile 0 放入 stage 0。
        load_aligned_tile_async(&As[0][0][0], &Bs[0][0][0],
                                A, B, N, K, 0,
                                blockIdx.y, blockIdx.x, tid);
        wait_for_all_async_copies();
        __syncthreads();
    }

    // K 轴流水线：快路径在计算当前 read_stage 时，把下一 tile 异步写入
    // write_stage。通用路径仍使用 stage 0 做同步加载和边界补零。
    for (int bk = 0; bk < K; bk += BK) {
        if constexpr (Aligned) {
            if (bk + BK < K) {
                load_aligned_tile_async(&As[write_stage][0][0],
                                        &Bs[write_stage][0][0],
                                        A, B, N, K, bk + BK,
                                        blockIdx.y, blockIdx.x, tid);
            }
        } else {
            const int global_a_row = blockIdx.y * BM + a_load_row;
            const int global_a_col = bk + a_load_col;
            const std::size_t a_index =
                static_cast<std::size_t>(global_a_row) * K + global_a_col;

            // 通用路径只有在连续 16 个 half 全部合法且对齐时才能使用 float4。
            // 尾块或行首未对齐时逐元素读取，越界位置补 0，不影响 GEMM 结果。
            const bool a_vector_ok = global_a_row < M &&
                                     global_a_col + 15 < K &&
                                     (a_index % 8 == 0);
            if (a_vector_ok) {
                #pragma unroll
                for (int vec = 0; vec < 16; vec += 8) {
                    *reinterpret_cast<float4*>(&As[0][a_load_row][a_load_col + vec]) =
                        *reinterpret_cast<const float4*>(&A[a_index + vec]);
                }
            } else {
                #pragma unroll
                for (int v = 0; v < 16; ++v) {
                    const int col = global_a_col + v;
                    As[0][a_load_row][a_load_col + v] =
                        (global_a_row < M && col < K)
                            ? A[static_cast<std::size_t>(global_a_row) * K + col]
                            : __float2half(0.0f);
                }
            }

            const int global_b_row = bk + b_load_row;
            const int global_b_col = blockIdx.x * BN + b_load_col;
            const std::size_t b_index =
                static_cast<std::size_t>(global_b_row) * N + global_b_col;
            const bool b_vector_ok = global_b_row < K &&
                                     global_b_col + 15 < N &&
                                     (b_index % 8 == 0);
            if (b_vector_ok) {
                #pragma unroll
                for (int vec = 0; vec < 16; vec += 8) {
                    *reinterpret_cast<float4*>(&Bs[0][b_load_row][b_load_col + vec]) =
                        *reinterpret_cast<const float4*>(&B[b_index + vec]);
                }
            } else {
                #pragma unroll
                for (int v = 0; v < 16; ++v) {
                    const int col = global_b_col + v;
                    Bs[0][b_load_row][b_load_col + v] =
                        (global_b_row < K && col < N)
                            ? B[static_cast<std::size_t>(global_b_row) * N + col]
                            : __float2half(0.0f);
                }
            }

            // 同步路径必须等整个 Block 完成协作加载后才能读取 stage 0。
            __syncthreads();
        }

        // BK=32 包含两个 k16 子阶段。每个阶段先装载 fragment，
        // 再执行 2x4=8 次 warp 级 mma_sync。
        #pragma unroll
        for (int k_step = 0; k_step < BK / WMMA_K; ++k_step) {
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int row = warp_m * 32 + i * WMMA_M;
                wmma::load_matrix_sync(
                    a_frag[i], &As[read_stage][row][k_step * WMMA_K], BK);
            }

            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int col = warp_n * 64 + j * WMMA_N;
                // ldm 必须与 B 在 Shared Memory 中的真实行跨度一致。
                // Aligned 路径传入 136，通用路径仍传入 128。
                wmma::load_matrix_sync(
                    b_frag[j], &Bs[read_stage][k_step * WMMA_K][col],
                    BSharedStride);
            }

            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j],
                                   c_frag[i][j]);
                }
            }
        }

        if constexpr (Aligned) {
            if (bk + BK < K) {
                // 当前 tile 计算完后等待下一 tile，再交换读写 stage。
                wait_for_all_async_copies();
                __syncthreads();
                const int old_read_stage = read_stage;
                read_stage = write_stage;
                write_stage = old_read_stage;
            }
        } else {
            // 通用路径下一轮会覆盖 stage 0，必须等所有 warp 消费完毕。
            __syncthreads();
        }
    }

    if constexpr (Aligned) {
        // 快路径中每个 16x16 输出 tile 都完全位于矩阵内，可由 WMMA 直接写回。
        // 这会绕过通用路径的 8 KiB shared-memory 输出暂存区和逐元素判断。
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int row = blockIdx.y * BM + warp_m * 32 + i * WMMA_M;
                const int col = blockIdx.x * BN + warp_n * 64 + j * WMMA_N;
                wmma::store_matrix_sync(&C[static_cast<std::size_t>(row) * N + col],
                                        c_frag[i][j], N,
                                        wmma::mem_row_major);
            }
        }
    } else {
        // WMMA store 没有逐元素 mask，只能一次写完整 16x16 tile。
        // 每个 warp 使用独占暂存区，随后由 32 个 lane 合作写回合法元素。
        __shared__ __align__(32) float C_tile[8][WMMA_M][WMMA_N];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                wmma::store_matrix_sync(&C_tile[warp_id][0][0], c_frag[i][j],
                                        WMMA_N, wmma::mem_row_major);
                // store_matrix_sync 是 warp 集体操作；写回 C 前确保暂存数据可见。
                __syncwarp();

                const int tile_row = blockIdx.y * BM + warp_m * 32 + i * WMMA_M;
                const int tile_col = blockIdx.x * BN + warp_n * 64 + j * WMMA_N;
                for (int index = lane_id; index < WMMA_M * WMMA_N; index += 32) {
                    const int row = tile_row + index / WMMA_N;
                    const int col = tile_col + index % WMMA_N;
                    if (row < M && col < N) {
                        C[static_cast<std::size_t>(row) * N + col] =
                            C_tile[warp_id][index / WMMA_N][index % WMMA_N];
                    }
                }
                // 同一 warp 的下一 fragment 会复用这块暂存区。
                __syncwarp();
            }
        }
    }
}

void launch_gemm(const half* A, const half* B, float* C,
                 int M, int N, int K, dim3 grid) {
    // 只在所有 Block 都没有尾块时选择快路径。模板参数使无用分支在编译期消失，
    // 而不是让每个 GPU 线程在运行期反复判断矩阵是否对齐。
    const bool aligned = M % BM == 0 && N % BN == 0 && K > 0 && K % BK == 0;
    if (aligned) {
        const dim3 aligned_block(16, 16);
        gemm_tensor_core_kernel<true><<<grid, aligned_block>>>(
            A, B, C, M, N, K);
    } else {
        // 通用尾块路径固定使用 8 个 warp，与 8 份输出暂存区一一对应。
        const dim3 general_block(16, 16);
        gemm_tensor_core_kernel<false><<<grid, general_block>>>(
            A, B, C, M, N, K);
    }
}

void print_aligned_kernel_resources() {
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(
        &attributes, gemm_tensor_core_kernel<true>));

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    constexpr int threads_per_block = 16 * 16;
    int active_blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_sm,
        gemm_tensor_core_kernel<true>,
        threads_per_block,
        0));

    // Occupancy API 给出“资源约束下最多能驻留多少 Block”，属于理论上限；
    // 它不代表运行时每个周期都有同样比例的 warp 真正在执行。
    const int active_warps =
        active_blocks_per_sm * threads_per_block / properties.warpSize;
    const int max_warps =
        properties.maxThreadsPerMultiProcessor / properties.warpSize;
    const double theoretical_occupancy =
        100.0 * active_warps / static_cast<double>(max_warps);

    std::cout << "=== Aligned kernel resources ===\n"
              << "Device: " << properties.name << '\n'
              << "Registers/thread: " << attributes.numRegs << '\n'
              << "Static shared memory/block: "
              << attributes.sharedSizeBytes / 1024.0 << " KiB\n"
              << "Active blocks/SM: " << active_blocks_per_sm << '\n'
              << "Active warps/SM: " << active_warps << " / " << max_warps << '\n'
              << "Theoretical occupancy: " << theoretical_occupancy << "%\n";
}

struct TestCase {
    int M;
    int N;
    int K;
    const char* name;
};

void fill_inputs(std::vector<half>& A, std::vector<half>& B, unsigned seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (half& value : A) value = __float2half(distribution(generator));
    for (half& value : B) value = __float2half(distribution(generator));
}

bool run_correctness_case(const TestCase& test, unsigned seed) {
    const std::size_t count_A = static_cast<std::size_t>(test.M) * test.K;
    const std::size_t count_B = static_cast<std::size_t>(test.K) * test.N;
    const std::size_t count_C = static_cast<std::size_t>(test.M) * test.N;
    // 在 C 前后放置固定字节模式；若 kernel 越界写，guard 检查会立即失败。
    constexpr std::size_t guard_elements = 64;
    constexpr std::uint32_t guard_pattern = 0xA5A5A5A5u;

    std::vector<half> h_A(count_A);
    std::vector<half> h_B(count_B);
    std::vector<float> h_C(count_C);
    fill_inputs(h_A, h_B, seed);

    half* d_A = nullptr;
    half* d_B = nullptr;
    float* d_C_storage = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, count_A * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, count_B * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C_storage,
                          (count_C + 2 * guard_elements) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), count_A * sizeof(half),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), count_B * sizeof(half),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C_storage, 0xA5,
                          (count_C + 2 * guard_elements) * sizeof(float)));

    float* const d_C = d_C_storage + guard_elements;
    const dim3 grid((test.N + BN - 1) / BN, (test.M + BM - 1) / BM);
    launch_gemm(d_A, d_B, d_C, test.M, test.N, test.K, grid);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<std::uint32_t> storage_bits(count_C + 2 * guard_elements);
    CUDA_CHECK(cudaMemcpy(storage_bits.data(), d_C_storage,
                          storage_bits.size() * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost));
    std::memcpy(h_C.data(), storage_bits.data() + guard_elements,
                count_C * sizeof(float));

    bool guards_ok = true;
    for (std::size_t i = 0; i < guard_elements; ++i) {
        guards_ok &= storage_bits[i] == guard_pattern;
        guards_ok &= storage_bits[guard_elements + count_C + i] == guard_pattern;
    }

    // CPU 使用由 FP16 输入转换出的 double 做参考累加，避免参考实现自身的
    // FP32 舍入误差掩盖 GPU 结果问题。
    double max_abs_error = 0.0;
    double max_error_ratio = 0.0;
    int worst_row = 0;
    int worst_col = 0;
    bool values_ok = true;

    for (int row = 0; row < test.M; ++row) {
        for (int col = 0; col < test.N; ++col) {
            double reference = 0.0;
            double magnitude = 0.0;
            for (int k = 0; k < test.K; ++k) {
                const double a = __half2float(h_A[static_cast<std::size_t>(row) * test.K + k]);
                const double b = __half2float(h_B[static_cast<std::size_t>(k) * test.N + col]);
                reference += a * b;
                magnitude += std::abs(a * b);
            }

            const double actual = h_C[static_cast<std::size_t>(row) * test.N + col];
            const double abs_error = std::abs(actual - reference);
            // 用乘积绝对值之和缩放误差，避免结果接近 0 时相对误差失真。
            const double tolerance = 2.0e-5 + 3.0e-4 * magnitude;
            const double error_ratio = abs_error / tolerance;
            if (!std::isfinite(actual) || error_ratio > 1.0) values_ok = false;
            if (error_ratio > max_error_ratio) {
                max_error_ratio = error_ratio;
                max_abs_error = abs_error;
                worst_row = row;
                worst_col = col;
            }
        }
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_storage));

    const bool passed = guards_ok && values_ok;
    std::cout << (passed ? "[PASS] " : "[FAIL] ") << test.name
              << " (M=" << test.M << ", N=" << test.N << ", K=" << test.K << ')'
              << ", max_abs_error=" << max_abs_error
              << ", tolerance_ratio=" << max_error_ratio
              << ", worst=(" << worst_row << ',' << worst_col << ')'
              << ", guards=" << (guards_ok ? "intact" : "CORRUPTED") << '\n';
    return passed;
}

bool run_correctness_suite() {
    // 第一项覆盖规则尺寸快路径，其余项目覆盖通用尾块路径。
    const TestCase tests[] = {
        {128, 128, 32, "aligned block"},
        {130, 131, 37, "M/N/K tails"},
        {17, 19, 7, "smaller than WMMA tile"},
        {1, 1, 1, "minimal matrix"},
        {257, 129, 65, "multi-block tails"},
    };

    std::cout << "=== WMMA GEMM correctness suite ===\n";
    bool passed = true;
    for (std::size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); ++i) {
        passed &= run_correctness_case(tests[i], 2024u + static_cast<unsigned>(i));
    }
    std::cout << (passed ? "All correctness tests PASSED.\n"
                         : "Correctness test FAILED.\n");
    return passed;
}

void run_benchmark() {
    constexpr int M = 2048;
    constexpr int N = 2048;
    constexpr int K = 2048;
    constexpr int iterations = 20;

    std::vector<half> h_A(static_cast<std::size_t>(M) * K);
    std::vector<half> h_B(static_cast<std::size_t>(K) * N);
    fill_inputs(h_A, h_B, 2024);

    half* d_A = nullptr;
    half* d_B = nullptr;
    float* d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, h_A.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, h_B.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, static_cast<std::size_t>(M) * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(half),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(half),
                          cudaMemcpyHostToDevice));

    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    print_aligned_kernel_resources();

    // 预热一次，排除首次 CUDA 上下文初始化和代码加载开销。
    launch_gemm(d_A, d_B, d_C, M, N, K, grid);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    // CUDA Event 记录同一 stream 上多次 kernel 的总 GPU 时间，再取平均值。
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        launch_gemm(d_A, d_B, d_C, M, N, K, grid);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const double average_ms = total_ms / iterations;
    const double tflops = (2.0 * M * N * K) / (average_ms / 1000.0) / 1.0e12;
    std::cout << "=== Tensor Core (WMMA) GEMM benchmark ===\n"
              << "Execution time: " << average_ms << " ms\n"
              << "Performance: " << tflops << " TFLOP/s\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

void run_cublas_benchmark() {
    constexpr int M = 2048;
    constexpr int N = 2048;
    constexpr int K = 2048;
    constexpr int warmup_iterations = 5;
    constexpr int benchmark_iterations = 20;

    std::vector<half> h_A(static_cast<std::size_t>(M) * K);
    std::vector<half> h_B(static_cast<std::size_t>(K) * N);
    fill_inputs(h_A, h_B, 2024);

    half* d_A = nullptr;
    half* d_B = nullptr;
    float* d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, h_A.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, h_B.size() * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, static_cast<std::size_t>(M) * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(half),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(half),
                          cudaMemcpyHostToDevice));

    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    auto launch_cublas_gemm = [&]() {
        // cuBLAS 按 column-major 解释矩阵，而我们的内存是 row-major：
        // row-major C=A*B 在同一块内存上等价于 column-major C^T=B^T*A^T。
        // 因此交换 A/B 的传入顺序，并把输出形状写成 N x M，无需真实转置。
        CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N,
            M,
            K,
            &alpha,
            d_B,
            CUDA_R_16F,
            N,
            d_A,
            CUDA_R_16F,
            K,
            &beta,
            d_C,
            CUDA_R_32F,
            N,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };

    // 多次预热让 cuBLAS 完成内部模块加载和算法初始化，这些开销不属于 GEMM。
    for (int i = 0; i < warmup_iterations; ++i) launch_cublas_gemm();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < benchmark_iterations; ++i) launch_cublas_gemm();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const double average_ms = total_ms / benchmark_iterations;
    const double tflops = (2.0 * M * N * K) / (average_ms / 1000.0) / 1.0e12;

    std::cout << "=== cuBLAS Tensor Core GEMM benchmark ===\n"
              << "Input: FP16, accumulation/output: FP32\n"
              << "Execution time: " << average_ms << " ms\n"
              << "Performance: " << tflops << " TFLOP/s\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

int main(int argc, char** argv) {
    bool correctness = true;
    bool benchmark = false;
    bool cublas_benchmark = false;
    for (int i = 1; i < argc; ++i) {
        // --benchmark：先做正确性回归再测试；--benchmark-only：跳过回归，
        // 供 nsys 等工具只采集固定尺寸的性能 kernel。
        if (std::string(argv[i]) == "--benchmark") benchmark = true;
        if (std::string(argv[i]) == "--benchmark-only") {
            correctness = false;
            benchmark = true;
        }
        if (std::string(argv[i]) == "--cublas-only") {
            correctness = false;
            cublas_benchmark = true;
        }
        if (std::string(argv[i]) == "--benchmark-all") {
            correctness = false;
            benchmark = true;
            cublas_benchmark = true;
        }
    }

    if (correctness && !run_correctness_suite()) return EXIT_FAILURE;
    if (benchmark) run_benchmark();
    if (cublas_benchmark) run_cublas_benchmark();
    return EXIT_SUCCESS;
}
