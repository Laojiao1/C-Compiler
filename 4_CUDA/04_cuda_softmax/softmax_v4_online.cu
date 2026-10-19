/**
 * nvcc -O3 -lineinfo -arch=sm_120 --ptxas-options=-v \
  softmax_v4_online.cu -o softmax_v4_online

  nsys profile --trace=cuda --stats=true --force-overwrite=true \
  --output=nsys/softmax_v4_online \
  -- ./softmax_v4_online

  ncu --set full --launch-skip 1 --launch-count 1 --force-overwrite \
  --export ncu/softmax_v4_online.ncu-rep \
  -- ./softmax_v4_online
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)      \
                      << " at line " << __LINE__ << std::endl;          \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

#define WARP_SIZE 32

// 定义 Online Softmax 的状态结构体 (Max, Sum)
struct MDState {
    float m; // 局部/全局最大值
    float d; // 局部/全局 Exp 累加和
};

// Online Softmax 的状态合并函数 (Online Reduction)
// 结合两个局部状态 (m_a, d_a) 与 (m_b, d_b) -> 合并为 (m_new, d_new)
__device__ __forceinline__ MDState combine_md(MDState a, MDState b) {
    if (a.m < b.m) {
        return {b.m, a.d * expf(a.m - b.m) + b.d};
    } else {
        return {a.m, b.d * expf(b.m - a.m) + a.d};
    }
}

// Warp 级的 Online MDState 规约 (基于 Register Shuffle)
__device__ __forceinline__ MDState warp_reduce_md(MDState state) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        MDState other;
        other.m = __shfl_down_sync(0xffffffff, state.m, mask);
        other.d = __shfl_down_sync(0xffffffff, state.d, mask);
        state = combine_md(state, other);
    }
    return state;
}

// Block 级的 Online MDState 规约
__device__ __forceinline__ MDState block_reduce_md(MDState state, MDState* s_mem) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    // 1. Warp 内 Online 规约
    state = warp_reduce_md(state);

    // 2. 各 Warp 的 0 号线程写入 SMem
    if (lane_id == 0) {
        s_mem[warp_id] = state;
    }
    __syncthreads();

    // 3. 由第 0 个 Warp 汇总各 Warp 的 MDState
    int num_warps = blockDim.x / WARP_SIZE;
    MDState block_state = (threadIdx.x < num_warps) ? s_mem[lane_id] : MDState{-FLT_MAX, 0.0f};

    if (warp_id == 0) {
        block_state = warp_reduce_md(block_state);
    }

    if (threadIdx.x == 0) {
        s_mem[0] = block_state;
    }
    __syncthreads();

    return s_mem[0];
}

// Single-Pass State Tracking Online Softmax Kernel
__global__ void softmax_online_kernel(const float* __restrict__ input, float* __restrict__ output, int M, int N) {
    extern __shared__ MDState s_mem_md[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (row >= M) return;

    const float* row_input = input + row * N;
    float* row_output = output + row * N;

    int num_vec = N / 4;
    int tail_start = num_vec * 4;

    // 阶段 1: 单次遍历 (Single-Pass Load)，流式更新线程局部的 (m, d) 状态
    MDState thread_md = {-FLT_MAX, 0.0f};

    const float4* vec_input = reinterpret_cast<const float4*>(row_input);
    for (int idx = tid; idx < num_vec; idx += block_size) {
        float4 val4 = vec_input[idx];
        
        // 依次将 4 个矢量化元素流式融合进 thread_md
        thread_md = combine_md(thread_md, {val4.x, 1.0f});
        thread_md = combine_md(thread_md, {val4.y, 1.0f});
        thread_md = combine_md(thread_md, {val4.z, 1.0f});
        thread_md = combine_md(thread_md, {val4.w, 1.0f});
    }

    // 处理尾部元素
    for (int col = tail_start + tid; col < N; col += block_size) {
        thread_md = combine_md(thread_md, {row_input[col], 1.0f});
    }

    
    // 阶段 2: 块级分层规约，得到整行的全局 (row_max, row_sum) 状态
    MDState row_md = block_reduce_md(thread_md, s_mem_md);

    float row_max = row_md.m;
    float inv_sum = 1.0f / row_md.d;


    // 阶段 3: 利用最终状态进行归一化并写回
    float4* vec_output = reinterpret_cast<float4*>(row_output);
    for (int idx = tid; idx < num_vec; idx += block_size) {
        float4 val4 = vec_input[idx];
        float4 out4;
        out4.x = expf(val4.x - row_max) * inv_sum;
        out4.y = expf(val4.y - row_max) * inv_sum;
        out4.z = expf(val4.z - row_max) * inv_sum;
        out4.w = expf(val4.w - row_max) * inv_sum;
        vec_output[idx] = out4;
    }

    for (int col = tail_start + tid; col < N; col += block_size) {
        row_output[col] = expf(row_input[col] - row_max) * inv_sum;
    }
}

void cpu_softmax(const float* input, float* output, int M, int N, bool safe = true) {
    for (int r = 0; r < M; ++r) {
        float max_val = safe ? -FLT_MAX : 0.0f;
        if (safe) {
            for (int c = 0; c < N; ++c) {
                max_val = std::max(max_val, input[r * N + c]);
            }
        }
        float sum_val = 0.0f;
        for (int c = 0; c < N; ++c) {
            float exp_val = std::exp(input[r * N + c] - max_val);
            output[r * N + c] = exp_val;
            sum_val += exp_val;
        }
        for (int c = 0; c < N; ++c) {
            output[r * N + c] /= sum_val;
        }
    }
}

bool check_accuracy(const float* h_out_gpu, const float* h_out_cpu, int total_elements, float atol = 1e-4f) {
    for (int i = 0; i < total_elements; ++i) {
        if (std::isnan(h_out_gpu[i]) || std::isinf(h_out_gpu[i])) {
            std::cerr << "Validation failed: GPU output contains NaN/Inf at index " << i << std::endl;
            return false;
        }
        if (std::abs(h_out_gpu[i] - h_out_cpu[i]) > atol) {
            std::cerr << "Validation failed at index " << i << ": GPU=" << h_out_gpu[i] 
                      << ", CPU=" << h_out_cpu[i] << ", Diff=" << std::abs(h_out_gpu[i] - h_out_cpu[i]) << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    const int M = 4096;
    const int N = 2048;
    const int total_elements = M * N;
    const size_t matrix_size = total_elements * sizeof(float);

    std::cout << "=== 4.4.5 Online Softmax Benchmark ===" << std::endl;
    std::cout << "Matrix Shape: [" << M << ", " << N << "]" << std::endl;

    std::vector<float> h_input(total_elements);
    std::vector<float> h_output_gpu(total_elements);
    std::vector<float> h_output_cpu(total_elements);

    for (int i = 0; i < total_elements; ++i) {
        h_input[i] = static_cast<float>(rand() % 1000) / 10.0f - 50.0f;
    }

    cpu_softmax(h_input.data(), h_output_cpu.data(), M, N, true);

    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, matrix_size));
    CHECK_CUDA(cudaMalloc(&d_output, matrix_size));
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), matrix_size, cudaMemcpyHostToDevice));

    int threads_per_block = 256;
    int blocks_per_grid = M;
    // 分配储存 MDState 结构体的 Shared Memory
    size_t shared_mem_size = (threads_per_block / WARP_SIZE) * sizeof(MDState);

    for (int i = 0; i < 5; ++i) {
        softmax_online_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(d_input, d_output, M, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int iterations = 50;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        softmax_online_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(d_input, d_output, M, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iterations;

    CHECK_CUDA(cudaMemcpy(h_output_gpu.data(), d_output, matrix_size, cudaMemcpyDeviceToHost));
    bool passed = check_accuracy(h_output_gpu.data(), h_output_cpu.data(), total_elements);

    std::cout << "Accuracy Check: " << (passed ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Average Execution Time: " << avg_ms << " ms" << std::endl;

    double total_bytes = 3.0 * matrix_size; 
    double bandwidth_gbps = (total_bytes / (avg_ms / 1000.0)) / 1e9;
    std::cout << "Effective Memory Bandwidth (v4 Online): " << bandwidth_gbps << " GB/s" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}