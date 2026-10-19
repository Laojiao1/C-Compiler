/**
 * nvcc -O3 -lineinfo -arch=sm_120 --ptxas-options=-v \
  softmax_v3_fused.cu -o softmax_v3_fused

  nsys profile --trace=cuda --stats=true --force-overwrite=true \
  --output=nsys/softmax_v3_fused \
  -- ./softmax_v3_fused

  ncu --set full --launch-skip 1 --launch-count 1 --force-overwrite \
  --export ncu/softmax_v3_fused.ncu-rep \
  -- ./softmax_v3_fused
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>

#include "../config.h"
#include "../cuda_ops.cuh"

#define CHECK_CUDA(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)      \
                      << " at line " << __LINE__ << std::endl;          \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

constexpr int WARP_SIZE = gpu_config::kWarpSize;

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, mask));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, mask);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_max(float val, float* s_mem) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    val = warp_reduce_max(val);

    if (lane_id == 0) {
        s_mem[warp_id] = val;
    }
    __syncthreads();

    int num_warps = blockDim.x / WARP_SIZE;
    float block_max = (threadIdx.x < num_warps) ? s_mem[lane_id] : -FLT_MAX;

    if (warp_id == 0) {
        block_max = warp_reduce_max(block_max);
    }

    if (threadIdx.x == 0) {
        s_mem[0] = block_max;
    }
    __syncthreads();

    const float result = s_mem[0];
    // s_mem 会被下一次 reduction 立即复用。前一个 barrier 只保证结果已经
    // 写好；这里再同步一次，保证所有线程都完成读取，避免快线程提前覆盖。
    __syncthreads();
    return result;
}

__device__ __forceinline__ float block_reduce_sum(float val, float* s_mem) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);

    if (lane_id == 0) {
        s_mem[warp_id] = val;
    }
    __syncthreads();

    int num_warps = blockDim.x / WARP_SIZE;
    float block_sum = (threadIdx.x < num_warps) ? s_mem[lane_id] : 0.0f;

    if (warp_id == 0) {
        block_sum = warp_reduce_sum(block_sum);
    }

    if (threadIdx.x == 0) {
        s_mem[0] = block_sum;
    }
    __syncthreads();

    const float result = s_mem[0];
    __syncthreads();
    return result;
}

// Fused Scale + Causal Mask + Softmax Kernel
// 模拟 Self-Attention 场景: 矩阵大小 [M, N]，其中 M 和 N 通常代表 Sequence Length
__global__ void fused_scale_mask_softmax_kernel(const float* __restrict__ input, 
                                                 float* __restrict__ output, 
                                                 float scale, 
                                                 int M, int N,
                                                 bool causal) {
    extern __shared__ float s_mem[];

    int row = blockIdx.x; // 当前行的 Token 索引
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (row >= M) return;

    const float* row_input = input + row * N;
    float* row_output = output + row * N;

    // 只有行跨度为 16 Byte 整数倍时，每一行的 float4 起点才都对齐。
    // 否则退回标量路径，不能只处理尾部，因为第二行起点可能未对齐。
    int num_vec = (N % 4 == 0) ? N / 4 : 0;
    int tail_start = num_vec * 4;


    // 阶段 1: 在 Global Memory 加载的同时，融合 Scale 与 Causal Mask，并求局部 Max
    // Causal Mask 规则: 当 col > row (Future Token) 时，加上 -1e4f 掩码
    float thread_max = -FLT_MAX;

    const float4* vec_input = reinterpret_cast<const float4*>(row_input);
    for (int idx = tid; idx < num_vec; idx += block_size) {
        float4 val4 = vec_input[idx];
        int col_base = idx * 4;

        // 寄存器内顺手融合 Scale & Causal Mask
        val4.x = (causal && col_base + 0 > row) ? -FLT_MAX : val4.x * scale;
        val4.y = (causal && col_base + 1 > row) ? -FLT_MAX : val4.y * scale;
        val4.z = (causal && col_base + 2 > row) ? -FLT_MAX : val4.z * scale;
        val4.w = (causal && col_base + 3 > row) ? -FLT_MAX : val4.w * scale;

        thread_max = fmaxf(thread_max, fmaxf(fmaxf(val4.x, val4.y), fmaxf(val4.z, val4.w)));
    }

    // 尾部元素的融合与 Max 规约
    for (int col = tail_start + tid; col < N; col += block_size) {
        float val = (causal && col > row) ? -FLT_MAX : row_input[col] * scale;
        thread_max = fmaxf(thread_max, val);
    }

    // 块级分层规约求整行全局 Max
    float row_max = block_reduce_max(thread_max, s_mem);


    // 阶段 2: 再次加载并融合计算 exp(x * scale + mask - row_max) 求行和
    float thread_sum = 0.0f;

    for (int idx = tid; idx < num_vec; idx += block_size) {
        float4 val4 = vec_input[idx];
        int col_base = idx * 4;

        val4.x = (causal && col_base + 0 > row) ? -FLT_MAX : val4.x * scale;
        val4.y = (causal && col_base + 1 > row) ? -FLT_MAX : val4.y * scale;
        val4.z = (causal && col_base + 2 > row) ? -FLT_MAX : val4.z * scale;
        val4.w = (causal && col_base + 3 > row) ? -FLT_MAX : val4.w * scale;

        thread_sum += expf(val4.x - row_max);
        thread_sum += expf(val4.y - row_max);
        thread_sum += expf(val4.z - row_max);
        thread_sum += expf(val4.w - row_max);
    }

    for (int col = tail_start + tid; col < N; col += block_size) {
        float val = (causal && col > row) ? -FLT_MAX : row_input[col] * scale;
        thread_sum += expf(val - row_max);
    }

    float row_sum = block_reduce_sum(thread_sum, s_mem);
    float inv_sum = 1.0f / row_sum;


    // 阶段 3: 归一化并写回概率矩阵
    float4* vec_output = reinterpret_cast<float4*>(row_output);
    for (int idx = tid; idx < num_vec; idx += block_size) {
        float4 val4 = vec_input[idx];
        int col_base = idx * 4;

        val4.x = (causal && col_base + 0 > row) ? -FLT_MAX : val4.x * scale;
        val4.y = (causal && col_base + 1 > row) ? -FLT_MAX : val4.y * scale;
        val4.z = (causal && col_base + 2 > row) ? -FLT_MAX : val4.z * scale;
        val4.w = (causal && col_base + 3 > row) ? -FLT_MAX : val4.w * scale;

        float4 out4;
        out4.x = expf(val4.x - row_max) * inv_sum;
        out4.y = expf(val4.y - row_max) * inv_sum;
        out4.z = expf(val4.z - row_max) * inv_sum;
        out4.w = expf(val4.w - row_max) * inv_sum;

        vec_output[idx] = out4;
    }

    for (int col = tail_start + tid; col < N; col += block_size) {
        float val = (causal && col > row) ? -FLT_MAX : row_input[col] * scale;
        row_output[col] = expf(val - row_max) * inv_sum;
    }
}

cudaError_t cuda_ops::launch_fused_scale_mask_softmax(
    const float* input,
    float* output,
    float scale,
    int rows,
    int cols,
    bool causal,
    cudaStream_t stream) {
    if (input == nullptr || output == nullptr || rows <= 0 || cols <= 0 || scale <= 0.0f) {
        return cudaErrorInvalidValue;
    }

    constexpr int threads = gpu_config::softmax::kThreads;
    constexpr std::size_t shared_bytes =
        (threads / gpu_config::kWarpSize) * sizeof(float);
    fused_scale_mask_softmax_kernel<<<rows, threads, shared_bytes, stream>>>(
        input, output, scale, rows, cols, causal);
    return cudaPeekAtLastError();
}

#ifndef CUDA_OPS_LIBRARY

// CPU Baseline (Golden Reference): 带 Scale 和 Causal Mask 的 Softmax
void cpu_fused_softmax(const float* input, float* output, float scale, int M, int N) {
    for (int r = 0; r < M; ++r) {
        float max_val = -FLT_MAX;
        for (int c = 0; c < N; ++c) {
            float val = (c > r) ? -FLT_MAX : input[r * N + c] * scale;
            max_val = std::max(max_val, val);
        }
        float sum_val = 0.0f;
        for (int c = 0; c < N; ++c) {
            float val = (c > r) ? -FLT_MAX : input[r * N + c] * scale;
            float exp_val = std::exp(val - max_val);
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
    const float scale = 0.125f; // 对应 1 / sqrt(64)
    const int total_elements = M * N;
    const size_t matrix_size = total_elements * sizeof(float);

    std::cout << "=== 4.4.4 Fused Scale + Mask + Softmax Benchmark ===" << std::endl;
    std::cout << "Matrix Shape: [" << M << ", " << N << "]" << std::endl;

    std::vector<float> h_input(total_elements);
    std::vector<float> h_output_gpu(total_elements);
    std::vector<float> h_output_cpu(total_elements);

    for (int i = 0; i < total_elements; ++i) {
        h_input[i] = static_cast<float>(rand() % 1000) / 10.0f - 50.0f;
    }

    cpu_fused_softmax(h_input.data(), h_output_cpu.data(), scale, M, N);

    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, matrix_size));
    CHECK_CUDA(cudaMalloc(&d_output, matrix_size));
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), matrix_size, cudaMemcpyHostToDevice));

    int threads_per_block = 256;
    int blocks_per_grid = M;
    size_t shared_mem_size = (threads_per_block / WARP_SIZE) * sizeof(float);

    for (int i = 0; i < 5; ++i) {
        fused_scale_mask_softmax_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(d_input, d_output, scale, M, N, true);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int iterations = 50;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        fused_scale_mask_softmax_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(d_input, d_output, scale, M, N, true);
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
    std::cout << "Effective Memory Bandwidth (v3 Fused): " << bandwidth_gbps << " GB/s" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
#endif
