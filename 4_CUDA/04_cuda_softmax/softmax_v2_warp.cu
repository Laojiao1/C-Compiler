/**
 * nvcc -O3 -lineinfo -arch=sm_120 --ptxas-options=-v \
  softmax_v2_warp.cu -o softmax_v2_warp

  nsys profile --trace=cuda --stats=true --force-overwrite=true \
  --output=nsys/softmax_v2_warp \
  -- ./softmax_v2_warp

  ncu --set full --launch-skip 1 --launch-count 1 --force-overwrite \
  --export ncu/softmax_v2_warp.ncu-rep \
  -- ./softmax_v2_warp
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

// Warp 级最大值规约 (基于 Register Shuffle)
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, mask));
    }
    return val;
}

// Warp 级求和规约 (基于 Register Shuffle)
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, mask);
    }
    return val;
}

// Block 级分层规约 - Max
__device__ __forceinline__ float block_reduce_max(float val, float* s_mem) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    // 1. Warp 内规约
    val = warp_reduce_max(val);

    // 2. Warp 0 号线程将局部规约值写入 Shared Memory
    if (lane_id == 0) {
        s_mem[warp_id] = val;
    }
    __syncthreads(); // 仅需一次同步，保证所有 Warp 都写完 SMem

    // 3. 第 0 个 Warp 汇总各 Warp 的结果,num_warp = 256/32 = 8
    // 第 0 个 Warp 的前 8 个线程，把 8 个 Warp 的局部最大值装载进了各自的 block_max 寄存器里
    int num_warps = blockDim.x / WARP_SIZE;
    float block_max = (threadIdx.x < num_warps) ? s_mem[lane_id] : -FLT_MAX;

    if (warp_id == 0) {
        block_max = warp_reduce_max(block_max);
    }

    // 广播最终的 Block Max 给所有线程 (写 SMem 并同步)
    if (threadIdx.x == 0) {
        s_mem[0] = block_max;
    }
    __syncthreads();

    return s_mem[0];
}

// Block 级分层规约 - Sum
__device__ __forceinline__ float block_reduce_sum(float val, float* s_mem) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    // 1. Warp 内规约
    val = warp_reduce_sum(val);

    // 2. Warp 0 号线程写入 Shared Memory
    if (lane_id == 0) {
        s_mem[warp_id] = val;
    }
    __syncthreads();

    // 3. 第 0 个 Warp 汇总
    int num_warps = blockDim.x / WARP_SIZE;
    float block_sum = (threadIdx.x < num_warps) ? s_mem[lane_id] : 0.0f;

    if (warp_id == 0) {
        block_sum = warp_reduce_sum(block_sum);
    }

    if (threadIdx.x == 0) {
        s_mem[0] = block_sum;
    }
    __syncthreads();

    return s_mem[0];
}

// Warp Shuffle + 矢量化访存 (float4) Softmax Kernel
__global__ void softmax_warp_vectorized_kernel(const float* __restrict__ input, float* __restrict__ output, int M, int N) {
    // 共享内存空间大幅缩减：只需保存 Warp 数量的 float 即可 (例如 256 线程 => 8 个 Warp => 32 字节)
    extern __shared__ float s_mem[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (row >= M) return;

    const float* row_input = input + row * N;
    float* row_output = output + row * N;

    int num_vec = N / 4; // float4 可读取的块数
    int tail_start = num_vec * 4; // 尾部非 4 对齐开始索引


    // ----- 阶段 1: 矢量化读取 + 局部 Max -----
    float thread_max = -FLT_MAX;

    // 1.1 矢量化循环 (float4 访存)
    const float4* vec_input = reinterpret_cast<const float4*>(row_input);
    for (int idx = tid; idx < num_vec; idx += block_size) {
        float4 val4 = vec_input[idx];
        thread_max = fmaxf(thread_max, fmaxf(fmaxf(val4.x, val4.y), fmaxf(val4.z, val4.w)));
    }

    // 1.2 处理非 4 整除的尾部元素 (Scalar Tail)
    for (int col = tail_start + tid; col < N; col += block_size) {
        thread_max = fmaxf(thread_max, row_input[col]);
    }

    // 1.3 分层规约计算整行全局 Max
    float row_max = block_reduce_max(thread_max, s_mem);


    // ----- 阶段 2: 矢量化读取 + 局部 ExpSum -----
    float thread_sum = 0.0f;

    // 2.1 矢量化累加 Sum
    for (int idx = tid; idx < num_vec; idx += block_size) {
        float4 val4 = vec_input[idx];
        thread_sum += expf(val4.x - row_max);
        thread_sum += expf(val4.y - row_max);
        thread_sum += expf(val4.z - row_max);
        thread_sum += expf(val4.w - row_max);
    }

    // 2.2 尾部累加 Sum
    for (int col = tail_start + tid; col < N; col += block_size) {
        thread_sum += expf(row_input[col] - row_max);
    }

    // 2.3 分层规约计算整行和
    float row_sum = block_reduce_sum(thread_sum, s_mem);
    float inv_sum = 1.0f / row_sum;


    // ----- 阶段 3: 矢量化写回 Global Memory -----
    float4* vec_output = reinterpret_cast<float4*>(row_output);
    for (int idx = tid; idx < num_vec; idx += block_size) {
        float4 val4 = vec_input[idx];
        float4 out4;
        out4.x = expf(val4.x - row_max) * inv_sum;
        out4.y = expf(val4.y - row_max) * inv_sum;
        out4.z = expf(val4.z - row_max) * inv_sum;
        out4.w = expf(val4.w - row_max) * inv_sum;
        vec_output[idx] = out4; // 一次性矢量化写入 16 字节
    }

    // 尾部标量写回
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

bool check_accuracy(const float* h_out_gpu, const float* h_out_cpu, int total_elements, float atol = 1e-5f) {
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
    const int N = 2048; // 可尝试支持非 4 倍数的序列长度，代码会自动处理尾部
    const int total_elements = M * N;
    const size_t matrix_size = total_elements * sizeof(float);

    std::cout << "=== 4.4.3 Warp Shuffle & Vectorized Softmax Benchmark ===" << std::endl;
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
    // SMem 大小调整为仅分配每 Warp 1 个 float 即可
    size_t shared_mem_size = (threads_per_block / WARP_SIZE) * sizeof(float);

    for (int i = 0; i < 5; ++i) {
        softmax_warp_vectorized_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(d_input, d_output, M, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int iterations = 50;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        softmax_warp_vectorized_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(d_input, d_output, M, N);
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
    std::cout << "Effective Memory Bandwidth (v2 Warp Vectorized): " << bandwidth_gbps << " GB/s" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}