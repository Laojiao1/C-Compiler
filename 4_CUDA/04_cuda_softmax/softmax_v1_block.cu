/**
 * nvcc -O3 -lineinfo -arch=sm_120 --ptxas-options=-v \
  softmax_v1_block.cu -o softmax_v1_block

  nsys profile --trace=cuda --stats=true --force-overwrite=true \
  --output=nsys/softmax_v1_block \
  -- ./softmax_v1_block

  ncu --set full --launch-skip 1 --launch-count 1 --force-overwrite \
  --export ncu/softmax_v1_block.ncu-rep \
  -- ./softmax_v1_block
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

// Single Block Row-wise Softmax Kernel
// 1 个 Block 负责 1 行 (M 行 = M 个 Blocks)
// 使用 Shared Memory 进行 Block 内的树状规约 (Tree Reduction)
__global__ void softmax_block_kernel(const float* __restrict__ input, float* __restrict__ output, int M, int N) {
    // 动态分配 Shared Memory，用于树状规约中间数据存储
    extern __shared__ float sdata[];

    int row = blockIdx.x; // 1 个 Block 对应 1 行
    int tid = threadIdx.x; // Block 内部的线程 ID
    int block_size = blockDim.x;

    if (row >= M) return;

    const float* row_input = input + row * N;
    float* row_output = output + row * N;

    // ----- 阶段 1: 寻找行最大值 (Row Max) -----
    // 每个线程首先在寄存器中计算自己负责的元素的局部最大值 (Block-stride Loop)
    float thread_max = -FLT_MAX;
    for (int col = tid; col < N; col += block_size) {
        thread_max = fmaxf(thread_max, row_input[col]);
    }

    // 将线程局部最大值写入 Shared Memory
    sdata[tid] = thread_max;
    __syncthreads(); // 必须同步，确保所有线程都将 thread_max 写入了 SMem

    // 在 Shared Memory 中进行折半树状规约 (Tree Reduction)
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads(); // 每一层规约都需要块内同步，避免读写冲突 (Bank Conflict / Race Condition)
    }

    // 规约结束后，sdata[0] 即为该行的全局最大值
    float row_max = sdata[0];


    // ----- 阶段 2: 计算 exp(x - row_max) 并求行和 (Row Sum) -----
    // 每个线程再次遍历自己负责的数据，在寄存器中累加求和 (FP32 Accumulation)
    float thread_sum = 0.0f;
    for (int col = tid; col < N; col += block_size) {
        thread_sum += expf(row_input[col] - row_max);
    }

    // 将线程局部和写入 Shared Memory
    sdata[tid] = thread_sum;
    __syncthreads();

    // 在 Shared Memory 中再次进行折半树状规约求 Sum
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // 规约结束后，sdata[0] 即为该行的 ExpSum
    float row_sum = sdata[0];
    float inv_sum = 1.0f / row_sum; // 预先计算倒数，减少重复除法运算


    // ----- 阶段 3: 归一化并写回 Global Memory -----
    for (int col = tid; col < N; col += block_size) {
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
    const int N = 2048;
    const int total_elements = M * N;
    const size_t matrix_size = total_elements * sizeof(float);

    std::cout << "=== 4.4.2 Row-wise Single Block Softmax Benchmark ===" << std::endl;
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

    // 1 个 Block 负责 1 行，总共 M 个 Block
    int threads_per_block = 256;
    int blocks_per_grid = M;
    size_t shared_mem_size = threads_per_block * sizeof(float);

    // 预热
    for (int i = 0; i < 5; ++i) {
        softmax_block_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(d_input, d_output, M, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int iterations = 50;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        softmax_block_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(d_input, d_output, M, N);
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

    // 有效内存带宽计算:
    // v1 版本对 DRAM 的物理读写仅包含: 读 2 次 X 矩阵 + 写 1 次 Out 矩阵 (忽略小规模 SMem 交互)
    // 注意: 计算实际 DRAM 有效带宽时，数据量为 3.0 * matrix_size
    double total_bytes = 3.0 * matrix_size; 
    double bandwidth_gbps = (total_bytes / (avg_ms / 1000.0)) / 1e9;
    std::cout << "Effective Memory Bandwidth (v1 Block): " << bandwidth_gbps << " GB/s" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}