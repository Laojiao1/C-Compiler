/**
 * nvcc -O3 -lineinfo -arch=sm_120 --ptxas-options=-v \
  softmax_v0_naive.cu -o softmax_v0_naive

  nsys profile --trace=cuda --stats=true --force-overwrite=true \
  --output=nsys/softmax_v0_naive \
  -- ./softmax_v0_naive

  ncu --set full --launch-skip 1 --launch-count 1 --force-overwrite \
  --export ncu/softmax_v0_naive.ncu-rep \
  -- ./softmax_v0_naive
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

// Kernel 1: 寻找每行的最大值，用于 Safe Softmax 减最大值防止 expf 溢出
// 瓶颈：每个线程串行扫描一行 N 个元素，未利用 Block 内线程级的并行规约
__global__ void find_row_max_kernel(const float* __restrict__ input, float* __restrict__ row_max, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float max_val = -FLT_MAX;
        for (int col = 0; col < N; ++col) {
            float val = input[row * N + col];
            if (val > max_val) {
                max_val = val;
            }
        }
        // 将局部最大值写回 Global Memory 临时数组，带来额外 DRAM Store 开销
        row_max[row] = max_val;
    }
}

// Kernel 2: 读取原始输入和上一步算出的 row_max，计算 exp(x - max) 并对行求和
// 瓶颈：被迫将中间结果 output_exp 完整写回 Global Memory，造成严重的内存写带宽浪费
__global__ void exp_and_sum_kernel(const float* __restrict__ input, 
                                   const float* __restrict__ row_max, 
                                   float* __restrict__ output_exp, 
                                   float* __restrict__ row_sum, 
                                   int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        // 从 Global Memory 读取该行对应的最大值
        float max_val = row_max[row];
        float sum_val = 0.0f;
        for (int col = 0; col < N; ++col) {
            int idx = row * N + col;
            float exp_val = expf(input[idx] - max_val);
            // 强行物化中间矩阵，将 exp 后的值写会 Global Memory 供下一个 Kernel 使用
            output_exp[idx] = exp_val;
            sum_val += exp_val;
        }
        // 将行和写回 Global Memory
        row_sum[row] = sum_val;
    }
}

// Kernel 3: 读取 output_exp 和 row_sum，逐元素除以行和完成归一化
// 瓶颈：再次从 Global Memory 重新读取上一阶段写出的 output_exp，增加 DRAM Load 开销
__global__ void normalize_kernel(const float* __restrict__ output_exp, 
                                 const float* __restrict__ row_sum, 
                                 float* __restrict__ output, 
                                 int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float sum_val = row_sum[row];
        // 取倒数改除法为乘法，优化 ALU 指令执行周期
        float inv_sum = 1.0f / sum_val;
        for (int col = 0; col < N; ++col) {
            int idx = row * N + col;
            output[idx] = output_exp[idx] * inv_sum;
        }
    }
}

// CPU 正确性验证
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

// 精确度检查
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
    const size_t vector_size = M * sizeof(float);

    std::cout << "=== 4.4.1 Naive Softmax Benchmark ===" << std::endl;
    std::cout << "Matrix Shape: [" << M << ", " << N << "]" << std::endl;

    std::vector<float> h_input(total_elements);
    std::vector<float> h_output_gpu(total_elements);
    std::vector<float> h_output_cpu(total_elements);

    for (int i = 0; i < total_elements; ++i) {
        h_input[i] = static_cast<float>(rand() % 1000) / 10.0f - 50.0f;
    }

    cpu_softmax(h_input.data(), h_output_cpu.data(), M, N, true);

    float *d_input, *d_output, *d_exp_temp, *d_max_temp, *d_sum_temp;
    CHECK_CUDA(cudaMalloc(&d_input, matrix_size));
    CHECK_CUDA(cudaMalloc(&d_output, matrix_size));
    // 为多 Kernel 中转物理矩阵分配 Global Memory 空间
    CHECK_CUDA(cudaMalloc(&d_exp_temp, matrix_size));
    CHECK_CUDA(cudaMalloc(&d_max_temp, vector_size));
    CHECK_CUDA(cudaMalloc(&d_sum_temp, vector_size));

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), matrix_size, cudaMemcpyHostToDevice));

    // 当前策略：按行映射线程，网格仅在 M 维度展开
    int threads_per_block = 256;
    int blocks_per_grid = (M + threads_per_block - 1) / threads_per_block;

    for (int i = 0; i < 5; ++i) {
        find_row_max_kernel<<<blocks_per_grid, threads_per_block>>>(d_input, d_max_temp, M, N);
        exp_and_sum_kernel<<<blocks_per_grid, threads_per_block>>>(d_input, d_max_temp, d_exp_temp, d_sum_temp, M, N);
        normalize_kernel<<<blocks_per_grid, threads_per_block>>>(d_exp_temp, d_sum_temp, d_output, M, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int iterations = 50;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        // 多 Kernel 启动带来的 Host-Device 通信开销与多次 Global Memory 读写开销
        find_row_max_kernel<<<blocks_per_grid, threads_per_block>>>(d_input, d_max_temp, M, N);
        exp_and_sum_kernel<<<blocks_per_grid, threads_per_block>>>(d_input, d_max_temp, d_exp_temp, d_sum_temp, M, N);
        normalize_kernel<<<blocks_per_grid, threads_per_block>>>(d_exp_temp, d_sum_temp, d_output, M, N);
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

    // 有效带宽计算：包含多次冗余的 DRAM 矩阵 Load/Store
    double total_bytes = 6.0 * matrix_size; 
    double bandwidth_gbps = (total_bytes / (avg_ms / 1000.0)) / 1e9;
    std::cout << "Effective Memory Bandwidth (Naive): " << bandwidth_gbps << " GB/s" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_exp_temp);
    cudaFree(d_max_temp);
    cudaFree(d_sum_temp);

    return 0;
}