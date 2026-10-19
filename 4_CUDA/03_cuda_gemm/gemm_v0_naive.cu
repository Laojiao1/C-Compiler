/**
 * nvcc -O3 -lineinfo -arch=sm_120 --ptxas-options=-v \
  gemm_v0_naive.cu -o gemm_v0_naive
 * 
 * nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --stats=true --force-overwrite=true --output=nsys/gemm_v0_naive ./gemm_v0_naive
 *
 * ncu --set full --launch-skip 1 --launch-count 1 --force-overwrite \
  --export ncu/gemm_v0_naive.ncu-rep ./gemm_v0_naive
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

// CUDA 错误检查宏
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)           \
                      << " at line " << __LINE__ << std::endl;                \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// 1. Naive GEMM Kernel: C = A * B
// A: M x K, B: K x N, C: M x N (均按 Row-Major 行优先存储)
__global__ 
void gemm_v0_naive(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if(row < M && col < N) {
        float sum = 0.0f;
        for(int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 2. CPU 验证
void gemm_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for(int i = 0; i < M; ++i) {
        for(int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for(int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// 3. 正确性验证函数 (相对误差检查)
bool verify_result(const float* host_C, const float* device_C, int size) {
    for (int i = 0; i < size; ++i) {
        float diff = std::abs(host_C[i] - device_C[i]);
        float ref = std::abs(host_C[i]);
        // 相对误差大于 1e-3 判定为不一致
        if (diff / (ref + 1e-5f) > 1e-3f) {
            std::cout << "Mismatch at index " << i << ": CPU=" << host_C[i] << ", GPU=" << device_C[i] << std::endl;
            return false;
        }
    }
    return true;
}

// 4. 主函数与 Benchmark
int main() {
    // 设定矩阵维度 M = N = K = 2048 (Naive 版本跑 4096 比较慢，2048 足够观察性能)
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;

    std::cout << "Matrix Dimensions: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Host 内存分配与初始化
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    std::vector<float> h_C_ref(M * N, 0.0f);

    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // Device 内存分配
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));

    // 数据 H2D 拷贝
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));

    // 线程网格配置：使用 32x32 的 Thread Block (每个 Block 1024 线程)
    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, 
                 (M + blockDim.y - 1) / blockDim.y);

    // Warm-up 预热 (排除冷启动耗时)
    gemm_v0_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 使用 CUDA Event 准确记录 Kernel 执行时间
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int iterations = 10;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        gemm_v0_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // 计算平均用时
    float total_time_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&total_time_ms, start, stop));
    float avg_time_ms = total_time_ms / iterations;

    // 计算 GFLOPs / TFLOPS
    // 标准 GEMM 的 FLOPs 计算公式: 2 * M * N * K
    double flops = 2.0 * M * N * K;
    double tflops = (flops / (avg_time_ms / 1000.0)) / 1e12;

    std::cout << "Kernel Execution Time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "Performance: " << tflops << " TFLOPS" << std::endl;

    // 数据 D2H 写回并校验正确性
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));

    // 采样 CPU 结果进行验证（因为全量 CPU 算太慢）
    std::cout << "Verifying correctness against CPU..." << std::endl;
    gemm_cpu(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);

    if (verify_result(h_C_ref.data(), h_C.data(), M * N)) {
        std::cout << "Result Verification PASSED!" << std::endl;
    } else {
        std::cout << "Result Verification FAILED!" << std::endl;
    }

    // 清理资源
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}