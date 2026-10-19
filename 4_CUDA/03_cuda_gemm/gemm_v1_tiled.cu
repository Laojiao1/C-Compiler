/**
 * nvcc -O3 -lineinfo -arch=sm_120 --ptxas-options=-v \
  gemm_v1_tiled.cu -o gemm_v1_tiled
 * 
 * nsys profile --trace=cuda --stats=true --force-overwrite=true \
  --output=nsys/gemm_v1_tiled -- ./gemm_v1_tiled

  ncu --set full --launch-skip 1 --launch-count 1 --force-overwrite \
  --export ncu/gemm_v1_tiled.ncu-rep -- ./gemm_v1_tiled
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cstdlib>

#include "../config.h"
#include "../cuda_ops.cuh"

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)           \
                      << " at line " << __LINE__ << std::endl;                \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

constexpr int TILE_SIZE = gpu_config::gemm::kTileM;
static_assert(gpu_config::gemm::kTileM == gpu_config::gemm::kTileN &&
              gpu_config::gemm::kTileM == gpu_config::gemm::kTileK,
              "gemm_v1_tiled currently requires a square tile");


// Shared Memory Tiling GEMM Kernel: C = A * B
__global__ void gemm_v1_tiled(const float* __restrict__ A, 
                              const float* __restrict__ B, 
                              float* __restrict__ C, 
                              int M, int N, int K) {
    // 静态分配 Shared Memory                            
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // 当前线程负责的 C 矩阵全局行列号
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    // 沿 K 维度按 TILE_SIZE 分块推进
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for(int i = 0; i < num_tiles; ++i) {

        // 协作加载 A 和 B 的 Tile 到 Shared Memory (处理边界)
        if(row < M && (i * TILE_SIZE + tx) < K) {
            As[ty][tx] = A[row * K + (i * TILE_SIZE + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }

        if((i * TILE_SIZE + ty) < K && col < N) {
            Bs[ty][tx] = B[(i * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        // 同步：等待整个 Block 加载完毕
        __syncthreads();

        // 在 Shared Memory 中做局部点积累加
        #pragma unroll
        for(int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // 同步：等待所有线程计算完毕，准备加载下一个 Tile
        __syncthreads();
    }
    
    // 写回结果
    if(row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Attention 的 K 以 [N, d] 行主序保存。这个变体在搬入 Shared Memory 时
// 原地完成 B^T 视图，避免为了 QK^T 额外物化一份转置矩阵。
__global__ void gemm_v1_tiled_transpose_b(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE_SIZE + ty;
    const int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        const int a_col = tile * TILE_SIZE + tx;
        const int b_col = tile * TILE_SIZE + ty;

        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        // Bs[k_local][n_local] 对应 B[n_global][k_global]。
        Bs[ty][tx] = (col < N && b_col < K) ? B[col * K + b_col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int inner = 0; inner < TILE_SIZE; ++inner) {
            sum += As[ty][inner] * Bs[inner][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

cudaError_t cuda_ops::launch_gemm_v1_tiled(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k,
    bool transpose_b,
    cudaStream_t stream) {
    if (a == nullptr || b == nullptr || c == nullptr || m <= 0 || n <= 0 || k <= 0) {
        return cudaErrorInvalidValue;
    }

    const dim3 block(TILE_SIZE, TILE_SIZE);
    const dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE,
                    (m + TILE_SIZE - 1) / TILE_SIZE);
    if (transpose_b) {
        gemm_v1_tiled_transpose_b<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
    } else {
        gemm_v1_tiled<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
    }
    return cudaPeekAtLastError();
}

#ifndef CUDA_OPS_LIBRARY

// CPU 采样校验
bool verify_samples(const float* A, const float* B, const float* device_C, 
                    int M, int N, int K, int num_samples = 1000) {
    for (int s = 0; s < num_samples; ++s) {
        int row = rand() % M;
        int col = rand() % N;

        float cpu_sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            cpu_sum += A[row * K + k] * B[k * N + col];
        }

        float gpu_val = device_C[row * N + col];
        float diff = std::abs(cpu_sum - gpu_val);
        float ref = std::abs(cpu_sum);

        if (diff / (ref + 1e-5f) > 1e-3f) {
            std::cout << "Mismatch at C[" << row << "][" << col << "]: "
                      << "CPU=" << cpu_sum << ", GPU=" << gpu_val << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;

    std::cout << "Matrix Dimensions: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);

    srand(2024);
    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, 
                 (M + TILE_SIZE - 1) / TILE_SIZE);

    // Warm-up
    gemm_v1_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int iterations = 10;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        gemm_v1_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_time_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&total_time_ms, start, stop));
    float avg_time_ms = total_time_ms / iterations;

    double flops = 2.0 * M * N * K;
    double tflops = (flops / (avg_time_ms / 1000.0)) / 1e12;

    std::cout << "Kernel Execution Time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "Performance: " << tflops << " TFLOPS" << std::endl;

    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));
    
    std::cout << "Verifying correctness against CPU (sampling 1000 points)..." << std::endl;
    if (verify_samples(h_A.data(), h_B.data(), h_C.data(), M, N, K)) {
        std::cout << "Result Verification PASSED!" << std::endl;
    } else {
        std::cout << "Result Verification FAILED!" << std::endl;
    }

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
#endif
