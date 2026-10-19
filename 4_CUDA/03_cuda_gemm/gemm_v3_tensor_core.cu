/**
 * nvcc -O3 -lineinfo -arch=sm_120 --ptxas-options=-v \
  gemm_v3_tensor_core.cu -o gemm_v3_tensor_core

  nsys profile \
  --trace=cuda \
  --stats=true \
  --force-overwrite=true \
  --output=nsys/gemm_v3_tensor_core \
  -- ./gemm_v3_tensor_core

  ncu --set full \
  --launch-skip 1 \
  --launch-count 1 \
  --force-overwrite \
  --export ncu/gemm_v3_tensor_core.ncu-rep \
  -- ./gemm_v3_tensor_core
 */
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <mma.h>             // WMMA 头文件
#include <cuda_fp16.h>        // half 精度头文件
#include <cmath>

using namespace nvcuda;

// 定义 Block 与 WMMA Tile 维度
#define BM 128
#define BN 128
#define BK 32

// WMMA Tile 尺寸
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// 辅助工具：GPU 错误检查
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

/**
 * GEMM v3.0: Tensor Core GEMM via WMMA API
 * 输入: half (FP16), 累加器: float (FP32)
 */
__global__ void gemm_v3_wmma(const half* __restrict__ A,
                             const half* __restrict__ B,
                             float* __restrict__ C,
                             int M, int N, int K) {
    // Shared Memory 开辟 (FP16)
    __shared__ half As[BM][BK];
    __shared__ half Bs[BK][BN];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tid = ty * blockDim.x + tx; // 0..255 (共 8 个 Warp)
    int warp_id = tid / 32;        // 0..7

    // 在 Block (128x128) 空间中分配 8 个 Warp
    // 每个 Block 沿 M 排 4 个 Warp，沿 N 排 2 个 Warp (4 x 2 = 8 个 Warp，为什么要这么分？—— 因为这是一个约定，是综合考虑后的结果)
    int warp_m = warp_id / 2; // 0..3 (控制 M 方向: 4 * 32 = 128)
    int warp_n = warp_id % 2; // 0..1 (控制 N 方向: 2 * 64 = 128)

    // 每一个 Warp 负责 32x64 的 C 矩阵区域 (即 2x4 个 WMMA_16x16 Tile)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[4];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[2][4];

    // 初始化 C 累加器 Fragment 为 0
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }


    // ----- 1. 合作搬运索引 (使用 half2/half4 快速填充 Shared Memory) ----- 
    // As[128][32]: 256 线程分担 128*32 = 4096 个 half -> 每人搬 16 个 half
    int a_load_row = tid / 2;  // 0..127
    int a_load_col = (tid % 2) * 16; // 0, 16 -> 2 个线程平分 2*16=32 列

    // Bs[32][128]: 256 线程分担 32*128 = 4096 个 half -> 每人搬 16 个 half
    int b_load_row = tid / 8;  // 0..31
    int b_load_col = (tid % 8) * 16; // 0, 16, ..., 112 -> 8 个线程平分 128 列

    // ----- 2. 计算矩阵乘累加 -----
    // ----- A. 外层循环：主循环沿 K 步进 BK=32，共需要 K/BK = 2048/32 = 64 次 -----
    for (int bk = 0; bk < K; bk += BK) {
        // 从 Global Memory 搬运到 Shared Memory (向量化拷贝以提升吞吐)
        int global_a_row = by * BM + a_load_row;
        int global_a_col = bk + a_load_col;
        if (global_a_row < M && global_a_col < K) {
            #pragma unroll
            for (int vec = 0; vec < 16; vec += 8) { // 每次搬运 8 个 half (128-bit)
                *reinterpret_cast<float4*>(&As[a_load_row][a_load_col + vec]) =
                    *reinterpret_cast<const float4*>(&A[global_a_row * K + (global_a_col + vec)]);
            }
        } else {
            #pragma unroll
            for (int v = 0; v < 16; ++v) 
                As[a_load_row][a_load_col + v] = __float2half(0.0f);
        }

        int global_b_row = bk + b_load_row;
        int global_b_col = bx * BN + b_load_col;
        if (global_b_row < K && global_b_col < N) {
            #pragma unroll
            for (int vec = 0; vec < 16; vec += 8) {
                *reinterpret_cast<float4*>(&Bs[b_load_row][b_load_col + vec]) =
                    *reinterpret_cast<const float4*>(&B[global_b_row * N + (global_b_col + vec)]);
            }
        } else {
            #pragma unroll
            for (int v = 0; v < 16; ++v) 
                Bs[b_load_row][b_load_col + v] = __float2half(0.0f);
        }

        __syncthreads();

        // ----- B. 中层循环：使用 WMMA 推进内部 Tile 阵列乘加计算 (Tensor Core) ，共需：BK / WMMA_K = 32/16 = 2 次循环-----
        // BK=32 包含 2 个 WMMA_K=16 的子阶段
        #pragma unroll
        for (int k_step = 0; k_step < BK / WMMA_K; ++k_step) {
            // 加载 A 片段 (来自 Shared Memory As)
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                int smem_a_row = warp_m * 32 + i * WMMA_M;
                int smem_a_col = k_step * WMMA_K;
                wmma::load_matrix_sync(a_frag[i], &As[smem_a_row][smem_a_col], BK);
            }

            // 加载 B 片段 (来自 Shared Memory Bs)
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int smem_b_row = k_step * WMMA_K;
                int smem_b_col = warp_n * 64 + j * WMMA_N;
                wmma::load_matrix_sync(b_frag[j], &Bs[smem_b_row][smem_b_col], BN);
            }

            // ----- C. 内层循环：调用 Tensor Core 硬件乘加 (WMMA MMA)：共需 2*4=8 次调用 -----
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }

        __syncthreads();
    }


    // ----- 3. 将 WMMA 计算结果 Fragment 写回 Global Memory C ----- 
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int global_c_row = by * BM + warp_m * 32 + i * WMMA_M;
            int global_c_col = bx * BN + warp_n * 64 + j * WMMA_N;
            if (global_c_row < M && global_c_col < N) {
                wmma::store_matrix_sync(&C[global_c_row * N + global_c_col],
                                        c_frag[i][j],
                                        N,
                                        wmma::mem_row_major);
            }
        }
    }
}

int main() {
    const int M = 2048, N = 2048, K = 2048;
    size_t bytes_A = M * K * sizeof(half);
    size_t bytes_B = K * N * sizeof(half);
    size_t bytes_C = M * N * sizeof(float);

    std::vector<half> h_A(M * K), h_B(K * N);
    std::vector<float> h_C(M * N);

    for (int i = 0; i < M * K; ++i) h_A[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
    for (int i = 0; i < K * N; ++i) h_B[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);

    half *d_A, *d_B;
    float *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));

    // 256 线程/Block (16x16 线程排布，共 8 个 Warp)
    dim3 blockDim(16, 16); 
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    // 预热
    gemm_v3_wmma<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    int iterations = 20;
    for (int i = 0; i < iterations; ++i) {
        gemm_v3_wmma<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    float avg_time = milliseconds / iterations;

    double flops = 2.0 * M * N * K;
    double tflops = (flops / (avg_time / 1000.0)) / 1e12;

    std::cout << "=== Tensor Core (WMMA) GEMM Benchmark ===" << std::endl;
    std::cout << "Execution Time: " << avg_time << " ms" << std::endl;
    std::cout << "Tensor Core Performance: " << tflops << " TFLOPS" << std::endl;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}
