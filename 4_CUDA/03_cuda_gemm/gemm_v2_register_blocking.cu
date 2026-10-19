/**
 * nvcc -O3 -lineinfo -arch=sm_120 --ptxas-options=-v \
  gemm_v2_register_blocking.cu -o gemm_v2_register_blocking

  nsys profile \
  --trace=cuda \
  --stats=true \
  --force-overwrite=true \
  --output=nsys/gemm_v2_register_blocking \
  -- ./gemm_v2_register_blocking

  ncu --set full \
  --launch-skip 1 \
  --launch-count 1 \
  --force-overwrite \
  --export ncu/gemm_v2_register_blocking.ncu-rep \
  -- ./gemm_v2_register_blocking
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cstdlib>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)           \
                      << " at line " << __LINE__ << std::endl;                \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// 参数配置：
// 一个 Block 计算 128x128 的 C 子块
// 沿着 K 维度的 Tile 大小为 8
// 一个 Thread 计算 8x8 的 C 子块
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8


// 5.4 Register Blocking GEMM Kernel
__global__ void gemm_v2_register_blocking(const float* __restrict__ A, 
                                           const float* __restrict__ B, 
                                           float* __restrict__ C, 
                                           int M, int N, int K) {
    // 1. 分配 Shared Memory
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x; // 0..15
    int ty = threadIdx.y; // 0..15

    // 计算当前 Thread 在 Block 内的线性索引 (0..255)
    int tid = ty * blockDim.x + tx;

    // 2. 分配寄存器累加器 (8x8) 并初始化为 0
    float c_reg[TM][TN] = {0.0f};

    // 3. 分配用于读取 Shared Memory 的寄存器缓存
    float a_reg[TM];
    float b_reg[TN];

    // 4. 计算当前线程负责的 C 矩阵起始行列索引
    int thread_row = ty * TM;
    int thread_col = tx * TN;

    // 协作加载逻辑推导:
    // 一个 Block 需加载 As(128x8=1024 floats) 和 Bs(8x128=1024 floats)
    // 线程数 = 16x16 = 256. 因此每个线程搬运 1024/256 = 4 个 float.
    int a_load_row = tid / BK;
    int a_load_col = tid % BK;
    int a_load_stride = (blockDim.x * blockDim.y) / BK; // 256 / 8 = 32

    int b_load_row = tid / BN;
    int b_load_col = tid % BN;
    int b_load_stride = (blockDim.x * blockDim.y) / BN; // 256 / 128 = 2

    // 5. 主循环：沿 K 轴推进
    for (int bk = 0; bk < K; bk += BK) {

        // --- Step 5.1: 协作加载 A(128*8) Tile 到 As ---
        #pragma unroll
        for (int i = 0; i < BM; i += a_load_stride) {
            int cur_row = by * BM + a_load_row + i;
            int cur_col = bk + a_load_col;
            if (cur_row < M && cur_col < K) {
                As[a_load_row + i][a_load_col] = A[cur_row * K + cur_col];
            } else {
                As[a_load_row + i][a_load_col] = 0.0f;
            }
        }

        // --- Step 5.2: 协作加载 B(8*128) Tile 到 Bs ---
        #pragma unroll
        for (int i = 0; i < BK; i += b_load_stride) {
            int cur_row = bk + b_load_row + i;
            int cur_col = bx * BN + b_load_col;
            if (cur_row < K && cur_col < N) {
                Bs[b_load_row + i][b_load_col] = B[cur_row * N + cur_col];
            } else {
                Bs[b_load_row + i][b_load_col] = 0.0f;
            }
        }

        __syncthreads();

        // --- Step 5.3: 在 Shared Memory 和 Registers 之间做外积计算 ---
        #pragma unroll
        for (int k = 0; k < BK; ++k) {

            // 从 As 加载一列 TM (8) 个元素到 a_reg
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_reg[i] = As[thread_row + i][k];
            }

            // 从 Bs 加载一行 TN (8) 个元素到 b_reg
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                b_reg[j] = Bs[k][thread_col + j];
            }

            // 寄存器外积累加 (8x8 次 FMA)
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    c_reg[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }

        __syncthreads();
    }

    // 6. 将寄存器里的 8x8 结果写回 Global Memory
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int global_row = by * BM + thread_row + i;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int global_col = bx * BN + thread_col + j;
            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = c_reg[i][j];
            }
        }
    }
}

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

    // 网格尺寸配置
    dim3 blockDim(BN / TN, BM / TM); // (16, 16) -> 256 threads
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    // Warm-up
    gemm_v2_register_blocking<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int iterations = 10;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        gemm_v2_register_blocking<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
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