/**
 * nvcc -O3 -lineinfo -arch=sm_120 --ptxas-options=-v \
  gemm_v2.5_vectorized.cu -o gemm_v2.5_vectorized

  nsys profile \
  --trace=cuda \
  --stats=true \
  --force-overwrite=true \
  --output=nsys/gemm_v2.5_vectorized \
  -- ./gemm_v2.5_vectorized

  
 */
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

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
 * GEMM v2.5: Double Buffering + Vectorized Load (float4)
 * 适配 sm_120 / Blackwell 架构的 CUDA Core 终极形态
 */
__global__ void __launch_bounds__(256)
gemm_v2_5_vectorized_pipeline(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C,
                               int M, int N, int K) {
    // 1. 开辟双缓冲区 (Double Buffering)
    __shared__ float As[2][BM][BK];
    __shared__ float Bs[2][BK][BN];

    int tx = threadIdx.x; // 0..15
    int ty = threadIdx.y; // 0..15
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tid = ty * blockDim.x + tx; // 0..255

    // ----- 向量化搬运索引计算 (使用 float4，每次搬运 4 个 float) -----
    // A 矩阵: 128 x 8，使用 float4 后相当于 128 行 x 2 列 (每列 4 个 float)
    int a_vec_col = tid % (BK / 4); // 0..1 (只用到前2个向量列)
    int a_vec_row = tid / (BK / 4); // 0..127 (256个线程刚好直接铺满 128 行，无需 stride 循环！)

    // B 矩阵: 8 x 128，使用 float4 后相当于 8 行 x 32 列
    int b_vec_col = tid % (BN / 4); // 0..31
    int b_vec_row = tid / (BN / 4); // 0..7 (256个线程刚好直接铺满 8 行 x 32 列，无需 stride 循环！)

    // 当前线程负责计算的 C 子块 (8x8) 的寄存器
    float c_reg[TM][TN] = {0.0f};

    // 寄存器缓存 (用于外积计算)
    float a_reg[TM];
    float b_reg[TN];


    // ----- 预热阶段 (Preamble): 将第 0 块 Tile 加载到 Buffer 0 -----
    int global_a_row = by * BM + a_vec_row;
    int global_a_col = 0 + a_vec_col * 4;
    if (global_a_row < M && (global_a_col + 3) < K) {
        // 使用 float4 一次性拉取 16 字节
        reinterpret_cast<float4*>(&As[0][a_vec_row][a_vec_col * 4])[0] =
            reinterpret_cast<const float4*>(&A[global_a_row * K + global_a_col])[0];
    } else {
        reinterpret_cast<float4*>(&As[0][a_vec_row][a_vec_col * 4])[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    int global_b_row = 0 + b_vec_row;
    int global_b_col = bx * BN + b_vec_col * 4;
    if (global_b_row < K && (global_b_col + 3) < N) {
        reinterpret_cast<float4*>(&Bs[0][b_vec_row][b_vec_col * 4])[0] =
            reinterpret_cast<const float4*>(&B[global_b_row * N + global_b_col])[0];
    } else {
        reinterpret_cast<float4*>(&Bs[0][b_vec_row][b_vec_col * 4])[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    __syncthreads();

    int write_stage = 1; // 写入指针指向 Buffer 1
    int read_stage  = 0; // 读取指针指向 Buffer 0


    // ----- 主流水线循环 (Software Pipeline Loop) -----
    for (int bk = BK; bk < K; bk += BK) {
        // 1. 预取下一个 Tile (bk) 到 Buffer [write_stage] (向量化加载)
        global_a_row = by * BM + a_vec_row;
        global_a_col = bk + a_vec_col * 4;
        if (global_a_row < M && (global_a_col + 3) < K) {
            reinterpret_cast<float4*>(&As[write_stage][a_vec_row][a_vec_col * 4])[0] =
                reinterpret_cast<const float4*>(&A[global_a_row * K + global_a_col])[0];
        } else {
            reinterpret_cast<float4*>(&As[write_stage][a_vec_row][a_vec_col * 4])[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        global_b_row = bk + b_vec_row;
        global_b_col = bx * BN + b_vec_col * 4;
        if (global_b_row < K && (global_b_col + 3) < N) {
            reinterpret_cast<float4*>(&Bs[write_stage][b_vec_row][b_vec_col * 4])[0] =
                reinterpret_cast<const float4*>(&B[global_b_row * N + global_b_col])[0];
        } else {
            reinterpret_cast<float4*>(&Bs[write_stage][b_vec_row][b_vec_col * 4])[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        // 2. 从 Buffer [read_stage] 计算当前 Tile 的 8x8 外积
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // 抓取 As 和 Bs 的元素到 Register
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_reg[i] = As[read_stage][ty * TM + i][k];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                b_reg[j] = Bs[read_stage][k][tx * TN + j];
            }
            // 外积广播乘加
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    c_reg[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }

        // 等待下一个 Tile 预取完成，同时确保当前 Tile 消费完毕
        __syncthreads();

        // 乒乓切换 Buffer 指针
        write_stage ^= 1;
        read_stage  ^= 1;
    }


    // ----- 收尾阶段 (Epilogue): 计算最后一个已预取好的 Tile -----
    #pragma unroll
    for (int k = 0; k < BK; ++k) {
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            a_reg[i] = As[read_stage][ty * TM + i][k];
        }
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            b_reg[j] = Bs[read_stage][k][tx * TN + j];
        }
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                c_reg[i][j] += a_reg[i] * b_reg[j];
            }
        }
    }


    // ----- 写回全局内存 C -----
    int c_row_start = by * BM + ty * TM;
    int c_col_start = bx * BN + tx * TN;

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; j += 4) { // 写回同样可以使用 float4 向量化写回！
            int cur_r = c_row_start + i;
            int cur_c = c_col_start + j;
            if (cur_r < M && (cur_c + 3) < N) {
                reinterpret_cast<float4*>(&C[cur_r * N + cur_c])[0] =
                    make_float4(c_reg[i][j], c_reg[i][j+1], c_reg[i][j+2], c_reg[i][j+3]);
            }
        }
    }
}


// ----- CPU 参考实现与验证逻辑 -----
void gemm_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    const int M = 2048, N = 2048, K = 2048;
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N);
    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));

    dim3 blockDim(BN / TN, BM / TM); // 16 x 16 = 256 threads
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM); // 16 x 16 blocks

    // 预热
    gemm_v2_5_vectorized_pipeline<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    int iterations = 20;
    for (int i = 0; i < iterations; ++i) {
        gemm_v2_5_vectorized_pipeline<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    float avg_time = milliseconds / iterations;

    double flops = 2.0 * M * N * K;
    double tflops = (flops / (avg_time / 1000.0)) / 1e12;

    std::cout << "Kernel Execution Time: " << avg_time << " ms" << std::endl;
    std::cout << "Performance: " << tflops << " TFLOPS" << std::endl;

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));

    // 正确性抽样校验
    std::cout << "Verifying correctness..." << std::endl;
    bool passed = true;
    for (int i = 0; i < 100; ++i) {
        int r = rand() % M;
        int c = rand() % N;
        float cpu_sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            cpu_sum += h_A[r * K + k] * h_B[k * N + c];
        }
        if (std::abs(h_C[r * N + c] - cpu_sum) > 1e-2f) {
            std::cout << "Mismatch at (" << r << "," << c << "): GPU=" 
                      << h_C[r * N + c] << ", CPU=" << cpu_sum << std::endl;
            passed = false;
            break;
        }
    }
    if (passed) std::cout << "Result Verification PASSED!" << std::endl;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}