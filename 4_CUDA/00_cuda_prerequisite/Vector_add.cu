#include <cuda_runtime.h>
#include <iostream>

/**
    本代码仅仅是教学用最小示例，重点验证线程与数据的映射思路；
    不作为后续正式算子的 correctness 和工程错误处理模板。
 */

// CUDA Kernel：向量加法
__global__ void vector_add(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    // 当前线程负责的全局元素下标
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 防止最后一个 Block 中多余线程越界访问
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 1000;
    const size_t bytes = N * sizeof(float);

    // CPU 内存
    float* h_A = new float[N];
    float* h_B = new float[N];
    float* h_C = new float[N];

    // 初始化输入数据
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // GPU Global Memory
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // 将数据从 CPU 内存（Host）拷贝到 GPU 显存（Device）
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // 设定每个 Block 包含 256 个线程（通常设为 32 的倍数，如 128, 256, 512）
    const int block_size = 256;

    // 向上取整计算需要的 Block 数量，确保总线程数 >= N
    // (1000 + 256 - 1) / 256 = 1255 / 256 = 4 个 Block
    const int grid_size = (N + block_size - 1) / block_size;

    // 启动 Kernel：传入执行配置 <<<Grid 数量, Block 尺寸>>>
    vector_add<<<grid_size, block_size>>>(d_A, d_B, d_C, N);

    // 显式同步：等待 GPU 将任务完全执行完毕
    cudaDeviceSynchronize();

    // 将数据从 GPU 显存（Device）拷贝到 CPU 内存（Host）
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // 简单检查结果
    for (int i = 0; i < 10; ++i) {
        std::cout << h_A[i] << " + "
                  << h_B[i] << " = "
                  << h_C[i] << '\n';
    }

    // 释放 GPU 内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // 释放 CPU 内存
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}