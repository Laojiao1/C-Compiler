/**
 * CUDA 官方示例: vecAdd_unifiedMemory
 * 
 * 编译指令：nvcc -arch=sm_120 vecAdd_unifiedMemory.cu -o vecAdd_unifiedMemory
 */
#include <cuda_runtime.h>
#include <cuda/cmath>
#include <iostream>
#include <cmath>

// vecAdd kernel
__global__ void vecAdd(float* A, float* B, float* C, int vectorLength) {
    // 计算此线程负责哪个元素
    int workIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if(workIndex < vectorLength) {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}

// 自定义初始化函数
void initArray(float* arr, int size) {
    for(int i = 0; i < size; ++i) {
        arr[i] = static_cast<float>(i);
    }
}

// 在 CPU 上进行计算，用于检查
void serialVecAdd(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

// 结果验证
bool vectorApproximatelyEqual(float* A, float* B, int length, float epsilon=0.00001) {
    for(int i=0; i<length; i++) {
        if(fabs(A[i] -B[i]) > epsilon) {
            printf("Index %d mismatch: %f != %f", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

// ----- 统一内存管理方式 -----
/**
 * 统一内存是 CUDA 运行时的一项特性，它允许 NVIDIA 驱动程序管理主机和设备之间的数据传输，NVIDIA 驱动程序会确保 GPU 或 CPU 在尝试访问内存时都能访问到该内存
 */
void unifiedMemExample(int vectorLength) {
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

    // 使用统一内存分配缓冲区：这些缓冲区既可以从 CPU 访问，也可以从 GPU 访问
    cudaMallocManaged(&A, vectorLength*sizeof(float));
    cudaMallocManaged(&B, vectorLength*sizeof(float));
    cudaMallocManaged(&C, vectorLength*sizeof(float));

    // 在 host（CPU）端进行初始化
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // 启动内核。统一内存将确保 A、B 和 C 可供 GPU 访问
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
    // 等待 kernel 执行完毕
    cudaDeviceSynchronize();

    // 为了进行比较，请在 CPU 上串行执行计算
    serialVecAdd(A, B, comparisonResult, vectorLength);

    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength)) {
        printf("Unified Memory: CPU and GPU answers match\n");
    } else {
        printf("Unified Memory: Error - CPU and GPU answers do not match\n");
    }

    // Clean Up
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(comparisonResult);
}

int main(int argc, char** argv) {
    int vectorLength = 1024;

    if(argc >= 2) {
        vectorLength = std::atoi(argv[1]);
    }

    unifiedMemExample(vectorLength);

    return 0;
}