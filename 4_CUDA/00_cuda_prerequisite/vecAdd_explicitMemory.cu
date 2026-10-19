/**
 * CUDA 官方示例: vecAdd_explicitMemory
 * 
 * 编译指令：nvcc -arch=sm_120 vecAdd_explicitMemory.cu -o vecAdd_explicitMemory
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

// ----- 显示内存管理方式 -----
/**
 * 显式管理内存分配和内存空间之间的数据迁移有助于提升应用程序性能，但会使代码更加冗长
 */
void explicitMemExample(int vectorLength) {
    // 分配 host（CPU）内存
    float* A = (float*)malloc(vectorLength*sizeof(float));
    float* B = (float*)malloc(vectorLength*sizeof(float));
    float* C = (float*)malloc(vectorLength*sizeof(float));
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

    // 分配 device（GPU）内存
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    /**
     * 使用 cudaMallocHost API 分配主机内存，当缓冲区用于 CPU 和 GPU 内存之间的复制时，这是最佳实践。
     * 
     * 该代码用于 cudaMallocHost 在 CPU 上分配内存。这会在主机上分配页面锁定内存，这意味着页面不会被交换到磁盘上，从而提高复制性能，对于异步内存传输至关重要。
     */
    cudaMallocHost(&A, vectorLength*sizeof(float));
    cudaMallocHost(&B, vectorLength*sizeof(float));
    cudaMallocHost(&C, vectorLength*sizeof(float));

    // 直接在 host（CPU）端进行初始化
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // ----- 开始分配和复制 -----
    // 在 GPU 上分配内存
    cudaMalloc(&d_A, vectorLength*sizeof(float));
    cudaMalloc(&d_B, vectorLength*sizeof(float));
    cudaMalloc(&d_C, vectorLength*sizeof(float));

    // 拷贝数据到 GPU
    cudaMemcpy(d_A, A, vectorLength*sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(d_B, B, vectorLength*sizeof(float), cudaMemcpyDefault);
    cudaMemset(d_C, 0, vectorLength*sizeof(float));
    // 结束分配和复制
    
    // 启动 kernel
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(d_A, d_B, d_C, vectorLength);
    // 等待 kernel 执行完毕
    cudaDeviceSynchronize();

    // 拷贝结果回 CPU
    cudaMemcpy(C, d_C, vectorLength*sizeof(float), cudaMemcpyDefault);

    // 为了进行比较，请在 CPU 上串行执行计算
    serialVecAdd(A, B, comparisonResult, vectorLength);

    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength)) {
        printf("Explicit Memory: CPU and GPU answers match\n");
    } else {
        printf("Explicit Memory: Error - CPU and GPU answers do not match\n");
    }

    // Clean Up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    free(comparisonResult);
}

int main(int argc, char** argv) {
    int vectorLength = 1024;

    if(argc >= 2){
        vectorLength = std::atoi(argv[1]);
    }

    explicitMemExample(vectorLength);

    return 0;
}