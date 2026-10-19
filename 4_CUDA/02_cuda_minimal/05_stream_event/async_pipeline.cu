/**
 * 编译：
 *  nvcc -O3 \
    -lineinfo \
    -arch=sm_120 \
    --ptxas-options=-v \
    async_pipeline.cu \
    -o async_pipeline

   Nsight Systems 采集:
    nsys profile \
    -t cuda \
    --stats=true \
    --force-overwrite=true \
    -o async_pipeline_report \
    ./async_pipeline
 */
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

#define CHECK_CUDA(call)                                                   \
do {                                                                       \
    cudaError_t err = call;                                                \
    if (err != cudaSuccess) {                                              \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << "\n";    \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while(0)


// 最简单 Vector Add Kernel
// 每个 Thread 负责一个元素，通过 blockIdx/threadIdx 映射数据位置
__global__ 
void vector_add_kernel(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] + B[idx];
}

void run_sync_pageable(int N, size_t bytes) {
    // CPU host 端
    std::vector<float> h_A(N, 1.0f), h_B(N, 2.0f), h_C(N);

    // GPU device 端，通过 cudaMalloc 申请显存空间
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));

    // Stream 是 GPU 工作队列：
    // 同一个 Stream 内 CUDA 操作严格按照提交顺序执行
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Event 是 GPU 时间线上的标记点：记录的是 GPU 执行时间
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 测量完整端到端时间
    auto start_chrono = std::chrono::high_resolution_clock::now();

    // 这里先使用普通 cudaMemcpy，将输入数据从 CPU 搬运到 GPU
    // 目的是先理解 Stream 顺序，不讨论真正异步传输
    // Copy 操作进入 Stream 队列，Host 不等待 Copy 完成
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    int block = 256;
    int grid = (N + block - 1) / block;

    // 记录起始时间点（入队）
    CHECK_CUDA(cudaEventRecord(start, stream));

    // 提交计算任务到指定 stream
    vector_add_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, N);

    // 检查 Kernel Launch 配置错误
    CHECK_CUDA(cudaGetLastError());

    // GPU 执行完 Kernel 后记录 stop 时间点（入队）
    CHECK_CUDA(cudaEventRecord(stop, stream));

    // 等待该 Stream 中所有任务完成
    // 注意：它只阻塞等待 stream 这一个流，比阻塞等待整个设备的 cudaDeviceSynchronize() 更高效
    CHECK_CUDA(cudaStreamSynchronize(stream));

    auto end_chrono = std::chrono::high_resolution_clock::now();

    float ms;
    // 计算 kernel 在 GPU 上记录的时间差，直接输出毫秒（ms）
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    // 计算完整端到端时间
    double total_ms = std::chrono::duration<double, std::milli>(end_chrono - start_chrono).count();

    // D2H 结果复制回来进行 CPU 正确性检查
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaStreamSynchronize(stream));

    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != 3.0f) {
            correct = false;
            break;
        }
    }
    
    std::cout << "run_sync_pageable()\n";
    std::cout << "Kernel time: " << ms << " ms\n";
    std::cout << "End-to-end time: " << total_ms << " ms\n";
    std::cout << "Result: " << (correct ? "PASS" : "FAIL") << "\n";
    std::cout << "-----------------------------------\n";

    // 销毁阶段：先销毁 Event 和 Stream 句柄，再通过 cudaFree 释放显存，防止显存泄漏
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaStreamDestroy(stream));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

}

void run_async_pinned_single_stream(int N, size_t bytes) {
    // Pinned Memory:
    // GPU 可以通过 DMA 直接访问这块 Host 内存
    float *h_A, *h_B, *h_C;
    CHECK_CUDA(cudaMallocHost(&h_A, bytes));
    CHECK_CUDA(cudaMallocHost(&h_B, bytes));
    CHECK_CUDA(cudaMallocHost(&h_C, bytes));

    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // 测量完整端到端时间
    auto start_chrono = std::chrono::high_resolution_clock::now();

    // GPU device 端，通过 cudaMalloc 申请显存空间
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));


    // Stream 是 GPU 工作队列：
    // 同一个 Stream 内 CUDA 操作严格按照提交顺序执行
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Event 是 GPU 时间线上的标记点：记录的是 GPU 执行时间
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 使用 cudaMemcpyAsync 将 Copy 提交到指定 Stream
    // Pinned Memory 允许 GPU Copy Engine 通过 DMA 直接访问 Host Memory
    // 注意：同一个 Stream 内仍然按照提交顺序执行
    CHECK_CUDA(cudaMemcpyAsync(d_A, h_A, bytes, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_B, h_B, bytes, cudaMemcpyHostToDevice, stream));

    int block = 256;
    int grid = (N + block - 1) / block;

    // 记录起始时间点（入队）
    CHECK_CUDA(cudaEventRecord(start, stream));

    // 提交计算任务到指定 stream
    vector_add_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, N);

    // 检查 Kernel Launch 配置错误
    CHECK_CUDA(cudaGetLastError());

    // GPU 执行完 Kernel 后记录 stop 时间点（入队）
    CHECK_CUDA(cudaEventRecord(stop, stream));

    // 等待该 Stream 中所有任务完成
    // 注意：它只阻塞等待 stream 这一个流，比阻塞等待整个设备的 cudaDeviceSynchronize() 更高效
    CHECK_CUDA(cudaStreamSynchronize(stream));

    auto end_chrono = std::chrono::high_resolution_clock::now();

    float ms;
    // 计算 kernel 在 GPU 上记录的时间差，直接输出毫秒（ms）
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    // 计算完整端到端时间
    double total_ms =
    std::chrono::duration<double, std::milli>(
        end_chrono - start_chrono
    ).count();

    // D2H 结果复制回来进行 CPU 正确性检查
    CHECK_CUDA(cudaMemcpyAsync(h_C, d_C, bytes, cudaMemcpyDeviceToHost, stream));

    CHECK_CUDA(cudaStreamSynchronize(stream));

    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != 3.0f) {
            correct = false;
            break;
        }
    }

    std::cout << "run_async_pinned_single_stream()\n";
    std::cout << "Kernel time: " << ms << " ms\n";
    std::cout << "End-to-end time: " << total_ms << " ms\n";
    std::cout << "Result: " << (correct ? "PASS" : "FAIL") << "\n";
    std::cout << "-----------------------------------\n";

    // 销毁阶段：先销毁 Event 和 Stream 句柄，再通过 cudaFree 释放显存，防止显存泄漏
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaStreamDestroy(stream));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    CHECK_CUDA(cudaFreeHost(h_A));
    CHECK_CUDA(cudaFreeHost(h_B));
    CHECK_CUDA(cudaFreeHost(h_C));

}

void run_async_pinned_multi_stream(int N, size_t bytes) {
    constexpr int NUM_STREAMS = 4;

    // Pinned Host Memory: 分配页锁定内存以支持真正的 DMA 跨流异步传输
    float *h_A, *h_B, *h_C;
    CHECK_CUDA(cudaMallocHost(&h_A, bytes));
    CHECK_CUDA(cudaMallocHost(&h_B, bytes));
    CHECK_CUDA(cudaMallocHost(&h_C, bytes));

    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Device Memory: 每个 Stream 将访问其对应的内存分块（Chunk），互不干扰
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));

    // 创建多流实现流水线并发
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }

    const int chunk_size = N / NUM_STREAMS;
    const size_t chunk_bytes = chunk_size * sizeof(float);

    int block = 256;
    int grid = (chunk_size + block - 1) / block;

    // 使用 CPU 计时器测算包含 H2D、Kernel 以及 D2H 的整体 Pipeline 端到端耗时
    auto start = std::chrono::high_resolution_clock::now();

    // 循环将不同分块的传输与计算交错提交到各个 Stream，重叠（Overlap）传输与计算时间
    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * chunk_size;

        // 当前 Chunk 的 H2D 异步数据传输
        CHECK_CUDA(cudaMemcpyAsync(d_A + offset, 
            h_A + offset, chunk_bytes, cudaMemcpyHostToDevice, streams[i]));
        CHECK_CUDA(cudaMemcpyAsync(d_B + offset, 
            h_B + offset, chunk_bytes, cudaMemcpyHostToDevice, streams[i]));

        // 当前 Chunk 的 Kernel 异步计算
        vector_add_kernel<<<grid, block, 0, streams[i]>>>(
            d_A + offset, d_B + offset, d_C + offset, chunk_size);

        CHECK_CUDA(cudaGetLastError());

        // 当前 Chunk 的 D2H 异步数据拷贝回 Host
        CHECK_CUDA(cudaMemcpyAsync(h_C + offset, 
            d_C + offset, chunk_bytes, cudaMemcpyDeviceToHost, streams[i]));
    }

    // 逐一同步各个 Stream，等待所有流中的传输和计算任务全部完成
    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // CPU 端结果正确性校验
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != 3.0f) {
            correct = false;
            break;
        }
    }

    std::cout << "run_async_pinned_multi_stream()\n";
    std::cout << "Streams: " << NUM_STREAMS << "\n";
    std::cout << "End-to-end time: " << total_ms << " ms\n";
    std::cout << "Result: " << (correct ? "PASS" : "FAIL") << "\n";
    std::cout << "-----------------------------------\n";

    // 资源释放阶段：销毁所有流并释放显存与页锁定内存
    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    CHECK_CUDA(cudaFreeHost(h_A));
    CHECK_CUDA(cudaFreeHost(h_B));
    CHECK_CUDA(cudaFreeHost(h_C));
}

// 查看当前 Kernel 在 GPU 上理论可以驻留多少 Block
void query_occupancy() {
    int block_size = 256;
    int dynamic_smem = 0;
    int blocks_per_sm = 0;

    // CUDA 根据 kernel 资源使用情况：registers/thread、shared memory、线程数量等 去估算一个 SM 可以同时驻留多少 block
    CHECK_CUDA(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm,
            vector_add_kernel,
            block_size,
            dynamic_smem
        )
    );

    int device;
    CHECK_CUDA(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    int warps_per_block = block_size / 32;
    int active_warps = blocks_per_sm * warps_per_block;

    // 理论最大 warp 数量，不同架构不同，这里直接使用 CUDA Runtime 查询
    int max_warps_per_sm = prop.maxThreadsPerMultiProcessor / 32;

    float occupancy = static_cast<float>(active_warps) / max_warps_per_sm;

    std::cout << "\n===== Occupancy Query =====\n";
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Block size: " << block_size << "\n";
    std::cout << "Active blocks / SM: " << blocks_per_sm << "\n";
    std::cout << "Warps / block: " << warps_per_block << "\n";
    std::cout << "Active warps / SM: " << active_warps << "\n";
    std::cout << "Max warps / SM: " << max_warps_per_sm << "\n";
    std::cout << "Theoretical Occupancy: " << occupancy * 100.0f << "%\n";
}

int main() {
    const int N = 1 << 24; // 16,777,216 个元素 (16M)
    const size_t bytes = N * sizeof(float); // 64 MB

    std::cout << "-----------------------------------\n";
    run_sync_pageable(N, bytes);
    run_async_pinned_single_stream(N, bytes);
    run_async_pinned_multi_stream(N, bytes);
    query_occupancy();
    
    return 0;
}