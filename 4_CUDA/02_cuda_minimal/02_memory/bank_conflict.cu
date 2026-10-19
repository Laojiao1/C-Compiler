#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

// 统一检查 CUDA Runtime API 错误
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t error = (call);                                        \
        if (error != cudaSuccess) {                                        \
            std::cerr << "CUDA error at " << __FILE__ << ':' << __LINE__   \
                      << ": " << cudaGetErrorString(error) << std::endl;    \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

constexpr int WARP_SIZE = 32;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;

// 重复执行 Shared Memory Load，使 Bank Conflict 的性能差异更明显
constexpr int LOAD_ITERATIONS = 1024;

// 无 Bank Conflict：Warp 内线程连续访问 32 个 float
__global__ void shared_contiguous_kernel(
    const float* input,
    float* output)
{
    // 每个 Warp 使用自己独立的一行，避免不同 Warp 互相覆盖
    __shared__ volatile float tile[WARPS_PER_BLOCK][WARP_SIZE];

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int global_id = blockIdx.x * blockDim.x + tid;

    // lane 0~31 分别访问 Bank 0~31，无 Bank Conflict
    tile[warp_id][lane_id] = input[global_id];

    // 确保整个 Block 已经完成 Shared Memory 写入
    __syncthreads();

    float sum = 0.0f;

    // volatile 强制保留重复的 Shared Memory Load
    for (int i = 0; i < LOAD_ITERATIONS; ++i) {
        sum += tile[warp_id][lane_id];
    }

    // 写回结果，防止整个循环被编译器删除
    output[global_id] = sum;
}

// 32-way Bank Conflict：Warp 内线程按列访问 tile[32][32]
__global__ void shared_conflict_kernel(
    const float* input,
    float* output)
{
    // 每个 Warp 拥有一个独立的 32×32 tile
    __shared__ volatile float
        tile[WARPS_PER_BLOCK][WARP_SIZE][WARP_SIZE];

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int global_id = blockIdx.x * blockDim.x + tid;

    // 线性下标为 lane_id * 32
    // Bank ID = (lane_id * 32) % 32 = 0
    // 一个 Warp 的 32 个线程访问 Bank 0 中的不同地址
    tile[warp_id][lane_id][0] = input[global_id];

    __syncthreads();

    float sum = 0.0f;

    // 每次 Load 都会形成典型的 32-way Bank Conflict
    for (int i = 0; i < LOAD_ITERATIONS; ++i) {
        sum += tile[warp_id][lane_id][0];
    }

    output[global_id] = sum;
}

// Padding 版本：将每行跨度从 32 改为 33
__global__ void shared_padded_kernel(
    const float* input,
    float* output)
{
    // 最后一列是 Padding，不存放本实验的有效数据
    __shared__ volatile float
        tile[WARPS_PER_BLOCK][WARP_SIZE][WARP_SIZE + 1];

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int global_id = blockIdx.x * blockDim.x + tid;

    // 线性下标为 lane_id * 33
    // Bank ID = (lane_id * 33) % 32 = lane_id
    // 32 个线程重新分散到 32 个不同 Bank
    tile[warp_id][lane_id][0] = input[global_id];

    __syncthreads();

    float sum = 0.0f;

    // 访问逻辑仍然是按列读取，但 Padding 已经消除冲突
    for (int i = 0; i < LOAD_ITERATIONS; ++i) {
        sum += tile[warp_id][lane_id][0];
    }

    output[global_id] = sum;
}

// 检查 Kernel 输出是否符合预期
bool check_result(
    const std::vector<float>& input,
    const std::vector<float>& output)
{
    for (std::size_t i = 0; i < output.size(); ++i) {
        float expected =
            input[i] * static_cast<float>(LOAD_ITERATIONS);

        // 浮点累加可能存在微小误差，使用相对误差判断
        float tolerance = 1e-4f * std::fabs(expected) + 1e-5f;

        if (std::fabs(output[i] - expected) > tolerance) {
            std::cerr << "Mismatch at index " << i
                      << ": expected " << expected
                      << ", got " << output[i] << std::endl;
            return false;
        }
    }

    return true;
}

// 使用 CUDA Event 测量 Kernel 的平均执行时间
template <typename Kernel>
float benchmark_kernel(
    Kernel kernel,
    const float* d_input,
    float* d_output,
    int blocks,
    int warmup_iterations,
    int benchmark_iterations)
{
    // 预热，减少首次启动和 GPU 频率变化的影响
    for (int i = 0; i < warmup_iterations; ++i) {
        kernel<<<blocks, THREADS_PER_BLOCK>>>(d_input, d_output);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < benchmark_iterations; ++i) {
        kernel<<<blocks, THREADS_PER_BLOCK>>>(d_input, d_output);
    }

    // 检查 Kernel Launch 配置错误
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / static_cast<float>(benchmark_iterations);
}

// 计算逻辑上的 Shared Memory 读取带宽
double calculate_shared_bandwidth_gbps(
    int element_count,
    float milliseconds)
{
    // 每个线程每轮读取一个 float
    double bytes =
        static_cast<double>(element_count) *
        LOAD_ITERATIONS *
        sizeof(float);

    double seconds = milliseconds * 1e-3;

    return bytes / seconds / 1e9;
}

int main()
{
    constexpr int warmup_iterations = 5;
    constexpr int benchmark_iterations = 30;

    cudaDeviceProp device_prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, 0));

    // 每个 SM 启动多个 Block，提供足够的并行工作
    int blocks = device_prop.multiProcessorCount * 4;
    int element_count = blocks * THREADS_PER_BLOCK;

    std::size_t bytes =
        static_cast<std::size_t>(element_count) * sizeof(float);

    std::vector<float> h_input(element_count);
    std::vector<float> h_output(element_count);

    // 使用较小的周期数据，便于检查累加结果
    for (int i = 0; i < element_count; ++i) {
        h_input[i] =
            static_cast<float>((i % 100) + 1) * 0.001f;
    }

    float* d_input = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));

    CUDA_CHECK(cudaMemcpy(
        d_input,
        h_input.data(),
        bytes,
        cudaMemcpyHostToDevice));

    std::cout << "Device: " << device_prop.name << '\n';
    std::cout << "SM count: "
              << device_prop.multiProcessorCount << '\n';
    std::cout << "Blocks: " << blocks << '\n';
    std::cout << "Threads per block: "
              << THREADS_PER_BLOCK << '\n';
    std::cout << "Load iterations: "
              << LOAD_ITERATIONS << '\n';

    // 1. 连续访问基准
    float contiguous_ms = benchmark_kernel(
        shared_contiguous_kernel,
        d_input,
        d_output,
        blocks,
        warmup_iterations,
        benchmark_iterations);

    CUDA_CHECK(cudaMemcpy(
        h_output.data(),
        d_output,
        bytes,
        cudaMemcpyDeviceToHost));

    if (!check_result(h_input, h_output)) {
        std::cerr << "Contiguous kernel failed." << std::endl;
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
        return EXIT_FAILURE;
    }

    // 2. 32×32 按列访问
    float conflict_ms = benchmark_kernel(
        shared_conflict_kernel,
        d_input,
        d_output,
        blocks,
        warmup_iterations,
        benchmark_iterations);

    CUDA_CHECK(cudaMemcpy(
        h_output.data(),
        d_output,
        bytes,
        cudaMemcpyDeviceToHost));

    if (!check_result(h_input, h_output)) {
        std::cerr << "Conflict kernel failed." << std::endl;
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
        return EXIT_FAILURE;
    }

    // 3. 32×33 Padding 后按列访问
    float padded_ms = benchmark_kernel(
        shared_padded_kernel,
        d_input,
        d_output,
        blocks,
        warmup_iterations,
        benchmark_iterations);

    CUDA_CHECK(cudaMemcpy(
        h_output.data(),
        d_output,
        bytes,
        cudaMemcpyDeviceToHost));

    if (!check_result(h_input, h_output)) {
        std::cerr << "Padded kernel failed." << std::endl;
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
        return EXIT_FAILURE;
    }

    double contiguous_bandwidth =
        calculate_shared_bandwidth_gbps(
            element_count,
            contiguous_ms);

    double conflict_bandwidth =
        calculate_shared_bandwidth_gbps(
            element_count,
            conflict_ms);

    double padded_bandwidth =
        calculate_shared_bandwidth_gbps(
            element_count,
            padded_ms);

    std::cout << "\nContiguous access:\n";
    std::cout << "  Average time: "
              << contiguous_ms << " ms\n";
    std::cout << "  Logical shared bandwidth: "
              << contiguous_bandwidth << " GB/s\n";

    std::cout << "\n32-way bank conflict:\n";
    std::cout << "  Average time: "
              << conflict_ms << " ms\n";
    std::cout << "  Logical shared bandwidth: "
              << conflict_bandwidth << " GB/s\n";

    std::cout << "\nPadded access:\n";
    std::cout << "  Average time: "
              << padded_ms << " ms\n";
    std::cout << "  Logical shared bandwidth: "
              << padded_bandwidth << " GB/s\n";

    std::cout << "\nConflict slowdown vs contiguous: "
              << conflict_ms / contiguous_ms << "x\n";

    std::cout << "Padding speedup vs conflict: "
              << conflict_ms / padded_ms << "x\n";

    std::cout << "All implementations passed.\n";

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return EXIT_SUCCESS;
}