#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t error = (call);                                        \
        if (error != cudaSuccess) {                                        \
            std::cerr << "CUDA error: " << cudaGetErrorString(error)       \
                      << " at " << __FILE__ << ":" << __LINE__ << '\n';    \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

constexpr int BLOCK_SIZE = 256;
constexpr int NUM_BLOCKS = 4096;
constexpr int WARMUP = 10;
constexpr int REPEAT = 100;

__global__ void reduction_naive(const float* input, float* result, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 每个线程先在寄存器中累加自己负责的元素
    float local_sum = 0.0f;

    for (int i = index; i < n; i += stride) {
        local_sum += input[i];
    }

    // 所有线程都竞争同一个 Global Memory 地址
    atomicAdd(result, local_sum);
}

int main()
{
    constexpr int N = 1 << 24;
    constexpr std::size_t BYTES =
        static_cast<std::size_t>(N) * sizeof(float);

    std::vector<float> h_input(N);

    // 使用 1.0f 便于当前阶段稳定验证归约结构
    std::fill(h_input.begin(), h_input.end(), 1.0f);

    double cpu_sum = 0.0;

    for (float value : h_input) {
        cpu_sum += static_cast<double>(value);
    }

    float* d_input = nullptr;
    float* d_results = nullptr;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_input), BYTES));

    // 每次测试使用独立结果槽，避免把 cudaMemset 计入 Kernel 时间
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_results),
        (WARMUP + REPEAT) * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        d_input,
        h_input.data(),
        BYTES,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(
        d_results,
        0,
        (WARMUP + REPEAT) * sizeof(float)));

    // 预热
    for (int i = 0; i < WARMUP; ++i) {
        reduction_naive<<<NUM_BLOCKS, BLOCK_SIZE>>>(
            d_input, d_results + i, N);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < REPEAT; ++i) {
        reduction_naive<<<NUM_BLOCKS, BLOCK_SIZE>>>(
            d_input,
            d_results + WARMUP + i,
            N);
    }

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_time_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &total_time_ms, start, stop));

    float gpu_sum = 0.0f;

    CUDA_CHECK(cudaMemcpy(
        &gpu_sum,
        d_results + WARMUP + REPEAT - 1,
        sizeof(float),
        cudaMemcpyDeviceToHost));

    double absolute_error =
        std::abs(static_cast<double>(gpu_sum) - cpu_sum);

    double relative_error =
        absolute_error / std::max(std::abs(cpu_sum), 1.0);

    float average_time_ms =
        total_time_ms / static_cast<float>(REPEAT);

    // Reduction 只读取一次输入，按有效读取带宽计算
    double bandwidth_gb_s =
        static_cast<double>(BYTES) /
        (average_time_ms * 1.0e6);

    bool passed =
        absolute_error <= 1.0 ||
        relative_error <= 1.0e-5;

    std::cout << "N: " << N << '\n';
    std::cout << "Blocks: " << NUM_BLOCKS << '\n';
    std::cout << "Threads per block: " << BLOCK_SIZE << '\n';
    std::cout << "Atomic operations per kernel: "
              << NUM_BLOCKS * BLOCK_SIZE << '\n';

    std::cout << "CPU sum: " << cpu_sum << '\n';
    std::cout << "GPU sum: " << gpu_sum << '\n';
    std::cout << "Absolute error: " << absolute_error << '\n';
    std::cout << "Relative error: " << relative_error << '\n';

    std::cout << "Average kernel time: "
              << average_time_ms << " ms\n";

    std::cout << "Effective read bandwidth: "
              << bandwidth_gb_s << " GB/s\n";

    std::cout << (passed
        ? "Naive reduction passed.\n"
        : "Naive reduction failed.\n");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaFree(d_input));

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}