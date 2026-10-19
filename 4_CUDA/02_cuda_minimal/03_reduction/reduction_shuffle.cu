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

constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

static_assert(BLOCK_SIZE % WARP_SIZE == 0, "BLOCK_SIZE must be a multiple of 32");

// 将同一个 Warp 中的 32 个寄存器值归约到 lane 0
__device__ __forceinline__ float warp_reduce_sum(float value) {
    // 调用位置处当前正在执行的 lane 集合
    unsigned int mask = __activemask();

    // 步长依次递减：32 -> 16 -> 8 -> 4 -> 2 -> 1
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        // 读取当前 lane 下方 offset 个位置的寄存器值
        value += __shfl_down_sync(mask, value, offset);
    }

    // 最终只有 lane 0 保存完整 Warp 归约结果
    return value;
}

__global__ void reduction_shuffle(const float* input, float* result, int n) {
    // 每个 Warp 只需要一个 Shared Memory 槽位
    __shared__ float warp_sums[NUM_WARPS];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    // 一：线程在自己的寄存器中累加输入
    float local_sum = 0.0f;

    for (int i = index; i < n; i += stride) {
        local_sum += input[i];
    }

    // 二：每个 Warp 在寄存器之间完成归约
    float warp_sum = warp_reduce_sum(local_sum);

    // 每个 Warp 只有 lane 0 写出归约结果
    if (lane_id == 0) {
        warp_sums[warp_id] = warp_sum;
    }

    // 第一个 Warp 读取结果前，必须等待其他 Warp 写入完成
    __syncthreads();

    // 三：只让 Warp 0 归约所有 Warp 的部分和
    if (warp_id == 0) {
        // lane 0～7 读取有效结果，其余 lane 使用加法单位元 0
        float block_sum = lane_id < NUM_WARPS ? warp_sums[lane_id] : 0.0f;

        block_sum = warp_reduce_sum(block_sum);

        // Thread 0 得到整个 Block 的结果
        if (lane_id == 0) {
            atomicAdd(result, block_sum);
        }
    }
}

int main() {
    constexpr int N = 1 << 24;
    constexpr std::size_t BYTES =
    static_cast<std::size_t>(N) * sizeof(float);

    std::vector<float> h_input(N);

    // 暂时使用 1.0f，便于验证归约结构
    std::fill(h_input.begin(), h_input.end(), 1.0f);

    double cpu_sum = 0.0;

    for (float value : h_input) {
        cpu_sum += static_cast<double>(value);
    }

    float* d_input = nullptr;
    float* d_results = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_input), BYTES));
    // 每次 Kernel 使用独立结果槽，避免把清零操作计入 Kernel 时间
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_results), (WARMUP + REPEAT) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), BYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_results, 0, (WARMUP + REPEAT) * sizeof(float)));

    // 预热
    for (int i = 0; i < WARMUP; ++i) {
        reduction_shuffle<<<NUM_BLOCKS, BLOCK_SIZE>>>(
            d_input,
            d_results + i,
            N);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < REPEAT; ++i) {
        reduction_shuffle<<<NUM_BLOCKS, BLOCK_SIZE>>>(
            d_input,
            d_results + WARMUP + i,
            N);
    }

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_time_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &total_time_ms,
        start,
        stop));

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

    // Reduction 只计算输入读取量
    double bandwidth_gb_s =
        static_cast<double>(BYTES) /
        (average_time_ms * 1.0e6);

    bool passed =
        absolute_error <= 1.0 ||
        relative_error <= 1.0e-5;

    std::cout << "N: " << N << '\n';
    std::cout << "Blocks: " << NUM_BLOCKS << '\n';
    std::cout << "Threads per block: " << BLOCK_SIZE << '\n';
    std::cout << "Shared memory per block: "
              << NUM_WARPS * sizeof(float)
              << " bytes\n";

    std::cout << "Atomic operations per kernel: "
              << NUM_BLOCKS << '\n';

    std::cout << "CPU sum: " << cpu_sum << '\n';
    std::cout << "GPU sum: " << gpu_sum << '\n';
    std::cout << "Absolute error: " << absolute_error << '\n';
    std::cout << "Relative error: " << relative_error << '\n';

    std::cout << "Average kernel time: "
              << average_time_ms << " ms\n";

    std::cout << "Effective read bandwidth: "
              << bandwidth_gb_s << " GB/s\n";

    std::cout << (passed
        ? "Warp shuffle reduction passed.\n"
        : "Warp shuffle reduction failed.\n");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaFree(d_input));

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}