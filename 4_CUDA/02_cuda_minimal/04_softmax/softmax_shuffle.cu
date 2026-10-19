/**
 *  nvcc -O3 -lineinfo -arch=sm_120 -Xptxas -v \
    softmax_shuffle.cu -o softmax_shuffle \
    2>&1 | tee softmax_shuffle_ptxas.txt
 * 
 *  ptxas info    : 0 bytes gmem
    ptxas info    : Compiling entry function '_Z22softmax_shuffle_kernelPKfPfii' for 'sm_120'
    ptxas info    : Function properties for _Z22softmax_shuffle_kernelPKfPfii
        0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
    ptxas info    : Used 22 registers, used 1 barriers
    ptxas info    : Compile time = 10.094 ms

 */
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t error = (call);                                           \
        if (error != cudaSuccess) {                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << " - " << cudaGetErrorString(error) << '\n';          \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

constexpr int ROWS = 4096;
constexpr int COLS = 1000;
constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
constexpr int WARMUP_ITERS = 10;
constexpr int TIMING_ITERS = 100;

// 当前 warp 中的 32 个线程全部参与 shuffle 数据交换与同步
constexpr unsigned FULL_MASK = 0xffffffffu; 

// Warp 内利用 Shuffle 指令快速计算 32 个线程的最大值，直接在寄存器中完成归约
__device__ __forceinline__
float warp_reduce_max(float value) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        value = fmaxf(
            value,
            __shfl_down_sync(FULL_MASK, value, offset));
    }

    return value;
}

// Warp 内利用 Shuffle 指令快速计算 32 个线程的累加和，无需访问 Shared Memory
__device__ __forceinline__
float warp_reduce_sum(float value) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(FULL_MASK, value, offset);
    }

    return value;
}

// 两级 Block Reduce Max：结合 Warp Shuffle 与极少量 Shared Memory 实现 Block 级最大值归约
__device__ __forceinline__
float block_reduce_max_shuffle(float value, float* warp_results) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    // ----- 第一步：每个 Warp 在寄存器中独立归约，求出本 Warp 内 32 个线程的最大值 -----
    value = warp_reduce_max(value);

    // 每个 Warp 只由 lane 0 将该 Warp 的部分最大值写到 Shared Memory
    if (lane_id == 0) warp_results[warp_id] = value;

    __syncthreads();

    // ----- 第二步：由 Warp 0 读取所有 Warp 留下的部分结果，做跨 Warp 的二次归约 -----
    float block_max = -CUDART_INF_F;

    if (warp_id == 0) {
        block_max = lane_id < num_warps ? warp_results[lane_id] : -CUDART_INF_F;

        // Warp 0 的全部 32 个 lane 都参与 Shuffle，没有对应部分结果的 lane 使用 Max 单位元 -∞
        block_max = warp_reduce_max(block_max);

        // 由 Warp 0 的 lane 0 将最终代表整个 Block 的全局最大值写回 Shared Memory 首位
        if (lane_id == 0) warp_results[0] = block_max;
    }

    // 等待 Warp 0 写回最终结果，确保全 Block 线程广播读取前数据已就位
    __syncthreads();

    block_max = warp_results[0];

    // 所有线程读取完成后，才允许下一次 Reduction 复用 warp_results。
    __syncthreads();

    return block_max;
}

// 同理，两级 Block Reduce Sum：结合 Warp Shuffle 与极少量 Shared Memory 实现 Block 级累加和归约
__device__ __forceinline__
float block_reduce_sum_shuffle(float value, float* warp_results) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    // ----- 第一步：每个 Warp 在寄存器中独立归约，求出本 Warp 内 32 个线程的累加和 -----
    value = warp_reduce_sum(value);

    // 每个 Warp 只由 lane 0 将局部和写入 Shared Memory
    if (lane_id == 0) warp_results[warp_id] = value;

    __syncthreads();

    // ----- 第二步：由 Warp 0 读取所有 Warp 的局部和，完成最终的跨 Warp 累加 -----
    float block_sum = 0.0f;

    if (warp_id == 0) {
        block_sum = lane_id < num_warps ? warp_results[lane_id] : 0.0f;

        // Warp 0 的全部 32 个 lane 都参与 Shuffle。没有对应部分结果的 lane 使用 Sum 单位元 0。
        block_sum = warp_reduce_sum(block_sum);

        // 由 Warp 0 的 lane 0 将整行最终指数和写回 Shared Memory 首位
        if (lane_id == 0) warp_results[0] = block_sum;
    }

    __syncthreads();

    block_sum = warp_results[0];

    // 确保所有线程都读取完最终结果，之后才允许结束或重用 Shared Memory
    __syncthreads();

    return block_sum;
}

__global__
void softmax_shuffle_kernel(const float* input,
                            float* output,
                            int rows,
                            int cols) {
    // 动态 Shared Memory，仅需存储 NUM_WARPS 个 float 元素（如 256 线程只需 8 个 float = 32 bytes）
    extern __shared__ float warp_results[];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    // 边界防护：超出行数的 Block 直接退出
    if (row >= rows) {
        return;
    }

    // 计算当前 Block 负责行在全局内存中的起始偏移量
    const int row_offset = row * cols;

    // ----- 第一步：每个线程用 Block-stride Loop 跨步遍历自己负责的元素，计算局部最大值 -----
    float local_max = -CUDART_INF_F;

    for (int col = tid; col < cols; col += blockDim.x) {
        local_max = fmaxf(local_max, input[row_offset + col]);
    }

    // 第一次两级 Shuffle Reduction：高效汇聚出整行的最大值 row_max
    const float row_max = block_reduce_max_shuffle(local_max, warp_results);

    // ----- 第二步：利用 row_max 算数值稳定的 exp(x - row_max)，存入 output 暂存并累加局部指数和 -----
    float local_sum = 0.0f;

    for (int col = tid; col < cols; col += blockDim.x) {
        const int index = row_offset + col;
        const float exp_value = expf(input[index] - row_max);

        output[index] = exp_value;
        local_sum += exp_value;
    }

    // 第二次两级 Shuffle Reduction：高效汇聚出整行的指数和 row_sum
    const float row_sum = block_reduce_sum_shuffle(local_sum, warp_results);

    // ----- 第三步：归一化，将中间指数结果除以 row_sum，生成真正的概率分布 -----
    for (int col = tid; col < cols; col += blockDim.x) {
        const int index = row_offset + col;
        output[index] /= row_sum;
    }
}

// CPU 端标量参考实现，用于正确性比对
void softmax_cpu(const std::vector<float>& input,
                 std::vector<float>& output,
                 int rows,
                 int cols) {
    for (int row = 0; row < rows; ++row) {
        const int row_offset = row * cols;

        float row_max =
            -std::numeric_limits<float>::infinity();

        // 找整行最大值
        for (int col = 0; col < cols; ++col) {
            row_max = std::max(
                row_max,
                input[row_offset + col]);
        }

        double row_sum = 0.0;

        // 计算 exp 减最大值并累加行和，CPU 使用 double 提升校验参考精度
        for (int col = 0; col < cols; ++col) {
            const int index = row_offset + col;
            const double value = std::exp(
                static_cast<double>(
                    input[index] - row_max));

            output[index] = static_cast<float>(value);
            row_sum += value;
        }

        // 归一化
        for (int col = 0; col < cols; ++col) {
            output[row_offset + col] =
                static_cast<float>(
                    output[row_offset + col] / row_sum);
        }
    }
}

// 结果验证函数：比对 GPU 和 CPU 的绝对误差、相对误差，并检查每行概率之和是否接近 1
bool verify_result(const std::vector<float>& gpu_output,
                   const std::vector<float>& cpu_output,
                   int rows,
                   int cols) {
    constexpr float ABS_TOLERANCE = 1e-5f;

    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    float max_row_sum_error = 0.0f;
    int max_error_index = -1;

    // 逐元素对比误差
    for (std::size_t i = 0; i < gpu_output.size(); ++i) {
        if (!std::isfinite(gpu_output[i])) {
            std::cerr
                << "Non-finite GPU result at index "
                << i << '\n';
            return false;
        }

        const float abs_error =
            std::abs(gpu_output[i] - cpu_output[i]);

        const float denominator =
            std::max(std::abs(cpu_output[i]), 1e-7f);

        const float rel_error =
            abs_error / denominator;

        if (abs_error > max_abs_error) {
            max_abs_error = abs_error;
            max_error_index = static_cast<int>(i);
        }

        max_rel_error =
            std::max(max_rel_error, rel_error);
    }

    // 校验 Softmax 行和属性
    for (int row = 0; row < rows; ++row) {
        double row_sum = 0.0;

        for (int col = 0; col < cols; ++col) {
            row_sum += gpu_output[row * cols + col];
        }

        max_row_sum_error = std::max(
            max_row_sum_error,
            static_cast<float>(
                std::abs(row_sum - 1.0)));
    }

    std::cout
        << "Max absolute error: "
        << max_abs_error << '\n';

    std::cout
        << "Max relative error: "
        << max_rel_error << '\n';

    std::cout
        << "Max row-sum error: "
        << max_row_sum_error << '\n';

    if (max_error_index >= 0) {
        std::cout
            << "Max-error position: row "
            << max_error_index / cols
            << ", col "
            << max_error_index % cols
            << '\n';
    }

    return max_abs_error <= ABS_TOLERANCE;
}

int main() {
    constexpr std::size_t NUM_ELEMENTS =
        static_cast<std::size_t>(ROWS) * COLS;

    constexpr std::size_t BYTES =
        NUM_ELEMENTS * sizeof(float);

    std::vector<float> host_input(NUM_ELEMENTS);
    std::vector<float> host_gpu_output(NUM_ELEMENTS);
    std::vector<float> host_cpu_output(NUM_ELEMENTS);

    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(
        -10.0f,
        10.0f);

    // 制造大数值整行偏移，验证数值防溢出稳定性
    for (int row = 0; row < ROWS; ++row) {
        float row_shift = 0.0f;

        if (row % 3 == 0) {
            row_shift = 100.0f;
        } else if (row % 3 == 1) {
            row_shift = -100.0f;
        }

        for (int col = 0; col < COLS; ++col) {
            host_input[row * COLS + col] =
                distribution(generator) + row_shift;
        }
    }

    std::cout << "Rows: " << ROWS << '\n';
    std::cout << "Cols: " << COLS << '\n';
    std::cout << "Block size: " << BLOCK_SIZE << '\n';
    std::cout << "Warps per block: " << NUM_WARPS << '\n';
    std::cout << "Input elements: " << NUM_ELEMENTS << '\n';

    std::cout
        << "Dynamic shared memory per block: "
        << NUM_WARPS * sizeof(float)
        << " bytes\n";

    float* device_input = nullptr;
    float* device_output = nullptr;

    // 分配 GPU 内存
    CHECK_CUDA(cudaMalloc(&device_input, BYTES));
    CHECK_CUDA(cudaMalloc(&device_output, BYTES));

    // 数据传输至 GPU 显存
    CHECK_CUDA(cudaMemcpy(
        device_input,
        host_input.data(),
        BYTES,
        cudaMemcpyHostToDevice));

    // 配置 Grid 与 Block，每个 Block 只需要 NUM_WARPS * sizeof(float) 的极小 Shared Memory
    const dim3 grid(ROWS);
    const dim3 block(BLOCK_SIZE);

    const std::size_t shared_bytes =
        NUM_WARPS * sizeof(float);

    // 单次运行验证算子逻辑与数值正确性
    softmax_shuffle_kernel<<<grid, block, shared_bytes>>>(
        device_input,
        device_output,
        ROWS,
        COLS);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // 拷贝结果回 Host
    CHECK_CUDA(cudaMemcpy(
        host_gpu_output.data(),
        device_output,
        BYTES,
        cudaMemcpyDeviceToHost));

    // 运行 CPU 端参考对比函数
    softmax_cpu(
        host_input,
        host_cpu_output,
        ROWS,
        COLS);

    // 校验正确性
    const bool passed = verify_result(
        host_gpu_output,
        host_cpu_output,
        ROWS,
        COLS);

    std::cout
        << (passed
            ? "Softmax correctness check passed.\n"
            : "Softmax correctness check failed.\n");

    // 预热 GPU
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        softmax_shuffle_kernel<<<grid, block, shared_bytes>>>(
            device_input,
            device_output,
            ROWS,
            COLS);
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // 使用 CUDA Event 进行精准内核耗时测算
    cudaEvent_t start;
    cudaEvent_t stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    // 循环测试 Timing
    for (int i = 0; i < TIMING_ITERS; ++i) {
        softmax_shuffle_kernel<<<grid, block, shared_bytes>>>(
            device_input,
            device_output,
            ROWS,
            COLS);
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;

    CHECK_CUDA(cudaEventElapsedTime(
        &total_ms,
        start,
        stop));

    std::cout
        << "Average kernel time: "
        << total_ms / TIMING_ITERS
        << " ms\n";

    // 清理 CUDA Event 与显存
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(device_input));
    CHECK_CUDA(cudaFree(device_output));

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}