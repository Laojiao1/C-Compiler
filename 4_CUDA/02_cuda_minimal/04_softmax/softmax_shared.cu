/**
 *  nvcc -O3 -lineinfo -arch=sm_120 -Xptxas -v \
    softmax_shared.cu -o softmax_shared \
    2>&1 | tee softmax_shared_ptxas.txt
 * 
 *  ptxas info    : 0 bytes gmem
    ptxas info    : Compiling entry function '_Z21softmax_shared_kernelPKfPfii' for 'sm_120'
    ptxas info    : Function properties for _Z21softmax_shared_kernelPKfPfii
        0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
    ptxas info    : Used 24 registers, used 1 barriers
    ptxas info    : Compile time = 8.714 ms
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
constexpr int WARMUP_ITERS = 10;
constexpr int TIMING_ITERS = 100;

// Shared Memory 树形 Reduce Max：在 Block 内部求解所有线程 local_max 的最大值
__device__ __forceinline__
float block_reduce_max_shared(float value, float* shared) {
    const int tid = threadIdx.x;

    // 将每个线程的局部最大值写入 Shared Memory
    shared[tid] = value;
    __syncthreads();

    // 折半归约循环：不断比较并保留较大的值，最终最大值汇聚到 shared[0]
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] = fmaxf(shared[tid], shared[tid + offset]);
        }
        __syncthreads();
    }

    float result = shared[0];

    // 确保所有线程都读取完结果，之后才能复用 Shared Memory
    __syncthreads();

    return result;
}

// Shared Memory 树形 Reduce Sum：在 Block 内部求解所有线程 local_sum 的累加和
__device__ __forceinline__
float block_reduce_sum_shared(float value, float* shared) {
    const int tid = threadIdx.x;

    // 将每个线程的局部指数和写入 Shared Memory
    shared[tid] = value;
    __syncthreads();

    // 折半归约循环：不断累加对应位置的值，最终整行指数和汇聚到 shared[0]
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }

    float result = shared[0];

    // 保证所有线程都已经取得最终 row_sum，之后才能安全复用或结束
    __syncthreads();

    return result;
}

__global__
void softmax_shared_kernel(const float* input,
                           float* output,
                           int rows,
                           int cols) {
    // 动态共享内存，大小由启动 Kernel 时的第三个参数指定（BLOCK_SIZE * sizeof(float)）
    extern __shared__ float shared[];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    // 边界防护：超出行范围的 Block 直接返回
    if (row >= rows) {
        return;
    }

    // 计算当前 Block 负责的行在全局内存中的起始地址偏移
    const int row_offset = row * cols;

    // 第一阶段：每个线程采用 Block-stride Loop 跨步遍历自己负责的元素，计算局部最大值
    // 单位元初始化为负无穷，保证任意实际数值都能正确更新最大值
    float local_max = -CUDART_INF_F;

    for (int col = tid; col < cols; col += blockDim.x) {
        local_max = fmaxf(local_max, input[row_offset + col]);
    }

    // 第一次 Block Reduction：汇聚所有线程的 local_max，得到整行的最大值 row_max
    const float row_max = block_reduce_max_shared(local_max, shared);

    // 第二阶段：利用 row_max 计算数值稳定的 exp(x - row_max)，写回 output 并求局部指数和
    float local_sum = 0.0f;

    for (int col = tid; col < cols; col += blockDim.x) {
        const int index = row_offset + col;
        const float exp_value = expf(input[index] - row_max);

        // 暂时将未经归一化的指数中间结果存入 output 数组，避免重复计算 expf
        output[index] = exp_value;
        local_sum += exp_value;
    }

    // 第二次 Block Reduction：汇聚所有线程的 local_sum，得到整行指数和 row_sum
    const float row_sum = block_reduce_sum_shared(local_sum, shared);

    // 第三阶段：归一化，将上一阶段算出的 exp 中间结果除以 row_sum 得到最终 Softmax 概率分布
    for (int col = tid; col < cols; col += blockDim.x) {
        const int index = row_offset + col;
        output[index] /= row_sum;
    }
}

// CPU 端 Softmax 参考实现：用于验证 GPU Kernel 的数值正确性
void softmax_cpu(const std::vector<float>& input,
                 std::vector<float>& output,
                 int rows,
                 int cols) {
    for (int row = 0; row < rows; ++row) {
        const int row_offset = row * cols;

        float row_max = -std::numeric_limits<float>::infinity();

        // 找整行最大值
        for (int col = 0; col < cols; ++col) {
            row_max = std::max(row_max, input[row_offset + col]);
        }

        double row_sum = 0.0;

        // 计算 exp(x - row_max) 并累加行和，使用 double 提高 CPU 端标量计算精度
        for (int col = 0; col < cols; ++col) {
            const int index = row_offset + col;
            const double value =
                std::exp(static_cast<double>(input[index] - row_max));

            output[index] = static_cast<float>(value);
            row_sum += value;
        }

        // 归一化
        for (int col = 0; col < cols; ++col) {
            output[row_offset + col] =
                static_cast<float>(output[row_offset + col] / row_sum);
        }
    }
}

// 结果验证函数：检查 absolute error、relative error 以及每行输出之和是否为 1
bool verify_result(const std::vector<float>& gpu_output,
                   const std::vector<float>& cpu_output,
                   int rows,
                   int cols) {
    constexpr float ABS_TOLERANCE = 1e-5f;

    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    float max_row_sum_error = 0.0f;
    int max_error_index = -1;

    // 逐元素对比 GPU 和 CPU 的输出结果
    for (std::size_t i = 0; i < gpu_output.size(); ++i) {
        if (!std::isfinite(gpu_output[i])) {
            std::cerr << "Non-finite GPU result at index " << i << '\n';
            return false;
        }

        const float abs_error =
            std::abs(gpu_output[i] - cpu_output[i]);

        const float denominator =
            std::max(std::abs(cpu_output[i]), 1e-7f);

        const float rel_error = abs_error / denominator;

        if (abs_error > max_abs_error) {
            max_abs_error = abs_error;
            max_error_index = static_cast<int>(i);
        }

        max_rel_error = std::max(max_rel_error, rel_error);
    }

    // 验证 Softmax 的数学性质：每一行的输出概率之和必须严格接近 1.0
    for (int row = 0; row < rows; ++row) {
        double row_sum = 0.0;

        for (int col = 0; col < cols; ++col) {
            row_sum += gpu_output[row * cols + col];
        }

        max_row_sum_error = std::max(
            max_row_sum_error,
            static_cast<float>(std::abs(row_sum - 1.0)));
    }

    std::cout << "Max absolute error: " << max_abs_error << '\n';
    std::cout << "Max relative error: " << max_rel_error << '\n';
    std::cout << "Max row-sum error: " << max_row_sum_error << '\n';

    if (max_error_index >= 0) {
        std::cout << "Max-error position: row "
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
    std::uniform_real_distribution<float> distribution(-10.0f, 10.0f);

    // 构造测试数据：加入大范围的整行数值偏移（+100 / -100），严苛验证减去 row_max 后的防溢出效果
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
    std::cout << "Input elements: " << NUM_ELEMENTS << '\n';

    std::cout << "Dynamic shared memory per block: "
              << BLOCK_SIZE * sizeof(float)
              << " bytes\n";

    float* device_input = nullptr;
    float* device_output = nullptr;

    // 在 GPU 上分配输入与输出内存
    CHECK_CUDA(cudaMalloc(&device_input, BYTES));
    CHECK_CUDA(cudaMalloc(&device_output, BYTES));

    // 将主机端数据拷贝至 GPU 显存
    CHECK_CUDA(cudaMemcpy(
        device_input,
        host_input.data(),
        BYTES,
        cudaMemcpyHostToDevice));

    // 网格与线程块配置：1 个 Block 负责 1 行，每个 Block 包含 BLOCK_SIZE 个线程
    const dim3 grid(ROWS);
    const dim3 block(BLOCK_SIZE);
    const std::size_t shared_bytes =
        BLOCK_SIZE * sizeof(float);

    // 首次运行 Kernel：用于验证算法的正确性
    softmax_shared_kernel<<<grid, block, shared_bytes>>>(
        device_input,
        device_output,
        ROWS,
        COLS);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // 将 GPU 输出结果拷贝回 Host 侧
    CHECK_CUDA(cudaMemcpy(
        host_gpu_output.data(),
        device_output,
        BYTES,
        cudaMemcpyDeviceToHost));

    // 运行 CPU 标量版 Softmax 算子
    softmax_cpu(
        host_input,
        host_cpu_output,
        ROWS,
        COLS);

    // 对比 GPU 与 CPU 结果的偏差
    const bool passed = verify_result(
        host_gpu_output,
        host_cpu_output,
        ROWS,
        COLS);

    std::cout << (passed
        ? "Softmax correctness check passed.\n"
        : "Softmax correctness check failed.\n");

    // 预热 GPU，确保 CUDA Context 已经完整初始化，排除指令加载开销
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        softmax_shared_kernel<<<grid, block, shared_bytes>>>(
            device_input,
            device_output,
            ROWS,
            COLS);
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // 创建 CUDA Event 用于精准记录硬件级耗时
    cudaEvent_t start;
    cudaEvent_t stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    // 循环执行多次 Kernel 测量平均耗时
    for (int i = 0; i < TIMING_ITERS; ++i) {
        softmax_shared_kernel<<<grid, block, shared_bytes>>>(
            device_input,
            device_output,
            ROWS,
            COLS);
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    std::cout << "Average kernel time: "
              << total_ms / TIMING_ITERS
              << " ms\n";

    // 释放 Event 与显存资源
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(device_input));
    CHECK_CUDA(cudaFree(device_output));

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}