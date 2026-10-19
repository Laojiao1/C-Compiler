#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

// 统一宏检查
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t error = (call);                                        \
        if (error != cudaSuccess) {                                        \
            std::cerr << "CUDA error at "                                  \
                      << __FILE__ << ":" << __LINE__ << "\n"                \
                      << "  code: " << static_cast<int>(error) << "\n"      \
                      << "  name: " << cudaGetErrorName(error) << "\n"      \
                      << "  message: " << cudaGetErrorString(error)         \
                      << std::endl;                                        \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)


__global__ void vector_add_grid_stride_kernel(
    const float* x,
    const float* y,
    float* output,
    std::size_t n
) {
    std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;

    std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * gridDim.x;

    for (std::size_t i = index; i < n; i += stride) {
        output[i] = x[i] + y[i];
    }
}


int main() {
    constexpr std::size_t n = 1'000'003; // 元素数量：故意采用不能被整除的长度，可以验证边界处理是否正确
    const std::size_t bytes = n * sizeof(float); // 实际分配和复制的字节数

    // Host 数据可以使用普通 c++ 容器来分配 Device Memory
    std::vector<float> h_x(n);
    std::vector<float> h_y(n);
    std::vector<float> h_output(n);

    for (std::size_t i = 0; i < n; ++i) {
        h_x[i] = static_cast<float>(i) * 0.001f;
        h_y[i] = static_cast<float>(i % 100) * 0.01f;
    }

    // Device 指针通过 cudaMalloc 分配
    float* d_x = nullptr;
    float* d_y = nullptr;
    float* d_output = nullptr;

    // CUDA 检查
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_x), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_y), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output), bytes));

    // Host -> Device：使用 cudaMemcpy 将输入同步复制到 GPU 显存
    CUDA_CHECK(cudaMemcpy(
        d_x,
        h_x.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_y,
        h_y.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    // 启动 kernel
    constexpr int block_size = 256;
    int grid_size = static_cast<int>((n + block_size - 1) / block_size);

    // ---- Runtime -----
    constexpr int warmup = 10;
    // 预热
    for(int i = 0; i < warmup; ++i) {
        vector_add_grid_stride_kernel<<<grid_size, block_size>>> (
            d_x,
            d_y,
            d_output,
            n
        );
    }

    // 由于 kernel launch 对 host 通常是异步的，因此提交并不代表 kernel 已经执行完毕，所以需要检查两类不同错误
    // Launch Error → 检查提交和启动是否成功
    // kernel 启动后立即检查，主要用来捕获启动阶段错误，比如 block 超过硬件限制，配置非法等等
    CUDA_CHECK(cudaGetLastError());

    // 异步执行错误 → 等待 GPU 完成，并检查执行阶段错误
    // 捕获非法现存访问，越界访问，未对其访问等 kernel 执行期间发生的其他错误
    CUDA_CHECK(cudaDeviceSynchronize());

    // CUDA Event 计时
    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    constexpr int repeat = 100;

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repeat; ++i) {
        vector_add_grid_stride_kernel<<<grid_size, block_size>>>(
            d_x,
            d_y,
            d_output,
            n
        );
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &total_ms,
        start,
        stop
    ));

    // 计算一次 Kernel 的平均 GPU 时间
    const float average_ms = total_ms / repeat;

    // 计时完成后释放 Event
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // 理论数据量
    const double transferred_bytes = static_cast<double>(n) * 3.0 * sizeof(float);

    // 将毫秒转换为秒
    const double average_seconds = static_cast<double>(average_ms) / 1000.0;

    // 计算有效带宽
    const double bandwidth_gb_s = transferred_bytes / average_seconds / 1.0e9;

    // 打印结果：
    std::cout << "Average kernel time: " << average_ms << " ms\n";
    std::cout << "Effective bandwidth: " << bandwidth_gb_s << " GB/s\n";

    // Device -> Host，取回最后一次 Kernel 的结果
    CUDA_CHECK(cudaMemcpy(
        h_output.data(),
        d_output,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    // CPU 正确性验证
    bool correct = true;
    constexpr float tolerance = 1e-6f;

    for (std::size_t i = 0; i < n; ++i) {
        const float expected = h_x[i] + h_y[i];
        const float difference = std::fabs(h_output[i] - expected);

        if (difference > tolerance) {
            std::cerr << "Mismatch at index " << i
                      << ": expected = " << expected
                      << ", actual = " << h_output[i]
                      << std::endl;

            correct = false;
            break;
        }
    }

    if (correct) {
        std::cout << "Vector Add passed." << std::endl;
    } else {
        std::cout << "Vector Add failed." << std::endl;
    }

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_output));

    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}