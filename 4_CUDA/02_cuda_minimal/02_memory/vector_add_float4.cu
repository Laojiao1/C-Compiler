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
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__   \
                      << " - " << cudaGetErrorString(error) << std::endl;   \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

// 标量版本：每个线程每轮处理一个 float
__global__ void vector_add_scalar(
    const float* a,
    const float* b,
    float* c,
    int n)
{
    // 当前线程在整个 Grid 中的全局线性索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 整个 Grid 的线程总数
    int stride = blockDim.x * gridDim.x;

    // Grid-stride Loop：
    // 同一轮中，相邻线程仍访问相邻元素，保持合并访存
    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

// float4 版本：每个线程每轮处理连续四个 float
__global__ void vector_add_float4(
    const float* a,
    const float* b,
    float* c,
    int n)
{
    // cudaMalloc 返回的基地址满足 float4 所需的 16 字节对齐
    // 转换后，一个 float4 元素对应原数组中的四个 float
    const float4* a4 = reinterpret_cast<const float4*>(a);
    const float4* b4 = reinterpret_cast<const float4*>(b);
    float4* c4 = reinterpret_cast<float4*>(c);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 可以完整组成 float4 的向量元素数量
    // 例如 n=10 时，n4=2，可以向量化处理前 8 个 float
    int n4 = n / 4;

    // i 是 float4 索引，不是原始 float 索引
    for (int i = idx; i < n4; i += stride) {
        float4 va = a4[i];
        float4 vb = b4[i];
        float4 vc;

        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;

        c4[i] = vc;
    }

    // 尾部边界处理：由于这里不够凑成一个float4，所以需要退回到最基础的单元素处理方式
    int tail_index = n4 * 4 + idx;

    if (tail_index < n) {
        c[tail_index] = a[tail_index] + b[tail_index];
    }
}

// 在 CPU 上检查计算结果
bool check_result(
    const std::vector<float>& a,
    const std::vector<float>& b,
    const std::vector<float>& c)
{
    for (std::size_t i = 0; i < c.size(); ++i) {
        float expected = a[i] + b[i];

        if (std::fabs(c[i] - expected) > 1e-5f) {
            std::cerr << "Mismatch at index " << i
                      << ": expected " << expected
                      << ", got " << c[i] << std::endl;
            return false;
        }
    }

    return true;
}

// 测量标量 Kernel 的平均执行时间
float benchmark_scalar(
    const float* d_a,
    const float* d_b,
    float* d_c,
    int n,
    int blocks,
    int threads,
    int warmup_iterations,
    int benchmark_iterations)
{
    // 预热：排除首次启动、缓存和频率状态等因素的影响
    for (int i = 0; i < warmup_iterations; ++i) {
        vector_add_scalar<<<blocks, threads>>>(d_a, d_b, d_c, n);
    }

    // 检查 Kernel 启动配置等同步前可发现的错误
    CUDA_CHECK(cudaGetLastError());

    // 等待预热 Kernel 全部执行完成
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // CUDA Event 在同一默认 Stream 中记录时间
    CUDA_CHECK(cudaEventRecord(start));

    // 执行
    for (int i = 0; i < benchmark_iterations; ++i) {
        vector_add_scalar<<<blocks, threads>>>(d_a, d_b, d_c, n);
    }

    CUDA_CHECK(cudaGetLastError());

    // stop 排在所有待测 Kernel 之后
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / benchmark_iterations;
}

// 测量 float4 Kernel 的平均执行时间
float benchmark_float4(
    const float* d_a,
    const float* d_b,
    float* d_c,
    int n,
    int blocks,
    int threads,
    int warmup_iterations,
    int benchmark_iterations)
{
    for (int i = 0; i < warmup_iterations; ++i) {
        vector_add_float4<<<blocks, threads>>>(d_a, d_b, d_c, n);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < benchmark_iterations; ++i) {
        vector_add_float4<<<blocks, threads>>>(d_a, d_b, d_c, n);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / benchmark_iterations;
}

// 根据算法必需的数据量计算有效带宽
double calculate_bandwidth_gbps(int n, float milliseconds)
{
    // 每个元素：
    // 读取 a：4 字节
    // 读取 b：4 字节
    // 写入 c：4 字节
    // 合计：12 字节
    double bytes =
        3.0 * static_cast<double>(n) * sizeof(float);

    double seconds = milliseconds * 1e-3;

    return bytes / seconds / 1e9;
}

int main()
{
    // 故意增加三个元素，使 n 不能被 4 整除
    // 用于验证 float4 Kernel 的标量尾部处理
    constexpr int n = (1 << 24) + 3;

    constexpr int threads = 256;
    constexpr int warmup_iterations = 10;
    constexpr int benchmark_iterations = 100;

    std::size_t bytes =
        static_cast<std::size_t>(n) * sizeof(float);

    // Host 端输入和输出数组
    std::vector<float> h_a(n);
    std::vector<float> h_b(n);
    std::vector<float> h_c(n);

    // 使用较小的周期数据，避免数值过大
    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i % 1000) * 0.001f;
        h_b[i] = static_cast<float>(i % 500) * 0.002f;
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    // 分配 Device Global Memory
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // 将两个输入数组复制到 GPU
    CUDA_CHECK(cudaMemcpy(
        d_a,
        h_a.data(),
        bytes,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_b,
        h_b.data(),
        bytes,
        cudaMemcpyHostToDevice));

    cudaDeviceProp device_prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, 0));

    // Grid-stride Loop 不要求 Grid 覆盖所有元素
    // 这里按 SM 数量配置足够多的常驻 Block
    int blocks = device_prop.multiProcessorCount * 32;

    std::cout << "Device: " << device_prop.name << '\n';
    std::cout << "N: " << n << '\n';
    std::cout << "Blocks: " << blocks << '\n';
    std::cout << "Threads per block: " << threads << '\n';

    // 标量版本性能测试
    float scalar_ms = benchmark_scalar(
        d_a,
        d_b,
        d_c,
        n,
        blocks,
        threads,
        warmup_iterations,
        benchmark_iterations);

    // 将标量版本结果复制回 Host 并检查正确性
    CUDA_CHECK(cudaMemcpy(
        h_c.data(),
        d_c,
        bytes,
        cudaMemcpyDeviceToHost));

    if (!check_result(h_a, h_b, h_c)) {
        std::cerr << "Scalar Vector Add failed." << std::endl;

        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));

        return EXIT_FAILURE;
    }

    // 清空输出，避免 float4 测试误用标量版本留下的数据
    CUDA_CHECK(cudaMemset(d_c, 0, bytes));

    // float4 版本使用与标量版本完全相同的 Grid 和计时配置
    float float4_ms = benchmark_float4(
        d_a,
        d_b,
        d_c,
        n,
        blocks,
        threads,
        warmup_iterations,
        benchmark_iterations);

    CUDA_CHECK(cudaMemcpy(
        h_c.data(),
        d_c,
        bytes,
        cudaMemcpyDeviceToHost));

    if (!check_result(h_a, h_b, h_c)) {
        std::cerr << "float4 Vector Add failed." << std::endl;

        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));

        return EXIT_FAILURE;
    }

    double scalar_bandwidth =
        calculate_bandwidth_gbps(n, scalar_ms);

    double float4_bandwidth =
        calculate_bandwidth_gbps(n, float4_ms);

    std::cout << "\nScalar kernel:\n";
    std::cout << "  Average time: " << scalar_ms << " ms\n";
    std::cout << "  Effective bandwidth: "
              << scalar_bandwidth << " GB/s\n";

    std::cout << "\nfloat4 kernel:\n";
    std::cout << "  Average time: " << float4_ms << " ms\n";
    std::cout << "  Effective bandwidth: "
              << float4_bandwidth << " GB/s\n";

    std::cout << "\nSpeedup: "
              << scalar_ms / float4_ms << "x\n";

    std::cout << "Both implementations passed.\n";

    // 释放 Device Memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return EXIT_SUCCESS;
}