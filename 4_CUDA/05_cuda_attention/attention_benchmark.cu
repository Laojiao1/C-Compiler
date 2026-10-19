#include "attention_api.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t status_ = (call);                                      \
        if (status_ != cudaSuccess) {                                            \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + \
                                     ":" + std::to_string(__LINE__) + ": " +   \
                                     cudaGetErrorString(status_));               \
        }                                                                       \
    } while (0)

namespace {

struct BenchmarkOptions {
    gpu_config::AttentionShape shape{
        gpu_config::benchmark::kDefaultSequenceLength,
        gpu_config::benchmark::kDefaultHeadDim,
        true};
    int warmup_iterations = gpu_config::benchmark::kDefaultWarmupIterations;
    int iterations = gpu_config::benchmark::kDefaultIterations;
};

struct VerificationResult {
    bool passed = true;
    float max_absolute_error = 0.0f;
    // normalized_error = abs_error / (atol + rtol * abs(reference))。
    // 它与 PASS 判据完全一致，<=1 表示通过；不会在 reference≈0 时虚高。
    float max_normalized_error = 0.0f;
    std::size_t worst_index = 0;
};

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t elements) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), elements * sizeof(float)));
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    float* get() { return data_; }
private:
    float* data_ = nullptr;
};

class CudaEvent {
public:
    CudaEvent() { CUDA_CHECK(cudaEventCreate(&event_)); }
    ~CudaEvent() { cudaEventDestroy(event_); }
    operator cudaEvent_t() const { return event_; }

private:
    cudaEvent_t event_{};
};

int parse_positive_int(const char* text, const char* option_name) {
    const std::string value(text);
    std::size_t consumed = 0;
    const int parsed = std::stoi(value, &consumed);
    if (consumed != value.size() || parsed <= 0) {
        throw std::invalid_argument(std::string(option_name) + " requires a positive integer");
    }
    return parsed;
}

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program
        << " [--n N] [--d D] [--iters I] [--warmup W] [--causal|--no-causal]\n";
}

BenchmarkOptions parse_options(int argc, char** argv) {
    BenchmarkOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        auto require_value = [&](const char* option_name) -> const char* {
            if (++index >= argc) {
                throw std::invalid_argument(std::string(option_name) + " requires a value");
            }
            return argv[index];
        };

        if (argument == "--n") {
            options.shape.sequence_length = parse_positive_int(require_value("--n"), "--n");
        } else if (argument == "--d") {
            options.shape.head_dim = parse_positive_int(require_value("--d"), "--d");
        } else if (argument == "--iters") {
            options.iterations = parse_positive_int(require_value("--iters"), "--iters");
        } else if (argument == "--warmup") {
            options.warmup_iterations = parse_positive_int(require_value("--warmup"), "--warmup");
        } else if (argument == "--causal") {
            options.shape.causal = true;
        } else if (argument == "--no-causal") {
            options.shape.causal = false;
        } else if (argument == "--help" || argument == "-h") {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }

    if (options.shape.head_dim > gpu_config::flash::kMaxHeadDim) {
        throw std::invalid_argument(
            "head dimension exceeds FlashAttention kMaxHeadDim=" +
            std::to_string(gpu_config::flash::kMaxHeadDim));
    }
    return options;
}

VerificationResult verify(
    const std::vector<float>& reference,
    const std::vector<float>& candidate) {
    VerificationResult result;
    for (std::size_t index = 0; index < reference.size(); ++index) {
        const float ref = reference[index];
        const float got = candidate[index];
        if (!std::isfinite(got)) {
            result.passed = false;
            result.worst_index = index;
            result.max_absolute_error = std::numeric_limits<float>::infinity();
            return result;
        }

        const float absolute_error = std::abs(got - ref);
        const float allowed = gpu_config::benchmark::kAbsoluteTolerance +
                              gpu_config::benchmark::kRelativeTolerance * std::abs(ref);
        const float normalized_error = absolute_error / allowed;
        if (absolute_error > result.max_absolute_error) {
            result.max_absolute_error = absolute_error;
            result.worst_index = index;
        }
        result.max_normalized_error =
            std::max(result.max_normalized_error, normalized_error);

        if (absolute_error > allowed) {
            result.passed = false;
        }
    }
    return result;
}

void print_verification(const char* name, const VerificationResult& result) {
    std::cout << "  " << std::left << std::setw(24) << name
              << (result.passed ? "PASS" : "FAIL")
              << "  max_abs=" << std::scientific << result.max_absolute_error
              << "  max_norm=" << result.max_normalized_error
              << "  worst_index=" << result.worst_index << '\n';
}

template <typename Launch>
float benchmark_kernel(Launch&& launch, int warmup_iterations, int iterations) {
    for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
        CUDA_CHECK(launch());
    }
    // 同步既隔离预热，也把异步 kernel 错误定位在计时之前。
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaEvent start;
    CudaEvent stop;
    CUDA_CHECK(cudaEventRecord(start));
    for (int iteration = 0; iteration < iterations; ++iteration) {
        CUDA_CHECK(launch());
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    return elapsed_ms / iterations;
}

double to_mib(double bytes) {
    return bytes / (1024.0 * 1024.0);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const BenchmarkOptions options = parse_options(argc, argv);
        const auto shape = options.shape;
        const std::size_t qkv_elements = shape.qkv_elements();
        const std::size_t score_elements = shape.score_elements();
        const std::size_t qkv_bytes = qkv_elements * sizeof(float);
        const std::size_t score_bytes = score_elements * sizeof(float);

        int device = 0;
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

        std::cout << "=== CUDA Attention: Unfused vs Tiled Online ===\n"
                  << "GPU: " << properties.name << '\n'
                  << "Shape: N=" << shape.sequence_length
                  << ", d=" << shape.head_dim
                  << ", causal=" << std::boolalpha << shape.causal << '\n'
                  << "Tiles: GEMM=" << gpu_config::gemm::kTileM << 'x'
                  << gpu_config::gemm::kTileN << 'x' << gpu_config::gemm::kTileK
                  << ", CUDA-Core Flash Br=" << gpu_config::flash::kQueryRows
                  << ", Bc=" << gpu_config::flash::kKeyCols
                  << ", Tensor Flash Br="
                  << gpu_config::flash_tensor_core::kQueryRows
                  << ", Bc=" << gpu_config::flash_tensor_core::kKeyCols << "\n\n";

        std::vector<float> host_q(qkv_elements);
        std::vector<float> host_k(qkv_elements);
        std::vector<float> host_v(qkv_elements);
        std::vector<float> host_reference(qkv_elements);
        std::vector<float> host_unfused(qkv_elements);
        std::vector<float> host_flash(qkv_elements);
        std::vector<float> host_tensor_core(qkv_elements);

        std::mt19937 generator(20260804);
        std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
        for (std::size_t index = 0; index < qkv_elements; ++index) {
            host_q[index] = distribution(generator);
            host_k[index] = distribution(generator);
            host_v[index] = distribution(generator);
        }

        const auto cpu_start = std::chrono::steady_clock::now();
        attention_cpu_reference(
            host_q.data(), host_k.data(), host_v.data(), host_reference.data(), shape);
        const auto cpu_stop = std::chrono::steady_clock::now();
        const double cpu_ms =
            std::chrono::duration<double, std::milli>(cpu_stop - cpu_start).count();

        DeviceBuffer device_q(qkv_elements);
        DeviceBuffer device_k(qkv_elements);
        DeviceBuffer device_v(qkv_elements);
        DeviceBuffer device_unfused(qkv_elements);
        DeviceBuffer device_flash(qkv_elements);
        DeviceBuffer device_tensor_core(qkv_elements);
        DeviceBuffer device_scores(score_elements);
        DeviceBuffer device_probabilities(score_elements);

        CUDA_CHECK(cudaMemcpy(device_q.get(), host_q.data(), qkv_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_k.get(), host_k.data(), qkv_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_v.get(), host_v.data(), qkv_bytes, cudaMemcpyHostToDevice));

        auto launch_unfused = [&]() {
            return launch_unfused_attention(
                device_q.get(), device_k.get(), device_v.get(), device_unfused.get(),
                device_scores.get(), device_probabilities.get(), shape);
        };
        auto launch_flash = [&]() {
            return launch_flash_attention_online(
                device_q.get(), device_k.get(), device_v.get(), device_flash.get(), shape);
        };
        const bool tensor_core_supported =
            shape.head_dim == gpu_config::flash_tensor_core::kHeadDim &&
            properties.major >= 7;
        auto launch_tensor_core = [&]() {
            return launch_flash_attention_tensor_core(
                device_q.get(), device_k.get(), device_v.get(),
                device_tensor_core.get(), shape);
        };

        CUDA_CHECK(launch_unfused());
        CUDA_CHECK(launch_flash());
        if (tensor_core_supported) {
            CUDA_CHECK(launch_tensor_core());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(
            host_unfused.data(), device_unfused.get(), qkv_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            host_flash.data(), device_flash.get(), qkv_bytes, cudaMemcpyDeviceToHost));
        if (tensor_core_supported) {
            CUDA_CHECK(cudaMemcpy(
                host_tensor_core.data(), device_tensor_core.get(), qkv_bytes,
                cudaMemcpyDeviceToHost));
        }

        std::cout << "Correctness (atol=" << gpu_config::benchmark::kAbsoluteTolerance
                  << ", rtol=" << gpu_config::benchmark::kRelativeTolerance << "):\n";
        const VerificationResult unfused_vs_cpu = verify(host_reference, host_unfused);
        const VerificationResult flash_vs_cpu = verify(host_reference, host_flash);
        const VerificationResult flash_vs_unfused = verify(host_unfused, host_flash);
        VerificationResult tensor_vs_cpu;
        print_verification("Unfused vs CPU", unfused_vs_cpu);
        print_verification("Flash vs CPU", flash_vs_cpu);
        print_verification("Flash vs Unfused", flash_vs_unfused);
        if (tensor_core_supported) {
            tensor_vs_cpu = verify(host_reference, host_tensor_core);
            print_verification("Tensor Core vs CPU", tensor_vs_cpu);
        } else {
            std::cout << "  " << std::left << std::setw(24)
                      << "Tensor Core vs CPU" << "SKIP  (requires d=64 and SM70+)\n";
        }
        std::cout << std::defaultfloat << "  CPU reference time: " << cpu_ms << " ms\n\n";

        const float unfused_ms = benchmark_kernel(
            launch_unfused, options.warmup_iterations, options.iterations);
        const float flash_ms = benchmark_kernel(
            launch_flash, options.warmup_iterations, options.iterations);
        float tensor_core_ms = 0.0f;
        if (tensor_core_supported) {
            tensor_core_ms = benchmark_kernel(
                launch_tensor_core, options.warmup_iterations, options.iterations);
        }

        std::cout << std::fixed << std::setprecision(4)
                  << "Kernel timing (" << options.iterations << " iterations):\n"
                  << "  Unfused: " << unfused_ms << " ms, "
                  << kUnfusedAttentionLaunches << " launches/iteration\n"
                  << "  Flash:   " << flash_ms << " ms, "
                  << kFlashAttentionLaunches << " launch/iteration\n"
                  << "  Flash speedup: " << unfused_ms / flash_ms << "x\n";
        if (tensor_core_supported) {
            std::cout << "  Tensor:  " << tensor_core_ms << " ms, "
                      << kTensorCoreAttentionLaunches << " launch/iteration\n"
                      << "  Tensor speedup vs Unfused: "
                      << unfused_ms / tensor_core_ms << "x\n"
                      << "  Tensor speedup vs CUDA-Core Flash: "
                      << flash_ms / tensor_core_ms << "x\n";
        }
        std::cout << '\n';

        const AttentionTraffic unfused_traffic = unfused_traffic_model(shape);
        const AttentionTraffic flash_traffic = flash_traffic_model(shape);
        const AttentionTraffic tensor_traffic = tensor_core_traffic_model(shape);
        const double materialization_lower_bound = 4.0 * score_bytes;
        std::cout
            << "Logical global-memory model (cache effects ignored):\n"
            << "  Unfused total:       " << to_mib(unfused_traffic.total_bytes) << " MiB\n"
            << "  Unfused S/P traffic: " << to_mib(unfused_traffic.intermediate_sp_bytes)
            << " MiB (this tiled implementation)\n"
            << "  S/P materialization lower bound: " << to_mib(materialization_lower_bound)
            << " MiB (write+read S and P once)\n"
            << "  Flash total:         " << to_mib(flash_traffic.total_bytes) << " MiB\n"
            << "  Flash S/P traffic:   0.0000 MiB (tile-local Shared Memory only)\n"
            << "  Tensor total:        " << to_mib(tensor_traffic.total_bytes) << " MiB\n"
            << "  Tensor S/P traffic:  0.0000 MiB (tile-local Shared Memory only)\n"
            << "  Unfused S+P workspace: " << to_mib(2.0 * score_bytes) << " MiB\n";

        const bool all_passed =
            unfused_vs_cpu.passed && flash_vs_cpu.passed && flash_vs_unfused.passed &&
            (!tensor_core_supported || tensor_vs_cpu.passed);
        std::cout << "\nOverall: " << (all_passed ? "PASS" : "FAIL") << '\n';
        return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
}
