/**
 * nvcc -O3 -lineinfo -arch=sm_120 --ptxas-options=-v \
  softmax_profiling_benchmark.cu -o softmax_profiling_benchmark

  nsys profile --trace=cuda --stats=true --force-overwrite=true \
  --output=nsys/softmax_profiling_benchmark \
  -- ./softmax_profiling_benchmark

  ncu --set full --launch-skip 1 --launch-count 1 --force-overwrite \
  --export ncu/softmax_profiling_benchmark.ncu-rep \
  -- ./softmax_profiling_benchmark
 */
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)      \
                      << " at line " << __LINE__ << std::endl;          \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

#define WARP_SIZE 32

struct MDState {
    float m;
    float d;
};

__device__ __forceinline__ MDState combine_md(MDState a, MDState b) {
    if (a.m < b.m) {
        return {b.m, a.d * expf(a.m - b.m) + b.d};
    } else {
        return {a.m, b.d * expf(b.m - a.m) + a.d};
    }
}

__device__ __forceinline__ MDState warp_reduce_md(MDState state) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        MDState other;
        other.m = __shfl_down_sync(0xffffffff, state.m, mask);
        other.d = __shfl_down_sync(0xffffffff, state.d, mask);
        state = combine_md(state, other);
    }
    return state;
}

__device__ __forceinline__ MDState block_reduce_md(MDState state, MDState* s_mem) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    state = warp_reduce_md(state);

    if (lane_id == 0) {
        s_mem[warp_id] = state;
    }
    __syncthreads();

    int num_warps = blockDim.x / WARP_SIZE;
    MDState block_state = (threadIdx.x < num_warps) ? s_mem[lane_id] : MDState{-FLT_MAX, 0.0f};

    if (warp_id == 0) {
        block_state = warp_reduce_md(block_state);
    }

    if (threadIdx.x == 0) {
        s_mem[0] = block_state;
    }
    __syncthreads();

    return s_mem[0];
}

// Kernel v0 Naive
__global__ void find_row_max_kernel(const float* __restrict__ input, float* __restrict__ row_max, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float max_val = -FLT_MAX;
        for (int col = 0; col < N; ++col) {
            float val = input[row * N + col];
            if (val > max_val) max_val = val;
        }
        row_max[row] = max_val;
    }
}

__global__ void exp_and_sum_kernel(const float* __restrict__ input, const float* __restrict__ row_max, float* __restrict__ output_exp, float* __restrict__ row_sum, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float max_val = row_max[row];
        float sum_val = 0.0f;
        for (int col = 0; col < N; ++col) {
            int idx = row * N + col;
            float exp_val = expf(input[idx] - max_val);
            output_exp[idx] = exp_val;
            sum_val += exp_val;
        }
        row_sum[row] = sum_val;
    }
}

__global__ void normalize_kernel(const float* __restrict__ output_exp, const float* __restrict__ row_sum, float* __restrict__ output, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float sum_val = row_sum[row];
        float inv_sum = 1.0f / sum_val;
        for (int col = 0; col < N; ++col) {
            int idx = row * N + col;
            output[idx] = output_exp[idx] * inv_sum;
        }
    }
}

// Kernel v1 Block
__global__ void softmax_block_kernel(const float* __restrict__ input, float* __restrict__ output, int M, int N) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (row >= M) return;
    const float* row_input = input + row * N;
    float* row_output = output + row * N;

    float thread_max = -FLT_MAX;
    for (int col = tid; col < N; col += block_size) {
        thread_max = fmaxf(thread_max, row_input[col]);
    }
    sdata[tid] = thread_max;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }
    float row_max = sdata[0];

    float thread_sum = 0.0f;
    for (int col = tid; col < N; col += block_size) {
        thread_sum += expf(row_input[col] - row_max);
    }
    sdata[tid] = thread_sum;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    float row_sum = sdata[0];
    float inv_sum = 1.0f / row_sum;

    for (int col = tid; col < N; col += block_size) {
        row_output[col] = expf(row_input[col] - row_max) * inv_sum;
    }
}

// Kernel v4 Online
__global__ void softmax_online_kernel(const float* __restrict__ input, float* __restrict__ output, int M, int N) {
    extern __shared__ MDState s_mem_md[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (row >= M) return;
    const float* row_input = input + row * N;
    float* row_output = output + row * N;

    int num_vec = N / 4;
    int tail_start = num_vec * 4;

    MDState thread_md = {-FLT_MAX, 0.0f};
    const float4* vec_input = reinterpret_cast<const float4*>(row_input);
    for (int idx = tid; idx < num_vec; idx += block_size) {
        float4 val4 = vec_input[idx];
        thread_md = combine_md(thread_md, {val4.x, 1.0f});
        thread_md = combine_md(thread_md, {val4.y, 1.0f});
        thread_md = combine_md(thread_md, {val4.z, 1.0f});
        thread_md = combine_md(thread_md, {val4.w, 1.0f});
    }
    for (int col = tail_start + tid; col < N; col += block_size) {
        thread_md = combine_md(thread_md, {row_input[col], 1.0f});
    }

    MDState row_md = block_reduce_md(thread_md, s_mem_md);
    float row_max = row_md.m;
    float inv_sum = 1.0f / row_md.d;

    float4* vec_output = reinterpret_cast<float4*>(row_output);
    for (int idx = tid; idx < num_vec; idx += block_size) {
        float4 val4 = vec_input[idx];
        float4 out4;
        out4.x = expf(val4.x - row_max) * inv_sum;
        out4.y = expf(val4.y - row_max) * inv_sum;
        out4.z = expf(val4.z - row_max) * inv_sum;
        out4.w = expf(val4.w - row_max) * inv_sum;
        vec_output[idx] = out4;
    }
    for (int col = tail_start + tid; col < N; col += block_size) {
        row_output[col] = expf(row_input[col] - row_max) * inv_sum;
    }
}

// 统一的 Profiling 测量函数
float benchmark_kernel(int version, float* d_input, float* d_output, float* d_exp_temp, float* d_max_temp, float* d_sum_temp, int M, int N, int iterations = 50) {
    int threads_per_block = 256;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < 5; ++i) {
        if (version == 0) {
            int grid = (M + threads_per_block - 1) / threads_per_block;
            find_row_max_kernel<<<grid, threads_per_block>>>(d_input, d_max_temp, M, N);
            exp_and_sum_kernel<<<grid, threads_per_block>>>(d_input, d_max_temp, d_exp_temp, d_sum_temp, M, N);
            normalize_kernel<<<grid, threads_per_block>>>(d_exp_temp, d_sum_temp, d_output, M, N);
        } else if (version == 1) {
            size_t smem = threads_per_block * sizeof(float);
            softmax_block_kernel<<<M, threads_per_block, smem>>>(d_input, d_output, M, N);
        } else if (version == 4) {
            size_t smem = (threads_per_block / WARP_SIZE) * sizeof(MDState);
            softmax_online_kernel<<<M, threads_per_block, smem>>>(d_input, d_output, M, N);
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        if (version == 0) {
            int grid = (M + threads_per_block - 1) / threads_per_block;
            find_row_max_kernel<<<grid, threads_per_block>>>(d_input, d_max_temp, M, N);
            exp_and_sum_kernel<<<grid, threads_per_block>>>(d_input, d_max_temp, d_exp_temp, d_sum_temp, M, N);
            normalize_kernel<<<grid, threads_per_block>>>(d_exp_temp, d_sum_temp, d_output, M, N);
        } else if (version == 1) {
            size_t smem = threads_per_block * sizeof(float);
            softmax_block_kernel<<<M, threads_per_block, smem>>>(d_input, d_output, M, N);
        } else if (version == 4) {
            size_t smem = (threads_per_block / WARP_SIZE) * sizeof(MDState);
            softmax_online_kernel<<<M, threads_per_block, smem>>>(d_input, d_output, M, N);
        }
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iterations;
}

int main() {
    std::cout << "=== 4.4.6 Softmax Profiling & Sequence Length Scaling Benchmark ===" << std::endl;
    std::cout << std::left << std::setw(10) << "N (SeqLen)"
              << std::setw(15) << "v0 Naive (ms)"
              << std::setw(15) << "v1 Block (ms)"
              << std::setw(15) << "v4 Online (ms)"
              << std::setw(20) << "v4 Bandwidth (GB/s)" 
              << std::setw(20) << "v4 Throughput (GFLOPS)" << std::endl;
    std::cout << std::string(90, '-') << std::endl;

    // 固定总元素数约为 800 万，动态调整 M 与 N
    const std::vector<int> N_list = {128, 512, 2048, 8192, 16384};
    const int total_target = 4096 * 2048;

    for (int N : N_list) {
        int M = total_target / N;
        size_t matrix_size = M * N * sizeof(float);
        size_t vector_size = M * sizeof(float);

        float *d_input, *d_output, *d_exp_temp, *d_max_temp, *d_sum_temp;
        CHECK_CUDA(cudaMalloc(&d_input, matrix_size));
        CHECK_CUDA(cudaMalloc(&d_output, matrix_size));
        CHECK_CUDA(cudaMalloc(&d_exp_temp, matrix_size));
        CHECK_CUDA(cudaMalloc(&d_max_temp, vector_size));
        CHECK_CUDA(cudaMalloc(&d_sum_temp, vector_size));

        float ms_v0 = benchmark_kernel(0, d_input, d_output, d_exp_temp, d_max_temp, d_sum_temp, M, N);
        float ms_v1 = benchmark_kernel(1, d_input, d_output, d_exp_temp, d_max_temp, d_sum_temp, M, N);
        float ms_v4 = benchmark_kernel(4, d_input, d_output, d_exp_temp, d_max_temp, d_sum_temp, M, N);

        // v4 有效带宽: (3 * matrix_size) / time
        double bytes_v4 = 3.0 * matrix_size;
        double bw_v4 = (bytes_v4 / (ms_v4 / 1000.0)) / 1e9;

        // v4 GFLOPS: (5 * M * N) / time
        double gflops_v4 = (5.0 * M * N / (ms_v4 / 1000.0)) / 1e9;

        std::cout << std::left << std::setw(10) << N
                  << std::setw(15) << std::fixed << std::setprecision(4) << ms_v0
                  << std::setw(15) << ms_v1
                  << std::setw(15) << ms_v4
                  << std::setw(20) << std::setprecision(2) << bw_v4
                  << std::setw(20) << std::setprecision(2) << gflops_v4 << std::endl;

        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_exp_temp);
        cudaFree(d_max_temp);
        cudaFree(d_sum_temp);
    }

    return 0;
}