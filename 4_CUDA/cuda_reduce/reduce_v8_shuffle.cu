#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define THREAD_PER_BLOCK 256
#define WARP_SIZE 32

// Warp 级 Shuffle 快速归约（5次折半寄存器交换，替代 volatile shared mem）
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template<unsigned int NUM_PER_BLOCK, unsigned int NUM_PER_THREAD>
__global__ void reduce_shuffle(const float *d_input, float *d_output) {
    // 仅需保存每个 Warp 的结果：256 线程只有 8 个 Warp
    constexpr int NUM_WARPS = THREAD_PER_BLOCK / WARP_SIZE;
    __shared__ float warp_sums[NUM_WARPS];

    unsigned int tid = threadIdx.x;
    unsigned int lane_id = tid % WARP_SIZE;
    unsigned int warp_id = tid / WARP_SIZE;

    // 1. 本地寄存器累加全局内存数据
    unsigned int base_idx = blockIdx.x * NUM_PER_BLOCK + tid;
    float sum = 0.0f;
    #pragma unroll
    for (int s = 0; s < NUM_PER_THREAD; s++) {
        sum += d_input[base_idx + s * THREAD_PER_BLOCK];
    }

    // 2. 第一阶段：每个 Warp 内部通过 Shuffle 归约
    sum = warp_reduce_sum(sum);

    // 每个 Warp 的 Lane 0 负责把结果写入 Shared Memory
    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    // 3. 第二阶段：由 Warp 0 将所有 Warp 的局部和做最终归约
    // 只有前 NUM_WARPS 个线程需要参与加载
    sum = (tid < NUM_WARPS) ? warp_sums[lane_id] : 0.0f;

    if (warp_id == 0) {
        sum = warp_reduce_sum(sum);
    }

    // 4. 写回当前 Block 的最终结果
    if (tid == 0) {
        d_output[blockIdx.x] = sum;
    }
}

bool check(float *out, float *res, int n) {
    for (int i = 0; i < n; i++) {
        if (fabs(out[i] - res[i]) > 1e-2) {
            return false;
        }
    }
    return true;
}

int main() {
    const int N = 32 * 1024 * 1024;
    size_t bytes_input = N * sizeof(float);

    float *input = (float *)malloc(bytes_input);
    float *d_input;
    cudaMalloc((void **)&d_input, bytes_input);

    constexpr int block_num = 1024;
    constexpr int num_per_block = N / block_num;
    constexpr int num_per_thread = num_per_block / THREAD_PER_BLOCK;

    size_t bytes_output = block_num * sizeof(float);
    float *output = (float *)malloc(bytes_output);
    float *d_output;
    cudaMalloc((void **)&d_output, bytes_output);
    float *result = (float *)malloc(bytes_output);

    for (int i = 0; i < N; i++) {
        input[i] = 2.0f * (float)drand48() - 1.0f;
    }

    // CPU 基准校验
    for (int i = 0; i < block_num; i++) {
        float cur = 0.0f;
        for (int j = 0; j < num_per_block; j++) {
            cur += input[i * num_per_block + j];
        }
        result[i] = cur;
    }

    cudaMemcpy(d_input, input, bytes_input, cudaMemcpyHostToDevice);

    dim3 Grid(block_num, 1);
    dim3 Block(THREAD_PER_BLOCK, 1);

    reduce_shuffle<num_per_block, num_per_thread><<<Grid, Block>>>(d_input, d_output);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, bytes_output, cudaMemcpyDeviceToHost);

    if (check(output, result, block_num)) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);
    free(input);
    free(output);
    free(result);

    return 0;
}