#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

#define THREAD_PER_BLOCK 256

__device__ void warpReduce(volatile float *sdata, unsigned int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduce0(float *d_input, float *d_output) {
    volatile __shared__ float sdata[THREAD_PER_BLOCK];

    unsigned int tid = threadIdx.x;

    unsigned int i = 2 * blockIdx.x * blockDim.x + tid;
    sdata[tid] = d_input[i] + d_input[i + blockDim.x];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s /= 2) {
        if(tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 当循环迭代到 s < 32 时，整个 Block 内只剩下 Warp 0（线程 0~31） 在工作
    // 此时可以把循环在 s=32 处截断，展开为纯 Warp 内计算，节省 __syncthreads()
    if(tid < 32) {
        warpReduce(sdata, tid);
    }

    if (tid == 0)
        d_output[blockIdx.x] = sdata[0];
}

bool check(float *out, float *res, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (fabs(out[i] - res[i]) > 1e-2)
            return false;
    }
    return true;
}

int main()
{
    const int N = 32 * 1024 * 1024;
    float *input = (float *)malloc(N * sizeof(float));
    float *d_input;
    cudaMalloc((void **)&d_input, N * sizeof(float));

    int block_num = N / THREAD_PER_BLOCK / 2;
    float *output = (float *)malloc(block_num * sizeof(float));
    float *d_output;
    cudaMalloc((void **)&d_output, block_num * sizeof(float));
    float *result = (float *)malloc(block_num * sizeof(float));

    for (int i = 0; i < N; i++)
    {
        input[i] = 2.0 * (float)drand48() - 1.0;
        // input[i] = 1;
    }

    for (int i = 0; i < block_num; i++)
    {
        float cur = 0;
        for (int j = 0; j < 2 * THREAD_PER_BLOCK; j++)
        {
            cur += input[i * 2 * THREAD_PER_BLOCK + j];
        }
        result[i] = cur;
    }

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(block_num, 1);
    dim3 Block(THREAD_PER_BLOCK, 1);

    reduce0<<<Grid, Block>>>(d_input, d_output);

    cudaMemcpy(output, d_output, block_num * sizeof(float), cudaMemcpyDeviceToHost);

    if (check(output, result, block_num))
        printf("the ans is right\n");
    else
    {
        printf("the ans is wrong\n");
        // for (int i = 0; i < block_num; i++)
        // {
        //     printf("%lf ", output[i]);
        // }
        // printf("\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);
}
