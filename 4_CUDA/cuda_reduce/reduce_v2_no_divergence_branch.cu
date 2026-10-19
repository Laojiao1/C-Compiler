#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

#define THREAD_PER_BLOCK 256

__global__ void reduce0(float *d_input, float *d_output)
{
    __shared__ float sdata[THREAD_PER_BLOCK];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = d_input[i];
    __syncthreads();

    // 优化2：将分散活跃改为连续活跃，让线程连续工作
    for (int s = 1; s < blockDim.x; s *= 2)
    {
        if(tid <  blockDim.x / (2 * s)) {
            int index = threadIdx.x * 2 * s;
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
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

    int block_num = N / THREAD_PER_BLOCK;
    float *output = (float *)malloc((N / THREAD_PER_BLOCK) * sizeof(float));
    float *d_output;
    cudaMalloc((void **)&d_output, (N / THREAD_PER_BLOCK) * sizeof(float));
    float *result = (float *)malloc((N / THREAD_PER_BLOCK) * sizeof(float));

    for (int i = 0; i < N; i++)
    {
        input[i] = 2.0 * (float)drand48() - 1.0;
        // input[i] = 1;
    }

    for (int i = 0; i < block_num; i++)
    {
        float cur = 0;
        for (int j = 0; j < THREAD_PER_BLOCK; j++)
        {
            cur += input[i * THREAD_PER_BLOCK + j];
        }
        result[i] = cur;
    }

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(N / THREAD_PER_BLOCK, 1);
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
