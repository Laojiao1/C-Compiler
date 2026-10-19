#pragma once

#include <cuda_runtime.h>

namespace cuda_ops {

    // C[M,N] = A[M,K] * B[N,K]^T when transpose_b=true;
    // otherwise C[M,N] = A[M,K] * B[K,N].
    cudaError_t launch_gemm_v1_tiled(
        const float* a,
        const float* b,
        float* c,
        int m,
        int n,
        int k,
        bool transpose_b,
        cudaStream_t stream = nullptr);

    // output = softmax(scale * input + causal_mask), row-wise.
    cudaError_t launch_fused_scale_mask_softmax(
        const float* input,
        float* output,
        float scale,
        int rows,
        int cols,
        bool causal,
        cudaStream_t stream = nullptr);

}  // namespace cuda_ops
