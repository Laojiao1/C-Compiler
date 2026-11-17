#include "attention_api.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "../cuda_ops.cuh"

// CPU
void attention_cpu_reference(
    const float* q, 
    const float* k, 
    const float* v, 
    float* output, 
    const gpu_config::AttentionShape& shape) {
    const int n = shape.sequence_length;
    const int d = shape.head_dim;

    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    const float negative_inf = -std::numeric_limits<float>::infinity();

    std::vector<float> scores(shape.score_elements());
    std::vector<float> p(shape.score_elements());

    // 计算 S = (Q * K^T) * scale + Mask
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float dot = 0.0f;
            for (int inner = 0; inner < d; ++inner) {
                // 直接改变遍历顺序，计算 QK^T
                dot += q[i * d + inner] * k[j * d + inner];
            }
            // mask
            scores[i * n + j] = (shape.causal && j > i) ? negative_inf : dot * scale;
        }
    }

    for(int i = 0; i < n; ++i) {
        float i_max = negative_inf;
        for(int j = 0; j < n; j++) {
            i_max = std::max(i_max, scores[i * n + j]);
        }
        float d = 0;
        for(int j = 0; j < n; ++j) {
            const float P = std::exp(scores[i * n + j] - i_max);
            p[i * n + j] = P;
            d += P;
        }
        for(int j = 0; j < n; ++j) {
            p[i * n + j] = p[i * n + j] / d;
        }
    }

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < d; j++) {
            float value = 0.0f;
            for(int inner = 0; inner < n; ++inner) {
                value += p[i * n + j] * v[i * d + j];
            }
            output[i * d + j] = value;
        }
    }
}

// unfused：3 kernel
cudaError_t launch_unfused_attention(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    float* scores,
    float* probabilities,
    const gpu_config::AttentionShape& shape,
    cudaStream_t stream
) {
    const int n = shape.sequence_length;
    const int d = shape.head_dim;

    if(n <= 0 || d <= 0) return cudaErrorInvalidValue;

    // S = Q * K^T
    cudaError_t status = cuda_ops::launch_gemm_v1_tiled(q, k, scores, n, n, d, true, stream);
    if(status != cudaSuccess) {
        return status;
    }

    // Scale + Mask + Row-wise Softmax
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    status = cuda_ops::launch_fused_scale_mask_softmax(scores, probabilities, scale, n, n, shape.causal, stream);
    if(status != cudaSuccess) {
        return status;
    }

    // O = P * V 
    return cuda_ops::launch_gemm_v1_tiled(probabilities, v, output, n, d, n, false, stream);
}

// flash-attention-online
constexpr int kWarpSize = 32;
constexpr int kQueryRows = 16;
constexpr int kKeyCols = 32;
constexpr int kKeySharedStride = kKeyCols + 1; // +1 防止 bank 冲突
constexpr int kMaxHeadDim = 128;
constexpr int kOutputValuesPerLane = (kMaxHeadDim + kWarpSize - 1)/kWarpSize;

__device__ __forceinline__ float warp_reduce_sum(float value) {
    #pragma unroll
    for(int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float warp_reduce_max(float value) {
    #pragma unroll
    for(int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

template <int HeadDim, bool Causal>
__global__ void flash_attention_online_kernel(
    const float* __restrict__ q, // 声明 restrict 告诉编译器无地址重叠，优化编译。建议只要是严格区分输入和输出的指针参数，都加上 restrict
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ output,
    int n, 
    int runtime_head_dim, // [n, d]
    float scale
) {
    const int d = HeadDim == 0 ? runtime_head_dim : HeadDim; // 动态传参
    // 存放 qkv 的共享内存
    extern __shared__ float shared[]; // extern 用来声明动态共享内存：这块共享内存的大小不在编译器写死，而是由 Host 端在启动 kernel 时动态传入

    // 将共享内存的空间分别切给 QKV
    float* q_tile = shared;
    float* k_tile = q_tile + kQueryRows * d;
    float* v_tile = k_tile + kKeySharedStride * d; // 默认K转置[d, kKeySharedStride]
    float* p_tile = v_tile + kKeyCols * d;

    // 全局内存加载到共享内存，以及因果掩码的跳块优化
    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    const int lane = tid % kWarpSize;
    const int q_start = blockIdx.x * kQueryRows; // 基地址
    const int q_index = q_start + warp_id; // 偏移
    const bool valid_query = (warp_id < kQueryRows) && (q_index < n);

    // 加载 Q
    for(int i = tid; i < kQueryRows * d; i += blockDim.x) {
       const int row = i / d;
       const int col = i % d;
       const int global_row = q_start + row;
       q_tile[i] = global_row < n ? q[global_row * d + col] : 0.0f;
    }
    __syncthreads();

    // K V tile + online softmax
    float row_max = -std::numeric_limits<float>::infinity();
    float row_sum = 0.0f;
    float o_reg[kOutputValuesPerLane] = {0.0f};

    // 总共需要处理的 Key tile 数
    const int all_keyTiles = (n + kKeyCols - 1) / kKeyCols;
    const int last_q_exclusive = min(n, q_start + kQueryRows);
    const int causal_key_tiles = (last_q_exclusive + kKeyCols - 1) / kKeyCols; // 因果跳块实际要处理的 Key tile 数
    const int key_tiles = Causal ? causal_key_tiles : all_keyTiles;

    // 外循环：一共需要处理几个 K 块 [n, d]
    for(int i = 0; i < key_tiles; ++i) {
        const int key_start = i * kKeyCols; // 基地址
        // 内循环：每次处理一个块
        for(int j = tid; j < kKeyCols * d; j += blockDim.x) {
            const int row = j / d;
            const int col = j % d;
            const int global_row = key_start + row;
            const bool valid_key = global_row < n;

            // 直接逻辑转置搬运 [d, bc+1]
            k_tile[col * kKeySharedStride + row] = valid_key ? k[global_row * d + col] : 0.0f;
            v_tile[j] = valid_key ? v[global_row * d + col] : 0.0f;
        }
        __syncthreads();

        if(valid_query) {
            const int key_index = key_start + lane;
            const bool keep = key_index < n && (!Causal || key_index <= q_index);
            float score = -std::numeric_limits<float>::infinity();
            if(keep) {
                // 流水线优化：采用4路累加器打断依赖链，防止流水线阻塞，当 dot3时，dot0 的结果正好算完
                float dot0 = 0.0f;
                float dot1 = 0.0f;
                float dot2 = 0.0f;
                float dot3 = 0.0f;
                int inner = 0;
                for(; inner + 3 < d; inner += 4) {
                    dot0 += q_tile[warp_id * d + inner + 0] * k_tile[(inner + 0) * kKeySharedStride + lane];
                    dot1 += q_tile[warp_id * d + inner + 1] * k_tile[(inner + 1) * kKeySharedStride + lane];
                    dot2 += q_tile[warp_id * d + inner + 2] * k_tile[(inner + 2) * kKeySharedStride + lane];
                    dot3 += q_tile[warp_id * d + inner + 3] * k_tile[(inner + 3) * kKeySharedStride + lane];
                }
                // 这里是故意这样写的，采用树形归约，减少指令发射
                float dot = (dot0 + dot1) + (dot2 + dot3);

                // 处理剩余维度
                for(; inner < d; ++inner) {
                    dot += q_tile[warp_id * d + inner] * k_tile[inner * kKeySharedStride + lane];
                }
                // 不要忘记缩放因子
                score = dot * scale;
            }

            // Online-Softmax 
            float tile_max = warp_reduce_max(score);
            tile_max = __shfl_sync(0xffffffffu, tile_max, 0); // 广播

            const float new_max = fmaxf(row_max, tile_max);
            const float alpha = expf(row_max - new_max);

            // 求分子
            const float p = keep ? expf(score - new_max) : 0.0f;
            p_tile[warp_id * kKeyCols + lane] = p;

            // 求分母
            float tile_sum = warp_reduce_sum(p);
            tile_sum = __shfl_sync(0xffffffffu, tile_sum, 0);
            const float new_sum = alpha * row_sum + tile_sum;

            __syncwarp();

            // 修正输出 output
            #pragma unroll
            for(int slot = 0; slot < kOutputValuesPerLane; ++slot) {
                const int o_col = slot * kWarpSize + lane;
                if(o_col < d) {
                    float pv = 0.0f;
                    #pragma unroll
                    for(int key_offset = 0; key_offset < kKeyCols; ++key_offset) {
                        // [kQueryRows, kKeyCols] * [kKeyCols, d] 
                        // [Br, Bc] * [Bc, d]
                        pv += p_tile[warp_id * kKeyCols + key_offset] * v_tile[key_offset * d + o_col];
                    }
                    o_reg[slot] = alpha * o_reg[slot] + pv;
                }
            }
            row_max = new_max;
            row_sum = new_sum;
        }
        __syncthreads();
    }

    // 最后在外部做唯一一次归一化除法
    if(valid_query) {
        const float inverse_sum = row_sum > 0.0f ? (1.0f / row_sum) : 0.0f;
        #pragma unroll
        for(int slot = 0; slot < kOutputValuesPerLane; ++slot) {
            const int o_col = slot * kWarpSize + lane;
            if(o_col < d) {
                output[q_index * d + o_col] = o_reg[slot] * inverse_sum;
            }
        }
    }
}