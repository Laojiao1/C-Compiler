#include "attention_api.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "../cuda_ops.cuh"

void attention_cpu_reference(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    const gpu_config::AttentionShape& shape) {

    const int n = shape.sequence_length;
    const int d = shape.head_dim;

    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    const float negative_infinity = -std::numeric_limits<float>::infinity();

    // 1. 显式分配内存，物化 S [N, N] 与 P [N, N] 矩阵
    std::vector<float> scores(shape.score_elements());
    std::vector<float> probabilities(shape.score_elements());

    // 计算得分矩阵 S = (Q * K^T) * scale + Mask
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            float dot = 0.0f;
            for (int inner = 0; inner < d; ++inner) {
                // 直接改变遍历顺序，计算 QK^T
                dot += q[row * d + inner] * k[col * d + inner];
            }
            // mask
            scores[row * n + col] =
                (shape.causal && col > row) ? negative_infinity : dot * scale;
        }
    }

    // 使用 Safe Softmax 减去 row_max 避免数值溢出
    for (int row = 0; row < n; ++row) {
        float row_max = negative_infinity;
        for (int col = 0; col < n; ++col) {
            row_max = std::max(row_max, scores[row * n + col]); // 不断更新 m
        }

        float denominator = 0.0f;
        for (int col = 0; col < n; ++col) {
            const float probability = std::exp(scores[row * n + col] - row_max); // 计算指数
            probabilities[row * n + col] = probability; // 存储指数
            denominator += probability; // 累加
        }
        for (int col = 0; col < n; ++col) {
            probabilities[row * n + col] /= denominator; // 计算 softmax
        }
    }

    // 计算最终输出 O = P * V
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < d; ++col) {
            float value = 0.0f;
            for (int inner = 0; inner < n; ++inner) {
                value += probabilities[row * n + inner] * v[inner * d + col]; // 计算 PV
            }
            output[row * d + col] = value; // 存储
        }
    }
}

// GPU 逻辑：采用3次Kernel，中间结果S和P必须完整写入HBM(全局显存)中，在下一个kernel读出来，读写瓶颈极其严重
cudaError_t launch_unfused_attention(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    float* scores,
    float* probabilities,
    const gpu_config::AttentionShape& shape,
    cudaStream_t stream) {

    const int n = shape.sequence_length;
    const int d = shape.head_dim;
    
    if (n <= 0 || d <= 0) {
        return cudaErrorInvalidValue;
    }

    // ----- 计算 S = Q * K^T -----
    // 复用阶段三的 tiled GEMM：transpose_b=true 表示 K 保持 [N,d]。存储但按 K^T 读取，不产生额外转置矩阵。
    cudaError_t status = cuda_ops::launch_gemm_v1_tiled(
        q, k, scores, n, n, d, true, stream);
    if (status != cudaSuccess) {
        return status;
    }

    // ----- 启动单独的 Softmax Kernel 完成 Scale + Mask + Row-wise Softmax ----- 
    // 直接复用阶段四的 fused scale + mask + softmax，S 与 P 仍是两个独立全局矩阵，所以这里只减少 launch，不改变 Unfused 的物化本质。
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    status = cuda_ops::launch_fused_scale_mask_softmax(
        scores, probabilities, scale, n, n, shape.causal, stream);
    if (status != cudaSuccess) {
        return status;
    }

    // 调用传统 GEMM 计算 O = P * V 
    return cuda_ops::launch_gemm_v1_tiled(
        probabilities, v, output, n, d, n, false, stream);
}

// unfused 显存流量分析函数
AttentionTraffic unfused_traffic_model(const gpu_config::AttentionShape& shape) {
    const double n = static_cast<double>(shape.sequence_length);
    const double d = static_cast<double>(shape.head_dim);
    const double tile = static_cast<double>(gpu_config::gemm::kTileM);
    const double n_tiles = std::ceil(n / tile);
    const double d_tiles = std::ceil(d / tile);

    // 1. QK 阶段读取 Q 和 K 的逻辑流量
    const double qk_reads = 2.0 * n_tiles * n * d;
    // 写出 S 矩阵 [N, N]
    const double qk_score_write = n * n;

    // 2. Fused Softmax 读取 S 三遍 (找 max、求 sum、算 exp)，写出 P 一遍：4 * N^2
    const double softmax_accesses = 4.0 * n * n;

    // 3. PV 阶段：P 矩阵被各个 d-tile 重复读取，V 矩阵被各个 row-tile 重复读取
    const double pv_p_reads = d_tiles * n * n;
    const double pv_v_reads = n_tiles * n * d;

    // 4. 写出 O 矩阵 [N, d]
    const double output_write = n * d;

    // 中间结果
    const double intermediate =
        qk_score_write + softmax_accesses + pv_p_reads;

    // 总流量
    const double total_floats =
        qk_reads + qk_score_write + softmax_accesses +
        pv_p_reads + pv_v_reads + output_write;

    return {total_floats * sizeof(float), intermediate * sizeof(float)};
}
