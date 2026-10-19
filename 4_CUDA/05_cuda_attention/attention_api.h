/**
 * 作为项目的头文件，对外暴露 CPU Reference、Unfused 注意力机制、Online FlashAttention（CUDA Core）以及 WMMA FlashAttention（Tensor Core）的调用接口与流量分析模型
 */
#pragma once

#include <cuda_runtime.h>

#include "../config.h"

struct AttentionTraffic {
    double total_bytes;
    double intermediate_sp_bytes;
};

// 1. CPU 参考对照实现：用于校验 GPU 算出的结果是否正确（正确性验证）
void attention_cpu_reference(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    const gpu_config::AttentionShape& shape);


// 2. 传统的 Unfused 注意力接口：用来作为 GPU 性能对标的“反面教材”
// 注意：它比别人多传入了 scores 和 probabilities 两个显存指针
cudaError_t launch_unfused_attention(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    float* scores,
    float* probabilities,
    const gpu_config::AttentionShape& shape,
    cudaStream_t stream = nullptr);


// 3. FlashAttention CUDA Core 版接口：通用 FP32 高速实现
cudaError_t launch_flash_attention_online(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    const gpu_config::AttentionShape& shape,
    cudaStream_t stream = nullptr);


// 4. FlashAttention Tensor Core 版接口：WMMA 硬件加速实现（目前固定 d=64）
cudaError_t launch_flash_attention_tensor_core(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    const gpu_config::AttentionShape& shape,
    cudaStream_t stream = nullptr);


// HBM 显存流量分析模型（理论计算）
AttentionTraffic unfused_traffic_model(const gpu_config::AttentionShape& shape);
AttentionTraffic flash_traffic_model(const gpu_config::AttentionShape& shape);
AttentionTraffic tensor_core_traffic_model(const gpu_config::AttentionShape& shape);

// Kernel 启动次数常量定义
// Unfused 传统实现需要 Launch 3 个 Kernel (MatMul -> Softmax -> MatMul)
constexpr int kUnfusedAttentionLaunches = 3;
// FlashAttention CUDA Core 版只需 Launch 1 个 Kernel (Fused 融合成单 Kernel
constexpr int kFlashAttentionLaunches = 1;
// FlashAttention Tensor Core 版同样只需 Launch 1 个 Kernel
constexpr int kTensorCoreAttentionLaunches = 1;
