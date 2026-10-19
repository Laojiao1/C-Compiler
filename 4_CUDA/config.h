/**
 * 定义整个工程的物理约束和硬件参数（如 Warp 大小、Tile 维度、线程块大小）。
 * 避免将 TILE_SIZE 硬编码在各个文件中
 */
#pragma once

#include <cstddef>

// 全工程只共享“硬件事实”和默认实验参数。
// 不同算子的 tile 含义不同，因此按命名空间分域，避免一个 TILE_SIZE
// 同时绑死 GEMM、Softmax 与 FlashAttention


// ----- 基础硬件参数与 GEMM 域 -----
namespace gpu_config {

    // GPU 线程束大小 32
    constexpr int kWarpSize = 32;

    // gemm 和 softmax 专门服务于 Unfused 算子，防止这些传统 GEMM 的 Tile 尺寸与 FlashAttention 的 Br/Bc 混合混淆
    namespace gemm {
        constexpr int kTileM = 32;
        constexpr int kTileN = 32;
        constexpr int kTileK = 32;
    }  // namespace gemm

    namespace softmax {
        constexpr int kThreads = 256;
    }  // namespace softmax


    // ----- CUDA Core FlashAttention 域 -----
    namespace flash {

        // 一个 warp 负责一行 Q；一个 block 共处理 kQueryRows 行。
        // 通过 #ifndef 宏保护，便于在调优时可通过 -DGPU_FLASH_QUERY_ROWS=... 等参数直接重载分块尺寸，无需修改源代码。
        #ifndef GPU_FLASH_QUERY_ROWS
        #define GPU_FLASH_QUERY_ROWS 16
        #endif
        #ifndef GPU_FLASH_KEY_COLS
        #define GPU_FLASH_KEY_COLS 32
        #endif

        constexpr int kQueryRows = GPU_FLASH_QUERY_ROWS; // Br
        constexpr int kKeyCols = GPU_FLASH_KEY_COLS; // Bc
        constexpr int kMaxHeadDim = 128; 
        constexpr int kThreads = kQueryRows * kWarpSize; // 16 * 32 = 512 线程

        // 编译期断言：确保线程数不超过 CUDA 每个 Block 最多 1024 个线程的硬性物理规则
        static_assert(kThreads <= 1024, "FlashAttention block exceeds CUDA thread limit");
    }  // namespace flash


    // ----- Tensor Core FlashAttention 域 -----
    namespace flash_tensor_core {

        // 第一版 Tensor Core 路径固定 d=64，把正确性、数据布局和 WMMA 流程
        // 独立验证清楚后，再扩展到其他 head dimension。
        #ifndef GPU_FLASH_TC_QUERY_ROWS
        #define GPU_FLASH_TC_QUERY_ROWS 32
        #endif
        constexpr int kQueryRows = GPU_FLASH_TC_QUERY_ROWS; // Br
        constexpr int kKeyCols = 32; // Bc
        constexpr int kHeadDim = 64; // D(第一版固定 d=64)
        constexpr int kWmmaTile = 16; // WMMA 16*16 硬件原生 Tile
        constexpr int kQkWarps =
            (kQueryRows / kWmmaTile) * (kKeyCols / kWmmaTile); // (32/16)*(32/16) = 4 warps

        constexpr int kPvWarps =
            (kQueryRows / kWmmaTile) * (kHeadDim / kWmmaTile); // (32/16)*(64/16) = 8 warps

        constexpr int kWarps = kQkWarps > kPvWarps ? kQkWarps : kPvWarps; // 取两者的最大值 max(4, 8) = 8 warps，确保同一个 Block 在计算这两个阶段时，都有足够的线程/Warp 资源来承载

        constexpr int kThreads = kWarps * kWarpSize; // 共 8 * 32 = 256 线程

        static_assert(kQueryRows % kWmmaTile == 0,
                    "Tensor Core Br must be a multiple of the WMMA tile");
        static_assert(kThreads <= 1024, "Tensor Core block exceeds CUDA thread limit");
    }  // namespace flash_tensor_core


    namespace benchmark {
        constexpr int kDefaultSequenceLength = 1024;  // 默认序列长度
        constexpr int kDefaultHeadDim = 64;           // 默认 Attention Head 维度 d
        constexpr int kDefaultWarmupIterations = 10;  // 默认预热轮数
        constexpr int kDefaultIterations = 100;       // 默认正式计时测量轮数
        constexpr float kAbsoluteTolerance = 3.0e-3f; // 绝对误差阈值
        constexpr float kRelativeTolerance = 3.0e-3f; // 相对误差阈值
    }  // namespace benchmark


    // ----- 运行时维度与元数据结构 -----
    struct AttentionShape {
        int sequence_length; // N
        int head_dim; // d
        bool causal; // 是否开启 Mask

        // 计算单个 Q K V 或 Output 矩阵的元素个数 N * d
        std::size_t qkv_elements() const {
            return static_cast<std::size_t>(sequence_length) * head_dim;
        }

        // 计算完全物化的注意力得分矩阵 S 或概率矩阵 P 的元素个数 N*N，用于 Unfused 基线的显存分配
        std::size_t score_elements() const {
            return static_cast<std::size_t>(sequence_length) * sequence_length;
        }
    };

}  // namespace gpu_config
