#include "attention_api.h"

#include <cuda_fp16.h>
#include <mma.h>

#include <cfloat>
#include <cmath>

namespace {

// ----- 阶段一：硬件常量与束内规约原语 -----
constexpr int kWarpSize = gpu_config::kWarpSize; // 32
constexpr int kQueryRows = gpu_config::flash::kQueryRows; // Br = 16
constexpr int kKeyCols = gpu_config::flash::kKeyCols; // Bc = 32
constexpr int kMaxHeadDim = gpu_config::flash::kMaxHeadDim; // 128
constexpr int kOutputValuesPerLane = (kMaxHeadDim + kWarpSize - 1) / kWarpSize; // 4
constexpr int kKeySharedStride = kKeyCols + 1; // +1 为了消除 Bank 冲突

static_assert(kKeyCols == kWarpSize,
              "The optimized score mapping requires one key per warp lane");

// ----- 两大规约操作：sum，max -----
// 通过 shuffle 指令，直接绕过 shared mem，快速找出32个线程中的 sum 和 max
__device__ __forceinline__ float warp_reduce_sum(float value) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float warp_reduce_max(float value) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}


// ===== 针对 CUDA Core 的 Flash Attention kernel =====
template <int HeadDim, bool Causal>
__global__ void flash_attention_online_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ output,
    int n,
    int runtime_head_dim,
    float scale) {

    // ----- 阶段二：动态 Shared Mem 布局与转置 Padding -----

    // 1. 动态确定 head dimension d (若 HeadDim==0 则使用运行时传入的 runtime_head_dim)
    const int d = HeadDim == 0 ? runtime_head_dim : HeadDim;

    // 2. 声明动态 Shared Memory 指针
    extern __shared__ float shared[];

    // 3. 将 Shared Memory 空间顺序切割给 Q, K, V, P 四个 Tile
    float* q_tile = shared; // [Br,d]
    float* k_tile = q_tile + kQueryRows * d;
    float* v_tile = k_tile + kKeySharedStride * d; // [Bc, d]
    // 这里只在 Shared Memory 暂存当前 tile 的概率，绝不写到 HBM 全局显存中
    float* p_tile = v_tile + kKeyCols * d; // [Br, Bc]

    // ----- 阶段三：全局内存加载与 Causal Mask 跳块优化 -----
    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    const int lane = tid % kWarpSize;
    const int query_start = blockIdx.x * kQueryRows;
    const int query_index = query_start + warp_id;
    const bool valid_query = query_index < n;

    // 块内所有线程协作将 Q 矩阵加载到 Shared Memory (q_tile)
    for (int index = tid; index < kQueryRows * d; index += blockDim.x) {
        const int row = index / d;
        const int col = index % d;
        const int global_row = query_start + row;
        q_tile[index] = global_row < n ? q[global_row * d + col] : 0.0f;
    }
    __syncthreads();

    // 对当前 K/V tile，Online Softmax 递推为：
    //   m_new = max(m_old, max(S_tile))
    //   alpha = exp(m_old - m_new)
    //   l_new = alpha * l_old + sum(exp(S_tile - m_new))
    //   O_acc = alpha * O_acc + exp(S_tile - m_new) * V_tile
    // 循环结束后只做一次 O = O_acc / l，因此中途不需要写出 S/P。
    // lane 只负责 lane, lane+32, ... 四个输出维度，避免旧版本每线程
    // float o[64] 带来的寄存器压力与 local-memory spill。
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    float o_reg[kOutputValuesPerLane] = {0.0f};

    // 1. 算出如果不管 Causal，全局一共有多少个 Key Tile
    const int all_key_tiles = (n + kKeyCols - 1) / kKeyCols;

    // 2. 找到当前 Block 所负责的所有 Query 中，最靠后（最未来）的那一行 Query 的后一位开区间界限
    const int last_query_exclusive = min(n, query_start + kQueryRows);

    // 3. 核心：算出当前 Block 实际上最多只需要处理到第几个 Key Tile
    const int causal_key_tiles = (last_query_exclusive + kKeyCols - 1) / kKeyCols;

    // 4. 三元表达式：如果开启了 Causal，就用剪枝后的上限；没开启就处理所有 Tile
    const int key_tiles = Causal ? causal_key_tiles : all_key_tiles;

    // 5. 最外层 Key Tile 循环：直接用计算出的 key_tiles 作为终止条件！
    for (int key_tile_index = 0; key_tile_index < key_tiles; ++key_tile_index) {
        const int key_start = key_tile_index * kKeyCols;

        for (int index = tid; index < kKeyCols * d; index += blockDim.x) {
            const int row = index / d;
            const int col = index % d;
            const int global_row = key_start + row;
            const bool valid_key = global_row < n;

            // K 在 Shared Memory 中转置为 [d, Bc+1]。这样计算 QK^T 时 一个 warp 的 32 个 lane 会读取连续 key。
            // +1 padding 消除转置写入时因 stride=32 产生的 shared-memory bank conflict
            k_tile[col * kKeySharedStride + row] = valid_key ? k[global_row * d + col] : 0.0f;
            v_tile[index] = valid_key ? v[global_row * d + col] : 0.0f;
        }
        __syncthreads();

        if (valid_query) {
            // lane 与当前 tile 的 key 一一对应。旧映射按 key 串行推进，每个 score 都要做一次 warp reduction；现在 32 个 score 并行，整个 tile 只为 Softmax 的 max/sum 各规约一次。
            const int key_index = key_start + lane;
            const bool keep = key_index < n && (!Causal || key_index <= query_index);

            // ----- 阶段四：最内层点积计算与 4 路指令级并行（ILP）展开 -----
            float score = -FLT_MAX;
            if (keep) {
                // 优化传统单累加器的流水线阻塞的问题，采用 4 路累加器打断依赖链
                // 当算 dot3 时，dot0 的结果正好算完出来
                float dot0 = 0.0f;
                float dot1 = 0.0f;
                float dot2 = 0.0f;
                float dot3 = 0.0f;
                int inner = 0;
                for (; inner + 3 < d; inner += 4) {
                    dot0 += q_tile[warp_id * d + inner + 0] *
                            k_tile[(inner + 0) * kKeySharedStride + lane];
                    dot1 += q_tile[warp_id * d + inner + 1] *
                            k_tile[(inner + 1) * kKeySharedStride + lane];
                    dot2 += q_tile[warp_id * d + inner + 2] *
                            k_tile[(inner + 2) * kKeySharedStride + lane];
                    dot3 += q_tile[warp_id * d + inner + 3] *
                            k_tile[(inner + 3) * kKeySharedStride + lane];
                }
                // 两两归并累加：循环结束后，把 4 个局部累加器的结果加起来，得到完整的向量点积
                float dot = (dot0 + dot1) + (dot2 + dot3);

                // 处理剩余维数
                for (; inner < d; ++inner) {
                    dot += q_tile[warp_id * d + inner] *
                           k_tile[inner * kKeySharedStride + lane];
                }
                // 乘上缩放因子
                score = dot * scale;
            }

            // ----- 阶段五：Online Softmax 递推更新、o_reg 缩放与最终归一化写出 -----
            // 1. 求当前 Tile 的局部最大值，并广播给整个 Warp
            float tile_max = warp_reduce_max(score);
            tile_max = __shfl_sync(0xffffffffu, tile_max, 0);

            // 2. 更新全局最大值并计算旧状态缩放因子 alpha
            const float new_max = fmaxf(row_max, tile_max);
            const float alpha = expf(row_max - new_max);

            // 3. 计算当前 Tile 概率，存入 p_tile，并求当前 Tile 分母和
            const float probability = keep ? expf(score - new_max) : 0.0f;
            p_tile[warp_id * kKeyCols + lane] = probability;
            float tile_sum = warp_reduce_sum(probability);
            tile_sum = __shfl_sync(0xffffffffu, tile_sum, 0);
            const float new_sum = alpha * row_sum + tile_sum;

            __syncwarp();

            // 4. 用 alpha 重缩放旧输出 o_reg，并加上 P_tile * V_tile
            #pragma unroll
            for (int slot = 0; slot < kOutputValuesPerLane; ++slot) {
                const int output_col = lane + slot * kWarpSize;
                if (output_col < d) {
                    float pv = 0.0f;
                    for (int key_offset = 0; key_offset < kKeyCols; ++key_offset) {
                        pv += p_tile[warp_id * kKeyCols + key_offset] *
                              v_tile[key_offset * d + output_col];
                    }
                    o_reg[slot] = alpha * o_reg[slot] + pv;
                }
            }
            row_max = new_max;
            row_sum = new_sum;
        }
        // 所有 warp 用完 K/V 后才能覆盖 Shared Memory 进入下一 tile。
        __syncthreads();
    }

    // 当最外层遍历完全部 K/V Tile 后，在循环体外面做全流程唯一一次归一化除法
    if (valid_query) {
        const float inverse_sum = 1.0f / row_sum;
        #pragma unroll
        for (int slot = 0; slot < kOutputValuesPerLane; ++slot) {
            const int output_col = lane + slot * kWarpSize;
            if (output_col < d) {
                // 将未归一化的累加器除以最终的总分母分母，得出最终归一化好的注意力结果，写回 HBM 全局显存
                output[query_index * d + output_col] = o_reg[slot] * inverse_sum;
            }
        }
    }
}


// kernel 启动包装器：屏蔽 CUDA 语法，将其封装成一个普通的 C++ 函数，并显式传递两个模板参数 <HeadDim, Causal> 给 Kernel，让 NVCC 编译器针对不同的 HeadDim 和 Causal 模式生成极度优化的 GPU 机器码
template <int HeadDim, bool Causal>
void launch_flash_specialization(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    int n,
    int runtime_head_dim,
    int grid,
    std::size_t shared_bytes,
    float scale,
    cudaStream_t stream) {
    flash_attention_online_kernel<HeadDim, Causal>
        <<<grid, gpu_config::flash::kThreads, shared_bytes, stream>>>(
            q, k, v, output, n, runtime_head_dim, scale);
}


// 根据 HeadDim 和 Causal 参数，选择对应的 kernel specialization 进行启动
template <bool Causal>
void dispatch_flash_head_dim(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    int n,
    int d,
    int grid,
    std::size_t shared_bytes,
    float scale,
    cudaStream_t stream) {

    /**
     * 编译优化：通过 switch 语句，实现了 “运行时变量 -> 编译期模板参数” 的转换
     * 目的是为了：
     * 1. 消除循环边界检查与循环展开
     * 2. 精确计算寄存器需求，防止 Local Memory 溢出
     * 3. 抹去无效条件分支（死代码消除）
     */
    switch (d) {
        case 32:
            launch_flash_specialization<32, Causal>(
                q, k, v, output, n, d, grid, shared_bytes, scale, stream);
            break;
        case 64:
            launch_flash_specialization<64, Causal>(
                q, k, v, output, n, d, grid, shared_bytes, scale, stream);
            break;
        case 128:
            launch_flash_specialization<128, Causal>(
                q, k, v, output, n, d, grid, shared_bytes, scale, stream);
            break;
        default:
            launch_flash_specialization<0, Causal>(
                q, k, v, output, n, d, grid, shared_bytes, scale, stream);
            break;
    }
}


// ===== 针对 Tensor Core 的 Flash Attention kernel =====
template <bool Causal>
__global__ void flash_attention_tensor_core_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ output,
    int n,
    float scale) {

    // ----- 阶段 1：WMMA 维度常量推导 与 Shared Memory 对齐布局 -----
    namespace wmma = nvcuda::wmma;
    constexpr int Br = gpu_config::flash_tensor_core::kQueryRows;
    constexpr int Bc = gpu_config::flash_tensor_core::kKeyCols;
    constexpr int D = gpu_config::flash_tensor_core::kHeadDim;
    constexpr int WmmaTile = gpu_config::flash_tensor_core::kWmmaTile;
    constexpr int Warps = gpu_config::flash_tensor_core::kWarps;
    constexpr int QueryTiles = Br / WmmaTile;
    constexpr int KeyTiles = Bc / WmmaTile;
    constexpr int OutputTiles = D / WmmaTile;
    constexpr int QkWarps = QueryTiles * KeyTiles;

    // half 的 WMMA leading dimension 必须保持 16-byte 粒度，且需要为16的整数倍，让相邻行错开从而解决Bank冲突，float store 同理。所以半字需要 +8，字需要 +4
    constexpr int QkvStride = D + 8;
    constexpr int PStride = Bc + 8;
    constexpr int ScoreStride = Bc + 4;
    constexpr int OutputStride = D + 4;

    // Q/K/V/P 用 half 降低 Shared Memory 流量并满足 Tensor Core 输入要求；
    // score、Online 状态与 O 始终用 FP32，避免跨 tile 递推不断损失精度。
    __shared__ __align__(16) half q_tile[Br * QkvStride];
    __shared__ __align__(16) half k_tile[Bc * QkvStride];
    __shared__ __align__(16) half v_tile[Bc * QkvStride];
    __shared__ __align__(16) half p_tile[Br * PStride];
    __shared__ __align__(16) float scores[Br * ScoreStride];
    __shared__ __align__(16) float output_acc[Br * OutputStride];
    __shared__ __align__(16) float pv_tile[Br * OutputStride];
    __shared__ float row_max[Br];
    __shared__ float row_sum[Br];
    __shared__ float alpha[Br];

    // ----- 阶段 2：全局内存加载、数据格式转换与状态初始化 ----- 
    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    const int lane = tid % kWarpSize;
    const int query_start = blockIdx.x * Br;

    // 1. 线程协作加载 QTile，并进行 FP32 -> FP16 (half) 转换
    for (int index = tid; index < Br * D; index += blockDim.x) {
        const int row = index / D;
        const int col = index % D;
        const int global_row = query_start + row;
        q_tile[row * QkvStride + col] = __float2half_rn(
            global_row < n ? q[global_row * D + col] : 0.0f);
        output_acc[row * OutputStride + col] = 0.0f;
    }
    // 2. Online Softmax 状态数组初始化 (维持 FP32 精度)
    if (tid < Br) {
        row_max[tid] = -FLT_MAX;
        row_sum[tid] = 0.0f;
        alpha[tid] = 0.0f;
    }
    __syncthreads();

    // ----- 阶段 3：K/V 协作加载与 Causal 跳块 -----
    // 跳块计算（与之前 CUDA Core 实现的逻辑一致）
    const int all_key_tiles = (n + Bc - 1) / Bc;
    const int last_query_exclusive = min(n, query_start + Br);
    const int causal_key_tiles = (last_query_exclusive + Bc - 1) / Bc;
    const int key_tiles = Causal ? causal_key_tiles : all_key_tiles;

    // 协作搬运
    for (int key_tile_index = 0; key_tile_index < key_tiles; ++key_tile_index) {
        const int key_start = key_tile_index * Bc;
        for (int index = tid; index < Bc * D; index += blockDim.x) {
            const int row = index / D;
            const int col = index % D;
            const int global_row = key_start + row;
            const bool valid = global_row < n;
            // FP32 -> FP16 实时转换
            k_tile[row * QkvStride + col] =
                __float2half_rn(valid ? k[global_row * D + col] : 0.0f);
            v_tile[row * QkvStride + col] =
                __float2half_rn(valid ? v[global_row * D + col] : 0.0f);
        }
        __syncthreads();

        // ----- 阶段 4：用 Tensor Core WMMA 加速算子 1（计算 $S = Q \cdot K^T$ 得分矩阵） -----
        // 每个 warp 产生一个 16x16 S 子块。K 原本按 [key,d] 存储，
        // 作为 col-major 的 [d,key] 矩阵读取即可得到 K^T 视图。
        if (warp_id < QkWarps) {
            // 子块坐标计算
            const int query_offset = (warp_id / KeyTiles) * WmmaTile; // 行 Q
            const int key_offset = (warp_id % KeyTiles) * WmmaTile; // 列 K

            // Fragment：一个由 Warp 内 32 个线程共同分担持有的“不透明矩阵寄存器块”
            wmma::fragment<wmma::accumulator, WmmaTile, WmmaTile, WmmaTile, float> score_fragment;
            // 把这 16 * 16 的矩阵块清零
            wmma::fill_fragment(score_fragment, 0.0f);

            for (int inner = 0; inner < D; inner += WmmaTile) { // inner 每次跳 16
                wmma::fragment<wmma::matrix_a, WmmaTile, WmmaTile, WmmaTile,
                               half, wmma::row_major> q_fragment;
                wmma::fragment<wmma::matrix_b, WmmaTile, WmmaTile, WmmaTile,
                               half, wmma::col_major> k_fragment; // 这里直接按照列优先存储，实现转置操作
                // Warp 内 32 个线程并发从 Shared Memory 中各自加载一部分数据到各自的寄存器中，组合拼装出 16*16 的 q_fragment 和 k_fragment。传入的步长 QkvStride (72) 确保了 16 字节对齐
                // 注意这里的指针二维偏移计算，满足：BasePtr + RowIndex * Stride + ColIndex
                wmma::load_matrix_sync(
                    q_fragment, q_tile + query_offset * QkvStride + inner,
                    QkvStride);
                wmma::load_matrix_sync(
                    k_fragment, k_tile + key_offset * QkvStride + inner,
                    QkvStride);

                // 执行乘累加 score_fragment = q_fragment * k_fragment + score_fragment
                wmma::mma_sync(
                    score_fragment, q_fragment, k_fragment, score_fragment);
            }
            // 把处于寄存器 Fragment 中的 16*16 结果矩阵写回 Shared Memory 的 scores 数组中（步长为 ScoreStride = 36）
            wmma::store_matrix_sync(
                scores + query_offset * ScoreStride + key_offset,
                score_fragment, ScoreStride,
                wmma::mem_row_major);
        }
        __syncthreads();

        // ----- 阶段 5：Softmax 状态递推、细粒度 Causal Mask 与 P_tile(FP16) 生成 -----
        // 四个 warp 各处理 4 行；lane 与 Bc=32 的 key 一一对应
        // 这里只做两次 shuffle reduction，softmax 状态仍保持 FP32
        #pragma unroll
        for (int group = 0; group < Br / Warps; ++group) {
            const int row = warp_id + group * Warps;
            const int query_index = query_start + row;
            const int key_index = key_start + lane;
            const bool valid_query = query_index < n;

            // 细粒度 Causal Mask 与缩放
            const bool keep = valid_query && key_index < n &&
                              (!Causal || key_index <= query_index);
            const float score =
                keep ? scores[row * ScoreStride + lane] * scale : -FLT_MAX;

            // Online Softmax 规约与 p_tile (FP16) 转换写入
            if (valid_query) {
                // Warp 内 Shuffle 规约：
                // 32 个线程求出当前行的最大值 tile_max，并通过 __shfl_sync 广播给 Warp 内所有线程
                float tile_max = warp_reduce_max(score);
                tile_max = __shfl_sync(0xffffffffu, tile_max, 0);

                // 状态更新：算出 m_new，缩放因子 alpha，和 P
                const float new_max = fmaxf(row_max[row], tile_max);
                const float row_alpha = expf(row_max[row] - new_max);
                const float probability = keep ? expf(score - new_max) : 0.0f;

                // 关键点：写入 p_tile 时转为 FP16
                // 因为下一个阶段，p_tile 将作为矩阵 A 喂给 Tensor Core 算 PV，而 Tensor Core 必须要求输入是 FP16
                p_tile[row * PStride + lane] = __float2half_rn(probability);

                float tile_sum = warp_reduce_sum(probability);
                tile_sum = __shfl_sync(0xffffffffu, tile_sum, 0);
                
                // 确保当前Warp内所有32个线程都读完了旧的row_max[row]状态，并完成了p_tile的写入
                __syncwarp(); 

                // lane 0 将新数据写回 Shared Mem 中
                if (lane == 0) {
                    alpha[row] = row_alpha;
                    row_sum[row] = row_alpha * row_sum[row] + tile_sum;
                    row_max[row] = new_max;
                }
            } else {
                p_tile[row * PStride + lane] = __float2half_rn(0.0f);
                if (lane == 0) {
                    alpha[row] = 0.0f;
                }
            }
        }
        __syncthreads();

        // ----- 阶段 6：用 WMMA 加速算子 2（计算 PV 矩阵）以及全局 Online Rescale 重缩放 -----
        const int query_offset = (warp_id / OutputTiles) * WmmaTile;
        const int output_col = (warp_id % OutputTiles) * WmmaTile;
        wmma::fragment<wmma::accumulator, WmmaTile, WmmaTile, WmmaTile, float>
            pv_fragment;
        wmma::fill_fragment(pv_fragment, 0.0f);
        for (int inner = 0; inner < Bc; inner += WmmaTile) {
            wmma::fragment<wmma::matrix_a, WmmaTile, WmmaTile, WmmaTile,
                           half, wmma::row_major> p_fragment;
            wmma::fragment<wmma::matrix_b, WmmaTile, WmmaTile, WmmaTile,
                           half, wmma::row_major> v_fragment;
            wmma::load_matrix_sync(
                p_fragment, p_tile + query_offset * PStride + inner, PStride);
            wmma::load_matrix_sync(
                v_fragment, v_tile + inner * QkvStride + output_col,
                QkvStride);
            wmma::mma_sync(pv_fragment, p_fragment, v_fragment, pv_fragment);
        }
        wmma::store_matrix_sync(
            pv_tile + query_offset * OutputStride + output_col,
            pv_fragment, OutputStride,
            wmma::mem_row_major);
        __syncthreads();

        // alpha 每行不同，无法直接作用于 WMMA fragment
        // 所以需要存回 Shared Memory 后逐元素完成 Online rescale，再进入下一个 K/V tile
        for (int index = tid; index < Br * D; index += blockDim.x) {
            const int row = index / D;
            const int col = index % D;
            output_acc[row * OutputStride + col] =
                alpha[row] * output_acc[row * OutputStride + col] +
                pv_tile[row * OutputStride + col];
        }
        __syncthreads();
    }

    // 代码在 Kernel 底部做全流程唯一一次除法归一化，并写回全局显存
    for (int index = tid; index < Br * D; index += blockDim.x) {
        const int row = index / D;
        const int col = index % D;
        const int global_row = query_start + row;
        if (global_row < n) {
            output[global_row * D + col] =
                output_acc[row * OutputStride + col] / row_sum[row];
        }
    }
}

}  // namespace


// CUDA Core 版入口
cudaError_t launch_flash_attention_online(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    const gpu_config::AttentionShape& shape,
    cudaStream_t stream) {
    const int n = shape.sequence_length;
    const int d = shape.head_dim;
    // 参数防御检查
    if (q == nullptr || k == nullptr || v == nullptr || output == nullptr ||
        n <= 0 || d <= 0 || d > kMaxHeadDim) {
        return cudaErrorInvalidValue;
    }

    // 动态计算线程块网格尺寸 Grid
    const int grid = (n + kQueryRows - 1) / kQueryRows;
    // 动态计算片上 Shared Memory 需求尺寸
    const std::size_t shared_floats =
        static_cast<std::size_t>(kQueryRows) * d +       // Q_tile 大小
        static_cast<std::size_t>(kKeySharedStride) * d + // K_tile 大小 (带Padding避开Bank冲突)
        static_cast<std::size_t>(kKeyCols) * d +         // V_tile 大小
        static_cast<std::size_t>(kQueryRows) * kKeyCols; // P_tile 大小
    const std::size_t shared_bytes = shared_floats * sizeof(float);

    // Causal 编译期模板分派与缩放因子计算
    const float scale = 1.0f / std::sqrt(static_cast<float>(d)); // 算 1/sqrt(d)
    if (shape.causal) {
        dispatch_flash_head_dim<true>(
            q, k, v, output, n, d, grid, shared_bytes, scale, stream);
    } else {
        dispatch_flash_head_dim<false>(
            q, k, v, output, n, d, grid, shared_bytes, scale, stream);
    }

    // 最后检查 Kernel 启动语法是否有错并返回状态
    return cudaPeekAtLastError();
}


// Tensor Core 版入口
cudaError_t launch_flash_attention_tensor_core(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    const gpu_config::AttentionShape& shape,
    cudaStream_t stream) {
    constexpr int D = gpu_config::flash_tensor_core::kHeadDim;
    constexpr int Br = gpu_config::flash_tensor_core::kQueryRows;

    // 静态维度限制检查
    if (q == nullptr || k == nullptr || v == nullptr || output == nullptr ||
        shape.sequence_length <= 0 || shape.head_dim != D) {
        return cudaErrorInvalidValue;
    }

    const int grid = (shape.sequence_length + Br - 1) / Br;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    if (shape.causal) {
        // 注意这里：共享内存大小为 0
        // 因为在 Tensor Core 内核内部，所有的 Shared Memory 数组（如 q_tile, k_tile, v_tile）都是使用静态大小声明的 (例如 __shared__ half q_tile[Br * QkvStride];)
        // 静态声明的共享内存在编译期就已经确定了字节数，因此在 launch 时填 0 即可，CUDA 运行时会自动分配
        flash_attention_tensor_core_kernel<true>
            <<<grid, gpu_config::flash_tensor_core::kThreads, 0, stream>>>(
                q, k, v, output, shape.sequence_length, scale);
    } else {
        flash_attention_tensor_core_kernel<false>
            <<<grid, gpu_config::flash_tensor_core::kThreads, 0, stream>>>(
                q, k, v, output, shape.sequence_length, scale);
    }
    return cudaPeekAtLastError();
}


// 针对 Flash Attention 的显存流量模型，计算总字节数与中间态字节数
AttentionTraffic flash_traffic_model(const gpu_config::AttentionShape& shape) {
    const double n = static_cast<double>(shape.sequence_length);
    const double d = static_cast<double>(shape.head_dim);
    const int query_tiles =
        (shape.sequence_length + gpu_config::flash::kQueryRows - 1) /
        gpu_config::flash::kQueryRows;

    double kv_rows_loaded = 0.0;
    for (int query_tile = 0; query_tile < query_tiles; ++query_tile) {
        if (!shape.causal) {
            kv_rows_loaded += n;
            continue;
        }
        const int last_query_exclusive = std::min(
            shape.sequence_length,
            (query_tile + 1) * gpu_config::flash::kQueryRows);
        const int key_tiles =
            (last_query_exclusive + gpu_config::flash::kKeyCols - 1) /
            gpu_config::flash::kKeyCols;
        kv_rows_loaded += std::min(
            shape.sequence_length,
            key_tiles * gpu_config::flash::kKeyCols);
    }

    // Q/O 各访问一次；K/V 的重复扫描次数由 Q tile 与 causal 跳块共同决定。
    const double total_floats = 2.0 * n * d + 2.0 * kv_rows_loaded * d;
    return {total_floats * sizeof(float), 0.0};
}


// 针对 Tensor Core 的显存流量模型，计算总字节数与中间态字节数
AttentionTraffic tensor_core_traffic_model(const gpu_config::AttentionShape& shape) {
    const double n = static_cast<double>(shape.sequence_length);
    const double d = static_cast<double>(shape.head_dim);
    constexpr int Br = gpu_config::flash_tensor_core::kQueryRows;
    constexpr int Bc = gpu_config::flash_tensor_core::kKeyCols;
    const int query_tiles = (shape.sequence_length + Br - 1) / Br;

    double kv_rows_loaded = 0.0;
    for (int query_tile = 0; query_tile < query_tiles; ++query_tile) {
        if (!shape.causal) {
            kv_rows_loaded += n;
            continue;
        }
        const int last_query_exclusive =
            std::min(shape.sequence_length, (query_tile + 1) * Br);
        const int key_tiles = (last_query_exclusive + Bc - 1) / Bc;
        kv_rows_loaded += std::min(shape.sequence_length, key_tiles * Bc);
    }

    const double total_floats = 2.0 * n * d + 2.0 * kv_rows_loaded * d;
    return {total_floats * sizeof(float), 0.0};
}
