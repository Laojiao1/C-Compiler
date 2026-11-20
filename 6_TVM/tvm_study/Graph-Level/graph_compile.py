import os
import tvm
from tvm import relay
import numpy as np


def main():
    print("=" * 60)
    print(" 图级编译器层：基于 Relay 构建并编译端到端网络图")
    print("=" * 60)

    M, N, K = 1024, 1024, 1024

    # 1. 声明网络拓扑关系 (Relay 表达)
    # 声明输入和权重的形状占位符
    x = relay.var("x", shape=(M, K), dtype="float32")
    w = relay.var("w", shape=(K, N), dtype="float32")
    bias = relay.var("bias", shape=(N,), dtype="float32")

    # ─── 拓扑层 1：线性矩阵乘 (Linear) ───
    linear_out = relay.nn.matmul(x, w)

    # ─── 拓扑层 2：偏置加法 (Bias Add) ───
    bias_out = relay.nn.bias_add(linear_out, bias)

    # ─── 拓扑层 3：激活层 (ReLU) ───
    relu_out = relay.nn.relu(bias_out)

    # 将拓扑节点打包为一个抽象语法树函数 (Function)
    func = relay.Function([x, w, bias], relu_out)

    # 将函数封装进完整的 IRModule
    ir_mod = tvm.IRModule.from_expr(func)

    print("\n[阶段 1] 优化前的原始拓扑网络图（高级 Relay 表达）:")
    print(ir_mod)

    # 2. 核心大招：呼叫图优化 Pass 兵团执行“算子融合”
    # 在 0.11.1 中，build 过程会自动在后台触发 FuseOps Pass
    # 为了让我们能肉眼看清融合后的形态，我们手动调用一次优化 Pass
    seq = tvm.transform.Sequential([
        relay.transform.InferType(),  # 智能推导张量类型（融合前必须做的 Pass）
        relay.transform.FuseOps(fuse_opt_level=3)  # 核心算子融合 Pass，级别 3 代表全量融合
    ])

    # 执行图变换
    optimized_mod = seq(ir_mod)

    print("\n" + "=" * 60)
    print("[阶段 2] 执行算子融合 Pass 后的计算图形态:")
    print("=" * 60)
    print(optimized_mod)

    # 3. 目标翻译：将整张融合后的拓扑图编译成硬件 Module
    print("\n正在调用 LLVM 将整张网络图转换为机器码动态库...")
    target = "llvm"

    # 使用 Relay 专用的图编译器进行离线 AOT 编译
    with tvm.transform.PassContext(opt_level=3):
        # relay.build 会同时输出图描述（json）和优化后的二进制 Module
        # 在这里我们主要提取它的二进制执行核心
        lib = relay.build(ir_mod, target=target, params={})

    # 4. 离线固化导出
    current_dir = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(current_dir, "so", "lib_network_graph.so")

    if not os.path.exists(os.path.dirname(so_path)):
        
        os.makedirs(os.path.dirname(so_path))

    # 直接对 relay.build 的返回值整个执行 export_library
    # TVM 会自动在后台将图 JSON 和参数打包成元数据，嵌入到 .so 的内部节点中
    lib.export_library(so_path)
    print(f"包含完整图拓扑元数据的物理全栈动态库已安全落盘: {so_path}")


if __name__ == "__main__":
    main()