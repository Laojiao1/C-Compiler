import os
import logging
import sys
import tvm
from tvm import relay
from tvm import autotvm
from tvm import runtime


def main():
    print("=" * 60)
    print(" 图级编译器层：基于 Relay 构建、自动提取 Task 并全栈编译")
    print("=" * 60)

    M, N, K = 1024, 1024, 1024

    # 1. 声明网络拓扑关系 (Relay 表达)
    x = relay.var("x", shape=(M, K), dtype="float32")
    w = relay.var("w", shape=(N, K), dtype="float32")  # 对齐 dense 的标准权重形状 (N, K)
    bias = relay.var("bias", shape=(N,), dtype="float32")

    # 核心拓扑：将 nn.matmul 升级为具备完整调优空间的 nn.dense
    # ─── 拓扑层 1：线性矩阵乘 (Linear) ───
    linear_out = relay.nn.dense(x, w)

    # ─── 拓扑层 2：偏置加法 (BiasAdd) ───
    bias_out = relay.nn.bias_add(linear_out, bias)

    # ─── 拓扑层 3：ReLU 激活 (ReLU) ───
    relu_out = relay.nn.relu(bias_out)

    # 将拓扑节点打包为一个抽象语法树函数 (Function)
    func = relay.Function([x, w, bias], relu_out)

    # 将函数封装进完整的 IRModule
    ir_mod = tvm.IRModule.from_expr(func)

    print("\n[阶段 1] 原始计算图拓扑构建完成（已升级为高级 dense 拓扑）。")

    # 2. 自动化 Task 提取
    target = "llvm"
    print("\n正在从计算图中自动提取算子调优任务...")
    tasks = autotvm.task.extract_from_program(ir_mod, target=target, params={})
    print(f"从图中成功提取出 {len(tasks)} 个重型算子调优任务。")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    so_dir = os.path.join(current_dir, "so")
    if not os.path.exists(so_dir):
        os.makedirs(so_dir)

    log_file = os.path.join(so_dir, "fused_network.log")

    # 3. 现场对齐调优
    for i, task in enumerate(tasks):
        print(f"\n任务 [{i + 1}/{len(tasks)}]: {task.name} | 空间大小: {len(task.config_space)}")

        # 限制测 10 组参数进行快速冲锋
        tuner = autotvm.tuner.RandomTuner(task)
        measure_option = autotvm.measure_option(
            builder=autotvm.LocalBuilder(),
            runner=autotvm.LocalRunner(number=4, repeat=1, min_repeat_ms=150)
        )
        tuner.tune(
            n_trial=10,
            measure_option=measure_option,
            callbacks=[autotvm.callback.log_to_file(log_file)]
        )

    # 4. 全栈联合编译（图级融合 + 现场算子黄金参数）
    print("\n正在进行最终全栈联合编译（图级融合 + 现场算子黄金参数）...")
    if os.path.exists(log_file):
        with autotvm.apply_history_best(log_file):
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(ir_mod, target=target, params={})
    else:
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(ir_mod, target=target, params={})

    # 5. 固化三要素分离部署包：结构清单 (JSON) + 计算核心 (.so) + 物理权重 (params)
    graph_json_path = os.path.join(so_dir, "lib_network_graph_pro.json")
    with open(graph_json_path, "w") as f:
        f.write(lib.get_graph_json())

    so_path = os.path.join(so_dir, "lib_network_graph_pro.so")
    lib.get_lib().export_library(so_path)

    params_path = os.path.join(so_dir, "lib_network_graph_pro.params")
    with open(params_path, "wb") as f:
        f.write(runtime.save_param_dict(lib.get_params()))

    print("\n" + "=" * 60)
    print("三要素分离部署包固化成功！")
    print(f" 1. 结构清单 (JSON): {graph_json_path}")
    print(f" 2. 计算核心 ( .so): {so_path}")
    print(f" 3. 物理权重 (params): {params_path}")
    print("=" * 60)


if __name__ == "__main__":
    logging.getLogger("tvm").addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger("tvm").setLevel(logging.INFO)
    main()