import os
import logging
import sys
import numpy as np
import tvm
from tvm import te
from tvm import autotvm

# 定义全局矩阵规模
M, N, K = 1024, 1024, 1024

# ========================================================
# 1. 声明参数化调度模板 (AutoTVM Template)
# ========================================================
# 用装饰器把普通的 te 计算包裹起来，使它具备参数空间配置能力
@autotvm.template("matmul_tune_task")
def matmul_tune_task():
    A = te.placeholder((M, K), name="A", dtype="float32")
    B = te.placeholder((K, N), name="B", dtype="float32")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")

    # 创建默认调度
    s = te.create_schedule(C.op)

    # 智能捞取当前任务的配置记录器
    cfg = autotvm.get_config()

    # 【核心改变：定义寻优空间】
    # 告诉 TVM：i 轴和 j 轴需要被切成两个子轴(outer, inner)，切成多大由你来猜
    cfg.define_split("tile_i", C.op.axis[0], num_outputs=2)
    cfg.define_split("tile_j", C.op.axis[1], num_outputs=2)

    # 执行参数化切块
    xo, xi = cfg["tile_i"].apply(s, C, C.op.axis[0])
    yo, yi = cfg["tile_j"].apply(s, C, C.op.axis[1])

    # 改变轴顺序，依然维持我们摸索出来的重排原则
    s[C].reorder(xo, yo, xi, yi)

    # 最内层 yi 轴开启向量化
    s[C].vectorize(yi)
    # xi 轴开启并行多核
    s[C].parallel(xi)

    return s, [A, B, C]


# ========================================================
# 2. 调优主控制引擎
# ========================================================
def run_tuning():
    target = "llvm"
    device = tvm.cpu(0)

    # 建立寻优任务，TVM 会自动扫描上面的函数，计算出总共有多少种参数组合
    task = autotvm.task.create("matmul_tune_task", args=(), target=target)
    print("=" * 60)
    print(f"检测到参数组合空间总数: {len(task.config_space)}")
    print("=" * 60)

    # 指定日志结果存放路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(current_dir, "log")
    log_file = os.path.join(log_dir, "matmul.txt")

    # 配置寻优测量参数：每组代码在硬件上重复跑 4 次，掐表拿平均值
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(number=4, repeat=1, min_repeat_ms=150)
    )

    # 选用最基础也最稳健的随机采样调优器 (RandomTuner)
    tuner = autotvm.tuner.RandomTuner(task)

    # 开始暴力拉网寻优（这里限制最多测 20 组，防止把实验时间拉得太长）
    print("正在启动硬件真实打分流水线，请静静观察每一组参数的真实吞吐量表现...")
    tuner.tune(
        n_trial=20,
        measure_option=measure_option,
        callbacks=[autotvm.callback.progress_bar(20), autotvm.callback.log_to_file(log_file)]
    )

    # ===================================================
    # 3. 捞取最优参数，重装上阵
    # ===================================================
    print("\n" + "=" * 60)
    print(" 寻优结束！正在提取全场跑分最高的‘黄金参数’进行最终编译...")
    print("=" * 60)

    # 使用刚刚生成的日志文件，自动应用胜出者的配置参数
    with autotvm.apply_history_best(log_file):
        with tvm.target.Target(target):
            s, arg_list = matmul_tune_task()
            # 打印查看获胜者的 TensorIR 结构体形态
            ir_mod = tvm.lower(s, arg_list, simple_mode=True)
            print(ir_mod)

            # 编译出终极 Module
            rt_mod = tvm.build(s, arg_list, target=target, name="tuned_gemm")

            # 物理验证性能
            a_np = np.random.uniform(size=(M, K)).astype("float32")
            b_np = np.random.uniform(size=(K, N)).astype("float32")
            c_np = np.zeros((M, N), dtype="float32")

            a_tvm = tvm.nd.array(a_np, device)
            b_tvm = tvm.nd.array(b_np, device)
            c_tvm = tvm.nd.array(c_np, device)

            evaluator = rt_mod.time_evaluator("tuned_gemm", device, number=10)
            execution_time = evaluator(a_tvm, b_tvm, c_tvm).mean
            gflops = (2 * M * N * K) / execution_time / 1e9
            print(f"AutoTVM 自动搜索到的最佳算子吞吐性能: {gflops:.2f} GFLOPS")


if __name__ == "__main__":
    # 配置全局日志打印，使我们在控制台能看到打分进度
    logging.getLogger("tvm").addHandler(logging.StreamHandler(sys.stdout))
    logging.getLogger("tvm").setLevel(logging.INFO)
    run_tuning()