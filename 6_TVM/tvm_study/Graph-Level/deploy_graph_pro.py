import os
import time
import numpy as np
import tvm
from tvm.contrib import graph_executor


def main():
    print("=" * 60)
    print(" 验证端：加载全栈分离包并计算 GFLOPS 算力吞吐")
    print("=" * 60)

    M, N, K = 1024, 1024, 1024
    dev = tvm.cpu(0)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    so_dir = os.path.join(current_dir, "so")

    # 1. 读取三要素
    with open(os.path.join(so_dir, "lib_network_graph_pro.json"), "r") as f:
        graph_json = f.read()

    mod_lib = tvm.runtime.load_module(os.path.join(so_dir, "lib_network_graph_pro.so"))

    with open(os.path.join(so_dir, "lib_network_graph_pro.params"), "rb") as f:
        param_bytes = f.read()

    # 2. 合体组装图执行器
    module = graph_executor.create(graph_json, mod_lib, dev)
    module.load_params(param_bytes)

    print("全栈图模块三要素分离加载成功！总线就绪。")

    # 3. 准备随机数据（注意：w_np 形状调整为对齐 dense 的 N, K）
    x_np = np.random.uniform(size=(M, K)).astype("float32")
    w_np = np.random.uniform(size=(N, K)).astype("float32")
    bias_np = np.random.uniform(size=(N,)).astype("float32")

    # 4. 数据搬运绑定
    module.set_input("x", tvm.nd.array(x_np, dev))
    module.set_input("w", tvm.nd.array(w_np, dev))
    module.set_input("bias", tvm.nd.array(bias_np, dev))

    # 5. 推理掐表
    print("\n硬件流水线拉满，开始直驱全栈网络图计算...")
    times = []
    for _ in range(10):
        t0 = time.time()
        module.run()
        t1 = time.time()
        times.append(t1 - t0)

    average_time_seconds = np.mean(times)
    average_time_ms = average_time_seconds * 1000

    # 6. 严苛数值对账与 GFLOPS 计算
    out = module.get_output(0).numpy()

    # 理论参考组：由于 w_np 形状是 (N, K)，矩阵乘法使用 w_np.T 转置进行对账
    ref_matmul = np.dot(x_np, w_np.T)
    ref_bias = ref_matmul + bias_np
    ref_relu = np.maximum(ref_bias, 0)

    try:
        np.testing.assert_allclose(out, ref_relu, rtol=1e-5)

        # 计算总浮点运算次数
        total_flops = 2 * M * N * K
        gflops = total_flops / average_time_seconds / 1e9

        print("=" * 60)
        print(" 数据正确性校验通过：全栈图级编译计算完全正确。")
        print("=" * 60)
        print(f" 纯运行期全图直驱平均推理耗时: {average_time_ms:.2f} 毫秒")
        print(f" 纯运行期全图直驱整体算力表现: {gflops:.2f} GFLOPS")
        print("=" * 60)
    except AssertionError as e:
        print("数据校验失败。")
        print(e)


if __name__ == "__main__":
    main()