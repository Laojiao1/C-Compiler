import os
import time
import numpy as np
import tvm
from tvm.contrib import graph_executor


def main():
    print("=" * 60)
    print(" 验证端：加载包含全栈融合的图动态库")
    print("=" * 60)

    M, N, K = 1024, 1024, 1024
    dev = tvm.cpu(0)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(current_dir, "so", "lib_network_graph.so")

    # Load compiled module
    lib = tvm.runtime.load_module(so_path)

    # 建立图执行器运行时环境
    module = graph_executor.GraphModule(lib["default"](dev))

    # 准备输入数据
    x_np = np.random.uniform(size=(M, K)).astype("float32")
    w_np = np.random.uniform(size=(K, N)).astype("float32")
    bias_np = np.random.uniform(size=(N,)).astype("float32")

    # 喂入数据
    module.set_input("x", tvm.nd.array(x_np, dev))
    module.set_input("w", tvm.nd.array(w_np, dev))
    module.set_input("bias", tvm.nd.array(bias_np, dev))

    # 掐表直驱
    t0 = time.time()
    module.run()
    t1 = time.time()

    # 获取输出完成严苛校验
    out = module.get_output(0).numpy()

    # 模拟标准理论计算以进行数值对账
    ref_matmul = np.dot(x_np, w_np)
    ref_bias = ref_matmul + bias_np
    ref_relu = np.maximum(ref_bias, 0)

    np.testing.assert_allclose(out, ref_relu, rtol=1e-5)
    print(f"数据校验通过。当前完整网络图运行耗时: {(t1 - t0) * 1000:.2f} 毫秒")

    # GFLOPS 计算
    gflops = 2 * M * K * N / (t1 - t0) * 1e-9
    print(f"当前完整网络图 GFLOPS: {gflops:.2f} GFLOPS")


if __name__ == "__main__":
    main()