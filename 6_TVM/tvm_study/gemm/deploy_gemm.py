import os
import time
import numpy as np
import tvm
import tvm.runtime as runtime

def main():
    print("=" * 60)
    print(" 纯净 Runtime 部署端：执行零编译直接物理加载")
    print("=" * 60)

    # 1. 定义矩阵维度
    M, N, K = 1024, 1024, 1024
    dev = tvm.cpu(0)

    # 2. 动态获取当前脚本所在目录，定位编译产物路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(current_dir, "so", "lib_kv_gemm.so")

    if not os.path.exists(so_path):
        raise FileNotFoundError(
            f"未在指定路径找到编译产物: {so_path}\n请确保先执行了编译导出脚本。"
        )

    print(f"检测到物理二进制文件，目标路径: {so_path}")

    # 3. 运行时核心解耦：直接从硬盘载入二进制机器码模块，无需编译器参与
    loaded_mod = tvm.runtime.load_module(so_path)

    # 从标准模块容器中提取已完成向量化和多核并行优化的 PackedFunc
    gemm_func = loaded_mod["my_gemm"]
    print("二进制模块成功载入内存，硬件控制总线已就绪。")

    # 4. 准备测试数据集（使用 NumPy 生成测试矩阵）
    a_np = np.random.uniform(size=(M, K)).astype("float32")
    b_np = np.random.uniform(size=(K, N)).astype("float32")
    c_np = np.zeros((M, N), dtype="float32")

    # 将 NumPy 阵列转换为 TVM Runtime 专用的轻量级 NDArray 容器，并将其绑定至 CPU 硬件内存
    a_tvm = tvm.nd.array(a_np, dev)
    b_tvm = tvm.nd.array(b_np, dev)
    c_tvm = tvm.nd.array(c_np, dev)

    # 5. 执行运行期硬直驱推理
    print("\n硬件执行单元启动，运行优化后的 GEMM 算子...")
    t0 = time.time()
    gemm_func(a_tvm, b_tvm, c_tvm)
    t1 = time.time()
    execution_time_seconds = t1 - t0
    execution_time_ms = execution_time_seconds * 1000

    # 6. 正确性校验与 GFLOPS 计算
    c_np_from_tvm = c_tvm.numpy()
    try:
        np.testing.assert_allclose(c_np_from_tvm, np.dot(a_np, b_np), rtol=1e-5)

        # 计算总浮点运算次数 (GEMM 公式: 2 * M * N * K)
        total_flops = 2 * M * N * K
        # 转换为每秒十亿次浮点运算 (GFLOPS)
        gflops = total_flops / execution_time_seconds / 1e9

        print("=" * 60)
        print(" 数据正确性校验通过：计算结果与标准数学矩阵完全吻合。")
        print("=" * 60)
        print(f" 纯运行期二进制直驱推理耗时: {execution_time_ms:.2f} 毫秒")
        print(f" 纯运行期二进制直驱吞吐性能: {gflops:.2f} GFLOPS")
        print("=" * 60)
    except AssertionError as e:
        print("数据校验失败：计算结果存在数值截断或优化越界。")
        print(e)


if __name__ == "__main__":
    main()