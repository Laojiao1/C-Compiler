"""
建议执行命令，保留运行日志：python relax_pass.py 2>&1 | tee relax_pass.log
"""
import tvm
from tvm import relax
from tvm.relax import script as R
from tvm.script import ir_module
import numpy as np


@ir_module
class Graph:
    @R.function # 告诉 TVM 的解析器，下面这段代码不要当成普通 Python 函数去执行，而是把它当成图纸，解析成 TVM 内部的计算图对象（relax.Function）
    def main(x: R.Tensor((4,), "float32")) -> R.Tensor((4,), "float32"):
        with R.dataflow(): # 可以理解为一个“纯粹的数据流区域”的代码块（里面全是数学计算，没有"副作用"，编译器可以放心对其优化）
            # R.const 的内容是编译器在构建 IR 的时候就已知的数据
            k = R.add(R.const([1, 2, 3, 4], "float32"), R.const([4, 3, 2, 1], "float32")) # 两个常量
            y = R.add(x, k) # 两个变量
            # 定义 z、unused、out，使用 R.multiply、R.add
            z = R.multiply(y, y)
            unused = R.add(x, R.const([1, 2, 3, 4], "float32"))
            out = R.add(z, y)
            R.output(out) # 数据流区域的输出，out 成为区域之外可用的值
        return out

print(Graph.script())

def analyze_uses(mod):
    # 直接访问 python 暴露的编译器数据结构
    func = mod["main"] # 获取函数体
    seq = func.body # 获取代码块
    bindings = seq.blocks[0].bindings # 这里 blocks 的长度为 1，因为只有一个 with R.dataflow()

    print(type(seq).__name__)
    print(type(bindings).__name__) # DataflowBlock

    for binding in bindings: # 这里的 block.bindings 是一个列表，存放代码块里的每一条赋值语句
        var = binding.var # 相当于等号左边被赋值的变量
        value = binding.value # 相当于等号右边具体的计算操作
        print(var.name, var.ty, type(value).__name__) # 打印变量名，变量类型信息，call（函数/算子调用）
        if isinstance(value, relax.Call): # call 类型包含三个重要的字段
            print("  op:", value.op.name) # 算子名称
            print("  args:", [type(arg).__name__ for arg in value.args]) # 传给这个算子的实际参数列表
            print("  uses:", [arg.name for arg in value.args if isinstance(arg, relax.Var)]) # 数据依赖分析，检查这行计算用到了那些上游变量，从而找出整张计算图的依赖关系

    # 先登记输入和所有定义，确保没有任何使用者的变量也能被统计
    uses = {param: [] for param in func.params}
    print(uses)
    for binding in bindings:
        uses[binding.var] = []

    # 每出现一次 var 操作数就记录一次，保留同一 Call 中的重复引用
    for binding in bindings:
        if isinstance(binding.value, relax.Call):
            for arg in binding.value.args:
                if isinstance(arg, relax.Var):
                    uses[arg].append(binding.var.name)

    # 函数返回值同样属于使用，否则 out 会被误判为死代码
    if isinstance(seq.body, relax.Var):
        uses[seq.body].append("return")

    for var, users in uses.items():
        print(f"{var.name}: uses={len(users)}, users={users}")

# analyze_uses(Graph)

# PASS
folded = relax.transform.FoldConstant()(Graph) # 常量折叠
dce_only = relax.transform.DeadCodeElimination()(Graph) # 死代码消除
optimized = relax.transform.DeadCodeElimination()(folded) # 常量折叠 + 死代码消除

for title, mod in [("Original", Graph), ("FoldConstant", folded),
                   ("DCE only", dce_only), ("Fold + DCE", optimized)]:
    print(f"\n===== {title} =====")
    print(mod.script())
    analyze_uses(mod)  # 针对变换后的 IR 重新统计

# 编译 + 执行
def run(mod, x_np):
    executable = tvm.compile(mod, target="llvm") # build
    vm = relax.VirtualMachine(executable, tvm.cpu()) # runtime
    return vm["main"](tvm.runtime.tensor(x_np)).numpy()

c0 = np.array([1, 2, 3, 4], dtype=np.float32)
c1 = np.array([4, 3, 2, 1], dtype=np.float32)
inputs = [
    np.array([-5, -4, -3, -2], dtype=np.float32),
    np.array([1, -2, 0.5, 3], dtype=np.float32),
]

for x_np in inputs:
    # NumPy 独立计算参考值，不依赖编译后的图
    y = x_np + (c0 + c1)
    expected = y * y + y
    before = run(Graph, x_np)
    after = run(optimized, x_np)

    np.testing.assert_allclose(before, expected)
    np.testing.assert_allclose(after, expected)
    print("Input:", x_np)
    print("Before:", before, "After:", after, "Expected:", expected)
    print("Correctness: PASS")


# 错误 shape 测试代码
wrong_x = np.ones((5,), dtype=np.float32)
try:
    run(optimized, wrong_x)
except (ValueError, RuntimeError) as exc:
    print("Shape rejected:", exc)
else:
    raise AssertionError("Expected shape (5,) to be rejected")


# 错误反例验证：改造计算图，把原来的 unused 真正参与输出计算，验证这个分支不能够被当成死代码进行消除
@ir_module
class LiveGraph:
    @R.function
    def main(x: R.Tensor((4,), "float32")) -> R.Tensor((4,), "float32"):
        with R.dataflow():
            k = R.add(R.const([1, 2, 3, 4], "float32"),
                      R.const([4, 3, 2, 1], "float32"))
            y = R.add(x, k)
            z = R.multiply(y, y)
            unused = R.add(x, R.const([1, 2, 3, 4], "float32"))
            out = R.add(z, y)
            final = R.add(out, unused)  # unused 现在出现在最终结果的依赖链上
            R.output(final)
        return final

live_dce = relax.transform.DeadCodeElimination()(LiveGraph)
print(live_dce.script())
analyze_uses(live_dce)

for x_np in inputs:
    y = x_np + (c0 + c1)
    excepted = y * y + y + (x_np + c0)
    np.testing.assert_allclose(run(LiveGraph, x_np), excepted)
    np.testing.assert_allclose(run(live_dce, x_np), excepted)
    print("Live branch correctness: PASS", excepted)