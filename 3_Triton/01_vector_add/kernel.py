import torch
import triton
import triton.language as tl

SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
MAX_BLOCK_SIZE = 65536

@triton.jit
def add_kernel(
    x_ptr,      # 指向第一个输入向量的指针
    y_ptr,      # 指向第二个输入向量的指针
    output_ptr, # 指向输出向量的指针
    n_elements, # 向量的总元素个数
    BLOCK_SIZE: tl.constexpr, # 常量，编译器在编译时必须知道它的具体数值
):
    # 第一步：我是谁？拿到当前 Block 的 id
    # axis=0 表示我们在一维网格上调度
    pid = tl.program_id(axis=0)
    
    # 第二步：我在哪？计算当前 Block 承包的数据块的物理起点
    # 比如 pid=2, BLOCK_SIZE=128，那么起点就是 256
    block_start = pid * BLOCK_SIZE
    
    # 获取属于当前 Block 的相对偏移序列：[0, 1, 2, ..., BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 第三步：构建 Mask 掩码，防止最后一个 Block 越界访问
    # 如果总元素是 1000，最后一个 block 会算到 1023，这里把超出的部分标为 False
    mask = offsets < n_elements
    
    # 第四步：干活！计算绝对指针位置，并执行完美的 Coalesced Memory Access
    # x_pointers 和 y_pointers 是一组连续的地址，硬件会发起合并内存事务
    x_pointers = x_ptr + offsets
    y_pointers = y_ptr + offsets
    
    # 从 Global Memory 读取连续 tile。Triton 的 blocked tensor 通常映射到寄存器，
    # 并不等同于显式申请 CUDA Shared Memory。
    # 未指定 other 时，被 mask 掉的 lane 值不应被依赖；后面的 store 也使用同一 mask。
    x = tl.load(x_pointers, mask=mask)
    y = tl.load(y_pointers, mask=mask)
    
    # 在寄存器/ALU中完成加法
    output = x + y
    
    # 计算输出的绝对指针
    output_pointers = output_ptr + offsets
    
    # 将寄存器中的结果按连续地址写回 Global Memory
    tl.store(output_pointers, output, mask=mask, other=0.0)

def add(x: torch.Tensor, y: torch.Tensor, BLOCK_SIZE: int = 1024):
    """
    Triton 算子的 Python 封装入口
    """
    # 先在 CPU 侧拦截非法输入。否则错误会拖到 Kernel 内部，变成错算或非法显存访问。
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("x 和 y 必须是 torch.Tensor")
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("x 和 y 必须位于 CUDA 设备")
    if x.device != y.device:
        raise ValueError("x 和 y 必须位于同一 CUDA 设备")
    if x.shape != y.shape:
        raise ValueError("x 和 y 的 shape 必须一致")
    if x.dtype != y.dtype:
        raise TypeError("x 和 y 的 dtype 必须一致")
    if x.dtype not in SUPPORTED_DTYPES:
        raise TypeError(f"不支持 dtype={x.dtype}，仅支持 FP16、BF16 和 FP32")
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("x 和 y 必须连续，当前 Kernel 按一维连续地址访问")
    if type(BLOCK_SIZE) is not int or BLOCK_SIZE <= 0:
        raise ValueError("BLOCK_SIZE 必须是正整数")
    if BLOCK_SIZE > MAX_BLOCK_SIZE or BLOCK_SIZE & (BLOCK_SIZE - 1):
        raise ValueError("BLOCK_SIZE 必须是不超过 65536 的 2 的幂")
    
    # 获取元素数量
    n_elements = x.numel()
    
    # 分配输出内存 (在 GPU 上提前占好坑)
    output = torch.empty_like(x)

    # 空张量没有 program instance 可启动，直接返回即可。
    if n_elements == 0:
        return output
    
    # 计算需要多少个 Block 才能覆盖整个数组
    # Triton 提供了一个辅助函数 cdiv 进行向上取整 (ceiling division)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    
    # 启动 Kernel！
    # 注意：这里的传参顺序必须和 add_kernel 定义的一致
    add_kernel[grid](
        x, y, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
