# lowering_notes

## Vector Add

本节以 `BLOCK_SIZE=256`、`num_warps=4`、`n_elements=1000` 的 Vector Add 为样本，对照实际编译产物追踪：

```text
Python DSL
→ TTIR
→ TTGIR
→ LLVM IR
→ PTX
→ cubin
→ SASS
```

相关文件：

- Kernel 与导出脚本：[`vector_add_ir/dump_vector_add.py`](vector_add_ir/dump_vector_add.py)
- 各层编译产物：[`vector_add_ir/IR`](vector_add_ir/IR)
- SASS 反汇编：[`vector_add_ir/IR/vector_add.sass`](vector_add_ir/IR/vector_add.sass)

导出脚本在保存 IR 之前执行：

```python
torch.testing.assert_close(output, x + y)
```

因此，本次分析建立在 correctness 已通过的 Kernel 上，而不是只根据生成指令推测程序是否正确。

### Python DSL -> TTIR

#### Python DSL 中描述了什么

Vector Add Kernel 的主体是：

```python
pid = tl.program_id(axis=0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements

x = tl.load(x_ptr + offsets, mask=mask)
y = tl.load(y_ptr + offsets, mask=mask)
output = x + y

tl.store(output_ptr + offsets, output, mask=mask)
```

这段代码表达的是一个 Triton Program 对 256 个逻辑元素执行的张量计算：

```text
取得 Program ID
→ 构造 256 个逻辑下标
→ 生成边界 Mask
→ 读取两个 Tensor
→ 逐元素相加
→ 写回结果
```

在这一层，我们只描述“一个 Program 处理一个 Tile”。源码没有显式写出 `threadIdx.x`，也没有规定每个 CUDA Thread 负责哪些元素。

`n_elements` 和 `BLOCK_SIZE` 都被声明为 `tl.constexpr`。本次编译使用：

```text
n_elements = 1000
BLOCK_SIZE = 256
```

所以它们会进入当前 Kernel 变体的编译期特化，而不是作为普通运行时标量参数参与每次计算。

#### `source` 不是 Python 源码副本

[`vector_add_source.txt`](vector_add_ir/IR/vector_add_source.txt) 来自：

```python
compiled_kernel.asm["source"]
```

在本次 Triton 环境中，它保存的是一份较早期的 Triton/MLIR 文本，而不是 `dump_vector_add.py` 的 Python 源码副本。这里仍能看到：

```mlir
%pid = tt.get_program_id x
%offsets_10 = tt.make_range {end = 256, start = 0}
%x_26 = tt.load ...
%output = arith.addf ...
tt.store ...
```

它还包含若干针对 32 位整数范围的扩展和检查，例如：

```mlir
arith.extsi
arith.cmpi sle
arith.cmpi sge
arith.andi
```

这些检查来自前端对整数运算语义的保守表达。在优化后的 TTIR 中，它们已经被证明无须保留并被消除。

需要注意，`compiled_kernel.asm` 中键名及其对应的精确编译时机可能随 Triton 版本变化。因此，这里只依据本次实际文件判断：

> `source` 比当前保存的 `ttir` 更早，不能把它当成另一份 Python 文件。

#### 优化后的 TTIR

[`vector_add.ttir`](vector_add_ir/IR/vector_add.ttir) 已经把 Python DSL 转换成以静态单赋值形式（Static Single Assignment，SSA）组织的 Tensor 级中间表示：

| Triton Python | 当前 TTIR |
| --- | --- |
| `tl.program_id(axis=0)` | `tt.get_program_id x` |
| `BLOCK_SIZE = 256` | `arith.constant 256 : i32` |
| `pid * BLOCK_SIZE` | `arith.muli` |
| `tl.arange(0, 256)` | `tt.make_range` |
| 标量基址广播 | `tt.splat` |
| `base + arange` | `arith.addi` |
| `offsets < n_elements` | `arith.cmpi slt` |
| 标量指针广播 | `tt.splat` |
| `ptr + offsets` | `tt.addptr` |
| `tl.load` | `tt.load` |
| `x + y` | `arith.addf` |
| `tl.store` | `tt.store` |

关键的数据流可以压缩为：

```mlir
%pid = tt.get_program_id x
%base = arith.muli %pid, %c256_i32
%range = tt.make_range {start = 0, end = 256}
%base_tensor = tt.splat %base
%offsets = arith.addi %base_tensor, %range
%mask = arith.cmpi slt, %offsets, dense<1000>
```

这里的 `tt.splat` 用于把一个标量复制成 Tensor 形状，使它可以参与逐元素 Tensor 运算：

```text
pid * 256                         scalar<i32>
tl.arange(0, 256)                 tensor<256xi32>
tt.splat(pid * 256)               tensor<256xi32>
base tensor + range tensor        tensor<256xi32>
```

同样，标量指针也先通过 `tt.splat` 扩展为指针 Tensor，再由 `tt.addptr` 生成 256 个逻辑地址：

```text
!tt.ptr<f32>
    ↓ tt.splat
tensor<256x!tt.ptr<f32>>
    ↓ tt.addptr
256 个逻辑元素地址
```

Mask 的类型是：

```mlir
tensor<256xi1>
```

它表示 256 个逻辑位置各自拥有一个布尔条件。此时 `tt.load`、`arith.addf` 和 `tt.store` 仍然操作完整的逻辑 Tensor。

#### 本阶段能确认什么

从 TTIR 可以直接确认：

- 当前 Program 的逻辑 Tile 包含 256 个元素；
- `n_elements=1000` 和 `BLOCK_SIZE=256` 已经成为编译期常量；
- 标量基址和标量指针通过 `tt.splat` 进入 Tensor 运算；
- Mask、Load、Add 和 Store 都保留 Tensor 语义；
- 较早 `source` 中的整数范围检查已被优化掉。

TTIR 还不能回答：

- 256 个元素具体分配给多少个 CUDA Thread；
- 每个 Thread 持有几个元素；
- Load/Store 最终使用标量还是向量指令；
- 线程访问是否形成了理想的硬件内存事务。

这些问题要从 TTGIR 及更低层编译产物继续追踪。

### TTIR -> TTGIR

#### Tensor 类型获得 Layout

在 [`vector_add.ttgir`](vector_add_ir/IR/vector_add.ttgir) 中，主要 Tensor 类型都增加了 `#blocked` 编码：

```mlir
tensor<256xi32, #blocked>
tensor<256xi1, #blocked>
tensor<256xf32, #blocked>
tensor<256x!tt.ptr<f32>, #blocked>
```

TTIR 中的：

```mlir
tensor<256xf32>
```

只表示一个包含 256 个 FP32 元素的逻辑 Tensor。TTGIR 中的：

```mlir
tensor<256xf32, #blocked>
```

则进一步指定这些逻辑元素如何分布到 GPU 的 Warp、Lane 和线程私有值上。

#### 本次 Blocked Layout

实际 TTGIR 给出的 Layout 是：

```mlir
#blocked = #ttg.blocked<{
  sizePerThread = [2],
  threadsPerWarp = [32],
  warpsPerCTA = [4],
  order = [0]
}>
```

各字段含义如下：

| 字段 | 当前值 | 含义 |
| --- | ---: | --- |
| `sizePerThread` | `[2]` | 每个线程持有两个逻辑元素 |
| `threadsPerWarp` | `[32]` | 每个 Warp 有 32 个线程 |
| `warpsPerCTA` | `[4]` | 一个 CTA 使用 4 个 Warp |
| `order` | `[0]` | 一维 Tensor 只有第 0 维 |

因此，一个 CTA 中参与该 Layout 的线程数是：

```text
32 threads/warp × 4 warps/CTA = 128 threads/CTA
```

一个 CTA 基本块覆盖的元素数是：

```text
2 elements/thread × 32 threads/warp × 4 warps/CTA
= 256 elements/CTA
```

它恰好覆盖当前 `tensor<256x...>`，不需要重复铺设 Layout 基本块。

模块属性也直接记录了：

```mlir
"ttg.num-warps" = 4
"ttg.threads-per-warp" = 32
```

#### Layout 在数据流中保持一致

从 `tt.make_range`、`tt.splat`、`arith.addi`、`tt.load`、`arith.addf` 到 `tt.store`，当前数据流一直使用同一个 `#blocked`：

```text
make_range  #blocked
     ↓
offsets     #blocked
     ↓
load        #blocked
     ↓
add         #blocked
     ↓
store       #blocked
```

当前 TTGIR 中没有出现：

```mlir
ttg.convert_layout
```

这可以直接证明该 TTGIR 没有显式 Layout Conversion。对于 Vector Add 这种逐元素算子，Load、Add 和 Store 可以沿用同一种元素分布，不需要为了后续运算重新分配线程间的数据。

#### 本阶段能确认什么

TTGIR 已经能够确认：

- 一个 Triton Program 对应一个处理 256 个元素的 CTA Tile；
- 当前 CTA 使用 4 个 Warp，即 128 个 CUDA Thread；
- 每个 Thread 持有两个逻辑元素；
- 整条逐元素数据流使用同一个 Blocked Layout；
- 当前 TTGIR 不需要 `ttg.convert_layout`。

TTGIR 给出了元素分布规则，但此时仍应继续检查 LLVM IR 和 PTX，确认该 Layout 最终如何变成线程索引和访存指令。

### TTGIR -> LLVM IR

#### Tensor 语义变成每线程代码

[`vector_add.ll`](vector_add_ir/IR/vector_add.ll) 已经不再保留 `tensor<256x...>` 和 `#blocked`。TTGIR 中隐含的线程映射被展开成读取 CUDA 特殊寄存器和计算线程私有地址的代码：

```llvm
%6 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
%7 = shl i32 %6, 8

%8 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
%9 = shl i32 %8, 1
%10 = and i32 %9, 254
%11 = or disjoint i32 %10, %7
```

其中：

```text
%6 = blockIdx.x
%7 = blockIdx.x << 8
   = blockIdx.x × 256

%8 = threadIdx.x
%9 = threadIdx.x << 1
   = threadIdx.x × 2
```

当前 `threadIdx.x` 的范围由 128 线程的启动约束限定，因此 `threadIdx.x × 2` 处于 `[0, 254]`。`and 254` 保留偶数低位，随后 `or disjoint` 合并 CTA 基址与线程基址。由于两部分使用互不重叠的位区间，它在当前条件下等价于：

```text
base_offset = blockIdx.x * 256 + threadIdx.x * 2
```

这正是 TTGIR 中：

```text
sizePerThread = 2
```

降低到线程索引后的具体形式。

#### 每个线程持有两个元素

LLVM IR 对 `x` 和 `y` 分别执行一次返回两个 32 位值的内联汇编 Load：

```llvm
ld.global.v2.b32 { $0, $1 }, [ $2 + 0 ]
```

Load 结果被拆成两个值：

```llvm
%16 = extractvalue { i32, i32 } %15, 0
%17 = extractvalue { i32, i32 } %15, 1
```

再分别转为 FP32 并执行两次加法：

```llvm
%26 = fadd float %18, %24
%27 = fadd float %19, %25
```

因此，当前每个 Thread 处理的逻辑元素是：

```text
element 0 = blockIdx.x * 256 + threadIdx.x * 2
element 1 = blockIdx.x * 256 + threadIdx.x * 2 + 1
```

TTGIR 中“每线程两个元素”的 Layout，已经在 LLVM IR 中变成两个线程私有标量值。

#### Mask 变成线程级 Predicate

TTIR 和 TTGIR 中的 Mask 是：

```text
tensor<256xi1>
```

LLVM IR 中则变成：

```llvm
%12 = icmp slt i32 %11, 1000
```

随后它作为 Predicate 传给向量 Load 和 Store。

当前 `n_elements=1000`，而线程基址始终是偶数。最后一个有效 Pair 是：

```text
{998, 999}
```

下一个 Pair 是：

```text
{1000, 1001}
```

所以同一个 Predicate 足以控制一对元素：两个元素要么同时有效，要么同时越界。这是当前 `n_elements=1000` 特化结果的一部分，不能脱离当前 Shape 推广为所有长度都必然采用一个 Predicate。

#### Kernel 启动约束

LLVM IR 函数属性包含：

```llvm
"nvvm.reqntid"="128"
```

它与 TTGIR 中：

```text
4 warps × 32 threads = 128 threads
```

完全对应。

LLVM 函数签名中还存在两个没有参与主体计算的附加指针参数。当前 Vector Add 的逻辑数据流只使用前三个全局内存指针；在没有更多编译器元数据证据时，不对附加参数的具体用途作猜测。

#### 本阶段能确认什么

LLVM IR 可以直接确认：

- `tt.get_program_id x` 已降低为读取 `ctaid.x`；
- Blocked Layout 已降低为 `tid.x` 参与的索引计算；
- 一个 CTA 使用 128 个线程；
- 每个线程计算两个相邻元素；
- Tensor 级 Load/Store 已变成每线程的二元素向量访存；
- Tensor 级 Add 已变成两个线程私有 FP32 标量加法；
- Mask 已变成控制访存的线程级 Predicate。

### LLVM IR -> PTX

#### Kernel 入口与虚拟寄存器

[`vector_add.ptx`](vector_add_ir/IR/vector_add.ptx) 的入口声明为：

```ptx
.visible .entry vector_add_kernel(...)
.reqntid 128
```

`.reqntid 128` 再次确认 Kernel 的 CTA 大小是 128 个线程。

PTX 还声明了：

```ptx
.reg .pred %p<2>;
.reg .b32  %r<13>;
.reg .b64  %rd<8>;
```

这些是 PTX 虚拟寄存器声明，不能直接当成最终 SASS 的物理寄存器用量。真实寄存器分配需要继续查看 cubin/SASS 或编译工具报告。

#### Program ID 与 Thread ID

LLVM NVVM Intrinsic 在 PTX 中变成特殊寄存器读取：

```ptx
mov.u32 %r7, %ctaid.x;
mov.u32 %r9, %tid.x;
```

完整索引计算是：

```ptx
shl.b32 %r8, %r7, 8;
shl.b32 %r10, %r9, 1;
and.b32 %r11, %r10, 254;
or.b32 %r12, %r11, %r8;
```

对应：

```text
Program base = %ctaid.x × 256
Thread base  = %tid.x × 2
Element base = Program base + Thread base
```

这条指令链把 Triton 的：

```python
pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
```

落实成了 CUDA Block 坐标、Thread 坐标和每线程两个元素的地址计算。

#### Mask 与 Predicated Memory Access

边界判断降低为：

```ptx
setp.lt.s32 %p1, %r12, 1000;
```

全局内存 Load 和 Store 都带有相同的 Predicate：

```ptx
@%p1 ld.global.v2.b32 ...
@%p1 st.global.v2.b32 ...
```

当 Predicate 为假时，对应线程不会执行这些全局内存访问。Load 前的寄存器被初始化为零：

```ptx
mov.u32 %r1, 0x0;
mov.u32 %r2, 0x0;
```

所以被 Mask 掉的 Load 不会留下未定义的 PTX 虚拟寄存器值；而 Store 同样受 Predicate 控制，不会写出有效范围。

#### Load、Add 与 Store

对每个输入，PTX 使用：

```ptx
ld.global.v2.b32
```

它表示每个线程从自己的起始地址读取两个连续的 32 位值。随后执行：

```ptx
add.f32 %r5, %r1, %r3;
add.f32 %r6, %r2, %r4;
```

最后使用：

```ptx
st.global.v2.b32
```

写回两个连续的 32 位结果。

需要区分两个概念：

- `v2` 直接证明单个线程执行二元素向量访存；
- 相邻线程的起始下标依次为 `0, 2, 4, ...`，直接证明一个 Warp 覆盖连续的元素区间。

这是一种适合合并访存（Coalesced Memory Access）的地址模式，但仅凭 PTX 不能确认最终产生多少个内存事务、缓存命中情况和实际带宽利用率。这些硬件现象仍需 Nsight Compute 验证。

#### 没有出现线程间通信

当前 PTX 中没有观察到：

```text
shfl.sync
ld.shared
st.shared
bar.sync
```

这与 TTGIR 中没有 `ttg.convert_layout` 相互印证。当前 Vector Add 的两个输入、加法结果和输出保持同一种线程分布，每个线程只处理自己的两个元素，不需要与其他线程交换数据。

#### 本阶段的证据边界

PTX 可以直接证明：

- CTA 请求 128 个线程；
- 使用 `%ctaid.x` 和 `%tid.x` 计算元素位置；
- 每个线程处理两个相邻 FP32 元素；
- Load/Store 使用 Predicate 实现边界保护；
- 使用 `ld.global.v2.b32` 和 `st.global.v2.b32`；
- 每个线程执行两次 `add.f32`；
- 没有显式 Shuffle、Shared Memory 或 Barrier 指令。

PTX 不能单独证明：

- 最终 SASS 是否与 PTX 保持一一对应；
- 实际物理寄存器用量；
- Global Memory 的真实事务数和吞吐；
- Kernel 是否已经达到带宽上限；
- 运行时延迟主要受哪个硬件单元限制。

### PTX -> cubin

#### cubin 是什么

[`vector_add.cubin`](vector_add_ir/IR/vector_add.cubin) 是面向目标 NVIDIA GPU 架构的二进制代码容器。当前 PTX 声明的目标是：

```ptx
.target sm_120a
```

PTX 仍属于虚拟指令集，而 cubin 中保存的是目标架构可装载的机器代码及相关元数据。GPU 最终执行的不是 `.ttir`、`.ttgir`、LLVM IR 或 PTX 文本，而是 cubin 中的机器指令。

#### 当前能确认什么

当前目录已经成功保存 `vector_add.cubin`，可以确认编译链生成了目标二进制产物。但 cubin 本身不是适合直接阅读的文本，不能仅根据二进制文件的存在判断具体 SASS 指令。

通过反汇编 cubin 得到可阅读的 [`vector_add.sass`](vector_add_ir/IR/vector_add.sass) 后，才能继续检查目标架构实际采用的机器指令。由此要明确区分：

```text
PTX
虚拟指令表示

cubin
目标架构二进制容器

SASS
从 cubin 解码得到的机器指令表示
```

### cubin -> SASS

#### 先找到真正的 Kernel 指令段

`vector_add.sass` 不只包含 Kernel 指令，还包含 ELF Section、调试信息、PTX 文本副本和 CUDA 元数据。当前真正需要阅读的部分从下面的 Section 开始：

```text
.text.vector_add_kernel
```

其中每一行左侧的十六进制数字是指令地址，例如：

```sass
/*0000*/ LDC ...
/*0010*/ S2R ...
/*0020*/ LDC.64 ...
```

本次目标架构是：

```text
sm_120a
```

SASS 元数据还记录了生成 cubin 的工具为 CUDA 12.9 的 `ptxas-blackwell`。因此，本节观察到的具体指令选择属于当前 Blackwell 目标，不能直接推广为所有 NVIDIA GPU 架构都会生成完全相同的 SASS。

#### PTX 与 SASS 不是逐行翻译

PTX 描述虚拟 NVIDIA GPU 指令，而 SASS 是目标架构实际执行的机器指令。后端可以在二者之间继续进行：

- 指令融合；
- Predicate 条件反转；
- 地址计算重组；
- 寄存器分配；
- 目标架构相关的指令选择。

当前 Vector Add 的关键对应关系如下：

| PTX | 当前 SASS | 作用 |
| --- | --- | --- |
| `mov.u32 ..., %tid.x` | `S2R R0, SR_TID.X` | 读取线程编号 |
| `mov.u32 ..., %ctaid.x` | `S2R R9, SR_CTAID.X` | 读取 CTA 编号 |
| `shl`、`and`、`or` | `SHF`、`LOP3.LUT`、`LEA` | 计算元素索引并重组部分指令 |
| `setp.lt index, 1000` | `ISETP.GE P0, index, 0x3e8` | 生成方向相反的越界 Predicate |
| `mul.wide` + `add.s64` | `IMAD.WIDE` | 融合字节偏移与指针地址计算 |
| `ld.global.v2.b32` | `LDG.E.64` | 读取两个连续的 FP32 元素 |
| 两条 `add.f32` | 两条 `FADD` | 分别完成两个 FP32 加法 |
| Predicated Store | `@P0 EXIT` + `STG.E.64` | 越界线程退出，有效线程写回 |
| `ret` | `EXIT` | 结束当前线程的 Kernel 执行 |

这张表不是一条 PTX 对应一条 SASS 的机械映射。例如，PTX 中独立存在的移位、按位操作和地址加法，在 SASS 中已经被重新组合；PTX 中的 Predicated Store 也没有直接变成一条带相同 Predicate 的 Store。

#### Program ID 与 Thread ID 的最终实现

SASS 首先从特殊寄存器读取线程编号和 CTA 编号：

```sass
S2R R0, SR_TID.X
S2R R9, SR_CTAID.X
```

然后执行：

```sass
SHF.L.U32 R0, R0, 0x1, RZ
LOP3.LUT R0, R0, 0xfe, RZ, 0xc0, !PT
LEA R9, R9, R0, 0x8
```

它们对应 PTX 中的线程下标乘 2、保留有效低位，以及 CTA 下标乘 256 后与线程基址合并。最终 `R9` 表示：

```text
element_base
= blockIdx.x * 256
+ threadIdx.x * 2
```

这里第一次直接看到：Triton 源码中的 `tl.program_id` 和 `tl.arange` 已经彻底消失，最终机器只需要 CTA ID、Thread ID 和整数地址计算。

#### Mask 的 Predicate 被反转

PTX 使用：

```ptx
setp.lt.s32 %p1, index, 1000
@%p1 ld.global.v2.b32 ...
@%p1 st.global.v2.b32 ...
```

其中 Predicate 为真代表当前 Pair 有效。

SASS 改为：

```sass
ISETP.GE.AND P0, PT, R9, 0x3e8, PT
@!P0 LDG.E.64 ...
@P0 EXIT
STG.E.64 ...
```

`0x3e8` 就是十进制的 `1000`。当前 `P0` 表示：

```text
element_base >= 1000
```

也就是“当前线程越界”。因此：

- `@!P0 LDG.E.64`：没有越界的线程才执行 Load；
- `@P0 EXIT`：越界线程在 Store 前直接结束；
- `STG.E.64`：执行到这里的线程都有效，所以 Store 不再需要 Predicate。

这说明高层的同一个 Mask 语义，在不同层可以采用不同控制流形式：

```text
TTIR / TTGIR
tensor<256xi1> Mask
        ↓
LLVM IR / PTX
Predicated Load 和 Store
        ↓
SASS
Predicated Load + 越界线程提前 EXIT
```

语义保持一致，但实际机器指令并不要求保持相同的表面结构。

#### 二元素向量访存变成 64 位指令

PTX 中每个线程执行：

```ptx
ld.global.v2.b32
```

SASS 中对应：

```sass
@!P0 LDG.E.64 R10, desc[UR4][R2.64]
@!P0 LDG.E.64 R12, desc[UR4][R4.64]
```

`v2.b32` 的总宽度是：

```text
2 × 32 bit = 64 bit
```

所以目标指令使用 `LDG.E.64`。第一次 Load 得到的两个 FP32 位模式放在以 `R10` 开始的寄存器对中，第二次 Load 放在以 `R12` 开始的寄存器对中：

```text
x0 → R10    x1 → R11
y0 → R12    y1 → R13
```

随后执行：

```sass
FADD R10, R12, R10
FADD R11, R13, R11
```

得到：

```text
R10 = x0 + y0
R11 = x1 + y1
```

最后：

```sass
STG.E.64 desc[UR4][R6.64], R10
```

一次写回 `R10:R11` 中的两个 FP32 结果。

`desc[UR4][R2.64]` 是当前 `sm_120a` 使用的目标相关地址形式。本阶段只需确认它是 64 位 Global Memory Load；描述符和 Uniform Register 的更底层机制不影响当前 Vector Add lowering 主线。

#### 地址计算被融合为 `IMAD.WIDE`

PTX 分别计算：

```ptx
mul.wide.s32 byte_offset, element_index, 4
add.s64 address, base_pointer, byte_offset
```

SASS 将乘 4 和 64 位地址加法融合为：

```sass
IMAD.WIDE R2, R9, 0x4, R2
IMAD.WIDE R4, R9, 0x4, R4
IMAD.WIDE R6, R9, 0x4, R6
```

三条指令分别生成 `x`、`y` 和 `output` 的访问地址。这里的 `0x4` 来自 FP32 元素大小：

```text
1 × float32 = 4 bytes
```

这再次说明，PTX 是后端继续优化的输入，而不是最终机器指令清单。

#### 实际寄存器与资源元数据

SASS 指令已经使用真实的编号寄存器，例如：

```text
R0
R2:R3
R4:R5
R6:R7
R9
R10:R11
R12:R13
```

反汇编元数据中的：

```text
EIATTR_REGCOUNT = 0x10
```

表示当前 Kernel 分配了 16 个通用寄存器。它与 PTX 中的虚拟寄存器声明不是同一个概念：PTX 虚拟寄存器数量只是中间表示的一部分，真正资源分配应以 cubin/SASS 元数据为准。

当前元数据还显示：

```text
Frame Size     = 0
Min Stack Size = 0
Required CTA   = (128, 1, 1)
```

这直接确认 Kernel 没有本地栈帧需求，并再次确认一个 CTA 使用 128 个线程。

知道寄存器数还不足以单独推导实际 Occupancy。真实并发驻留还要结合目标 SM 的寄存器容量、CTA 限制以及其他资源约束，最终可通过 Nsight Compute 验证。

#### 没有线程间通信指令

当前 `.text.vector_add_kernel` 中没有观察到：

```text
Shuffle
Shared Memory Load/Store
Barrier
```

SASS 文件中虽然存在名为 `.nv.shared.reserved.0` 的保留 Section，但其出现不能证明 Kernel 执行了共享内存访问。判断真实数据通路应以 Kernel 指令段为准；当前指令段只有 Global Memory Load/Store，没有共享内存指令。

这与前面各层形成一致证据：

```text
TTGIR
没有 ttg.convert_layout
        ↓
PTX
没有 shfl / ld.shared / st.shared / bar
        ↓
SASS
没有 Shuffle / Shared Memory / Barrier 机器指令
```

#### 暂不深入调度控制标记

反汇编文本在指令右侧还包含：

```text
&wr=...
&req=...
?trans...
?WAIT...
```

它们反映当前架构上的依赖、等待和调度控制信息。它们已经属于 Blackwell 机器指令调度细节，本阶段只需要知道这些标记服务于正确的依赖管理和指令调度，不继续解析具体编码。

#### 本阶段的证据边界

SASS 可以直接确认：

- 实际使用 `SR_TID.X` 和 `SR_CTAID.X`；
- PTX 的部分索引与地址计算被重组或融合；
- 二元素向量访存落实为 64 位 `LDG.E.64` 和 `STG.E.64`；
- 两个元素分别使用两条 `FADD`；
- 越界控制落实为 Predicated Load 和提前 `EXIT`；
- 当前 Kernel 的通用寄存器分配数为 16；
- 没有真实的 Shuffle、Shared Memory 或 Barrier 指令。

SASS 仍不能单独证明：

- 一个 Warp 的内存访问最终产生多少硬件事务；
- Global Memory 的实际吞吐率；
- 指令依赖是否造成了显著 Stall；
- 16 个寄存器是否成为 Occupancy 的限制因素；
- Kernel 的主要性能瓶颈。

这些问题需要结合 Nsight Compute 的实测指标判断。

### Vector Add Lowering 总结

当前样本形成的完整证据链是：

```text
Python DSL
一个 Program 处理 256 个逻辑元素
        ↓
TTIR
tensor<256x...> 表达 Tensor 级 Load / Add / Store
        ↓
TTGIR
sizePerThread=2
threadsPerWarp=32
warpsPerCTA=4
        ↓
LLVM IR
blockIdx.x * 256 + threadIdx.x * 2
每线程两个线程私有值
        ↓
PTX
128 threads/CTA
ld.global.v2.b32
两次 add.f32
st.global.v2.b32
        ↓
cubin
目标架构可装载的二进制代码
        ↓
SASS
LDG.E.64
两次 FADD
STG.E.64
16 个通用寄存器
```

其中，已经由 IR 或指令直接证明的事实包括：

- 一个 Program 的逻辑 Tile 是 256 个元素；
- 一个 CTA 使用 4 个 Warp、共 128 个线程；
- 每个线程负责两个相邻元素；
- Mask 在 SASS 中最终成为 Predicated Load 和越界线程提前退出；
- 当前数据流没有 Layout Conversion 和线程间通信；
- PTX 的单线程二元素向量访存最终成为 64 位 Global Memory 指令；
- SASS 报告当前 Kernel 使用 16 个通用寄存器。

仍需 Profiler 补充验证的内容包括：

- 实际合并访存事务；
- Global Memory 吞吐和带宽利用率；
- 活跃 Warp、Occupancy 与指令延迟；
- 当前 Vector Add 的真实性能瓶颈。

## Softmax

本节使用经典的数值稳定 Softmax 作为第一个线程间通信样本。实验配置是：

```text
n_rows     = 128
n_cols     = 1000
BLOCK_SIZE = 1024
num_warps  = 8
```

一个 Triton Program 处理一整行，Host 端启动 128 个 Program。每行的数学过程是：

```text
row_max    = max(row)
numerator  = exp(row - row_max)
denominator = sum(numerator)
output     = numerator / denominator
```

完整编译链为：

```text
Python DSL
→ TTIR
→ TTGIR
→ LLVM IR
→ PTX
→ cubin
→ SASS
```

相关文件：

- Kernel 与导出脚本：[`softmax_ir/dump_softmax.py`](softmax_ir/dump_softmax.py)
- Python/Triton 源码快照：[`softmax_ir/IR/softmax_source.txt`](softmax_ir/IR/softmax_source.txt)
- TTIR：[`softmax_ir/IR/softmax.ttir`](softmax_ir/IR/softmax.ttir)
- TTGIR：[`softmax_ir/IR/softmax.ttgir`](softmax_ir/IR/softmax.ttgir)
- LLVM IR：[`softmax_ir/IR/softmax.ll`](softmax_ir/IR/softmax.ll)
- PTX：[`softmax_ir/IR/softmax.ptx`](softmax_ir/IR/softmax.ptx)
- cubin：[`softmax_ir/IR/softmax.cubin`](softmax_ir/IR/softmax.cubin)
- SASS：[`softmax_ir/IR/softmax.sass`](softmax_ir/IR/softmax.sass)

导出脚本先执行 Kernel，再与 PyTorch 的实现比较：

```python
y_ref = torch.softmax(x, dim=1)

torch.testing.assert_close(
    y,
    y_ref,
    rtol=1e-4,
    atol=1e-5,
)
```

只有 correctness 通过后，脚本才会获取并保存编译产物。因此，本节仍然遵守“先验证计算结果，再阅读 IR 和指令”的顺序。

### Python DSL -> TTIR

#### 为什么先观察非 Online Softmax

当前 Kernel 没有使用 Online Softmax，而是显式执行两次 Reduction：

```text
第一次 Reduction：max
第二次 Reduction：sum
```

这样可以把以下结构直接暴露在 IR 中：

```text
局部规约
→ Warp 内规约
→ 跨 Warp 合并
→ Reduction 结果广播
```

当前实现仍然是一个融合 Kernel，并不等于把输入行从 Global Memory 读取两次。实际编译结果只读取一次输入行，计算出的四个 Numerator 在线程私有寄存器中保留到 Sum Reduction 结束。

Online Softmax 会把最大值与归一化因子的更新组合进同一个在线状态。本节暂不展开该算法，只用结构更直接的经典 Softmax 观察 Reduction Lowering。

#### `softmax_source.txt` 是真实 DSL 源码

本次脚本通过：

```python
inspect.getsource(softmax_kernel.fn)
```

保存 [`softmax_source.txt`](softmax_ir/IR/softmax_source.txt)，所以它是实际的 Python/Triton Kernel 源码。

这与 Vector Add 的 `vector_add_source.txt` 不同。后者来自 `compiled_kernel.asm["source"]`，在当时环境中保存的是一份较早期 Triton IR。阅读不同导出脚本时，不能只根据文件名中的 `source` 判断内容属于哪一层。

#### 一个 Program 处理一整行

源码中的：

```python
row_idx = tl.program_id(axis=0)
col_offsets = tl.arange(0, BLOCK_SIZE)
row_start = row_idx * n_cols
```

建立了二维矩阵到一维 Program Grid 的映射：

```text
Program 0 → Row 0
Program 1 → Row 1
...
Program 127 → Row 127
```

当前 `n_cols` 和 `BLOCK_SIZE` 都是 `tl.constexpr`，因此：

```text
n_cols = 1000
BLOCK_SIZE = 1024
```

被特化进当前 Kernel 变体。TTIR 的 Kernel 参数只剩输入、输出两个指针，没有运行时的 `n_cols` 和 `BLOCK_SIZE` 参数。

#### Mask 与 `-inf`

因为真实列数是 1000，而逻辑 Tile 是 1024，所以末尾 24 个位置必须被 Mask：

```python
mask = col_offsets < n_cols

row = tl.load(
    input_ptrs,
    mask=mask,
    other=-float("inf"),
)
```

TTIR 中的 `-inf` 表示为：

```mlir
arith.constant dense<0xFF800000> : tensor<1024xf32>
```

Load 仍然是一个 Tensor 级操作：

```mlir
%row = tt.load %input_ptrs, %mask, %negative_inf
```

越界位置参与 Max Reduction 时保持中性：

```text
max(valid_value, -inf) = valid_value
```

最大值计算完成后，越界位置的 Numerator 是：

```text
exp(-inf - row_max) = 0
```

因此，它们也不会增加后续的指数和。最终 Store 仍使用原始 Mask，避免写出真实行边界。

#### `tl.max` 变成带 Combine Region 的 `tt.reduce`

源码：

```python
row_max = tl.max(row, axis=0)
```

在 TTIR 中表示为：

```mlir
%row_max = "tt.reduce"(%row) <{axis = 0}> ({
^bb0(%lhs: f32, %rhs: f32):
  %result = arith.maxnumf %lhs, %rhs : f32
  tt.reduce.return %result : f32
}) : (tensor<1024xf32>) -> f32
```

这里要把 Reduction 分成两部分理解：

```text
tt.reduce
定义对哪个 Tensor、沿哪个维度规约

Combine Region
定义任意两个局部结果如何合并
```

`tt.reduce` 并没有规定必须使用 Shuffle 或 Shared Memory。它只表达：

> 将 `tensor<1024xf32>` 沿 axis 0 合并为一个 `f32`，合并规则是 `maxnumf`。

线程组织和通信方案要到后续 Lowering 才会展开。

#### Reduction 标量重新广播为 Tensor

`row_max` 的结果是一个标量：

```mlir
f32
```

而 `row` 仍然是：

```mlir
tensor<1024xf32>
```

所以执行逐元素减法前需要：

```mlir
%row_max_tensor = tt.splat %row_max
  : f32 -> tensor<1024xf32>

%centered = arith.subf %row, %row_max_tensor
```

这表示所有 1024 个逻辑位置都要使用同一个行最大值。

#### `tl.sum` 使用相同的 Reduction 骨架

第二次 Reduction：

```python
denominator = tl.sum(numerator, axis=0)
```

同样变成 `tt.reduce`，区别只是 Combine Region：

```mlir
%denominator = "tt.reduce"(%numerator) <{axis = 0}> ({
^bb0(%lhs: f32, %rhs: f32):
  %result = arith.addf %lhs, %rhs : f32
  tt.reduce.return %result : f32
}) : (tensor<1024xf32>) -> f32
```

之后 `denominator` 再通过 `tt.splat` 广播为 1024 元素，参与逐元素除法。

#### 本阶段能确认什么

TTIR 可以直接确认：

- 一个 Program 逻辑上处理 1024 个位置；
- 真实有效列数为 1000，末尾 24 个位置使用 Mask；
- 越界 Load 的 `other` 值是 `-inf`；
- Max 和 Sum 是两个独立的 `tt.reduce`；
- Max 的 Combine Region 使用 `arith.maxnumf`；
- Sum 的 Combine Region 使用 `arith.addf`；
- 两个 Reduction 都从 Tensor 得到标量，再广播回 Tensor；
- 当前是经典数值稳定 Softmax，不是 Online Softmax。

TTIR 还不能确认：

- 1024 个元素分给多少个线程；
- 每个线程先规约多少个元素；
- Reduction 是否使用 Warp Shuffle；
- 跨 Warp 结果如何交换；
- 是否需要 Shared Memory 和 Barrier。

### TTIR -> TTGIR

#### 本次 Blocked Layout

实际 TTGIR 使用：

```mlir
#blocked = #ttg.blocked<{
  sizePerThread = [4],
  threadsPerWarp = [32],
  warpsPerCTA = [8],
  order = [0]
}>
```

因此，一个 CTA 中的线程数是：

```text
32 threads/warp × 8 warps/CTA
= 256 threads/CTA
```

一个 Layout 基本块覆盖：

```text
4 elements/thread
× 32 threads/warp
× 8 warps/CTA
= 1024 elements/CTA
```

它恰好覆盖当前整行逻辑 Tensor。

模块属性也直接记录：

```mlir
"ttg.num-warps" = 8
"ttg.threads-per-warp" = 32
```

#### 每个线程负责四个元素

从 Layout 可以先得到：

```text
一个线程持有四个逻辑元素
```

而更低层 LLVM IR 和 PTX 进一步确认具体映射是：

```text
Thread t
→ {4t, 4t+1, 4t+2, 4t+3}
```

例如：

```text
Thread 0   → {0, 1, 2, 3}
Thread 1   → {4, 5, 6, 7}
...
Thread 249 → {996, 997, 998, 999}
Thread 250 → {1000, 1001, 1002, 1003}，全部被 Mask
```

这种分配既让每个线程获得四个可先在寄存器中局部规约的值，也让相邻线程覆盖连续的 Global Memory 区间。

#### `tt.reduce` 仍然保持抽象

TTGIR 中的 Reduction 仍然写成：

```mlir
"tt.reduce"(%row) ...
"tt.reduce"(%numerator) ...
```

与 TTIR 相比，关键变化不是 Reduction 已经变成 Shuffle，而是输入 Tensor 获得了 `#blocked`：

```mlir
tensor<1024xf32, #blocked>
```

编译器现在同时知道：

```text
需要把 1024 个值规约为一个标量
+
这些值分布在 8 个 Warp、256 个线程中
+
每个线程持有 4 个值
```

这组信息为后续选择通信方案提供了依据。

#### Shared Memory 并非来自 `convert_layout`

当前 TTGIR 中没有：

```mlir
ttg.convert_layout
```

但 LLVM IR、PTX 和 SASS 中确实出现了 Shared Memory 与 Barrier。这说明：

> 编译器生成 Shared Memory 不只可能因为 Layout Conversion，也可能因为跨 Warp Reduction 本身需要 CTA 范围的通信。

当前共享内存的职责是：

```text
收集 8 个 Warp Partial
→ 让 Warp 0 合并
→ 广播最终 Reduction 结果
```

不是把一个 Tensor 从 Layout A 转换到 Layout B。

#### 本阶段能确认什么

TTGIR 可以直接确认：

- 当前 CTA 使用 8 个 Warp、256 个线程；
- 每个线程持有 4 个逻辑元素；
- 1024 个位置正好由一个 Blocked Layout 基本块覆盖；
- Max 和 Sum 都要对跨越 8 个 Warp 的 Tensor 做 Reduction；
- 当前数据流没有显式 Layout Conversion。

TTGIR 只能让我们推断“跨 Warp Reduction 必须解决通信问题”。到底使用何种指令，仍需从 LLVM IR、PTX 和 SASS 确认。

### TTGIR -> LLVM IR

#### Tensor Load 变成每线程四元素 Load

LLVM IR 读取：

```llvm
%row_idx = call @llvm.nvvm.read.ptx.sreg.ctaid.x()
%tid = call @llvm.nvvm.read.ptx.sreg.tid.x()
```

并计算：

```text
row_start   = blockIdx.x * 1000
thread_base = threadIdx.x * 4
```

最终每个线程的输入位置是：

```text
blockIdx.x * 1000
+ threadIdx.x * 4
+ {0, 1, 2, 3}
```

四个 FP32 元素通过一次返回四个 32 位值的内联汇编 Load 进入线程：

```llvm
ld.global.v4.b32
```

Load 执行前，四个默认值都被初始化为 `0xFF800000`，即 `-inf`。

Mask 只比较 Pair 基址：

```llvm
thread_base < 1000
```

在当前特化 Shape 中这是安全的，因为 `1000` 能被 4 整除：

```text
最后一个有效向量：{996, 997, 998, 999}
下一个向量：      {1000, 1001, 1002, 1003}
```

所以四个元素总是一起有效或一起越界。不能据此推断任意 `n_cols` 都会使用一个 Predicate 控制四个元素；编译器会根据具体 Shape 和可向量化条件决定实现。

#### Reduction 第一层：线程内局部规约

每个线程先对自己持有的四个元素执行 Max：

```llvm
local_max = max(x0, x1)
local_max = max(local_max, x2)
local_max = max(local_max, x3)
```

这一步不需要线程间通信，只使用当前线程的四个寄存器值。

Sum Reduction 也采用相同结构：

```llvm
local_sum = n0 + n1 + n2 + n3
```

因此，完整 Reduction 并不是一开始就让 1024 个元素跨线程交换，而是先把问题压缩：

```text
1024 个元素
→ 256 个线程局部结果
```

#### Reduction 第二层：Warp 内 Butterfly Shuffle

每个线程得到一个局部结果后，使用：

```llvm
@llvm.nvvm.shfl.sync.bfly.i32
```

在一个 Warp 的 32 个 Lane 之间交换值。

第一次 Warp Reduction 使用 Offset：

```text
16 → 8 → 4 → 2 → 1
```

每次 Shuffle 后都执行 Combine：

```text
Max Reduction → maxnum
Sum Reduction → fadd
```

经过五轮 Butterfly 后，每个 Warp 的 32 个线程局部结果被合并为一个 Warp Partial。

此时问题从：

```text
256 个线程局部结果
```

继续压缩为：

```text
8 个 Warp Partial
```

Warp Shuffle 只能直接在同一个 Warp 内通信，不能把 Warp 0 的寄存器直接交给 Warp 7，所以还需要第三层。

#### Reduction 第三层：Shared Memory 合并 8 个 Warp

LLVM IR 计算：

```text
lane_id = threadIdx.x & 31
warp_id = (threadIdx.x & 255) >> 5
```

每个 Warp 只有 Lane 0 执行：

```text
shared[warp_id] = warp_partial
```

8 个 Warp 因此在共享暂存区中留下：

```text
shared[0]
shared[1]
...
shared[7]
```

一共使用 8 个 FP32 Slot，对应 32 字节的地址范围。随后执行 CTA Barrier，确保所有 Warp Partial 已经写入。

接着由 Thread 0～7，也就是 Warp 0 的前 8 个 Lane，读取这 8 个值：

```text
Lane 0 ← shared[0]
Lane 1 ← shared[1]
...
Lane 7 ← shared[7]
```

因为只剩 8 个值，第二轮 Warp 内 Reduction 只需要：

```text
4 → 2 → 1
```

最终由 Thread 0 写入：

```text
shared[0] = final_reduction_result
```

再经过一次 CTA Barrier，所有线程才能安全读取 `shared[0]`。

#### 一次跨 Warp Reduction 的完整结构

Max 或 Sum 的单次完整 Reduction 可以概括为：

```text
每线程 4 个值
        ↓
线程内局部 Reduction
        ↓
每 Warp 32 个局部结果
        ↓ 5 次 SHFL Butterfly
8 个 Warp Partial
        ↓ Lane 0 写 Shared Memory
CTA Barrier
        ↓
Warp 0 前 8 个 Lane 读取
        ↓ 3 次 SHFL Butterfly
1 个 CTA 结果
        ↓ Thread 0 写 shared[0]
CTA Barrier
        ↓
所有线程读取最终标量
```

这就是本节第一次完整观察到的：

```text
Register
→ Warp Shuffle
→ Shared Memory
→ Barrier
→ Register
```

#### 为什么 Max 与 Sum 之间还有一次 Barrier

Max Reduction 完成后，所有线程从 `shared[0]` 读取行最大值，并开始计算：

```text
exp(x - row_max)
```

后续 Sum Reduction 会复用同一块 Shared Memory 写入新的 Warp Partial。如果某个 Warp 过早开始写，而另一个 Warp 还没有读取旧的 `row_max`，就会破坏正确性。

因此，在 Sum Reduction 第一次覆盖共享暂存区前，还有一轮 CTA Barrier：

```text
所有线程完成对 row_max 的读取和使用
        ↓
Barrier
        ↓
允许 Sum Reduction 重用 Shared Memory
```

当前 LLVM IR 中总共出现 5 次 CTA Barrier：

| Barrier | 作用 |
| ---: | --- |
| 1 | 等待 8 个 Max Warp Partial 写完 |
| 2 | 等待最终 Row Max 写入 `shared[0]` |
| 3 | 确保所有线程已读取 Row Max，再复用 Shared Memory |
| 4 | 等待 8 个 Sum Warp Partial 写完 |
| 5 | 等待最终 Denominator 写入 `shared[0]` |

#### `tl.exp` 降低为以 2 为底的指数

TTIR 中：

```mlir
math.exp
```

在 LLVM IR 中变成：

```text
x * log2(e)
→ llvm.nvvm.ex2.approx.f
```

这是利用恒等式：

```text
e^x = 2^(x * log2(e))
```

每个线程持有四个 Centered Value，所以会生成四组乘法和 `ex2`。

#### 分母与 Numerator 都留在线程内

Sum Reduction 得到最终 Denominator 后，所有线程从 `shared[0]` 读取它。此前计算出的四个 Numerator 没有写回 Global Memory，而是继续作为线程私有值保留：

```text
n0 / denominator
n1 / denominator
n2 / denominator
n3 / denominator
```

最后通过一次四元素向量 Store 写回。

因此，虽然算法包含 Max 和 Sum 两次逻辑 Reduction，但当前 Kernel 的 Global Memory 数据流仍然是：

```text
一次 Input Load
→ 全部 Softmax 计算
→ 一次 Output Store
```

#### 本阶段能确认什么

LLVM IR 可以直接确认：

- 一个 CTA 使用 256 个线程；
- 每个线程读取四个连续 FP32 元素；
- 每个线程先执行四元素局部 Max/Sum；
- Warp Reduction 使用 Butterfly Shuffle；
- 8 个 Warp Partial 通过 Shared Memory 合并；
- Max 和 Sum 各有两次跨 Warp Barrier；
- 两次 Reduction 之间还有一次用于安全复用 Shared Memory 的 Barrier；
- `tl.exp` 使用 `ex2.approx` 路径；
- Numerator 保留在线程私有值中，没有回写 Global Memory；
- 输出使用四元素向量 Store。

### LLVM IR -> PTX

#### Kernel 启动与向量访存

PTX 入口声明：

```ptx
.extern .shared .align 16 .b8 global_smem[];
.reqntid 256
```

`.reqntid 256` 对应：

```text
8 warps × 32 threads = 256 threads
```

输入与输出分别使用：

```ptx
ld.global.v4.b32
st.global.v4.b32
```

`v4.b32` 表示每个线程处理：

```text
4 × 32 bit = 128 bit
```

相邻线程的起始位置相差四个 FP32，因此一个 Warp 覆盖连续的 128 个 FP32 元素。这是适合合并访存的地址模式，但实际内存事务数和带宽利用率仍需 Profiler 验证。

#### Max Reduction 的 PTX 指令序列

每个线程先执行三条：

```ptx
max.f32
```

把四个元素规约为一个局部最大值。随后使用五条：

```ptx
shfl.sync.bfly.b32
```

Offset 依次为：

```text
16, 8, 4, 2, 1
```

每个 Warp 的结果由 Lane 0 通过：

```ptx
st.shared.b32
```

写入共享暂存区。`bar.sync 0` 保证 8 个 Warp 都已完成写入。

Warp 0 的前 8 个 Lane 再执行：

```ptx
ld.shared.b32
```

并通过 Offset：

```text
4, 2, 1
```

完成第二轮 Shuffle。Thread 0 写回最终 Row Max，第二个 `bar.sync 0` 之后所有线程读取 `global_smem[0]`。

#### Sum Reduction 复用相同结构

Sum Reduction 的通信骨架与 Max 相同：

```text
3 次线程内 add
5 次完整 Warp Shuffle
Shared Store
Barrier
Shared Load
3 次 Warp 0 Shuffle
Shared Store
Barrier
Shared Load
```

区别在于 Combine 指令从：

```ptx
max.f32
```

变为：

```ptx
add.f32
```

#### 静态指令数量

扫描当前 PTX Kernel 主体可以得到：

| 指令类别 | 静态出现次数 | 来源 |
| --- | ---: | --- |
| `shfl.sync.bfly` | 16 | Max 8 次 + Sum 8 次 |
| `st.shared` | 4 | 每次 Reduction 两次 |
| `ld.shared` | 4 | 每次 Reduction 两次 |
| `bar.sync` | 5 | Max 2 次 + 中间复用 1 次 + Sum 2 次 |
| `ex2.approx` | 4 | 每线程四个 Numerator |
| `div.full` | 4 | 每线程四个输出 |

这些数字是静态指令实例数量，不表示一个 CTA 执行期间只有 16 次 Lane 数据交换或 4 次 Shared Memory 访问。每条 SIMD/SIMT 指令会由满足条件的线程共同执行，Predication 也会改变动态参与范围。

#### PTX 仍然不是最终机器指令

PTX 可以直接确认：

- Shuffle 的 Butterfly Offset；
- Shared Memory Load/Store；
- 5 次 CTA Barrier；
- 四元素 Global Memory 向量访存；
- `ex2.approx.f32` 和 `div.full.f32`。

但 PTX 仍然使用虚拟寄存器，也没有决定最终机器指令是否进一步融合或展开。特别是：

```ptx
div.full.f32
```

在当前 SASS 中并不存在同名的一条 `FDIV`，说明仍需继续追踪 cubin。

### PTX -> cubin

#### 从虚拟 ISA 到目标架构二进制

[`softmax.cubin`](softmax_ir/IR/softmax.cubin) 是面向：

```text
sm_120a
```

生成的目标二进制容器。它保存可装载的机器代码及相关元数据，而不是适合人工逐行阅读的文本。

这一阶段的关键变化是：

```text
PTX 虚拟指令
        ↓
目标架构指令选择与寄存器分配
        ↓
cubin 机器码
```

具体生成了哪些机器指令，必须通过 SASS 反汇编确认。

### cubin -> SASS

#### Global Memory 访问变成 128 位机器指令

PTX 中：

```ptx
ld.global.v4.b32
```

在 SASS 中变成：

```sass
@!P1 LDG.E.128 R4, desc[UR6][R8.64]
```

四个 FP32 输入分别进入：

```text
R4, R5, R6, R7
```

Load 前，SASS 先用：

```sass
MOV R4, 0xff800000
MOV R5, 0xff800000
MOV R6, 0xff800000
MOV R7, 0xff800000
```

准备四个 `-inf` 默认值。越界线程的 `LDG.E.128` 被 Predicate 禁止后，这四个值仍保持为 `-inf`。

最终 Store 对应：

```sass
STG.E.128 desc[UR6][R2.64], R12
```

一次写回四个 FP32 结果。

#### Local Max 与 Warp Max

线程内四元素 Max 使用：

```sass
FMNMX
FMNMX
FMNMX
```

之后 Warp 内规约使用：

```sass
SHFL.BFLY
```

Butterfly Offset 在指令中仍然清晰可见：

```text
0x10, 0x8, 0x4, 0x2, 0x1
```

这与 PTX 中的 `16, 8, 4, 2, 1` 完全对应。

#### 跨 Warp 通信成为 `STS / LDS / BAR`

PTX 的 Shared Memory 与 Barrier 指令，在当前 `sm_120a` SASS 中变成：

| PTX | SASS |
| --- | --- |
| `st.shared.b32` | `STS` |
| `ld.shared.b32` | `LDS` |
| `bar.sync 0` | `BAR.SYNC.DEFER_BLOCKING 0x0` |

Max Reduction 中可以直接观察：

```sass
STS
BAR.SYNC.DEFER_BLOCKING
LDS
SHFL.BFLY 0x4
SHFL.BFLY 0x2
SHFL.BFLY 0x1
STS
BAR.SYNC.DEFER_BLOCKING
LDS
```

Sum Reduction 再出现一次相同的通信骨架。

SASS 主体中静态出现：

```text
SHFL.BFLY                 16 次
STS                        4 次
LDS                        4 次
BAR.SYNC.DEFER_BLOCKING    5 次
```

这与 PTX 的静态指令数量一致，直接确认线程间通信没有在最终机器码阶段被消除。

#### 第三个 Barrier 保护共享暂存区复用

在 Max 的最终结果被所有线程读取后，SASS 先执行四组中心化和乘 `log2(e)`，随后出现：

```sass
BAR.SYNC.DEFER_BLOCKING 0x0
```

之后才进入 Sum Reduction 的 Shared Store。

它对应 LLVM IR/PTX 中 Max 与 Sum 之间的第三个 Barrier，保证旧的 Row Max 不会在其他 Warp 尚未读取完成时被新的 Sum Warp Partial 覆盖。

#### `tl.exp` 最终使用 `MUFU.EX2`

PTX 的：

```ptx
ex2.approx.f32
```

在 SASS 中变成：

```sass
MUFU.EX2
```

当前每个线程有四个 Numerator，因此静态出现四条 `MUFU.EX2`。

周围还存在 Predicate、条件缩放和额外乘法，用于处理目标指令输入范围和特殊数值情况。不能把一条高层 `tl.exp` 简化成“一定只对应一条 MUFU 指令”；当前真实机器序列比 PTX 的表面形式更复杂。

#### `div.full.f32` 最终没有变成 `FDIV`

PTX 中四个输出使用：

```ptx
div.full.f32
```

但当前 SASS 中没有 `FDIV`。后端采用：

```text
读取 Denominator
→ 根据数值范围进行条件缩放
→ MUFU.RCP 计算倒数
→ 四条 FMUL 乘 Numerator
```

关键指令是：

```sass
MUFU.RCP
FMUL
FMUL
FMUL
FMUL
```

因此：

```text
numerator / denominator
```

在机器层被重写为：

```text
numerator × reciprocal(denominator)
```

并配合额外缩放处理数值范围。这是 PTX 与 SASS 不一一对应的又一个直接例子。

#### Mask 最终变成提前退出

SASS 使用：

```sass
ISETP.GE.U32.AND P1, ..., thread_base, 0x3e8, ...
```

其中：

```text
0x3e8 = 1000
```

`P1` 表示当前四元素向量已经越界：

```text
thread_base >= 1000
```

输入 Load 使用：

```sass
@!P1 LDG.E.128
```

输出 Store 前使用：

```sass
@P1 EXIT
STG.E.128
```

所以越界线程跳过 Input Load，并在 Output Store 前提前结束；有效线程则完成 128 位写回。

即使越界线程不访问 Global Memory，它们仍然不能在 Reduction 之前提前退出，因为 CTA Barrier 要求参与该同步路径的线程保持一致。越界线程必须携带 `-inf` 和后续的零 Numerator 参与 Reduction，直到最后不再需要 CTA 级同步时才安全退出。

#### SASS 直接证明了什么

当前 SASS 可以直接确认：

- 每个线程使用一次 128 位输入 Load 和一次 128 位输出 Store；
- Max 使用 `FMNMX`；
- 两次 Reduction 总共保留 16 条 `SHFL.BFLY`；
- 跨 Warp 通信使用 `STS` 和 `LDS`；
- CTA 同步使用 5 条 `BAR.SYNC.DEFER_BLOCKING`；
- `tl.exp` 最终使用 `MUFU.EX2`；
- PTX 除法最终使用 `MUFU.RCP` 与乘法序列；
- Mask 在最后落实为 Predicated Load 和越界线程提前退出。

SASS 仍然不能单独证明：

- 5 条 Barrier 实际造成多少等待周期；
- Shared Memory 是否成为吞吐瓶颈；
- Shuffle 与依赖链造成多少 Stall；
- Numerator 的长生命周期带来多大寄存器压力；
- `num_warps=8` 是否是当前 Shape 的最优选择；
- Kernel 最终受计算吞吐、特殊函数单元还是内存带宽限制。

这些结论仍需 Nsight Compute 和受控 Benchmark 验证。

### Softmax Lowering 总结

当前 Softmax 的完整 Lowering 证据链是：

```text
Python DSL
一行经典稳定 Softmax
Max → Exp → Sum → Divide
        ↓
TTIR
两个 tt.reduce
maxnumf Combine / addf Combine
        ↓
TTGIR
sizePerThread = 4
threadsPerWarp = 32
warpsPerCTA = 8
        ↓
LLVM IR
每线程四元素局部 Reduction
→ Warp Shuffle
→ Shared Memory
→ CTA Barrier
        ↓
PTX
16 × shfl.sync.bfly
4 × st.shared
4 × ld.shared
5 × bar.sync
        ↓
cubin
sm_120a 目标二进制
        ↓
SASS
SHFL.BFLY
STS / LDS
BAR.SYNC.DEFER_BLOCKING
MUFU.EX2 / MUFU.RCP
```

与 Vector Add 相比，Softmax 新增的核心不是更多算术指令，而是跨线程的数据依赖：

```text
Vector Add
每个线程独立完成自己的元素
线程之间不需要通信

Softmax
每个输出依赖整行 Max 和 Sum
线程必须共同生成两个全局行标量
```

一次跨 8 个 Warp 的 Reduction 被编译为三级结构：

```text
线程内
4 个值 → 1 个局部值

Warp 内
32 个局部值 → 1 个 Warp Partial

CTA 内
8 个 Warp Partial → 1 个行级结果
```

本次已经由 IR、PTX 和 SASS 直接证明：

- 一个 Program/CTA 处理一整行；
- 一个 CTA 使用 8 个 Warp、256 个线程；
- 每个线程负责四个连续元素；
- 输入与输出使用 128 位向量访存；
- Max 和 Sum 都采用“局部规约 + Warp Shuffle + Shared Memory 合并”；
- 两次 Reduction 总共使用 16 条静态 Shuffle 指令；
- Shared Memory 暂存 8 个 Warp Partial；
- 5 次 Barrier 分别保护 Partial 合并、结果广播和暂存区复用；
- Shared Memory 来自 Reduction Lowering，而不是 `convert_layout`；
- Numerator 不回写 Global Memory，而是保留到分母计算完成；
- `tl.exp` 最终使用 `MUFU.EX2`；
- PTX `div.full` 最终成为 `MUFU.RCP`、条件缩放与乘法。

当前还不能只凭指令确定：

- 线程间通信占总时间的比例；
- Barrier 是否是主要瓶颈；
- 实际 Shared Memory 吞吐与冲突情况；
- 128 位访存最终对应多少内存事务；
- 当前寄存器压力和 Occupancy；
- 8 Warp 是否优于 4 Warp 或其他配置。

因此，Softmax 建立的最终分析方法是：

```text
TTIR
确认 Reduction 数学语义
        ↓
TTGIR
确认 Reduction 跨越的线程与 Warp 范围
        ↓
LLVM IR / PTX
确认通信算法与同步结构
        ↓
SASS
确认真实 Shuffle、Shared Memory、Barrier 和特殊函数指令
        ↓
Nsight Compute
验证这些机制的实际硬件代价
```

## Matmul

本节以 FP16 输入、FP32 累加、FP16 输出的 Tiled Matmul 为观察样本。实验配置为：

```text
M = 128
N = 128
K = 128

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32

Grid = (2, 2)
```

每个 Triton Program 负责计算一个 `64 × 64` 的 C Tile，并沿 K 维以 32 为步长循环累加。

重新生成编译产物前，测试已经使用：

```python
torch.testing.assert_close(
    C,
    ref,
    atol=2e-2,
    rtol=2e-2,
)
```

只有 correctness 断言通过后，脚本才会执行 IR 导出。因此，本节分析的编译产物来自已经通过数值验证的 Kernel。

相关文件：

- Kernel 与导出脚本：[`matmul_ir/dump_matmul.py`](matmul_ir/dump_matmul.py)
- 较早期编译 IR：[`matmul_ir/IR/matmul.source`](matmul_ir/IR/matmul.source)
- TTIR：[`matmul_ir/IR/matmul.ttir`](matmul_ir/IR/matmul.ttir)
- TTGIR：[`matmul_ir/IR/matmul.ttgir`](matmul_ir/IR/matmul.ttgir)
- LLVM IR：[`matmul_ir/IR/matmul.llir`](matmul_ir/IR/matmul.llir)
- PTX：[`matmul_ir/IR/matmul.ptx`](matmul_ir/IR/matmul.ptx)
- cubin：[`matmul_ir/IR/matmul.cubin`](matmul_ir/IR/matmul.cubin)
- SASS：[`matmul_ir/IR/matmul.sass`](matmul_ir/IR/matmul.sass)

Matmul 的 Lowering 不只回答 `tl.dot` 变成了哪条 Tensor Core 指令，还包含三条相互连接的数据通路：

```text
Global Memory
→ Shared Memory 双缓冲

Shared Memory
→ Tensor Core Fragment
→ MMA

MMA Accumulator
→ Store-friendly Layout
→ Global Memory
```

### Python DSL -> TTIR

在 TTIR 这一层，编译器**完全不关心 GPU 硬件细节**（不知道 SM、Warp、Register，也不管 Shared Memory）。它只做高层逻辑的标准化。

#### 一个 Program 负责一个 C Tile

Kernel 使用二维 Program Grid：

```python
pid_m = tl.program_id(axis=0)
pid_n = tl.program_id(axis=1)
```

当前 Grid 是：

```text
(ceil_div(128, 64), ceil_div(128, 64))
= (2, 2)
```

因此四个 Program 分别覆盖：

```text
Program (0, 0) → C[0:64,   0:64]
Program (0, 1) → C[0:64,  64:128]
Program (1, 0) → C[64:128, 0:64]
Program (1, 1) → C[64:128,64:128]
```

TTIR 中对应：

```mlir
%pid_m = tt.get_program_id x
%pid_n = tt.get_program_id y
```

并分别构造 64 个 M 坐标和 64 个 N 坐标。

#### `matmul.source` 仍然不是 Python 源码

当前脚本遍历 `compiled.asm` 并把每个键直接写入文件：

```python
for key, value in asm.items():
    file_path = ... f"matmul.{key}"
```

所以 [`matmul.source`](matmul_ir/IR/matmul.source) 来自：

```text
compiled.asm["source"]
```

它是一份较早期的 Triton 编译 IR，而不是 `dump_matmul.py` 的 Python 源码副本。与优化后的 TTIR 相比，其中还包含更多整数扩展和范围检查。

#### 构造 A Tile 与 B Tile

一个 K Tile 的逻辑形状是：

```text
A Tile：64 × 32
B Tile：32 × 64
```

源码通过：

```python
offs_m[:, None]
offs_k[None, :]
```

形成 A 的二维坐标，通过：

```python
offs_k[:, None]
offs_n[None, :]
```

形成 B 的二维坐标。

TTIR 使用：

```text
tt.expand_dims
→ tt.broadcast
→ arith.muli / arith.addi
→ tt.addptr
```

把一维坐标扩展为二维 Pointer Tensor：

```mlir
tensor<64x32x!tt.ptr<f16>>
tensor<32x64x!tt.ptr<f16>>
```

这时只表达逻辑上的 Tile 地址，还没有决定哪些线程执行具体 Load。

#### K 维循环变成 `scf.for`

源码：

```python
for k in range(0, K, BLOCK_K):
```

在 TTIR 中变成：

```mlir
scf.for %k = 0 to %K step 32
```

当前 `K=128`，所以运行时处理四个 K Tile：

```text
K Tile 0：  0:32
K Tile 1： 32:64
K Tile 2： 64:96
K Tile 3： 96:128
```

`K` 是普通运行时参数，不是 `tl.constexpr`；而 `BLOCK_K=32` 是编译期常量。因此循环步长已经固定，但循环上界仍来自 Kernel 参数。

#### Mask 与零填充

A 的 Mask 同时检查：

```text
M Coordinate < M
K Coordinate < K
```

B 的 Mask 同时检查：

```text
K Coordinate < K
N Coordinate < N
```

越界位置使用：

```text
other = 0.0
```

这是矩阵乘法 Reduction 的中性值：

```text
x × 0 = 0
```

所以非对齐 Shape 仍可使用完整的 `64 × 32` 与 `32 × 64` 逻辑 Tile。

当前测试 Shape 恰好能被三个 Block Size 整除，但 Mask 仍保留在通用 Kernel 中。不能因为本次所有 Predicate 都为真，就认为编译器已经从通用代码中删除边界语义。

#### `tl.dot` 的 TTIR 语义

源码：

```python
acc += tl.dot(a, b)
```

在 TTIR 中表示为：

```mlir
%next_acc = tt.dot %a, %b, %acc
  : tensor<64x32xf16>
  * tensor<32x64xf16>
  -> tensor<64x64xf32>
```

这直接确认：

```text
A 输入：FP16
B 输入：FP16
Accumulator：FP32
```

`tt.dot` 表达的是一个矩阵乘加：

```text
Accumulator = A Tile × B Tile + Accumulator
```

但 TTIR 还没有指定：

- 是否使用 Tensor Core；
- 每个 Warp 计算 C Tile 的哪一部分；
- A/B 是否经过 Shared Memory；
- 数据如何进入 MMA Fragment；
- 最终选择哪一种 MMA 指令形状。

TTIR 中显示的：

```text
inputPrecision = tf32
```

不能解释为当前输入已经变成 TF32。实际输入类型仍是 `f16`，后续 PTX 也明确生成 `f16 × f16 → f32` 的 MMA 指令。判断真实计算类型应结合操作数类型和最终指令，而不能只孤立阅读这个属性。

#### FP32 Accumulator 转 FP16

K 循环完成后，TTIR 使用：

```mlir
arith.truncf
tensor<64x64xf32>
→ tensor<64x64xf16>
```

然后构造 C 的二维 Pointer Tensor 与边界 Mask，执行 Tensor 级 `tt.store`。

这说明当前精度策略是：

```text
FP16 Input
→ FP32 Accumulation
→ FP16 Output
```

#### 本阶段能确认什么

TTIR 可以直接确认：

- 一个 Program 负责一个 `64 × 64` C Tile；
- K 维按 32 分块；
- A/B Tile 分别是 `64 × 32` 和 `32 × 64`；
- A/B 使用二维 Mask，越界值填零；
- `tt.dot` 执行 FP16 输入、FP32 累加；
- K 循环使用 `scf.for`；
- 最终 Accumulator 从 FP32 截断为 FP16。

TTIR 无法确认（留给下一阶段）：

1. **没有线程分工**：这 64x64 个数到底由几个 Warp、几个 Thread 算？
2. **没有存储层级**：数据是直接从 Global 读到寄存器，还是经过 Shared Memory？
3. **没有硬件指令**：`tt.dot` 到底是用 CUDA 标量 ALU 算，还是用 Tensor Core 算？

### TTIR -> TTGIR

在这一阶段，Triton 编译器彻底脱离了抽象的张量算术，开始根据 GPU 硬件（如 NVIDIA Ampere/Hopper 架构）的物理特性，**将数据分配给具体的线程，并搭建内存流水线**。

#### 当前 CTA 使用四个 Warp

源码没有显式传入 `num_warps`，但实际 TTGIR 模块记录：

```mlir
"ttg.num-warps" = 4
"ttg.threads-per-warp" = 32
```

所以当前编译变体使用：

```text
4 warps/CTA × 32 threads/warp
= 128 threads/CTA
```

这是本次真实编译结果，不代表所有 `64 × 64 × 32` Matmul 在所有 Triton 版本中都会自动选择四个 Warp。

#### Matmul 同时使用多类 Layout

TTGIR 中首次同时出现：

```text
#blocked / #blocked1
#shared / #shared1
#ttg.dot_op
#mma
```

它们分别服务于不同阶段：

| Layout 类型    | 代表属性                                      | 物理职责与设计意图                                           |
| -------------- | --------------------------------------------- | ------------------------------------------------------------ |
| **`#blocked`** | `sizePerThread=[1,8]`, `threadsPerWarp=[8,4]` | **全局显存（Global Memory）高效访存 Layout**。 确保每个线程在连续地址上一次抓取 8 个 FP16（16 Bytes / 128 bit），精准触发 GPU 的 合并访存。 |
| **`#shared`**  | `swizzled_shared`, `vec=8, maxPhase=4`        | **共享内存（Shared Memory）地址重排 Layout**。 通过按位异或（XOR）改变物理存储地址，确保 32 个线程并发读取矩阵 Fragment 时不会碰撞到同一个 SMEM Bank（彻底消除 Bank Conflict）。 |
| **`#dot_op`**  | `opIdx=0/1`, `parent=#mma`, `kWidth=2`        | **Tensor Core 输入操作数 Layout**。 表达 Shared Memory 数据被 `ldmatrix` 加载进线程私有寄存器（Fragment）后的特定排列格式。 |
| **`#mma`**     | `versionMajor=2`, `warpsPerCTA=[2,2]`         | **Tensor Core 累加器（Accumulator）输出 Layout**。 定义 $64 \times 64$ 的 FP32 中间结果如何打碎分布在 4 个 Warp（$2 \times 2$ 排列）、128 个线程的私有寄存器中。 |

Matmul 的核心难点由此显现：

> Global Memory、Shared Memory、Tensor Core Operand 和最终 Store 对数据 Layout 的要求并不相同。

#### A 与 B 使用不同 Blocked Layout

A 的 Global-load Layout 是：

```mlir
#blocked1 = #ttg.blocked<{
  sizePerThread = [1, 8],
  threadsPerWarp = [8, 4],
  warpsPerCTA = [4, 1],
  order = [1, 0]
}>
```

B 使用：

```mlir
#blocked = #ttg.blocked<{
  sizePerThread = [1, 8],
  threadsPerWarp = [4, 8],
  warpsPerCTA = [4, 1],
  order = [1, 0]
}>
```

两者都让每个线程一次拥有连续的 8 个 FP16 值，也就是：

```text
8 × 2 Bytes = 16 Bytes
```

这与后续 128-bit Async Copy 相呼应。

A 与 B 的线程布局不同，是因为它们的矩阵形状和连续访问维度不同。TTGIR 可以确认 Layout 差异和连续宽度，但实际内存事务与合并效率仍需 PTX/SASS 和 Profiler 共同验证。

#### MMA Layout 与 Warp 排列

Accumulator 使用：

```mlir
#mma = #ttg.nvidia_mma<{
  versionMajor = 2,
  versionMinor = 0,
  warpsPerCTA = [2, 2],
  instrShape = [16, 8]
}>
```

`warpsPerCTA=[2,2]` 表示四个 Warp 沿 C Tile 的 M/N 方向组成一个 `2 × 2` Warp Grid。

`instrShape=[16,8]` 描述 MMA 指令在 M/N 方向的基本形状。K 方向的实际指令尺寸需要继续结合 `dot_op` 和 PTX：

```ptx
mma.sync.aligned.m16n8k16
```

MMA Layout 不是普通二维 Blocked Layout。一个 Warp 的 Accumulator Fragment 会分散到 32 个 Lane 的多组寄存器中，不能把它简单理解成“每个 Warp 持有一个连续数组”。

#### Swizzled Shared Layout

A 的 Shared Layout：

```mlir
#shared = #ttg.swizzled_shared<{
  vec = 8,
  perPhase = 2,
  maxPhase = 4,
  order = [1, 0]
}>
```

B 的 Shared Layout：

```mlir
#shared1 = #ttg.swizzled_shared<{
  vec = 8,
  perPhase = 1,
  maxPhase = 8,
  order = [1, 0]
}>
```

这里的 Swizzle 改变逻辑坐标到 Shared Memory 地址的映射，使同一个 Tile 能以适合 Matrix Fragment Load 的方式排列。

从 IR 可以直接确认编译器选择了 Swizzled Layout；是否完全避免 Bank Conflict、实际冲突程度如何，仍需 Nsight Compute 验证。

#### A/B Shared Memory 双缓冲

TTGIR 分配：

```mlir
!ttg.memdesc<2x64x32xf16, #shared, ...>
!ttg.memdesc<2x32x64xf16, #shared1, ...>
```

第一个维度为 2，表示两个 Stage。

A 的逻辑空间：

```text
2 stages × 64 × 32 × 2 Bytes
= 8192 Bytes
```

B 的逻辑空间：

```text
2 stages × 32 × 64 × 2 Bytes
= 8192 Bytes
```

合计：

```text
16 KiB
```

两组 Stage 允许：

```text
一个 Stage 供当前 MMA 使用
另一个 Stage 预取未来的 K Tile
```

这就是双缓冲（Double Buffering）的基本结构。

#### Global-to-Shared 异步复制

TTGIR 使用：

```text
ttg.async_copy_global_to_local
ttg.async_commit_group
ttg.async_wait
```

数据路径是：

```text
Blocked Global Layout
→ Async Copy
→ Swizzled Shared Layout
```

每次 Copy 记录：

```mlir
contiguity = 8
```

即每个线程负责连续 8 个 FP16，正好是 16 Bytes。

当前编译结果最终使用传统 `cp.async` 路径，而不是 Tensor Memory Accelerator（TMA）。这一点要到 PTX/SASS 才能最终确认。

#### Software Pipeline 的 Prologue、Loop 与 Epilogue

TTGIR 在主循环前先发起前两个 Stage 的 A/B Async Copy：

```text
Stage 0 ← K Tile 0
Stage 1 ← K Tile 1
```

循环中：

```text
等待当前 Stage
→ 从 Shared Memory 读取当前 A/B Fragment
→ 执行 tt.dot
→ 向另一 Stage 预取未来 K Tile
```

循环结束后：

```mlir
ttg.async_wait {num = 0}
```

等待剩余异步操作完成。

这形成：

```text
Prologue：填充 Pipeline
Steady State：计算与预取交错
Epilogue：排空 Pipeline
```

IR 证明编译器建立了异步 Pipeline；它是否在硬件上充分隐藏 Global Memory 延迟，仍需 Profiler 观察。

#### Shared Memory 到 MMA Operand Fragment

等待当前 Stage 就绪后，TTGIR 使用：

```mlir
ttg.local_load
```

将 A/B Shared Tile 转换为：

```mlir
#ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>
```

其中：

```text
opIdx = 0 → A Operand
opIdx = 1 → B Operand
```

这一步表达：

```text
Swizzled Shared Layout
→ Tensor Core Operand Fragment Layout
```

更低层将通过 `ldmatrix`/`LDSM` 实现。

#### `tt.dot` 已经绑定 MMA Layout

循环中的 `tt.dot` 现在具有：

```text
A：#ttg.dot_op
B：#ttg.dot_op
Accumulator：#mma
```

因此，到 TTGIR 层已经可以确认编译器选择 NVIDIA MMA 路径，但最终使用 `mma.sync` 还是其他目标架构机制，仍需检查生成指令。

#### 输出端出现真实 `convert_layout`

FP32 Accumulator 先截断为 FP16，但仍保持 `#mma`：

```mlir
tensor<64x64xf16, #mma>
```

最终 Global Store 需要 `#blocked`：

```mlir
tensor<64x64xf16, #blocked>
```

所以 TTGIR 插入：

```mlir
%blocked_c = ttg.convert_layout %mma_c
  : tensor<64x64xf16, #mma>
  -> tensor<64x64xf16, #blocked>
```

这不是只改变类型标签。后续 PTX/SASS 会直接证明它通过：

```text
MMA Registers
→ Shared Memory
→ Blocked Registers
```

完成真实数据重排。

#### 本阶段能确认什么

- **确立了并发与计算模式**：当前 CTA 使用 4 个 Warp（128 线程），按 $[2, 2]$ 网格划分 C Tile；确定使用 NVIDIA 原生 MMA 路径。
- **构建了高效内存流水线**：开辟 $16\text{ KiB}$ 共享内存做双缓冲，使用 `async_copy` 建立流水线掩盖内存延迟。
- **厘清了 Layout 演进路径**：

$$
\text{Global (\#blocked)} \xrightarrow{\text{Async Copy}} \text{Shared (\#shared)} \xrightarrow{\text{Local Load}} \text{Fragment (\#dot\_op)}\xrightarrow{\text{tt.dot}} \text{Accumulator (\#mma)} \xrightarrow{\text{convert\_layout}} \text{Global Store (\#blocked)}
$$



- A/B Global Load 使用不同的 Blocked Layout；
- A/B Shared Tile 使用不同的 Swizzled Layout；
- A/B 各拥有两个 Shared Memory Stage；
- Global-to-Shared 使用异步 Copy；
- Shared-to-Register 使用 `dot_op` Fragment；
- Accumulator 使用 NVIDIA MMA Layout；
- 四个 Warp 按 `[2,2]` 分布；
- 输出端存在 `#mma → #blocked` 的 `convert_layout`。

### TTGIR -> LLVM IR

#### 高层 Tensor 开始变成每线程流水线

LLVM IR 已不再保留完整的 Python Tensor 表达，而是展开为：

```text
读取 ctaid.x / ctaid.y / tid.x
→ 计算每线程 Global 地址
→ 计算 Swizzled Shared 地址
→ 发起 Async Copy
→ 等待与同步
→ 加载 MMA Fragment
→ 执行矩阵乘加
```

这一层的地址计算非常多。阅读时不需要逐行追踪每一个 `and/xor/shift`，而应识别三类结果：

- Global Memory 地址；
- Swizzled Shared Memory 地址；
- MMA Fragment 的 Lane/Register 选择。

#### Async Copy Token 变成依赖控制

TTGIR 的 Async Token 不保存 A/B 数据，它只描述异步操作之间的依赖：

```text
哪些 Copy 已经提交
哪些数据必须等待
允许保留多少个未完成 Group
```

LLVM/NVVM 层开始把这些语义变成 `cp.async`、Commit、Wait 和 Barrier 相关的目标操作。

#### Shared Memory 与 MMA Fragment

`ttg.local_load` 被降低为 Matrix Load 路径。A 与 B 的 Shared Layout 不同，因此会分别生成普通和转置形式的 Matrix Fragment Load。

`tt.dot` 被展开为多条 Warp 级 MMA 操作。每条 MMA 消费 A/B Fragment 中的一部分，并更新一组 FP32 Accumulator 寄存器。

#### Accumulator 的长生命周期

FP32 Accumulator 在整个 K 循环中作为 Loop-carried Value 存活：

```text
K Tile 0 更新
→ K Tile 1 更新
→ K Tile 2 更新
→ K Tile 3 更新
→ 最终截断
```

这会带来明显的寄存器需求，但仅凭 LLVM IR 不能确定最终物理寄存器数，也不能直接断言 Occupancy 已经受限。

#### 输出 Layout Conversion 开始物化

`#mma → #blocked` 的转换需要重新分配每个线程持有的输出元素。LLVM/PTX 最终选择 Shared Memory 作为中间交换区：

```text
MMA Fragment Registers
→ Matrix Store 到 Shared Memory
→ CTA Synchronization
→ Vector Load 回线程寄存器
→ Global Store
```

这正是 `ttg.convert_layout` 产生真实线程间数据重排的实例。

### LLVM IR -> PTX

#### 线程配置

PTX 入口包含：

```ptx
.reqntid 128
```

对应当前 TTGIR 的：

```text
4 warps × 32 threads = 128 threads
```

#### Async Copy 使用 `cp.async`

Global-to-Shared 数据搬运最终使用：

```ptx
cp.async.cg.shared.global
```

并配合：

```ptx
cp.async.commit_group
cp.async.wait_group
bar.sync
```

当前 PTX 中静态出现：

| 指令 | 静态数量 |
| --- | ---: |
| `cp.async.cg.shared.global` | 12 |
| `cp.async.commit_group` | 6 |
| `cp.async.wait_group` | 2 |

每条 `cp.async` 复制 16 Bytes，与 TTGIR 的 `contiguity=8` 和 FP16 元素大小一致。

这些是静态指令实例。部分指令位于 K 循环中，当前 `K=128` 时会动态执行多轮，不能把静态计数当作整个 CTA 的总执行次数。

当前指令是 `cp.async.cg.shared.global`，没有出现 TMA 指令。因此，本样本直接证明使用的是经典 Async Copy Pipeline，而非 TMA Pipeline。

#### Shared-to-Register 使用 `ldmatrix`

A Fragment 使用：

```ptx
ldmatrix.sync.aligned.m8n8.x4.shared.b16
```

B Fragment 使用：

```ptx
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16
```

`trans` 形式对应 B Operand 所需的数据排列。

当前 PTX 静态包含 8 条 `ldmatrix`。这些指令把 Shared Memory 中的 FP16 Tile 协同加载到 Warp 的 MMA Operand Registers 中。

#### `tt.dot` 变成 Tensor Core MMA

最终 PTX 指令是：

```ptx
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
```

字段直接说明：

```text
Instruction Tile：M16 × N8 × K16
A Operand：FP16
B Operand：FP16
Accumulator：FP32
Result：FP32
```

当前 PTX 静态包含 16 条 `mma.sync`。它们位于 K Pipeline 的计算循环中，负责更新不同的 C Fragment。

`mma.sync` 的出现可以直接证明 Tensor Core 路径已经被选择；实际 Tensor Core 利用率仍需 Profiler 验证。

#### Pipeline 中共有七条 CTA Barrier

当前 PTX 静态包含 7 条：

```ptx
bar.sync 0
```

它们服务于不同的数据依赖：

- 确保 Async Copy 的 Shared Tile 可以安全消费；
- 保护双缓冲 Stage 的切换和复用；
- 保护输出 `convert_layout` 的 Shared Memory 写后读。

不能只根据“共有 7 条 Barrier”就断言 Barrier 是性能瓶颈。不同 Barrier 的动态执行次数和等待时间不同，需要结合控制流与 Profiler 数据分析。

#### 输出转换使用 `stmatrix`

MMA Accumulator 截断成 FP16 后，PTX 使用：

```ptx
stmatrix.sync.aligned.m8n8.x4.shared.b16
```

把分布在 MMA Fragment Registers 中的数据按 Matrix Fragment 形式写入 Shared Memory。

当前静态包含 4 条 `stmatrix`，分两组完成。每组写入后，线程通过：

```ptx
ld.shared.v4.b32
```

以适合 Global Store 的 Blocked Layout 重新读取。

这一段的数据流是：

```text
#mma Registers
→ stmatrix
→ Shared Memory
→ Barrier
→ ld.shared.v4.b32
→ #blocked Registers
```

它直接证明当前 `ttg.convert_layout` 不是零成本转换。

#### 每线程写回 32 个 FP16

当前 PTX 最终执行四条：

```ptx
st.global.v4.b32
```

一次 `v4.b32` 是：

```text
4 × 32 bit = 16 Bytes = 8 个 FP16
```

四次 Store 因此对应：

```text
4 × 8 FP16 = 32 FP16/thread
```

整个 CTA：

```text
128 threads × 32 FP16
= 4096 FP16
= 64 × 64 C Tile
```

这与一个 Program 的逻辑输出 Tile 完全对应。

#### PTX 静态结构汇总

| 机制 | 关键 PTX | 静态数量 |
| --- | --- | ---: |
| Global-to-Shared | `cp.async.cg.shared.global` | 12 |
| Async Commit | `cp.async.commit_group` | 6 |
| Async Wait | `cp.async.wait_group` | 2 |
| Shared-to-Fragment | `ldmatrix` | 8 |
| Tensor Core | `mma.sync.m16n8k16` | 16 |
| CTA 同步 | `bar.sync` | 7 |
| Fragment-to-Shared | `stmatrix` | 4 |
| Shared-to-Blocked | `ld.shared.v4.b32` | 4 |
| Global Store | `st.global.v4.b32` | 4 |

### PTX -> cubin

#### 目标架构代码生成

当前 cubin 面向：

```text
sm_120a
```

这一阶段继续完成：

```text
PTX 虚拟寄存器
→ 物理寄存器分配

PTX Async/MMA 指令
→ 目标架构机器指令

PTX 依赖语义
→ Scoreboard、Wait 与调度控制
```

具体结果必须通过 SASS 反汇编确认。

### cubin -> SASS

#### `cp.async` 最终成为 `LDGSTS`

PTX：

```ptx
cp.async.cg.shared.global
```

SASS：

```sass
LDGSTS.E.BYPASS.128
```

`LDGSTS` 表达从 Global Memory Load 并写入 Shared Memory 的异步数据通路。当前 SASS 静态包含 12 条，与 PTX 的 12 条 `cp.async` 对应。

Mask 最终作为 Predicate 参数控制无效 Tile 位置，越界值由异步复制路径按零填充语义处理。

#### Async Wait 最终成为 `DEPBAR`

两个 PTX：

```ptx
cp.async.wait_group
```

在 SASS 中对应两个：

```sass
DEPBAR.LE
```

它们等待异步 Copy 的依赖 Scoreboard 达到指定状态。

`cp.async.commit_group` 不一定生成一条同名机器指令。部分 Group/依赖信息会编码进机器级 Scoreboard 和控制字段，因此不能要求每一条 PTX Commit 都找到同名 SASS。

#### `ldmatrix` 最终成为 `LDSM`

PTX 的普通与转置 Matrix Load：

```text
ldmatrix ... m8n8.x4
ldmatrix ... m8n8.x4.trans
```

在 SASS 中分别表现为：

```sass
LDSM.16.M88.4
LDSM.16.MT88.4
```

当前静态包含 8 条 `LDSM`。

其中：

```text
M88  → 普通 Matrix Load
MT88 → Transposed Matrix Load
```

这是真实 Shared Memory Tile 被转换为 Tensor Core Operand Fragment 的机器指令证据。

#### `mma.sync` 最终成为 `HMMA`

PTX：

```ptx
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
```

SASS：

```sass
HMMA.16816.F32
```

当前静态包含 16 条 `HMMA.16816.F32`。

因此，Tensor Core 路径的证据链完整成立：

```text
Triton DSL
tl.dot
        ↓
TTIR
tt.dot
        ↓
TTGIR
#mma + #ttg.dot_op
        ↓
PTX
mma.sync.m16n8k16
        ↓
SASS
HMMA.16816.F32
```

`HMMA` 的存在证明机器代码使用 Tensor Core 指令，但不能单独证明 Tensor Core 已经满负载运行。

#### 输出 `convert_layout` 最终成为 `STSM + LDS`

PTX 的：

```ptx
stmatrix.sync.aligned.m8n8.x4.shared.b16
```

在 SASS 中成为：

```sass
STSM.16.M88.4
```

当前静态包含 4 条 `STSM`。

随后 Shared Memory 中的数据通过：

```sass
LDS.128
```

重新进入 Store-friendly Registers。输出转换部分对应四条 128-bit Shared Load。

最后使用四条：

```sass
STG.E.128
```

写入 Global Memory。

完整机器数据通路是：

```text
MMA Accumulator Registers
        ↓
STSM.16.M88.4
        ↓
Shared Memory
        ↓
BAR.SYNC.DEFER_BLOCKING
        ↓
LDS.128
        ↓
Blocked Registers
        ↓
STG.E.128
```

这直接回答了 `ttg.convert_layout` 中“数据到底怎么搬”：

> 当前 `#mma → #blocked` 转换确实借助 Shared Memory 和 CTA Barrier 完成，而不是只在同一线程内重新排列寄存器。

#### 七条 Barrier 保留到机器码

PTX 的 7 条 `bar.sync` 在 SASS 中仍对应 7 条：

```sass
BAR.SYNC.DEFER_BLOCKING
```

这证明相关 CTA 同步没有在机器码阶段被消除。

但指令存在只能证明同步语义，不能说明每条 Barrier 实际等待了多久。某些 Barrier 到达时其他 Warp 可能已经准备完毕，真实 Stall 必须由 Profiler 测量。

#### SASS 直接证明了什么

当前 SASS 可以直接确认：

- Global-to-Shared Async Copy 使用 12 条 `LDGSTS.E.BYPASS.128`；
- 两个 Async Wait 落实为 `DEPBAR.LE`；
- Shared-to-MMA Fragment 使用 8 条 `LDSM`；
- Tensor Core 计算使用 16 条 `HMMA.16816.F32`；
- 输出 Fragment 使用 4 条 `STSM` 写入 Shared Memory；
- 输出重排使用 128-bit Shared Load；
- 最终使用四条 `STG.E.128` 写回 C Tile；
- 七条 CTA Barrier 保留在机器码中。

SASS 仍然不能单独确认：

- Async Copy 与 HMMA 实际重叠了多少周期；
- Shared Swizzle 的真实 Bank Conflict 情况；
- Tensor Core Pipe 的活跃比例；
- Accumulator 的寄存器压力是否限制 Occupancy；
- Barrier 造成多少 Warp Stall；
- 当前 Tile 和 Warp 配置是否最优；
- Kernel 是计算受限还是访存受限。

### Matmul Lowering 总结

#### 静态指令对照表汇总

| **阶段 / 动作**   | **PTX 虚拟指令 (静态数量)**      | **SASS 硬件指令 (静态数量)**  | **物理功能与作用**                            |
| ----------------- | -------------------------------- | ----------------------------- | --------------------------------------------- |
| **异步搬运**      | `cp.async.cg.shared.global` (12) | `LDGSTS.E.BYPASS.128` (12)    | 显存直通共享内存，不占用通用寄存器            |
| **依赖等待**      | `cp.async.wait_group` (2)        | `DEPBAR.LE` (2)               | 等待异步搬运完成，驱动流水线                  |
| **加载 Fragment** | `ldmatrix...` (8)                | `LDSM.16.M88.4 / MT88.4` (8)  | 从 SMEM 加载数据并拆散塞入 Tensor Core 寄存器 |
| **硬件矩阵乘**    | `mma.sync.m16n8k16...` (16)      | `HMMA.16816.F32` (16)         | 触发 Tensor Core 硬件爆炒算乘加               |
| **输出拼图**      | `stmatrix...` (4)                | `STSM.16.M88.4` (4)           | 将 `#mma` 碎片写回 SMEM 做布局重组            |
| **布局读回**      | `ld.shared.v4.b32` (4)           | `LDS.128` (4)                 | 从 SMEM 按 `#blocked` 连续读回 128-bit        |
| **写回显存**      | `st.global.v4.b32` (4)           | `STG.E.128` (4)               | 128-bit 合并写入 Global Memory                |
| **线程同步**      | `bar.sync` (7)                   | `BAR.SYNC.DEFER_BLOCKING` (7) | 保护流水线 Stage 切换与输出重排读写           |

#### Matmul 的完整 Lowering 可以压缩为三条证据链

第一条：输入流水线：把数据从 Global Memory 搬运到 Tensor Core 的 Registers 里，并且全程不能让 Tensor Core 等待

```text
Python DSL
	tl.load A/B Tile
        ↓
TTIR（认领数据）
    tensor<64x32xf16>
    tensor<32x64xf16>
        ↓
TTGIR（确定运输策略）
    Blocked Global Layout
    → async_copy（异步搬运）
    → Double-buffered Swizzled Shared Layout （双缓冲 + 地址重排）
        ↓
PTX（转化为通用指令）
    cp.async.cg.shared.global（异步指令：把数据从显存搬运到共享内存中）
    → ldmatrix（从共享内存中读取数据，并按照Tensor Core喜爱的特定格式拼好，放入到线程的寄存器中）
        ↓
SASS（GPU硬件执行指令）
    LDGSTS.E.BYPASS.128（Load Global to Shared Store，显存直通共享内存）
    → LDSM（硬件级矩阵加载指令）
```

第二条： Tensor Core 计算：利用 GPU 内置的专用矩阵乘法硬件（Tensor Core），极速算完 $C = A \times B + C$。

```text
Python DSL
	tl.dot
        ↓
TTIR（明确计算精度，A和B采用半精度float，为了防止溢出，acc采用单精度float）
    tt.dot
    FP16 × FP16 + FP32 Accumulator
        ↓
TTGIR（分配硬件 layout）
    #ttg.dot_op
    → #mma [将数据从普通的操作数布局（#dot_op），映射到了 Tensor Core 专属的 #mma（Matrix Multiply-Accumulate）物理分布布局。这意味着 64×64 的矩阵被拆散，分给了各个线程的寄存器]
        ↓
PTX（调用 NV 硬件规范）
	mma.sync.m16n8k16 [告诉硬件，在一个 Warp（32 个线程）内，同步触发一次 16×8×16 尺寸的矩阵乘累加硬件操作]
        ↓
SASS（芯片内部机器指令）
	HMMA.16816.F32 [GPU 芯片内部硬件矩阵单元（Tensor Core）真正执行的物理指令！这一步只花几个时钟周期，就能瞬间完成成百上千次乘加运算]
```

第三条：输出布局转换：把散落在各个线程寄存器里的计算结果，重新拼回连续的大数组，安全、高速地写回 Global Memory

```text
TTGIR（解决冲突问题）
    #mma FP16 Result [刚算完的数据为mma布局，碎片化分布在各个线程的私有寄存器中，如果直接写回显存，会造成极其可怕的非合并访存]
    → ttg.convert_layout [因此必须做 convert_layout]
    → #blocked [转换成按行/列连续排列的 #blocked 布局]
        ↓
PTX / SASS（借助共享内存做重组中转）：
PTX
    stmatrix [各个线程先把手里碎片化的#mma寄存器数据，像拼图一样按照正确位置写入共享内存中]
    → Shared Memory [线程重新从 Shared Memory 里按连续的整块（128-bit，即 8 个 FP16） 读回自己的寄存器]
    → ld.shared.v4
        ↓
SASS
    STSM [各个线程先把手里碎片化的#mma寄存器数据，像拼图一样按照正确位置写入共享内存中]
    → BAR [Barrier 屏障：所有线程停下等待，保证大家都拼完了，共享内存里此时已经是一块完整且连续的 C 矩阵]
    → LDS.128 [线程重新从 Shared Memory 里按连续的整块（128-bit，即 8 个 FP16） 读回自己的寄存器]
    → STG.E.128 [此时每个线程手里拿到的都是连续大块数据了，发起一次极速的合并写回，将结果存入 Global Memory]
```

与 Vector Add 和 Softmax 相比：

```text
Vector Add
重点：Tensor 元素如何直接映射到线程

Softmax
重点：线程如何通过 Shuffle 和 Shared Memory 共同 Reduction

Matmul
重点：多种 Layout 如何连接 Global Memory、Shared Memory 和 Tensor Core
```

当前已经由 IR、PTX 和 SASS 直接证明：

- 一个 Program 负责 `64 × 64` C Tile；
- 当前 CTA 使用四个 Warp、128 个线程；
- K 维以 32 为 Tile 循环；
- A/B 采用 FP16，Accumulator 采用 FP32；
- A/B 使用 16 KiB 的双 Stage Shared Memory 逻辑空间；
- Global-to-Shared 使用 `cp.async/LDGSTS`；
- Shared-to-Fragment 使用 `ldmatrix/LDSM`；
- `tl.dot` 最终使用 `mma.sync/HMMA` Tensor Core 路径；
- 输出端 `convert_layout` 通过 `stmatrix/STSM + Shared Memory + LDS` 真实重排；
- 最终每个线程写回 32 个 FP16 输出；
- Pipeline 和输出转换相关的 7 条 Barrier 保留到 SASS。

仍需 Nsight Compute 或 Benchmark 验证：

- Global Load 是否达到理想合并访存；
- Async Copy 是否有效隐藏内存延迟；
- Shared Memory 是否存在明显 Bank Conflict；
- Tensor Core 利用率与 HMMA Pipe Active；
- 寄存器数量和 Occupancy；
- Barrier Stall；
- 不同 `BLOCK_M/BLOCK_N/BLOCK_K` 与 `num_warps` 的性能差异。

Matmul 最终建立的分析方法是：

```text
先看 TTGIR Layout
确认数据在哪个存储层级、由哪些线程持有
        ↓
再看 PTX/SASS
确认 Async Copy、Matrix Load、MMA 和 Layout Conversion
        ↓
最后看 Profiler
验证 Pipeline、Tensor Core 与 Shared Memory 的实际效率
```





