# WSL2 开发环境清单

盘点日期为 2026-11-13（北京时间）。范围为当前工作区使用的 Ubuntu WSL2、Conda 登记环境、Linux 用户目录和工作区中发现的 Python venv，以及可从 WSL 调用的开发工具。版本来自本机命令或安装元数据。GPU 实测采用小型验证程序。

## 常用入口

| 场景 | 环境或命令 | 已验证 |
| --- | --- | --- |
| CUDA C++ | `/usr/local/cuda/bin/nvcc` | 使用 `-arch=sm_120` 编译并运行一个最小 CUDA kernel |
| PyTorch、Triton 与单层量化 | `conda activate triton` | PyTorch FP16 16×16 GEMM 通过，Triton 可导入 |
| vLLM 和量化部署 | `conda activate vllm` | vLLM CLI 版本查询和 PyTorch FP16 16×16 GEMM 通过 |
| CUDA 时间线 | `/usr/local/bin/nsys` | 采集最小 CUDA 程序并生成 `.nsys-rep` |
| CUDA 单算子分析 | `/usr/local/NVIDIA-Nsight-Compute/ncu` | 在 CC 12.0 上成功采集 `LaunchStats` |

## 系统、硬件与路径

| 项目 | 实测 |
| --- | --- |
| 工作发行版 | `Ubuntu-20.04`，WSL2，盘点时运行中且为默认发行版 |
| 其他已注册发行版 | `docker-desktop`，WSL2，盘点时停止 |
| Linux 用户空间 | Ubuntu 20.04.6 LTS，x86_64，glibc 2.31，Bash 5.0.17 |
| 内核 | `6.6.87.2-microsoft-standard-WSL2` |
| 工作区 | Windows `E:\Code\C++Compiler` 对应 WSL `/mnt/e/Code/C++Compiler` |
| CPU | AMD Ryzen 9 9955HX，WSL 可见 16 核、32 逻辑线程 |
| 内存 | WSL 约 15 GiB RAM、4 GiB swap |
| GPU | 单张 RTX 5060 Laptop GPU，8151 MiB 总显存，Compute Capability 12.0 |
| GPU 设备与驱动库 | `/dev/dxg`；`/usr/lib/wsl/lib` 下的 `libcuda.so.1`、`libnvidia-ml.so.1` |
| NVIDIA 驱动与管理工具 | 驱动 576.65，WSL `nvidia-smi` 575.60，显示驱动 CUDA 版本 12.9 |

## C++ 与常用开发工具

| 工具 | 版本与位置 |
| --- | --- |
| GCC / G++ | 9.4.0，`/usr/bin/gcc`、`/usr/bin/g++`，Ubuntu 版本 `9.4.0-1ubuntu1~20.04.2` |
| GNU Make | 4.2.1，`/usr/bin/make` |
| GDB / gdbserver | 9.2，`/usr/bin/gdb`、`/usr/bin/gdbserver` |
| GNU Binutils | 2.34，`/usr/bin` 下的 `ld`、`as`、`objdump`、`nm`、`readelf` |
| LLVM 运行库 | 系统包 `libllvm12`，12.0.0 |
| Ninja | `triton` 内 1.13.2，`vllm` 内二进制报告 `1.13.2.git.kitware.jobserver-pipe-1` |
| Git | 2.25.1，`/usr/bin/git` |
| 调试与终端 | strace 5.5、ltrace 0.7.3、tmux 3.0a |
| 网络与归档 | curl 7.68.0、wget 1.20.3、rsync 3.1.3、GNU tar 1.30 |
| Java / sbt | 默认 Java 为 OpenJDK 17.0.15（有 JDK），另有 OpenJDK 11.0.27 JRE。sbt runner 1.12.11，`/usr/bin/sbt` |
| Node.js / npm | Windows 互操作入口 `/mnt/d/Development/nodejs/node.exe` 为 24.15.0，同目录的 `npm` 为 11.12.1 |
| Docker Desktop 入口 | Windows 工具目录已进入 WSL 的 PATH。盘点时调用 `docker version` 提示启用该发行版的 WSL integration |

## CUDA Toolkit 与 NVIDIA 工具

系统 CUDA Toolkit **12.8.0** 通过 Ubuntu 包安装，`/usr/local/cuda` 最终指向 `/usr/local/cuda-12.8`。驱动报告的 CUDA 12.9、系统 Toolkit 12.8 和各 Python 环境的 PyTorch CUDA 构建版本分别记录。CUDA 头文件中已确认存在 `cuda_fp8.h` 和 `cuda_fp4.h`。

| 工具 | 版本 | 路径与验证 |
| --- | --- | --- |
| `nvcc`、`ptxas` | 12.8.61 | `/usr/local/cuda/bin`。`nvcc -arch=sm_120` 实际编译和执行通过 |
| `cuobjdump`、`nvdisasm` | 12.8.55 | `/usr/local/cuda/bin`，版本查询通过 |
| Compute Sanitizer | 2025.1.0.0 | `/usr/local/cuda/bin/compute-sanitizer`，版本查询通过 |
| CUDA-GDB | 12.8，基于 GDB 13.2 | `/usr/local/cuda/bin/cuda-gdb`，版本查询通过 |
| `nvprof` | 12.8.57 | `/usr/local/cuda/bin/nvprof`，版本查询通过 |
| Nsight Systems CLI | 2026.1.2.63 | `/usr/local/bin/nsys` 指向 `/opt/nvidia/nsight-systems-cli/2026.1.2/target-linux-x64/nsys`。实跑 CUDA trace 并生成报告 |
| Nsight Systems Toolkit 版 | 2024.6.2.225 | `/usr/local/cuda/bin/nsys`。对应 GUI 入口为 `/usr/local/bin/nsys-ui`，指向 `/opt/nvidia/nsight-systems/2024.6.2/host-linux-x64/nsys-ui` |
| Nsight Compute | 2025.2.1.0 | 交互 shell 默认 `/usr/local/NVIDIA-Nsight-Compute/ncu`，实际位于 `/usr/local/NVIDIA-Nsight-Compute-2025.2`。成功采集 SM 12.0 上的 kernel Launch Statistics |
| Nsight Compute Toolkit 版 | 2025.1.0.0 | `/usr/local/cuda/bin/ncu`，另有 `/opt/nvidia/nsight-compute/2025.1.0` 安装目录 |

`nsys status --environment` 显示 Linux `perf_event_open` 可用，`perf_event_paranoid=2`。当前用户的 CPU 进程树采样检查通过，系统范围 CPU 采样检查失败。CUDA trace 的实测通过，采集参数为 `--trace=cuda --sample=none --cpuctxsw=none`。

系统 Toolkit 中的主要库位于 `/usr/local/cuda/targets/x86_64-linux/lib`。下表为系统包版本。

| 组件 | 版本 |
| --- | --- |
| CUDA Runtime / CUPTI | 12.8.57 / 12.8.57 |
| NVRTC / NVTX | 12.8.61 / 12.8.55 |
| cuBLAS（含 cuBLASLt） | 12.8.3.14 |
| cuFFT / cuRAND | 11.3.3.41 / 10.3.9.55 |
| cuSOLVER / cuSPARSE / cuFile | 11.7.2.55 / 12.5.7.53 / 1.13.0.11 |

## Python 和虚拟环境

系统解释器 `/usr/bin/python3` 为 **3.8.10**。交互式 Bash 可用 Conda，默认 Python 来自 Miniconda base。自动执行可用解释器绝对路径，或先运行 `source ~/miniconda3/etc/profile.d/conda.sh` 再 `conda activate`。

| 环境 | 根目录 | Python | 关键验证 |
| --- | --- | --- | --- |
| Conda `base` | `/home/laojiao/miniconda3` | 3.14.6 | Conda 26.5.3，解释器和包元数据可读取 |
| Conda `triton` | `/home/laojiao/miniconda3/envs/triton` | 3.10.20 | PyTorch 2.11.0+cu128、Triton 3.6.0。构建包含 `sm_120`，FP16 GEMM 通过 |
| Conda `triton31` | `/home/laojiao/miniconda3/envs/triton31` | 3.10.20 | PyTorch 2.5.0+cu124、Triton 3.1.0。CUDA 设备查询返回可用，PyTorch 构建架构最高为 `sm_90`，在本机执行 FP16 GEMM 报 `no kernel image is available for execution on the device` |
| Conda `vllm` | `/home/laojiao/miniconda3/envs/vllm` | 3.12.14 | vLLM 0.31.0+cu129，PyTorch 2.13.0+cu129、Triton 3.7.1。CLI 与 FP16 GEMM 通过，具体模型与量化 kernel 留到实验时验证 |
| venv `tvm_study` | `/home/laojiao/.virtualenvs/tvm_study` | 3.8.10 | 可直接执行 `bin/python`，该 venv 的发行包为 pip、setuptools、wheel |

GPU GEMM 验证用两个 16×16 的 FP16 全 1 矩阵相乘，并同步后读取单个结果。成功环境返回 16.0。

### 各环境关键包

版本以各自解释器的包元数据为准。PyTorch 表中的 CUDA 后缀来自实际导入后的版本查询。

| 环境 | 已安装的关键包与版本 |
| --- | --- |
| `base` | conda 26.5.3、pip 26.1.2、setuptools 82.0.1、wheel 0.47.0、packaging 26.0 |
| `triton` | torch 2.11.0+cu128、triton 3.6.0、torchvision 0.26.0+cu128、torchaudio 2.11.0+cu128、transformers 5.17.0、tokenizers 0.23.2、huggingface_hub 1.32.0、safetensors 0.8.0、numpy 2.2.6、pandas 2.3.3、matplotlib 3.10.9、pytest 9.1.1、pip 26.1.2 |
| `triton31` | torch 2.5.0+cu124、triton 3.1.0、numpy 1.26.4、pip 26.1.2 |
| `vllm` | vllm 0.31.0+cu129、torch 2.13.0+cu129、triton 3.7.1、torchvision 0.28.0+cu129、torchaudio 2.11.0+cu129、torchcodec 0.16.0+cu129、transformers 5.17.0、tokenizers 0.23.3、huggingface_hub 1.33.0、safetensors 0.8.0、sentencepiece 0.2.2、numpy 2.3.5、numba 0.65.0、llvmlite 0.47.0、FastAPI 0.136.3、uvicorn 0.54.0、uv 0.9.12、pip 26.2.1 |
| `tvm_study` | pip 25.0.1、setuptools 75.3.4、wheel 0.45.1 |

| 环境 | GPU 运行库和推理相关包 |
| --- | --- |
| `triton` | `cuda-toolkit` 12.8.1、`cuda-bindings` 12.9.4、`nvidia-cuda-runtime-cu12` 12.8.90、`nvidia-cublas-cu12` 12.8.4.1、`nvidia-cudnn-cu12` 9.19.0.56、`nvidia-nccl-cu12` 2.28.9、`nvidia-cuda-nvrtc-cu12` 12.8.93、`nvidia-cuda-cupti-cu12` 12.8.90 |
| `triton31` | `nvidia-cuda-runtime-cu12` 12.4.127、`nvidia-cublas-cu12` 12.4.5.8、`nvidia-cudnn-cu12` 9.1.0.70、`nvidia-nccl-cu12` 2.21.5、`nvidia-cuda-nvrtc-cu12` 12.4.127、`nvidia-cuda-cupti-cu12` 12.4.127 |
| `vllm` | `cuda-toolkit` 12.9.1、`cuda-python` 和 `cuda-bindings` 12.9.10、`nvidia-cuda-runtime-cu12` 12.9.79、`nvidia-cuda-nvcc-cu12` 12.9.86、`nvidia-cuda-nvrtc-cu12` 12.9.86、`nvidia-cuda-cupti-cu12` 12.9.79、`nvidia-cublas-cu12` 12.9.1.4、`nvidia-cudnn-cu12` 9.20.0.48、`nvidia-nccl-cu12` 2.29.7、`nvidia-cutlass-dsl` 4.7.1、`cuda-tile` 1.6.0、`compressed-tensors` 0.17.0、`flashinfer-python` 0.7.0.post1、`humming-kernels` 0.1.16、`tilelang` 0.1.12、`apache-tvm-ffi` 0.1.11 |

vLLM 另有本地源码目录 `/mnt/e/Code/C++Compiler/5_LLM_Serving/04_vllm/vllm-official`。上表的 vLLM 版本取自 Conda 环境实际执行的 CLI。

## FPGA 工具

| 工具 | 版本或位置 |
| --- | --- |
| Verilator | 5.017 devel，`~/oss-cad-suite/bin/verilator` |
| Icarus Verilog | 13.0 devel，`~/oss-cad-suite/bin/iverilog` |
| Yosys | 0.34+43，`~/oss-cad-suite/bin/yosys` |
| 另一份 OSS CAD Suite 目录 | `~/osscad20231022/oss-cad-suite/bin` |

## 更新这份清单

平时按上面的绝对路径或环境名直接使用。更换驱动、升级 Toolkit、切换环境或发现版本异常时，只复查受影响的条目。

```bash
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader
/usr/local/cuda/bin/nvcc --version
/usr/local/bin/nsys --version
/usr/local/NVIDIA-Nsight-Compute/ncu --version
~/miniconda3/bin/conda env list
~/miniconda3/envs/triton/bin/python -m pip show torch triton transformers
~/miniconda3/envs/vllm/bin/python -m pip show torch triton vllm compressed-tensors
~/miniconda3/envs/vllm/bin/vllm --version
```
