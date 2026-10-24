# 5.3 KV Cache 与请求调度

按学习路线的三小节组织。5.3.1、5.3.2 的核心实现，以及 5.3.3 的三组真实服务验证均已完成。用户统一复盘后，再讨论 5.4，不自动进入下一阶段。
第一组使用 B 的成功重试记录；容量组使用首次成功记录；前缀组使用按用户授权卸载新增 `flash-attn` 后重新采集的匹配结果。原始失败和被替代记录保留，不参与正式比较。
2026-10-23 进行了目录整理，只调整归属、导入和运行入口，不改变核心算法或历史结果。

```text
03_scheduler_kv_cache/
├── README.md
├── stage_1_paged_kv/          # 5.3.1：分页 KV、前缀共享与数值 reference
│   ├── paged_kv.py
│   ├── paged_kv_store.py
│   ├── attention_reference.py
│   ├── prefix_cache.py
│   ├── tests/                # 51 项
│   └── results/              # 该小节日志与历史指纹
├── stage_2_scheduler/         # 5.3.2：CPU 调度、完成回收、显式抢占与重算
│   ├── scheduler.py
│   ├── scheduler_simulator.py
│   ├── teaching_scheduler.py
│   ├── record_scheduler_traces.py
│   ├── tests/                # 37 项
│   └── results/              # 调度轨迹、88 项里程碑日志与历史指纹
└── stage_3_serving/           # 5.3.3：真实服务实验的辅助工具
    ├── serve_stream_probe.sh
    ├── stream_probe.py
    ├── prepare_mixed_workload.py
    ├── mixed_probe.py
    ├── budget_experiment.py  # 参数核验、2 轮预热、3 轮测量与 A/B、B/C 汇总
    ├── cache_experiment.py   # 容量 2/3 条边界、独立前缀冷/热采集
    ├── run_cache_experiments.sh  # 顺序管理脚本自己创建的服务
    ├── tests/                # 52 项，不访问真实模型服务
    └── results/              # 第一组记录，以及下列分目录
        ├── cache_validation_20261023/               # 正式容量、旧关闭组、首次启动失败
        ├── cache_validation_20261023_prefix_retry1/ # 原配置重试的启动失败日志
        └── cache_validation_20261023_prefix_clean/  # 正式 prefix_off / prefix_on 匹配对照
```

第二阶段通过包导入复用第一阶段的 KV 实现，没有复制另一套块管理。第三阶段调用真实 vLLM 服务，不使用 CPU 模拟器代替 GPU 推理。

## 当前学习边界

第一、二阶段需要理解块管理、引用、调度与恢复的核心逻辑。第三阶段重点是流程、控制变量、计时边界与结论；HTTP、线程、解析和保存属于辅助工作，不要求逐行研究。

讲义和实验记录继续分开存于既有阶段五笔记目录。5.3 结束后先复盘，再讨论 5.4 的内容，不提前展开完整源码主链。

## 运行规则：始终从本目录进入

以下命令在 WSL 中执行。使用已有环境，不重新安装依赖：

```bash
conda activate vllm
cd /mnt/e/Code/C++Compiler/5_LLM_Serving/03_scheduler_kv_cache
```

Python 工具使用 `python -m 阶段包.模块` 运行，不再使用根目录旧脚本名，也不必切换到阶段子目录。

## 功能测试

全部 140 项：

```bash
python -m unittest discover -s . -v
```

仅测试某一阶段：

```bash
python -m unittest discover -s stage_1_paged_kv/tests -t . -v
python -m unittest discover -s stage_2_scheduler/tests -t . -v
python -m unittest discover -s stage_3_serving/tests -t . -v
```

单组示例：`python -m unittest -v stage_2_scheduler.tests.test_preemption`。
第三阶段测试只使用人工事件与临时假服务，不创建 vLLM 引擎。

## 5.3.2 轨迹入口

```bash
python -m stage_2_scheduler.record_scheduler_traces --case all
```

已有[调度轨迹](stage_2_scheduler/results/scheduler_traces_20261022.json)保持原样，不需要为了目录整理重新生成历史记录。

## 5.3.3 服务与采集入口

若已有相同配置的服务，继续复用，不能再启动第二个引擎。只有需要启动时才运行：

```bash
bash stage_3_serving/serve_stream_probe.sh
```

单请求采集示例：

```bash
python -m stage_3_serving.stream_probe \
  --output stage_3_serving/results/single_stream_new.json
```

固定工作量已生成并搬到第三阶段；只有需要生成新文件时才运行准备程序：

```bash
python -m stage_3_serving.prepare_mixed_workload \
  --output stage_3_serving/results/mixed_workload_new.json
```

继续使用现有工作量做一轮混合预跑：

```bash
python -m stage_3_serving.mixed_probe \
  --workload stage_3_serving/results/mixed_workload_20261023.json \
  --output stage_3_serving/results/mixed_pilot_new.json
```

采集器不会覆盖旧文件。上述 `*_new.json` 是新运行文件名示例，运行前确认未占用。混合预跑只验收长度与客户端观察窗口重叠，不证明 GPU 每轮混合组批，也不是正式预算对照。

## 5.3.3 第一组正式对照：预算与分块

问题：长输入加入后，怎样分配 Prefill 工作，会怎样影响已有 Decode、长请求首正文延迟与整组吞吐？本组不测试缓存容量边界，也不开启前缀缓存。

| 配置 | `max_num_batched_tokens` | chunked prefill | 固定 KV 池 |
|---|---:|---|---|
| A | 512 | 开 | 1 GiB，块大小 16 |
| B | 4096 | 开 | 同 A |
| C | 4096 | 关 | 同 A |

只比较 A/B 与 B/C。1 GiB 是本组采用的保守实验配置，不是最优值。按本模型 BF16 的 28 KiB/Token 预测为 2340 块、37440 个位置，三份成功记录均已由 `/metrics` 核对一致。显式 KV 字节预算控制缓存大小，不能再把 `gpu_memory_utilization=0.70` 当成整个服务占用的硬上限。

本次 B 首次在初始指标读取时超时，尚未预热；失败文件原样保留，成功比较使用 `budget_B_20261023_retry1.json`。详细流程、三轮波动、结论与限制见[第一组实验记录](<E:/桌面/Postgraduate/Compiler/C++编译器/阶段5：大模型推理优化/5.3 KVCache 与 请求调度/5.3.3 控制变量实验—实验记录.md>)。下面文件名已被本次结果占用，复跑必须换新名称，不覆盖历史结果。

### 一套流程，依次用于 A、B、C

1. 两个终端均激活已有 `vllm` 环境，并进入本 README 所在根目录。
2. 在原服务终端按 Ctrl+C，等待旧服务与 EngineCore 退出。不要用另一个终端同时再启动引擎。启动器发现旧进程会拒绝启动，不会自动结束它。
3. 在服务终端运行当前配置的启动命令，等到启动完成、服务可用。
4. 在采集终端运行对应采集命令。期间不要发送其它请求或运行其它 GPU 实验。程序自动完成同一工作量的 2 轮预热、3 轮正式采集，预热不计入汇总。
5. 看到 `BUDGET_EXPERIMENT_PASS` 后保留文件；结束当前服务，按同样步骤换下一配置。若启动或验收失败，先保留错误排查，不修改个别配置的公共参数来凑结果。

服务终端，每次只执行一条：

```bash
bash stage_3_serving/serve_stream_probe.sh A
# A 测量完成并退出后，再执行 B；B 完成并退出后，再执行 C。
bash stage_3_serving/serve_stream_probe.sh B
bash stage_3_serving/serve_stream_probe.sh C
```

启动完成后，在采集终端确认指标接口返回 200，再开始采集：

```bash
curl --noproxy '*' -fsS --connect-timeout 3 --max-time 10 \
  -o /dev/null -w 'metrics HTTP %{http_code}\n' \
  http://127.0.0.1:8000/metrics
```

采集终端，配置名称和文件名必须与当前服务对应：

```bash
python -m stage_3_serving.budget_experiment run --config A \
  --workload stage_3_serving/results/mixed_workload_20261023.json \
  --output stage_3_serving/results/budget_A_20261023.json

python -m stage_3_serving.budget_experiment run --config B \
  --workload stage_3_serving/results/mixed_workload_20261023.json \
  --output stage_3_serving/results/budget_B_20261023_retry1.json

python -m stage_3_serving.budget_experiment run --config C \
  --workload stage_3_serving/results/mixed_workload_20261023.json \
  --output stage_3_serving/results/budget_C_20261023.json
```

三份结果全部完成后：

```bash
python -m stage_3_serving.budget_experiment compare \
  stage_3_serving/results/budget_A_20261023.json \
  stage_3_serving/results/budget_B_20261023_retry1.json \
  stage_3_serving/results/budget_C_20261023.json
```

程序核对本机 `/proc` 的显式启动参数和进程身份，保存实际 KV 配置、包版本、GPU 显存快照及原始指标。不会启动/停止服务，也不要求开启开发接口。各轮原始请求、失败、事件门控与时间定义沿用预跑工具；不满足长度或客户端重叠验收就停止，不挑选成功轮次。文件不覆盖，重跑需使用新文件名，并相应调整比较入口。

汇总关注长请求的平均客户端 TTFT、短请求的平均客户端 TPOT、整组输出吞吐：对三轮值给出中位数与范围，逐请求数据仍全部保存。范围重叠且波动明显时，不强行宣称提升；三个样本也不构成统计显著性或稳定容量证明。比较输出的百分比是“右侧相对左侧”，延迟越小越好、吞吐越大越好。

所有工具核验、空闲等待、指标读取和 GPU 快照都在每轮 HTTP 计时窗口之外。客户端重叠仍不等于 GPU 逐轮混合组批；预算和开关影响的是完整服务表现，不从这些数据单独推断具体 kernel 的耗时。本 README 保留操作入口，实际观测和结论单独归档于实验记录。

## 后两组已完成：用户无需重复执行

容量组固定 128 MiB KV、块大小 16、预算 512、分块开启、前缀关闭，只验证 N=2/3。总块数 292，其中本地 vLLM 保留 1 个 null block，普通请求可分配 291 块。单请求末段需 104 块，2 条需 208、3 条需 312。三轮实测 N=2 无抢占，N=3 各一次抢占、利用率到达 100%，全部请求最终完成。

前缀组固定 1 GiB KV，其余参数一致，关闭/开启各独立启动。新 workload 包含共享前缀 1024、私有后缀 256、固定输出 64 的请求；无关输入预热、每轮新目标、独立 primer。三轮冷请求均零命中，热单请求均命中 1024/1280 Token，热四请求均命中 4096/5120 Token，即 Token 加权 80%。primer 成本单列，不计入热态吞吐。关闭组的查询分母为 0，命中率不适用。

正式证据：

- [容量结果](stage_3_serving/results/cache_validation_20261023/capacity.json)：2 条/3 条的平均 E2E 三轮中位数约 1935/2680 ms，输出吞吐约 130.58/102.60 Token/s；这是容量边界观测，不将性能差异全部归因于抢占。
- [关闭前缀缓存](stage_3_serving/results/cache_validation_20261023_prefix_clean/prefix_off.json)和[开启前缀缓存](stage_3_serving/results/cache_validation_20261023_prefix_clean/prefix_on.json)：填充后单请求 TTFT 中位数约 151.31→41.34 ms；四请求热态输出吞吐约 186.19→277.97 Token/s。两组输入和 29 条对应正文全部一致，不代替逐层数值验证。
- [独立工作量](stage_3_serving/results/cache_validation_20261023_prefix_clean/workload.json)：不使用第一组普通混合 workload；目标和后缀在第一个 Token 区分，防止额外命中。

本次只按用户明确授权卸载了临时安装的 `flash-attn 2.8.3.post2`，原因是其 GLIBC_2.32 依赖使新服务无法初始化；没有保存包副本、升级其他依赖或修改框架源码。第一次关闭组与安装时间重叠，未与恢复后的开启组配对。详情、失败路径和全部命令见既有[实验记录](<E:/桌面/Postgraduate/Compiler/C++编译器/阶段5：大模型推理优化/5.3 KVCache 与 请求调度/5.3.3 控制变量实验—实验记录.md>)第 9—12 节。

以下仅为复现入口，不需要用户再运行。目录名须未占用，且没有现有引擎：

```bash
# 完整顺序执行三种服务配置。
bash stage_3_serving/run_cache_experiments.sh stage_3_serving/results/cache_validation_NEW

# 也可只跑匹配的前缀两组；不能使用已存在的结果目录。
bash stage_3_serving/run_cache_experiments.sh stage_3_serving/results/prefix_NEW prefix_off prefix_on
```

脚本使用已有绝对 Python/模型路径，只启动和关闭自己的服务；其他引擎或端口存在时停止，不接管用户进程。服务日志保留在各运行目录，输出文件不覆盖。容量组以约 10 Hz 采样运行/等待/利用率，可能影响绝对耗时；前缀组不进行该在线采样。所有结论局限于本次小工作量，不视为线上稳态、硬件极限或通用最优值。

## 历史证据与布局迁移

- 5.3.1：[用户提供的 51 项日志](stage_1_paged_kv/results/raw/tests_pass_20261021.log)、[当时归档指纹](stage_1_paged_kv/results/source_fingerprints_20261021.json)。
- 5.3.2：[当前代码当时直接复跑的 88 项日志](stage_2_scheduler/results/raw/tests_pass_20261022.log)、[当时归档指纹](stage_2_scheduler/results/source_fingerprints_20261022.json)。
- 5.3.3：[单请求采集](stage_3_serving/results/single_stream_probe_20261023.json)、[工作量](stage_3_serving/results/mixed_workload_20261023.json)、[2048 预算混合预跑](stage_3_serving/results/mixed_pilot_2048_20261023.json)。
- 第一组正式对照：[A](stage_3_serving/results/budget_A_20261023.json)、[B 成功重试](stage_3_serving/results/budget_B_20261023_retry1.json)、[C](stage_3_serving/results/budget_C_20261023.json)；另保留 [B 初始采集失败](stage_3_serving/results/budget_B_20261023.json)。

这些文件只搬迁，没有重写其内容。旧日志、指纹和 JSON 内的源码/工作量路径记录的是实验当时的平铺布局，不能冒充整理后的当前路径。查询时按本 README 的阶段归属映射；旧笔记中的历史命令也不再直接作为当前入口。

KV 引用计数约定仍为“请求引用数 + 缓存条目引用数”；`gather()` 和 CPU 调度轨迹都只是教学 reference，不提供 GPU kernel 或真实引擎性能结论。
