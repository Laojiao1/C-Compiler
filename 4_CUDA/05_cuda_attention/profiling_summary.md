# CUDA Attention Profiling Summary

测试环境：RTX 5060 Laptop GPU（SM 12.0）、CUDA 12.8、Nsight Systems
2026.1.2、Nsight Compute 2025.2.1。默认问题规模为 `N=1024, d=64,
causal=true`。

## 最终结果

| 实现 | Kernel 时间 | Launch | 逻辑全局流量 | 完整 S/P |
|---|---:|---:|---:|---:|
| Unfused | 0.2647 ms | 3 | 52.25 MiB | 物化 |
| CUDA-Core Online | 0.1354 ms | 1 | 17.00 MiB | 不物化 |
| Tensor-Core Online | 0.0779 ms | 1 | 8.75 MiB | 不物化 |

Tensor-Core 版本相对 Unfused 加速约 `3.40x`，相对 CUDA-Core Online
加速约 `1.74x`。FP16 tile、FP32 Online 状态的最大绝对误差约为
`1.21e-4`，normalized error 为 `0.0359 < 1`。

## Nsight Systems

最终时间线中，每种 kernel 共采集 111 次：

- QK tiled GEMM：平均 133.817 us；
- fused scale/mask/softmax：平均 15.119 us；
- PV tiled GEMM：平均 116.331 us；
- CUDA-Core Online：平均 136.201 us；
- Tensor-Core Online：平均 76.958 us。

报告：`nsys/attention_final_n1024_d64.nsys-rep`。

## Nsight Compute 驱动的优化

### CUDA-Core Online

- 56 registers/thread，22.78 KiB dynamic Shared Memory；
- theoretical occupancy 66.7%，achieved occupancy 47.2%；
- 约 52% 的 warp 发射间隔来自 MIO throttle；
- DRAM throughput 仅 0.86%，L2 hit rate 93.9%。

结论：它不是 DRAM-bound。标量 Shared Memory 指令、特殊函数和寄存器压力
限制了继续扩展，因此改用 WMMA，而不是继续减少 HBM 流量。

报告：`ncu/attention_flash_n1024_d64.ncu-rep`。

### Tensor-Core Online

初版标准行主序布局产生平均 5.8-way shared-load bank conflict，共 811,895
次冲突。给 half tile 的 leading dimension 增加 8 个元素、给 FP32 tile 增加
4 个元素后，Nsight Compute 不再报告 shared-load conflict，shared excessive
wavefront 从 62% 降至约 9%。

最终 `Br=32` 版本：

- 256 threads/block，32 blocks；
- 56 registers/thread，38.78 KiB static Shared Memory；
- achieved occupancy 18.9%；
- executed instructions 约 3.08M；
- short-scoreboard/MIO dependency 约占发射间隔的 34.7%。

剩余瓶颈来自 Shared Memory 容量、block 级同步与不足一个完整调度 wave。继续
优化需要 warp specialization、异步 copy/pipeline 或更底层的 MMA 布局，不再是
简单调整 `Br/Bc` 可以解决的问题。

最终报告：`ncu/attention_tensor_final_n1024_d64.ncu-rep`。

> Nsight Compute 需要多次 replay，报告中的 Duration 会被 profiling 放大；性能
> 对比使用 CUDA Event 与 Nsight Systems 的非 replay 时间。
