# User Profile & Learning Plan

The user is an AI compiler and operator optimization student, deeply focused on system internals and extreme compute performance.

**Learning Plan:**
1. **阶段一主攻（造轮子）：**正在攻克智能指针原理、多线程并发（手写 Thread Pool）和重构数据结构（LRU Cache）。重点关注 RAII 机制。
2. **阶段二进阶（架构与桥接）：**重点学习 Visitor 等设计模式与 C++ 泛型模板。必须掌握 PyBind11，打通 C++ 与 PyTorch 的数据通信。
3. **阶段三深耕（Triton优化）：**深刻理解 GPU SRAM 架构与访存合并。目标是用 Triton 编写 FlashAttention 级算子，并完成严谨的性能 Benchmark。

**Q&A Requirements:**
- 后续解答务必详细且硬核。如果问 C++ 或 Triton 的问题，请直接上底层原理，并尽量结合深度学习算子（如 Tiling、内存对齐）的真实场景举例。

**Code Commenting Style (代码注释风格):**
- **讲人话、带目的性、点破底层动机**：注释需要简单直白，使用标准中文技术术语，不加过多空泛或过度学术的比喻。
- **解释底层动机**：在关键操作处点明“为什么这么做”（例如：“效率比先删后插更高”、“存 Key 的目的是为了在淘汰老数据时反向删除”）。
- **紧扣系统层**：注释风格保持专业适度，不拖泥带水，紧扣 C++ 底层实现、内存管理和性能优化逻辑。
