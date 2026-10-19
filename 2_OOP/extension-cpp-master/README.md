# PyTorch 中的 C++/CUDA 扩展

本仓库包含两个用于 PyTorch 的 C++/CUDA 扩展示例：

1. **extension_cpp** - 使用标准的 ATen/LibTorch API
2. **extension_cpp_stable** - 使用 [LibTorch 稳定版 ABI (Stable ABI)](https://pytorch.org/docs/main/notes/libtorch_stable_abi.html)

这两个扩展都展示了如何编写一个名为 `mymuladd` 的自定义算子（custom op），该算子同时具备自定义的 CPU 和 CUDA 内核。

## extension_cpp（标准 ATen API）

使用完整的 ATen/LibTorch API。这是编写 PyTorch 扩展的传统方式。
有关更多详细信息，请参阅[此教程](https://pytorch.org/tutorials/advanced/cpp_custom_ops.html)。

## extension_cpp_stable（稳定版 ABI）

使用 LibTorch 稳定版 ABI，以确保构建的扩展可以在任何 $\ge$ 2.10.0 的 PyTorch 版本上运行，而无需为每个 PyTorch 版本重新编译。

`extension_cpp_stable` 示例需要 PyTorch 2.10+ 版本。

## 构建方法

构建 extension_cpp（标准 API）：

```
cd extension_cpp
pip install --no-build-isolation -e .

```

构建 extension_cpp_stable（稳定版 ABI）：

```
cd extension_cpp_stable
pip install --no-build-isolation -e .

```

## 测试方法

测试这两个扩展：

```
python test/test_extension.py

```

## 作者

[Peter Goldsborough](https://github.com/goldsborough), [Richard Zou](https://github.com/zou3519)