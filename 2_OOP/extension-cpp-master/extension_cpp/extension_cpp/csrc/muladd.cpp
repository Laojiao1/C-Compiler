#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <vector>

extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}

namespace extension_cpp {

at::Tensor mymuladd_cpu(const at::Tensor& a, const at::Tensor& b, double c) {
  // 1. 各种防御性检查 (Check) 
  TORCH_CHECK(a.sizes() == b.sizes()); // 确保形状一致
  TORCH_CHECK(a.dtype() == at::kFloat); // 确保类型是 float32
  TORCH_CHECK(b.dtype() == at::kFloat); // 确保类型是 float32
  // 确保在 CPU 上
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);

  // 2. 准备内存数据
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();

  // 创建一个空的 Tensor 来装结果
  at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());

  // 获取指向底层数据的 C++ 原生指针 (float*)
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();

  // 3. 核心计算逻辑：写一个纯粹的 C++ for 循环
  for (int64_t i = 0; i < result.numel(); i++) {
    result_ptr[i] = a_ptr[i] * b_ptr[i] + c;
  }

  // 4. 返回结果
  return result;
}

at::Tensor mymul_cpu(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  for (int64_t i = 0; i < result.numel(); i++) {
    result_ptr[i] = a_ptr[i] * b_ptr[i];
  }
  return result;
}

// An example of an operator that mutates one of its inputs.
void myadd_out_cpu(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(b.sizes() == out.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_CHECK(out.dtype() == at::kFloat);
  TORCH_CHECK(out.is_contiguous());
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CPU);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = out.data_ptr<float>();
  for (int64_t i = 0; i < out.numel(); i++) {
    result_ptr[i] = a_ptr[i] + b_ptr[i];
  }
}

// 1. 定义算子 (Operator Definition)
TORCH_LIBRARY(extension_cpp, m) {
  // 定义了命名空间为 extension_cpp
  // 注册了一个名为 mymuladd 的算子，并声明了它的输入输出类型签名
  m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
  m.def("mymul(Tensor a, Tensor b) -> Tensor");
  m.def("myadd_out(Tensor a, Tensor b, Tensor(a!) out) -> ()");
}

// 2. 注册 CPU 的具体实现 (Operator Implementation)
TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
  // 当传入 CPU Tensor 时，把 mymuladd 路由到 C++ 函数 &mymuladd_cpu
  m.impl("mymuladd", &mymuladd_cpu);
  m.impl("mymul", &mymul_cpu);
  m.impl("myadd_out", &myadd_out_cpu);
}

}
