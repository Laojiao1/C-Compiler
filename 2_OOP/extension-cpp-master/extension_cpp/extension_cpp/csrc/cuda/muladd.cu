#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

namespace extension_cpp {

__global__ void muladd_kernel(int numel, const float* a, const float* b, float c, float* result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) result[idx] = a[idx] * b[idx] + c;
}

at::Tensor mymuladd_cuda(const at::Tensor& a, const at::Tensor& b, double c) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);

  // 确保在 CUDA 上
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
  
  // 准备内存数据
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = at::empty(a_contig.sizes(), a_contig.options());

  // 获取指向底层数据的 C++ 原生指针 (float*)
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();

  // 获取张量的总元素个数
  int numel = a_contig.numel();
  
  // 获取当前 CUDA 流
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
  // 启动 CUDA 核函数
  // 每个线程块里有256个线程
  muladd_kernel<<<(numel+255)/256, 256, 0, stream>>>(numel, a_ptr, b_ptr, c, result_ptr);

  return result;
}

// __global__ 表示这个函数运行在 GPU 上，但是由 CPU 调用
__global__ void mul_kernel(int numel, const float* a, const float* b, float* result) {
  // 计算当前线程的全局唯一索引 (1D Grid 1D Block 常见写法)
  // 这里就是用来替代 for(int i=0) 的变量的
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  // 防御性越界检查：因为线程总数可能会略微多于我们实际的元素个数 numel
  if (idx < numel) 
    // 只有一行计算！没有 for 循环！每个线程只算属于自己的那一个数据！
    result[idx] = a[idx] * b[idx]; 
}

at::Tensor mymul_cuda(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = at::empty(a_contig.sizes(), a_contig.options());
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  int numel = a_contig.numel();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  mul_kernel<<<(numel+255)/256, 256, 0, stream>>>(numel, a_ptr, b_ptr, result_ptr);
  return result;
}

__global__ void add_kernel(int numel, const float* a, const float* b, float* result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) result[idx] = a[idx] + b[idx];
}

void myadd_out_cuda(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(b.sizes() == out.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_CHECK(out.dtype() == at::kFloat);
  TORCH_CHECK(out.is_contiguous());
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CUDA);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = out.data_ptr<float>();
  int numel = a_contig.numel();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  add_kernel<<<(numel+255)/256, 256, 0, stream>>>(numel, a_ptr, b_ptr, result_ptr);
}


// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m) {
  // 当传入 CUDA Tensor 时，把 mymuladd 路由到 C++ 函数 &mymuladd_cuda
  m.impl("mymuladd", &mymuladd_cuda);
  m.impl("mymul", &mymul_cuda);
  m.impl("myadd_out", &myadd_out_cuda);
}

}
