#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdio>

#include "gopeft_cuda.h"

namespace {

thread_local char last_error[256];

struct context {
  cublasHandle_t blas;
};

__device__ __constant__ float nf4_codebook[16] = {-1.f, -0.6961928f, -0.52507305f, -0.3949175f,
  -0.28444138f, -0.18477343f, -0.09105004f, 0.f, 0.0795803f, 0.1609302f,
  0.2461123f, 0.33791524f, 0.44070983f, 0.562617f, 0.72295684f, 1.f};

int cuda_status(cudaError_t status) {
  if (status == cudaSuccess) return 0;
  std::snprintf(last_error, sizeof(last_error), "%s", cudaGetErrorString(status));
  return 1;
}

int cublas_status(cublasStatus_t status) {
  if (status == CUBLAS_STATUS_SUCCESS) return 0;
  std::snprintf(last_error, sizeof(last_error), "cuBLAS status %d", static_cast<int>(status));
  return 1;
}

__global__ void add_row(float* dst, const float* row, int count, int cols) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) dst[index] += row[index % cols];
}

__global__ void dropout(float* dst, int count, float probability, unsigned long long seed) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  unsigned long long value = seed + static_cast<unsigned long long>(index);
  value ^= value >> 12;
  value ^= value << 25;
  value ^= value >> 27;
  float random = static_cast<float>((value * 2685821657736338717ULL) >> 40) * (1.f / 16777216.f);
  dst[index] = random < probability ? 0.f : dst[index] / (1.f - probability);
}

__global__ void dropout_mask(float* dst, int count, float probability, unsigned long long seed) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  unsigned long long value = seed + static_cast<unsigned long long>(index);
  value ^= value >> 12;
  value ^= value << 25;
  value ^= value >> 27;
  float random = static_cast<float>((value * 2685821657736338717ULL) >> 40) * (1.f / 16777216.f);
  dst[index] = random < probability ? 0.f : 1.f / (1.f - probability);
}

__global__ void multiply(float* dst, const float* src, int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) dst[index] *= src[index];
}

__device__ float nf4(unsigned char code) {
  return nf4_codebook[code];
}

__global__ void quantized_linear(float* dst, const float* input, const unsigned char* codes,
    const float* scales, const unsigned char* scale_codes, const float* scale_scales,
    int rows, int out, int in, int block_size, int scale_block_size, int scheme,
    float alpha, float beta) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  __shared__ float input_tile[16][32];
  float sum = 0.f;
  int offset = col * in;
  for (int start = 0; start < in; start += 32) {
    for (int position = threadIdx.y * blockDim.x + threadIdx.x; position < 16 * 32; position += blockDim.x * blockDim.y) {
      int tile_row = position / 32;
      int tile_col = position % 32;
      int source_row = blockIdx.y * blockDim.y + tile_row;
      int source_col = start + tile_col;
      input_tile[tile_row][tile_col] = source_row < rows && source_col < in ? input[source_row * in + source_col] : 0.f;
    }
    __syncthreads();
    if (row < rows && col < out) {
      int count = in - start < 32 ? in - start : 32;
      for (int local = 0; local < count; ++local) {
        int index = offset + start + local;
        unsigned char packed = codes[index >> 1];
        unsigned char code = (index & 1) ? (packed >> 4) : (packed & 0x0f);
        int block = index / block_size;
        float scale = scale_block_size ? static_cast<float>(scale_codes[block]) * scale_scales[block / scale_block_size] : scales[block];
        float weight = scheme ? nf4(code) * scale : static_cast<float>(static_cast<signed char>(code << 4) >> 4) * scale;
        sum += input_tile[threadIdx.y][local] * weight;
      }
    }
    __syncthreads();
  }
  if (row < rows && col < out) {
    int index = row * out + col;
    dst[index] = alpha * sum + beta * dst[index];
  }
}

}

extern "C" int peft_cuda_device_count(int* count) {
  return cuda_status(cudaGetDeviceCount(count));
}

extern "C" int peft_cuda_new(void** pointer, int device) {
  if (cuda_status(cudaSetDevice(device))) return 1;
  context* value = new context{};
  if (cublas_status(cublasCreate(&value->blas))) {
    delete value;
    return 1;
  }
  *pointer = value;
  return 0;
}

extern "C" int peft_cuda_free_context(void* pointer) {
  context* value = static_cast<context*>(pointer);
  if (value == nullptr) return 0;
  int status = cublas_status(cublasDestroy(value->blas));
  delete value;
  return status;
}

extern "C" int peft_cuda_alloc(void** pointer, size_t bytes) { return cuda_status(cudaMalloc(pointer, bytes)); }
extern "C" int peft_cuda_free(void* pointer) { return pointer == nullptr ? 0 : cuda_status(cudaFree(pointer)); }
extern "C" int peft_cuda_h2d(void* dst, const void* src, size_t bytes) { return cuda_status(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice)); }
extern "C" int peft_cuda_d2h(void* dst, const void* src, size_t bytes) { return cuda_status(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost)); }
extern "C" int peft_cuda_d2d(void* dst, const void* src, size_t bytes) { return cuda_status(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice)); }
extern "C" int peft_cuda_zero(void* dst, size_t bytes) { return cuda_status(cudaMemset(dst, 0, bytes)); }

extern "C" int peft_cuda_dropout(void* dst, int count, float probability, uint64_t seed) {
	dropout<<<(count + 255) / 256, 256>>>(static_cast<float*>(dst), count, probability, seed);
	return cuda_status(cudaGetLastError());
}

extern "C" int peft_cuda_dropout_mask(void* dst, int count, float probability, uint64_t seed) {
  dropout_mask<<<(count + 255) / 256, 256>>>(static_cast<float*>(dst), count, probability, seed);
  return cuda_status(cudaGetLastError());
}

extern "C" int peft_cuda_multiply(void* dst, const void* src, int count) {
  multiply<<<(count + 255) / 256, 256>>>(static_cast<float*>(dst), static_cast<const float*>(src), count);
  return cuda_status(cudaGetLastError());
}

extern "C" int peft_cuda_gemm(void* raw_context, void* dst, const void* a, const void* b,
    int m, int n, int k, int a_cols, int b_cols, int transpose_a, int transpose_b,
    float alpha, float beta) {
  context* value = static_cast<context*>(raw_context);
  cublasOperation_t op_a = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t op_b = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  return cublas_status(cublasSgemm(value->blas, op_a, op_b, n, m, k, &alpha,
      static_cast<const float*>(b), b_cols, static_cast<const float*>(a), a_cols,
      &beta, static_cast<float*>(dst), n));
}

extern "C" int peft_cuda_axpy(void* raw_context, void* dst, const void* src, int count, float alpha) {
  context* value = static_cast<context*>(raw_context);
  return cublas_status(cublasSaxpy(value->blas, count, &alpha, static_cast<const float*>(src), 1,
      static_cast<float*>(dst), 1));
}

extern "C" int peft_cuda_add_row(void* dst, const void* row, int rows, int cols) {
  int count = rows * cols;
  add_row<<<(count + 255) / 256, 256>>>(static_cast<float*>(dst), static_cast<const float*>(row), count, cols);
  return cuda_status(cudaGetLastError());
}

extern "C" int peft_cuda_quantized_linear(void*, void* dst, const void* input, const void* codes,
    const void* scales, const void* scale_codes, const void* scale_scales, int rows, int out,
    int in, int block_size, int scale_block_size, int scheme, float alpha, float beta) {
  dim3 block(16, 16);
  dim3 grid((out + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
  quantized_linear<<<grid, block>>>(static_cast<float*>(dst), static_cast<const float*>(input),
      static_cast<const unsigned char*>(codes), static_cast<const float*>(scales),
      static_cast<const unsigned char*>(scale_codes), static_cast<const float*>(scale_scales),
      rows, out, in, block_size, scale_block_size, scheme, alpha, beta);
  return cuda_status(cudaGetLastError());
}

extern "C" const char* peft_cuda_last_error(void) { return last_error; }
