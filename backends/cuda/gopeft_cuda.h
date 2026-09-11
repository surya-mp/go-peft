#ifndef GOPEFT_CUDA_H
#define GOPEFT_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int peft_cuda_device_count(int* count);
int peft_cuda_new(void** context, int device);
int peft_cuda_free_context(void* context);
int peft_cuda_alloc(void** pointer, size_t bytes);
int peft_cuda_free(void* pointer);
int peft_cuda_h2d(void* dst, const void* src, size_t bytes);
int peft_cuda_d2h(void* dst, const void* src, size_t bytes);
int peft_cuda_d2d(void* dst, const void* src, size_t bytes);
int peft_cuda_zero(void* dst, size_t bytes);
int peft_cuda_dropout(void* dst, int count, float probability, uint64_t seed);
int peft_cuda_dropout_mask(void* dst, int count, float probability, uint64_t seed);
int peft_cuda_multiply(void* dst, const void* src, int count);
int peft_cuda_gemm(void* context, void* dst, const void* a, const void* b, int m, int n, int k, int a_cols, int b_cols, int transpose_a, int transpose_b, float alpha, float beta);
int peft_cuda_axpy(void* context, void* dst, const void* src, int count, float alpha);
int peft_cuda_add_row(void* dst, const void* row, int rows, int cols);
int peft_cuda_quantized_linear(void* context, void* dst, const void* input, const void* codes, const void* scales, const void* scale_codes, const void* scale_scales, int rows, int out, int in, int block_size, int scale_block_size, int scheme, float alpha, float beta);
const char* peft_cuda_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
