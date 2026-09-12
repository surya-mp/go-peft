# Native CUDA validation — 2026-09-11

GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU (GPU 0, 6144 MiB).
Driver: 591.74. WSL NVIDIA-SMI: 590.52.01; reported driver CUDA capability: 13.1.
Validated toolkit: CUDA 13.2, nvcc V13.2.86; cuBLAS log version 13.4.1.
Go: go1.26.5 linux/amd64. GCC: Ubuntu 15.2.0-16ubuntu1.

## Reproduce environment

On the validated machine, source `/home/surya-mp/.local/lib/go-peft-toolchain/cuda-env.sh` before the commands below.
Its contents at validation time were:

```bash
export CUDA_HOME=/home/surya-mp/.local/lib/go-peft-toolchain/cuda-13.2-packages/root/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:/home/surya-mp/.local/lib/go-peft-toolchain/go/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CGO_ENABLED=1
export CGO_LDFLAGS="-L$CUDA_HOME/lib64"
export GOFLAGS=-mod=readonly
export NVCC_PREPEND_FLAGS=-arch=sm_86
```

Adjust these installation paths when reproducing on another machine.
This selects the user-local CUDA 13.2 toolkit, Go, CGO_ENABLED=1, CUDA library paths,
GOFLAGS=-mod=readonly and NVCC_PREPEND_FLAGS=-arch=sm_86 for this GPU.
The system /usr/local/cuda still points to CUDA 13.1, whose headers fail with this glibc.

## Commands and results

- git status --short: inspected before changes; pre-existing changes preserved.
- nvidia-smi: sandbox NVML access failed; outside sandbox succeeded.
- nvcc --version / go version: initially missing; installed versions above verified.
- sh scripts/build_cuda.sh: initially CRLF shell failure; then CUDA 13.1 compilation failed
  with 17 incompatible C/C++ linkage declarations and rsqrt/rsqrtf exception mismatches.
  Added C-linkage guards to the bridge header. CUDA 13.1 then reported only the two
  math-header errors (full output: [go-peft-cuda-13.1-build.log](#go-peft-cuda-131-buildlog)).
  User-local CUDA 13.2 compiled successfully, exit 0 ([go-peft-cuda-build.log](#go-peft-cuda-buildlog)).
- ls/file/ar/nm: gopeft_cuda.o (76 KiB, ELF x86-64) and libgopeftcuda.a (78 KiB)
  exist under backends/cuda; archive contains gopeft_cuda.o and native CUDA entry points.
  References include __cudaLaunchKernel and cublasSgemm_v2.
- go env GOOS GOARCH CGO_ENABLED CC: linux, amd64, 1, gcc.
- go list -tags cuda: GoFiles=[train.go], CgoFiles=[cuda.go],
  IgnoredGoFiles=[cuda_stub.go train_stub.go], TestGoFiles=[cuda_test.go].
- go test -count=1 -v -tags cuda ./backends/cuda: PASS, 6 tests, 0 skips, 0 failures.
  Complete output: [go-peft-cuda-native-tests.log](#go-peft-cuda-native-testslog).
- go test -count=1 -v -tags cuda ./...: PASS, 43 top-level tests, 0 skips, 0 failures.
  Complete output: [go-peft-cuda-full-suite.log](#go-peft-cuda-full-suitelog).
- go run -tags cuda ./examples/lora/cuda: exit 0, [1 2].
  Output: [go-peft-lora-example.log](#go-peft-lora-examplelog). cuBLAS trace: [go-peft-lora-cublas.log](#go-peft-lora-cublaslog).
- go run -tags cuda ./examples/qlora/cuda: exit 0, [5.637332 14.675482].
  Output: [go-peft-qlora-example.log](#go-peft-qlora-examplelog). cuBLAS trace: [go-peft-qlora-cublas.log](#go-peft-qlora-cublaslog).
- go test -tags cuda -list '^Benchmark' ./backends/cuda: no benchmarks listed.
  Repository benchmark functions use CPU or MLX; no CUDA benchmark was invented.

## Native execution evidence and numerical results

Both examples call cuda.New -> NewDevice(0), cudaGetDeviceCount, cudaSetDevice,
and cublasCreate. They fail on initialization errors and never enable DisableCUDA.
Runtime keeps the supplied CUDA engine. cuBLAS logs show cublasCreate_v2 and
cublasSgemm_v2 on GPU=0 (three SGEMMs for LoRA, two for QLoRA).
QLoRA calls peft_cuda_quantized_linear, which launches quantized_linear<<<...>>>;
returned values are copied from device memory by cudaMemcpyDeviceToHost.
Native tests also exercise int4/NF4, double quantization, training and dropout kernels.
No CUDA test skipped. No observed numerical failures; forward/reference comparisons
passed the existing 1e-4 tolerance and training checks passed.
CPU reference calculations and host-side setup/loss aggregation exist in the code;
they are not a fallback for the CUDA matrix products or kernels.

## Files changed by this task

- scripts/build_cuda.sh: CRLF -> LF only, verified against the saved original.
- backends/cuda/gopeft_cuda.h: guarded extern "C" around bridge declarations;
  original CRLF line endings and other contents preserved.
- Generated artifacts: backends/cuda/gopeft_cuda.o, backends/cuda/libgopeftcuda.a.

Before adding this report, SHA256 comparison against the resumed worktree baseline confirmed every other tracked
file unchanged (including go.mod and go.sum). No reset, clean or unrelated refactoring.
User-local Go/CUDA installation and environment are under
`/home/surya-mp/.local/lib/go-peft-toolchain`. The captured logs are embedded below so
this report does not depend on temporary files. This report records the tested worktree,
including its pre-existing changes, rather than a clean commit.

## Captured logs

The example runs enabled `CUBLAS_LOGINFO_DBG=1` and set `CUBLAS_LOGDEST_DBG`
to the respective cuBLAS log path. The successful build produced no output; its
recorded exit status was 0.

### go-peft-cuda-13.1-build.log

```text
/usr/include/x86_64-linux-gnu/bits/mathcalls.h(206): error: exception specification is incompatible with that of previous function "rsqrt" (declared at line 629 of /usr/local/cuda/bin/../targets/x86_64-linux/include/crt/math_functions.h)
   extern double rsqrt (double __x) noexcept (true); extern double __rsqrt (double __x) noexcept (true);
                                    ^

/usr/include/x86_64-linux-gnu/bits/mathcalls.h(206): error: exception specification is incompatible with that of previous function "rsqrtf" (declared at line 653 of /usr/local/cuda/bin/../targets/x86_64-linux/include/crt/math_functions.h)
   extern float rsqrtf (float __x) noexcept (true); extern float __rsqrtf (float __x) noexcept (true);
                                   ^

2 errors detected in the compilation of "/mnt/c/Users/spram/Downloads/Projects/go-lib/go-peft/backends/cuda/gopeft_cuda.cu".
```

### go-peft-cuda-build.log

No output (exit status 0).

### go-peft-cuda-native-tests.log

```text
=== RUN   TestLoRAForward
--- PASS: TestLoRAForward (0.59s)
=== RUN   TestQLoRAQuantizedLinear
--- PASS: TestQLoRAQuantizedLinear (0.00s)
=== RUN   TestAdapterTraining
--- PASS: TestAdapterTraining (0.01s)
=== RUN   TestAdapterTrainingDropout
--- PASS: TestAdapterTrainingDropout (0.01s)
=== RUN   TestQLoRAAdapterTraining
--- PASS: TestQLoRAAdapterTraining (0.02s)
=== RUN   TestQLoRAAdapterTrainingDropout
--- PASS: TestQLoRAAdapterTrainingDropout (0.00s)
PASS
ok  	github.com/surya-mp/go-peft/backends/cuda	0.873s
```

### go-peft-cuda-full-suite.log

```text
=== RUN   TestCPUGemm
--- PASS: TestCPUGemm (0.00s)
=== RUN   TestCPUDropoutIsDeterministic
--- PASS: TestCPUDropoutIsDeterministic (0.00s)
=== RUN   TestRuntimeRequiresCUDA
--- PASS: TestRuntimeRequiresCUDA (0.00s)
=== RUN   TestRuntimeUsesProvidedCUDAEngine
--- PASS: TestRuntimeUsesProvidedCUDAEngine (0.00s)
=== RUN   TestRuntimeCanExplicitlyUseCPU
--- PASS: TestRuntimeCanExplicitlyUseCPU (0.00s)
PASS
ok  	github.com/surya-mp/go-peft/backend	0.004s
=== RUN   TestLoRAForward
--- PASS: TestLoRAForward (0.34s)
=== RUN   TestQLoRAQuantizedLinear
--- PASS: TestQLoRAQuantizedLinear (0.00s)
=== RUN   TestAdapterTraining
--- PASS: TestAdapterTraining (0.01s)
=== RUN   TestAdapterTrainingDropout
--- PASS: TestAdapterTrainingDropout (0.01s)
=== RUN   TestQLoRAAdapterTraining
--- PASS: TestQLoRAAdapterTraining (0.02s)
=== RUN   TestQLoRAAdapterTrainingDropout
--- PASS: TestQLoRAAdapterTrainingDropout (0.00s)
PASS
ok  	github.com/surya-mp/go-peft/backends/cuda	0.459s
?   	github.com/surya-mp/go-peft/examples/lora/cpu	[no test files]
?   	github.com/surya-mp/go-peft/examples/lora/cuda	[no test files]
?   	github.com/surya-mp/go-peft/examples/lora/huggingface	[no test files]
?   	github.com/surya-mp/go-peft/examples/lora/injection	[no test files]
?   	github.com/surya-mp/go-peft/examples/qlora/cpu	[no test files]
?   	github.com/surya-mp/go-peft/examples/qlora/cuda	[no test files]
?   	github.com/surya-mp/go-peft/examples/qlora/huggingface	[no test files]
?   	github.com/surya-mp/go-peft/examples/qlora/injection	[no test files]
=== RUN   TestLoRAAdapterRoundTrip
--- PASS: TestLoRAAdapterRoundTrip (0.00s)
=== RUN   TestQLoRAAdapterRoundTrip
--- PASS: TestQLoRAAdapterRoundTrip (0.00s)
=== RUN   TestPEFTParityAdapterStateValidation
=== RUN   TestPEFTParityAdapterStateValidation/missing
=== RUN   TestPEFTParityAdapterStateValidation/unexpected
=== RUN   TestPEFTParityAdapterStateValidation/shape
--- PASS: TestPEFTParityAdapterStateValidation (0.00s)
    --- PASS: TestPEFTParityAdapterStateValidation/missing (0.00s)
    --- PASS: TestPEFTParityAdapterStateValidation/unexpected (0.00s)
    --- PASS: TestPEFTParityAdapterStateValidation/shape (0.00s)
=== RUN   TestPEFTParityPortableBiasState
--- PASS: TestPEFTParityPortableBiasState (0.00s)
PASS
ok  	github.com/surya-mp/go-peft/format/huggingface	0.008s
=== RUN   TestRoundTripIsDeterministic
--- PASS: TestRoundTripIsDeterministic (0.00s)
=== RUN   TestReadRejectsTrailingData
--- PASS: TestReadRejectsTrailingData (0.00s)
PASS
ok  	github.com/surya-mp/go-peft/format/safetensors	0.007s
=== RUN   TestConfigValidate
=== RUN   TestConfigValidate/rank
=== RUN   TestConfigValidate/negative_alpha
=== RUN   TestConfigValidate/nan_alpha
=== RUN   TestConfigValidate/infinite_alpha
=== RUN   TestConfigValidate/negative_dropout
=== RUN   TestConfigValidate/unit_dropout
=== RUN   TestConfigValidate/nan_dropout
=== RUN   TestConfigValidate/missing_targets
=== RUN   TestConfigValidate/empty_target
=== RUN   TestConfigValidate/bias
--- PASS: TestConfigValidate (0.00s)
    --- PASS: TestConfigValidate/rank (0.00s)
    --- PASS: TestConfigValidate/negative_alpha (0.00s)
    --- PASS: TestConfigValidate/nan_alpha (0.00s)
    --- PASS: TestConfigValidate/infinite_alpha (0.00s)
    --- PASS: TestConfigValidate/negative_dropout (0.00s)
    --- PASS: TestConfigValidate/unit_dropout (0.00s)
    --- PASS: TestConfigValidate/nan_dropout (0.00s)
    --- PASS: TestConfigValidate/missing_targets (0.00s)
    --- PASS: TestConfigValidate/empty_target (0.00s)
    --- PASS: TestConfigValidate/bias (0.00s)
=== RUN   TestMatches
--- PASS: TestMatches (0.00s)
=== RUN   TestInjectDoesNotReplaceOnInvalidModule
--- PASS: TestInjectDoesNotReplaceOnInvalidModule (0.00s)
=== RUN   TestInject
--- PASS: TestInject (0.00s)
=== RUN   TestLinearForwardMergeAndUnmerge
--- PASS: TestLinearForwardMergeAndUnmerge (0.00s)
=== RUN   TestLinearForwardIntoMatchesForward
--- PASS: TestLinearForwardIntoMatchesForward (0.00s)
=== RUN   TestLinearBias
--- PASS: TestLinearBias (0.00s)
=== RUN   TestPEFTParityLoRALifecycle
--- PASS: TestPEFTParityLoRALifecycle (0.00s)
=== RUN   TestPEFTParityZeroDropout
--- PASS: TestPEFTParityZeroDropout (0.00s)
=== RUN   TestPEFTParityTrainingDropoutAndBiasModes
--- PASS: TestPEFTParityTrainingDropoutAndBiasModes (0.00s)
PASS
ok  	github.com/surya-mp/go-peft/lora	0.006s
=== RUN   TestParameterOwnsShape
--- PASS: TestParameterOwnsShape (0.00s)
=== RUN   TestParameterClone
--- PASS: TestParameterClone (0.00s)
=== RUN   TestParameterCount
--- PASS: TestParameterCount (0.00s)
PASS
ok  	github.com/surya-mp/go-peft/peft	0.006s
=== RUN   TestConfigValidate
=== RUN   TestConfigValidate/lora
=== RUN   TestConfigValidate/block_size
=== RUN   TestConfigValidate/quantization
=== RUN   TestConfigValidate/int4_double_quant
=== RUN   TestConfigValidate/scale_block_size
--- PASS: TestConfigValidate (0.00s)
    --- PASS: TestConfigValidate/lora (0.00s)
    --- PASS: TestConfigValidate/block_size (0.00s)
    --- PASS: TestConfigValidate/quantization (0.00s)
    --- PASS: TestConfigValidate/int4_double_quant (0.00s)
    --- PASS: TestConfigValidate/scale_block_size (0.00s)
=== RUN   TestConfigAllowsInt4WithoutDoubleQuant
--- PASS: TestConfigAllowsInt4WithoutDoubleQuant (0.00s)
=== RUN   TestInject
--- PASS: TestInject (0.00s)
=== RUN   TestForwardMatchesQuantizedReference
--- PASS: TestForwardMatchesQuantizedReference (0.00s)
=== RUN   TestMergeIsRejected
--- PASS: TestMergeIsRejected (0.00s)
=== RUN   TestNF4DoubleQuantForward
--- PASS: TestNF4DoubleQuantForward (0.00s)
=== RUN   TestPEFTParityQLoRAContract
--- PASS: TestPEFTParityQLoRAContract (0.00s)
PASS
ok  	github.com/surya-mp/go-peft/qlora	0.006s
=== RUN   TestQuantize
--- PASS: TestQuantize (0.00s)
=== RUN   TestQuantizationErrorIsBounded
--- PASS: TestQuantizationErrorIsBounded (0.00s)
=== RUN   TestQuantizedStorageCopiesBuffers
--- PASS: TestQuantizedStorageCopiesBuffers (0.00s)
PASS
ok  	github.com/surya-mp/go-peft/quantization/int4	0.006s
=== RUN   TestQuantizeCodebookAndDot
--- PASS: TestQuantizeCodebookAndDot (0.00s)
=== RUN   TestDoubleQuantizationReducesScaleStorage
--- PASS: TestDoubleQuantizationReducesScaleStorage (0.00s)
=== RUN   TestQuantizedStorageCopiesDoubleQuantBuffers
--- PASS: TestQuantizedStorageCopiesDoubleQuantBuffers (0.00s)
PASS
ok  	github.com/surya-mp/go-peft/quantization/nf4	0.006s
```

### go-peft-lora-example.log

```text
[1 2]
```

### go-peft-lora-cublas.log

```text
I! cuBLAS (v13.4.1) function cublasStatus_t cublasCreate_v2(cublasContext**) called:
i!  handle: type=cublasHandle_t; val=POINTER (IN HEX:0x0x1f9448f0)
i! Time: 2026-09-11T18:11:38 elapsed from start 0.016667 minutes or 1.000000 seconds
i!Process=10673; Thread=123934205857792; GPU=0; Handle=POINTER (IN HEX:0x(nil))
i! COMPILED WITH: GNU GCC/G++ / 8.5.0 20210514 (Red Hat 8.5.0-26)
I! cuBLAS (v13.4.1) function cublasStatus_t cublasSgemm_v2(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int) called:
i!  handle: type=cublasHandle_t; val=POINTER (IN HEX:0x0x1f9560f0)
i!  transa: type=cublasOperation_t; val=CUBLAS_OP_T(1)
i!  transb: type=cublasOperation_t; val=CUBLAS_OP_N(0)
i!  m: type=int; val=2
i!  n: type=int; val=1
i!  k: type=int; val=2
i!  alpha: type=float; val=POINTER (IN HEX:0x0x7ffda216dedc)
i!  A: type=float; val=POINTER (IN HEX:0x0x504e20400)
i!  lda: type=int; val=2
i!  B: type=float; val=POINTER (IN HEX:0x0x504e20600)
i!  ldb: type=int; val=2
i!  beta: type=float; val=POINTER (IN HEX:0x0x7ffda216ded8)
i!  C: type=float; val=POINTER (IN HEX:0x0x504e20c00)
i!  ldc: type=int; val=2
i! Time: 2026-09-11T18:11:38 elapsed from start 0.016667 minutes or 1.000000 seconds
i!Process=10673; Thread=123934205857792; GPU=0; Handle=POINTER (IN HEX:0x0x1f9560f0); StreamId=POINTER (IN HEX:0x(nil)) (defaultStream); MathMode=CUBLAS_DEFAULT_MATH
i! COMPILED WITH: GNU GCC/G++ / 8.5.0 20210514 (Red Hat 8.5.0-26)
I! cuBLAS (v13.4.1) function cublasStatus_t cublasSgemm_v2(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int) called:
i!  handle: type=cublasHandle_t; val=POINTER (IN HEX:0x0x1f9560f0)
i!  transa: type=cublasOperation_t; val=CUBLAS_OP_T(1)
i!  transb: type=cublasOperation_t; val=CUBLAS_OP_N(0)
i!  m: type=int; val=1
i!  n: type=int; val=1
i!  k: type=int; val=2
i!  alpha: type=float; val=POINTER (IN HEX:0x0x7ffda216dedc)
i!  A: type=float; val=POINTER (IN HEX:0x0x504e20800)
i!  lda: type=int; val=2
i!  B: type=float; val=POINTER (IN HEX:0x0x504e20600)
i!  ldb: type=int; val=2
i!  beta: type=float; val=POINTER (IN HEX:0x0x7ffda216ded8)
i!  C: type=float; val=POINTER (IN HEX:0x0x504e20e00)
i!  ldc: type=int; val=1
i! Time: 2026-09-11T18:11:38 elapsed from start 0.016667 minutes or 1.000000 seconds
i!Process=10673; Thread=123934205857792; GPU=0; Handle=POINTER (IN HEX:0x0x1f9560f0); StreamId=POINTER (IN HEX:0x(nil)) (defaultStream); MathMode=CUBLAS_DEFAULT_MATH
i! COMPILED WITH: GNU GCC/G++ / 8.5.0 20210514 (Red Hat 8.5.0-26)
I! cuBLAS (v13.4.1) function cublasStatus_t cublasSgemm_v2(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int) called:
i!  handle: type=cublasHandle_t; val=POINTER (IN HEX:0x0x1f9560f0)
i!  transa: type=cublasOperation_t; val=CUBLAS_OP_T(1)
i!  transb: type=cublasOperation_t; val=CUBLAS_OP_N(0)
i!  m: type=int; val=2
i!  n: type=int; val=1
i!  k: type=int; val=1
i!  alpha: type=float; val=POINTER (IN HEX:0x0x7ffda216dedc)
i!  A: type=float; val=POINTER (IN HEX:0x0x504e20a00)
i!  lda: type=int; val=1
i!  B: type=float; val=POINTER (IN HEX:0x0x504e20e00)
i!  ldb: type=int; val=1
i!  beta: type=float; val=POINTER (IN HEX:0x0x7ffda216ded8)
i!  C: type=float; val=POINTER (IN HEX:0x0x504e20c00)
i!  ldc: type=int; val=2
i! Time: 2026-09-11T18:11:38 elapsed from start 0.016667 minutes or 1.000000 seconds
i!Process=10673; Thread=123934205857792; GPU=0; Handle=POINTER (IN HEX:0x0x1f9560f0); StreamId=POINTER (IN HEX:0x(nil)) (defaultStream); MathMode=CUBLAS_DEFAULT_MATH
i! COMPILED WITH: GNU GCC/G++ / 8.5.0 20210514 (Red Hat 8.5.0-26)
I! cuBLAS (v13.4.1) function cublasStatus_t cublasDestroy_v2(cublasHandle_t) called:
i!  handle: type=cublasHandle_t; val=POINTER (IN HEX:0x0x1f9560f0)
i! Time: 2026-09-11T18:11:38 elapsed from start 0.016667 minutes or 1.000000 seconds
i!Process=10673; Thread=123934205857792; GPU=0; Handle=POINTER (IN HEX:0x0x1f9560f0); StreamId=POINTER (IN HEX:0x(nil)) (defaultStream); MathMode=CUBLAS_DEFAULT_MATH
i! COMPILED WITH: GNU GCC/G++ / 8.5.0 20210514 (Red Hat 8.5.0-26)
```

### go-peft-qlora-example.log

```text
[5.637332 14.675482]
```

### go-peft-qlora-cublas.log

```text
I! cuBLAS (v13.4.1) function cublasStatus_t cublasCreate_v2(cublasContext**) called:
i!  handle: type=cublasHandle_t; val=POINTER (IN HEX:0x0x2813cad0)
i! Time: 2026-09-11T18:11:50 elapsed from start 0.000000 minutes or 0.000000 seconds
i!Process=10819; Thread=134298931191808; GPU=0; Handle=POINTER (IN HEX:0x(nil))
i! COMPILED WITH: GNU GCC/G++ / 8.5.0 20210514 (Red Hat 8.5.0-26)
I! cuBLAS (v13.4.1) function cublasStatus_t cublasSgemm_v2(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int) called:
i!  handle: type=cublasHandle_t; val=POINTER (IN HEX:0x0x2814e2d0)
i!  transa: type=cublasOperation_t; val=CUBLAS_OP_T(1)
i!  transb: type=cublasOperation_t; val=CUBLAS_OP_N(0)
i!  m: type=int; val=2
i!  n: type=int; val=1
i!  k: type=int; val=3
i!  alpha: type=float; val=POINTER (IN HEX:0x0x7fff8e9892ac)
i!  A: type=float; val=POINTER (IN HEX:0x0x504e20800)
i!  lda: type=int; val=3
i!  B: type=float; val=POINTER (IN HEX:0x0x504e20600)
i!  ldb: type=int; val=3
i!  beta: type=float; val=POINTER (IN HEX:0x0x7fff8e9892a8)
i!  C: type=float; val=POINTER (IN HEX:0x0x504e20e00)
i!  ldc: type=int; val=2
i! Time: 2026-09-11T18:11:50 elapsed from start 0.000000 minutes or 0.000000 seconds
i!Process=10819; Thread=134298931191808; GPU=0; Handle=POINTER (IN HEX:0x0x2814e2d0); StreamId=POINTER (IN HEX:0x(nil)) (defaultStream); MathMode=CUBLAS_DEFAULT_MATH
i! COMPILED WITH: GNU GCC/G++ / 8.5.0 20210514 (Red Hat 8.5.0-26)
I! cuBLAS (v13.4.1) function cublasStatus_t cublasSgemm_v2(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int) called:
i!  handle: type=cublasHandle_t; val=POINTER (IN HEX:0x0x2814e2d0)
i!  transa: type=cublasOperation_t; val=CUBLAS_OP_T(1)
i!  transb: type=cublasOperation_t; val=CUBLAS_OP_N(0)
i!  m: type=int; val=2
i!  n: type=int; val=1
i!  k: type=int; val=2
i!  alpha: type=float; val=POINTER (IN HEX:0x0x7fff8e9892ac)
i!  A: type=float; val=POINTER (IN HEX:0x0x504e20a00)
i!  lda: type=int; val=2
i!  B: type=float; val=POINTER (IN HEX:0x0x504e20e00)
i!  ldb: type=int; val=2
i!  beta: type=float; val=POINTER (IN HEX:0x0x7fff8e9892a8)
i!  C: type=float; val=POINTER (IN HEX:0x0x504e20c00)
i!  ldc: type=int; val=2
i! Time: 2026-09-11T18:11:50 elapsed from start 0.000000 minutes or 0.000000 seconds
i!Process=10819; Thread=134298931191808; GPU=0; Handle=POINTER (IN HEX:0x0x2814e2d0); StreamId=POINTER (IN HEX:0x(nil)) (defaultStream); MathMode=CUBLAS_DEFAULT_MATH
i! COMPILED WITH: GNU GCC/G++ / 8.5.0 20210514 (Red Hat 8.5.0-26)
I! cuBLAS (v13.4.1) function cublasStatus_t cublasDestroy_v2(cublasHandle_t) called:
i!  handle: type=cublasHandle_t; val=POINTER (IN HEX:0x0x2814e2d0)
i! Time: 2026-09-11T18:11:50 elapsed from start 0.000000 minutes or 0.000000 seconds
i!Process=10819; Thread=134298931191808; GPU=0; Handle=POINTER (IN HEX:0x0x2814e2d0); StreamId=POINTER (IN HEX:0x(nil)) (defaultStream); MathMode=CUBLAS_DEFAULT_MATH
i! COMPILED WITH: GNU GCC/G++ / 8.5.0 20210514 (Red Hat 8.5.0-26)
```

## Latest follow-up

The [fresh native CUDA and complete-suite output](cuda-context-fix-2026-09-11.md#fresh-validation-after-benchmark-publication) records the post-fix rerun after benchmark publication. Earlier results above remain a historical record.
