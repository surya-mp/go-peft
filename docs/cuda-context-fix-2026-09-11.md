# CUDA cuBLAS context failure: diagnosis and fix — 2026-09-11

The [latest test rerun and published benchmark output](#fresh-validation-after-benchmark-publication) are appended below.

## Result

Fixed the intermittent `cuBLAS status 14` failure in the native backend.
The exact benchmark command from [benchmarks.md](benchmarks.md) now passes without
`CUDA_MODULE_LOADING=EAGER`, thread pinning, retries, or a CPU fallback.

| Benchmark | ns/op | B/op | allocs/op |
| --- | ---: | ---: | ---: |
| LoRA | 121,168 | 65,537 | 1 |
| QLoRA | 268,001 | 65,536 | 1 |

These are single-run local measurements of the existing fixtures, including output
transfer, on the RTX 3050 6GB Laptop GPU. Five repetitions of each benchmark also
passed with explicit lazy loading. Performance varies between runs.

## Root cause and evidence

A Go goroutine may enter separate cgo calls on different OS threads. The bridge
selected the CUDA device only when creating the engine. A later cuBLAS call could
therefore execute on a thread without a current CUDA context.
[NVIDIA documents that a cuBLAS handle is tied to the current CUDA context at creation](https://docs.nvidia.com/cuda/cublas/).

A temporary interposition wrapper queried `cuCtxGetCurrent` immediately before
`cublasSgemm_v2`, without setting the context. It caught the failing A projection:

```text
TRACE before SGEMM current_context=(nil) context_status=0
TRACE cuBLAS=14 errno=0 thread=18768 m=16 n=32 k=512 handle=0x6ffb9c000c90 CUDApeek=0:no error sync=0:no error device=0 deviceStatus=0
--- FAIL: BenchmarkLoRAForward-16
    benchmark_test.go:33: lora: A projection: cuda: gemm: cuBLAS status 14
```

The runtime error query and device synchronization after failure both reported
success. A null context did not fail every call, which explains why small tests,
eager loading, and some benchmark repetitions appeared to work. The exact internal
cuBLAS dispatch path that makes the error intermittent was not inspected.

Disabling Go asynchronous preemption or garbage collection did not eliminate the
failure. A standalone C++ diagnostic completed 100 same-thread and 100 worker-thread
handle lifecycles without failures; each lifecycle selected its device before SGEMM.

### Correction to the earlier investigation

The previous report said thread pinning did not help. Its temporary Go overlay used
`/mnt/c/users/.../downloads/projects/...`, while `go list` resolved the package as
`/mnt/c/Users/.../Downloads/Projects/...`. The overlay did not apply.

The corrected overlay used the path from `go list` and logged
`OS-thread pin diagnostic active`, proving it was active. All five repetitions of
both benchmarks then passed. This was a diagnostic control; no thread-pinning change
was retained in the benchmark or production Go code.

## Minimal implementation change

In [gopeft_cuda.cu](../backends/cuda/gopeft_cuda.cu), the native context now remembers
the selected device. `peft_cuda_gemm`, `peft_cuda_axpy`, and context destruction call
`cudaSetDevice(value->device)` before using the cuBLAS handle. Device selection and
cuBLAS execution occur within the same C call, so Go cannot move between OS threads
between those operations. Failure to select the device returns a CUDA error.

No retry hides the original error; no global module-loading environment is changed.
The numerical kernels and benchmark implementation are unchanged.

[TestGemmOnDifferentOSThread](../backends/cuda/cuda_test.go) holds the creating thread
while another locked OS thread performs GEMM and AXPY using its engine. It checks a
nonzero `32 x 512` result against an identity matrix reference and rejects nonfinite
values. The focused test also passed before the fix in isolated runs, consistent
with the intermittent failure; the captured failing trace and repeated benchmark
controls provide the before/after reproduction evidence.

## Validation after the fix

Environment: Ubuntu 26.04 / WSL2; NVIDIA GeForce RTX 3050 6GB Laptop GPU; driver
591.74; user-local CUDA 13.2, nvcc 13.2.86, cuBLAS 13.4.1; Go 1.26.5 linux/amd64;
GCC 15.2.0. Native code built for `sm_86` using the previously installed environment.

```sh
. /home/surya-mp/.local/lib/go-peft-toolchain/cuda-env.sh
sh scripts/build_cuda.sh
CUDA_MODULE_LOADING=LAZY go test -count=1 -v -tags cuda ./backends/cuda
CUDA_MODULE_LOADING=LAZY go test -count=5 -run '^$' -tags cuda \
  -bench 'Benchmark(LoRA|QLoRA)Forward$' -benchmem ./backends/cuda
CUDA_MODULE_LOADING=LAZY go test -count=1 -v -tags cuda ./...
CUDA_MODULE_LOADING=LAZY go run -tags cuda ./examples/lora/cuda
CUDA_MODULE_LOADING=LAZY go run -tags cuda ./examples/qlora/cuda
unset CUDA_MODULE_LOADING LD_PRELOAD
go test -run '^$' -tags cuda -bench 'Benchmark(LoRA|QLoRA)Forward$' -benchmem ./backends/cuda
```

All commands exited 0. Results:

- Native suite: seven tests passed, no skips.
- Full suite: 62 top-level tests and three fuzz seed suites passed, no failures/skips.
- Examples: `[1 2]` and `[5.637332 14.675482]`.
- Benchmarks: five repetitions of each passed with lazy loading.
- Five further fresh-process LoRA benchmarks passed with the context diagnostic
  wrapper: no cuBLAS errors and no missing current CUDA context before SGEMM.
- The exact documented benchmark command passed without environment workarounds.

The native CUDA-tagged implementation and real GPU were used throughout. No observed
numerical failures remain. This validates the reported single-GPU failure; it does
not establish general multi-GPU or concurrent-engine thread safety.

## Changed files and preservation

- `backends/cuda/gopeft_cuda.cu`: retain device and select it before cuBLAS calls.
- `backends/cuda/cuda_test.go`: cross-OS-thread numerical regression coverage.
- This report: diagnosis, evidence, and final validation.
- `cuda-revalidation-benchmarks-2026-09-11.md`: resolution/correction notice only.
- Rebuilt generated `backends/cuda/gopeft_cuda.o` and `backends/cuda/libgopeftcuda.a`.

SHA256 comparison confirmed every other pre-existing tracked file unchanged.
Existing source CRLF line endings were preserved. No installation, module, or
benchmark implementation changes were made. Temporary instrumentation, overlays,
and the standalone diagnostic live under `/tmp/go-peft-cuda-debug`.

## Captured logs

### context-fresh-1.log

```text
goos: linux
goarch: amd64
pkg: github.com/surya-mp/go-peft/backends/cuda
cpu: 13th Gen Intel(R) Core(TM) i5-13450HX
BenchmarkLoRAForward-16    	TRACE before SGEMM current_context=(nil) context_status=0
TRACE cuBLAS=14 errno=0 thread=18768 m=16 n=32 k=512 handle=0x6ffb9c000c90 CUDApeek=0:no error sync=0:no error device=0 deviceStatus=0
--- FAIL: BenchmarkLoRAForward-16
    benchmark_test.go:33: lora: A projection: cuda: gemm: cuBLAS status 14
FAIL
```

### pinned-benchmarks.log

```text
goos: linux
goarch: amd64
pkg: github.com/surya-mp/go-peft/backends/cuda
cpu: 13th Gen Intel(R) Core(TM) i5-13450HX
BenchmarkLoRAForward-16     	   10000	    117259 ns/op	   65537 B/op	       1 allocs/op
--- BENCH: BenchmarkLoRAForward-16
    benchmark_test.go:65: OS-thread pin diagnostic active
    benchmark_test.go:16: OS-thread pin diagnostic active
    benchmark_test.go:16: OS-thread pin diagnostic active
BenchmarkLoRAForward-16     	   10000	    114532 ns/op	   65537 B/op	       1 allocs/op
--- BENCH: BenchmarkLoRAForward-16
    benchmark_test.go:65: OS-thread pin diagnostic active
    benchmark_test.go:16: OS-thread pin diagnostic active
    benchmark_test.go:16: OS-thread pin diagnostic active
BenchmarkLoRAForward-16     	    8976	    112933 ns/op	   65536 B/op	       1 allocs/op
--- BENCH: BenchmarkLoRAForward-16
    benchmark_test.go:65: OS-thread pin diagnostic active
    benchmark_test.go:16: OS-thread pin diagnostic active
    benchmark_test.go:16: OS-thread pin diagnostic active
BenchmarkLoRAForward-16     	    9252	    116687 ns/op	   65536 B/op	       1 allocs/op
--- BENCH: BenchmarkLoRAForward-16
    benchmark_test.go:65: OS-thread pin diagnostic active
    benchmark_test.go:16: OS-thread pin diagnostic active
    benchmark_test.go:16: OS-thread pin diagnostic active
BenchmarkLoRAForward-16     	   10789	    116509 ns/op	   65536 B/op	       1 allocs/op
--- BENCH: BenchmarkLoRAForward-16
    benchmark_test.go:65: OS-thread pin diagnostic active
    benchmark_test.go:16: OS-thread pin diagnostic active
    benchmark_test.go:16: OS-thread pin diagnostic active
    benchmark_test.go:16: OS-thread pin diagnostic active
BenchmarkQLoRAForward-16    	    3675	    275952 ns/op	   65536 B/op	       1 allocs/op
--- BENCH: BenchmarkQLoRAForward-16
    benchmark_test.go:65: OS-thread pin diagnostic active
    benchmark_test.go:38: OS-thread pin diagnostic active
    benchmark_test.go:38: OS-thread pin diagnostic active
BenchmarkQLoRAForward-16    	    4365	    269813 ns/op	   65536 B/op	       1 allocs/op
--- BENCH: BenchmarkQLoRAForward-16
    benchmark_test.go:65: OS-thread pin diagnostic active
    benchmark_test.go:38: OS-thread pin diagnostic active
    benchmark_test.go:38: OS-thread pin diagnostic active
BenchmarkQLoRAForward-16    	    4430	    275127 ns/op	   65536 B/op	       1 allocs/op
--- BENCH: BenchmarkQLoRAForward-16
    benchmark_test.go:65: OS-thread pin diagnostic active
    benchmark_test.go:38: OS-thread pin diagnostic active
    benchmark_test.go:38: OS-thread pin diagnostic active
BenchmarkQLoRAForward-16    	    4533	    268405 ns/op	   65536 B/op	       1 allocs/op
--- BENCH: BenchmarkQLoRAForward-16
    benchmark_test.go:65: OS-thread pin diagnostic active
    benchmark_test.go:38: OS-thread pin diagnostic active
    benchmark_test.go:38: OS-thread pin diagnostic active
BenchmarkQLoRAForward-16    	    4317	    262594 ns/op	   65536 B/op	       1 allocs/op
--- BENCH: BenchmarkQLoRAForward-16
    benchmark_test.go:65: OS-thread pin diagnostic active
    benchmark_test.go:38: OS-thread pin diagnostic active
    benchmark_test.go:38: OS-thread pin diagnostic active
PASS
ok  	github.com/surya-mp/go-peft/backends/cuda	13.005s
```

### native-fixed.log

```text
=== RUN   TestLoRAForward
--- PASS: TestLoRAForward (0.37s)
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
=== RUN   TestGemmOnDifferentOSThread
--- PASS: TestGemmOnDifferentOSThread (0.01s)
PASS
ok  	github.com/surya-mp/go-peft/backends/cuda	0.518s
```

### benchmarks-fixed.log

```text
goos: linux
goarch: amd64
pkg: github.com/surya-mp/go-peft/backends/cuda
cpu: 13th Gen Intel(R) Core(TM) i5-13450HX
BenchmarkLoRAForward-16     	    9787	    121116 ns/op	   65536 B/op	       1 allocs/op
BenchmarkLoRAForward-16     	   10230	    119028 ns/op	   65536 B/op	       1 allocs/op
BenchmarkLoRAForward-16     	   10015	    117196 ns/op	   65536 B/op	       1 allocs/op
BenchmarkLoRAForward-16     	    9087	    124392 ns/op	   65536 B/op	       1 allocs/op
BenchmarkLoRAForward-16     	    9799	    122123 ns/op	   65536 B/op	       1 allocs/op
BenchmarkQLoRAForward-16    	    4357	    269673 ns/op	   65536 B/op	       1 allocs/op
BenchmarkQLoRAForward-16    	    4222	    266070 ns/op	   65536 B/op	       1 allocs/op
BenchmarkQLoRAForward-16    	    4617	    269889 ns/op	   65536 B/op	       1 allocs/op
BenchmarkQLoRAForward-16    	    4364	    266767 ns/op	   65536 B/op	       1 allocs/op
BenchmarkQLoRAForward-16    	    4141	    266150 ns/op	   65536 B/op	       1 allocs/op
PASS
ok  	github.com/surya-mp/go-peft/backends/cuda	16.067s
```

### full-suite-fixed.log

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
ok  	github.com/surya-mp/go-peft/backend	0.006s
=== RUN   TestLoRAForward
--- PASS: TestLoRAForward (0.22s)
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
=== RUN   TestGemmOnDifferentOSThread
--- PASS: TestGemmOnDifferentOSThread (0.01s)
PASS
ok  	github.com/surya-mp/go-peft/backends/cuda	0.368s
=== RUN   TestTargetsAndPlan
--- PASS: TestTargetsAndPlan (0.00s)
=== RUN   TestInspectCommandsRequirePaths
--- PASS: TestInspectCommandsRequirePaths (0.00s)
PASS
ok  	github.com/surya-mp/go-peft/cmd/go-peft	0.031s
?   	github.com/surya-mp/go-peft/examples/lora/cpu	[no test files]
?   	github.com/surya-mp/go-peft/examples/lora/cuda	[no test files]
?   	github.com/surya-mp/go-peft/examples/lora/huggingface	[no test files]
?   	github.com/surya-mp/go-peft/examples/lora/injection	[no test files]
?   	github.com/surya-mp/go-peft/examples/qlora/cpu	[no test files]
?   	github.com/surya-mp/go-peft/examples/qlora/cuda	[no test files]
?   	github.com/surya-mp/go-peft/examples/qlora/huggingface	[no test files]
?   	github.com/surya-mp/go-peft/examples/qlora/injection	[no test files]
=== RUN   TestRead
--- PASS: TestRead (0.00s)
=== RUN   TestReadRejectsUnsupportedVersion
--- PASS: TestReadRejectsUnsupportedVersion (0.00s)
=== RUN   TestReadRejectsMissingArchitectureAndInvalidTensorOffset
=== RUN   TestReadRejectsMissingArchitectureAndInvalidTensorOffset/architecture
=== RUN   TestReadRejectsMissingArchitectureAndInvalidTensorOffset/offset
--- PASS: TestReadRejectsMissingArchitectureAndInvalidTensorOffset (0.00s)
    --- PASS: TestReadRejectsMissingArchitectureAndInvalidTensorOffset/architecture (0.00s)
    --- PASS: TestReadRejectsMissingArchitectureAndInvalidTensorOffset/offset (0.00s)
=== RUN   TestSkipValueScalarsAndNestedArray
--- PASS: TestSkipValueScalarsAndNestedArray (0.00s)
=== RUN   TestTypeString
--- PASS: TestTypeString (0.00s)
=== RUN   FuzzRead
=== RUN   FuzzRead/seed#0
=== RUN   FuzzRead/seed#1
--- PASS: FuzzRead (0.00s)
    --- PASS: FuzzRead/seed#0 (0.00s)
    --- PASS: FuzzRead/seed#1 (0.00s)
PASS
ok  	github.com/surya-mp/go-peft/format/gguf	0.030s
=== RUN   TestPEFTGoldenAdapterFixture
--- PASS: TestPEFTGoldenAdapterFixture (0.01s)
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
ok  	github.com/surya-mp/go-peft/format/huggingface	0.014s
=== RUN   TestLoadShardedModel
--- PASS: TestLoadShardedModel (0.00s)
=== RUN   TestInspectRejectsEscapingShard
--- PASS: TestInspectRejectsEscapingShard (0.00s)
=== RUN   TestLoadSingleModel
--- PASS: TestLoadSingleModel (0.00s)
=== RUN   TestLoadRejectsMismatchedIndexAndCallbackError
--- PASS: TestLoadRejectsMismatchedIndexAndCallbackError (0.00s)
=== RUN   FuzzInspectManifest
=== RUN   FuzzInspectManifest/seed#0
=== RUN   FuzzInspectManifest/seed#1
--- PASS: FuzzInspectManifest (0.00s)
    --- PASS: FuzzInspectManifest/seed#0 (0.00s)
    --- PASS: FuzzInspectManifest/seed#1 (0.00s)
PASS
ok  	github.com/surya-mp/go-peft/format/huggingface/model	0.033s
=== RUN   TestRoundTripIsDeterministic
--- PASS: TestRoundTripIsDeterministic (0.00s)
=== RUN   TestVisitDecodesFloat16AndBFloat16
--- PASS: TestVisitDecodesFloat16AndBFloat16 (0.00s)
=== RUN   TestVisitSkipsUnwantedTensor
--- PASS: TestVisitSkipsUnwantedTensor (0.00s)
=== RUN   TestVisitPropagatesVisitorAndRejectsUnsupportedDType
--- PASS: TestVisitPropagatesVisitorAndRejectsUnsupportedDType (0.00s)
=== RUN   TestReadRejectsTrailingData
--- PASS: TestReadRejectsTrailingData (0.00s)
=== RUN   FuzzReadAndVisit
=== RUN   FuzzReadAndVisit/seed#0
=== RUN   FuzzReadAndVisit/seed#1
--- PASS: FuzzReadAndVisit (0.00s)
    --- PASS: FuzzReadAndVisit/seed#0 (0.00s)
    --- PASS: FuzzReadAndVisit/seed#1 (0.00s)
PASS
ok  	github.com/surya-mp/go-peft/format/safetensors	0.030s
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
ok  	github.com/surya-mp/go-peft/lora	0.030s
=== RUN   TestParameterOwnsShape
--- PASS: TestParameterOwnsShape (0.00s)
=== RUN   TestParameterClone
--- PASS: TestParameterClone (0.00s)
=== RUN   TestParameterCount
--- PASS: TestParameterCount (0.00s)
PASS
ok  	github.com/surya-mp/go-peft/peft	0.030s
=== RUN   TestPlanLlamaAllLinear
--- PASS: TestPlanLlamaAllLinear (0.00s)
=== RUN   TestPlanPhi3AndFailures
--- PASS: TestPlanPhi3AndFailures (0.00s)
PASS
ok  	github.com/surya-mp/go-peft/profiles	0.030s
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
ok  	github.com/surya-mp/go-peft/qlora	0.029s
=== RUN   TestQuantize
--- PASS: TestQuantize (0.00s)
=== RUN   TestQuantizationErrorIsBounded
--- PASS: TestQuantizationErrorIsBounded (0.00s)
=== RUN   TestQuantizedStorageCopiesBuffers
--- PASS: TestQuantizedStorageCopiesBuffers (0.00s)
PASS
ok  	github.com/surya-mp/go-peft/quantization/int4	0.031s
=== RUN   TestQuantizeCodebookAndDot
--- PASS: TestQuantizeCodebookAndDot (0.00s)
=== RUN   TestDoubleQuantizationReducesScaleStorage
--- PASS: TestDoubleQuantizationReducesScaleStorage (0.00s)
=== RUN   TestQuantizedStorageCopiesDoubleQuantBuffers
--- PASS: TestQuantizedStorageCopiesDoubleQuantBuffers (0.00s)
PASS
ok  	github.com/surya-mp/go-peft/quantization/nf4	0.029s
=== RUN   TestRunnerAccumulatesClipsAndCheckpoints
--- PASS: TestRunnerAccumulatesClipsAndCheckpoints (0.00s)
PASS
ok  	github.com/surya-mp/go-peft/trainer	0.029s
```

### lora-fixed.log

```text
[1 2]
```

### qlora-fixed.log

```text
[5.637332 14.675482]
```

### documented-benchmarks-fixed.log

```text
goos: linux
goarch: amd64
pkg: github.com/surya-mp/go-peft/backends/cuda
cpu: 13th Gen Intel(R) Core(TM) i5-13450HX
BenchmarkLoRAForward-16     	    8352	    121168 ns/op	   65537 B/op	       1 allocs/op
BenchmarkQLoRAForward-16    	    3829	    268001 ns/op	   65536 B/op	       1 allocs/op
PASS
ok  	github.com/surya-mp/go-peft/backends/cuda	2.419s
```

### fixed-fresh-0.log

```text
goos: linux
goarch: amd64
pkg: github.com/surya-mp/go-peft/backends/cuda
cpu: 13th Gen Intel(R) Core(TM) i5-13450HX
BenchmarkLoRAForward-16    	   10048	    119840 ns/op	   65536 B/op	       1 allocs/op
PASS
```

### fixed-fresh-1.log

```text
goos: linux
goarch: amd64
pkg: github.com/surya-mp/go-peft/backends/cuda
cpu: 13th Gen Intel(R) Core(TM) i5-13450HX
BenchmarkLoRAForward-16    	   10000	    121248 ns/op	   65536 B/op	       1 allocs/op
PASS
```

### fixed-fresh-2.log

```text
goos: linux
goarch: amd64
pkg: github.com/surya-mp/go-peft/backends/cuda
cpu: 13th Gen Intel(R) Core(TM) i5-13450HX
BenchmarkLoRAForward-16    	    9591	    121859 ns/op	   65536 B/op	       1 allocs/op
PASS
```

### fixed-fresh-3.log

```text
goos: linux
goarch: amd64
pkg: github.com/surya-mp/go-peft/backends/cuda
cpu: 13th Gen Intel(R) Core(TM) i5-13450HX
BenchmarkLoRAForward-16    	    9836	    124028 ns/op	   65536 B/op	       1 allocs/op
PASS
```

### fixed-fresh-4.log

```text
goos: linux
goarch: amd64
pkg: github.com/surya-mp/go-peft/backends/cuda
cpu: 13th Gen Intel(R) Core(TM) i5-13450HX
BenchmarkLoRAForward-16    	    8398	    124631 ns/op	   65537 B/op	       1 allocs/op
PASS
```

## Fresh validation after benchmark publication

Rerun on 2026-09-11 America/Chicago (2026-09-12 UTC), after publishing the
131,399 ns/op LoRA and 267,842 ns/op QLoRA measurements in
[benchmarks.md](benchmarks.md). The earlier measurements in this report describe
the fix-validation run; the publication run's output is also retained below.

The native library was rebuilt successfully. Both test commands exited 0:
seven native tests passed, and the complete CUDA-tagged suite passed 62 top-level
tests plus three fuzz seed suites across 14 tested packages. There were no failures,
skips, or cached test results. The cross-OS-thread regression passed.

The same user-local CUDA 13.2 / Go 1.26.5 environment and RTX 3050 GPU were used.
No eager-loading override or diagnostic interposition library was enabled.
This rerun validates tests; the benchmark output below is from the preceding
publication run, not a newly measured benchmark in this test-only rerun.

```sh
. /home/surya-mp/.local/lib/go-peft-toolchain/cuda-env.sh
unset CUDA_MODULE_LOADING LD_PRELOAD
sh scripts/build_cuda.sh
go test -count=1 -v -tags cuda ./backends/cuda
go test -count=1 -v -tags cuda ./...
```

### Fresh native CUDA test output

```text
=== RUN   TestLoRAForward
--- PASS: TestLoRAForward (0.58s)
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
=== RUN   TestGemmOnDifferentOSThread
--- PASS: TestGemmOnDifferentOSThread (0.02s)
PASS
ok  	github.com/surya-mp/go-peft/backends/cuda	0.901s
```

### Fresh complete CUDA-tagged suite output

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
ok  	github.com/surya-mp/go-peft/backend	0.006s
=== RUN   TestLoRAForward
--- PASS: TestLoRAForward (0.33s)
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
=== RUN   TestGemmOnDifferentOSThread
--- PASS: TestGemmOnDifferentOSThread (0.01s)
PASS
ok  	github.com/surya-mp/go-peft/backends/cuda	0.486s
=== RUN   TestTargetsAndPlan
--- PASS: TestTargetsAndPlan (0.00s)
=== RUN   TestInspectCommandsRequirePaths
--- PASS: TestInspectCommandsRequirePaths (0.00s)
PASS
ok  	github.com/surya-mp/go-peft/cmd/go-peft	0.030s
?   	github.com/surya-mp/go-peft/examples/lora/cpu	[no test files]
?   	github.com/surya-mp/go-peft/examples/lora/cuda	[no test files]
?   	github.com/surya-mp/go-peft/examples/lora/huggingface	[no test files]
?   	github.com/surya-mp/go-peft/examples/lora/injection	[no test files]
?   	github.com/surya-mp/go-peft/examples/qlora/cpu	[no test files]
?   	github.com/surya-mp/go-peft/examples/qlora/cuda	[no test files]
?   	github.com/surya-mp/go-peft/examples/qlora/huggingface	[no test files]
?   	github.com/surya-mp/go-peft/examples/qlora/injection	[no test files]
=== RUN   TestRead
--- PASS: TestRead (0.00s)
=== RUN   TestReadRejectsUnsupportedVersion
--- PASS: TestReadRejectsUnsupportedVersion (0.00s)
=== RUN   TestReadRejectsMissingArchitectureAndInvalidTensorOffset
=== RUN   TestReadRejectsMissingArchitectureAndInvalidTensorOffset/architecture
=== RUN   TestReadRejectsMissingArchitectureAndInvalidTensorOffset/offset
--- PASS: TestReadRejectsMissingArchitectureAndInvalidTensorOffset (0.00s)
    --- PASS: TestReadRejectsMissingArchitectureAndInvalidTensorOffset/architecture (0.00s)
    --- PASS: TestReadRejectsMissingArchitectureAndInvalidTensorOffset/offset (0.00s)
=== RUN   TestSkipValueScalarsAndNestedArray
--- PASS: TestSkipValueScalarsAndNestedArray (0.00s)
=== RUN   TestTypeString
--- PASS: TestTypeString (0.00s)
=== RUN   FuzzRead
=== RUN   FuzzRead/seed#0
=== RUN   FuzzRead/seed#1
--- PASS: FuzzRead (0.00s)
    --- PASS: FuzzRead/seed#0 (0.00s)
    --- PASS: FuzzRead/seed#1 (0.00s)
PASS
ok  	github.com/surya-mp/go-peft/format/gguf	0.030s
=== RUN   TestPEFTGoldenAdapterFixture
--- PASS: TestPEFTGoldenAdapterFixture (0.01s)
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
ok  	github.com/surya-mp/go-peft/format/huggingface	0.021s
=== RUN   TestLoadShardedModel
--- PASS: TestLoadShardedModel (0.00s)
=== RUN   TestInspectRejectsEscapingShard
--- PASS: TestInspectRejectsEscapingShard (0.00s)
=== RUN   TestLoadSingleModel
--- PASS: TestLoadSingleModel (0.00s)
=== RUN   TestLoadRejectsMismatchedIndexAndCallbackError
--- PASS: TestLoadRejectsMismatchedIndexAndCallbackError (0.00s)
=== RUN   FuzzInspectManifest
=== RUN   FuzzInspectManifest/seed#0
=== RUN   FuzzInspectManifest/seed#1
--- PASS: FuzzInspectManifest (0.00s)
    --- PASS: FuzzInspectManifest/seed#0 (0.00s)
    --- PASS: FuzzInspectManifest/seed#1 (0.00s)
PASS
ok  	github.com/surya-mp/go-peft/format/huggingface/model	0.030s
=== RUN   TestRoundTripIsDeterministic
--- PASS: TestRoundTripIsDeterministic (0.00s)
=== RUN   TestVisitDecodesFloat16AndBFloat16
--- PASS: TestVisitDecodesFloat16AndBFloat16 (0.00s)
=== RUN   TestVisitSkipsUnwantedTensor
--- PASS: TestVisitSkipsUnwantedTensor (0.00s)
=== RUN   TestVisitPropagatesVisitorAndRejectsUnsupportedDType
--- PASS: TestVisitPropagatesVisitorAndRejectsUnsupportedDType (0.00s)
=== RUN   TestReadRejectsTrailingData
--- PASS: TestReadRejectsTrailingData (0.00s)
=== RUN   FuzzReadAndVisit
=== RUN   FuzzReadAndVisit/seed#0
=== RUN   FuzzReadAndVisit/seed#1
--- PASS: FuzzReadAndVisit (0.00s)
    --- PASS: FuzzReadAndVisit/seed#0 (0.00s)
    --- PASS: FuzzReadAndVisit/seed#1 (0.00s)
PASS
ok  	github.com/surya-mp/go-peft/format/safetensors	0.029s
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
ok  	github.com/surya-mp/go-peft/lora	0.029s
=== RUN   TestParameterOwnsShape
--- PASS: TestParameterOwnsShape (0.00s)
=== RUN   TestParameterClone
--- PASS: TestParameterClone (0.00s)
=== RUN   TestParameterCount
--- PASS: TestParameterCount (0.00s)
PASS
ok  	github.com/surya-mp/go-peft/peft	0.029s
=== RUN   TestPlanLlamaAllLinear
--- PASS: TestPlanLlamaAllLinear (0.00s)
=== RUN   TestPlanPhi3AndFailures
--- PASS: TestPlanPhi3AndFailures (0.00s)
PASS
ok  	github.com/surya-mp/go-peft/profiles	0.029s
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
ok  	github.com/surya-mp/go-peft/qlora	0.029s
=== RUN   TestQuantize
--- PASS: TestQuantize (0.00s)
=== RUN   TestQuantizationErrorIsBounded
--- PASS: TestQuantizationErrorIsBounded (0.00s)
=== RUN   TestQuantizedStorageCopiesBuffers
--- PASS: TestQuantizedStorageCopiesBuffers (0.00s)
PASS
ok  	github.com/surya-mp/go-peft/quantization/int4	0.031s
=== RUN   TestQuantizeCodebookAndDot
--- PASS: TestQuantizeCodebookAndDot (0.00s)
=== RUN   TestDoubleQuantizationReducesScaleStorage
--- PASS: TestDoubleQuantizationReducesScaleStorage (0.00s)
=== RUN   TestQuantizedStorageCopiesDoubleQuantBuffers
--- PASS: TestQuantizedStorageCopiesDoubleQuantBuffers (0.00s)
PASS
ok  	github.com/surya-mp/go-peft/quantization/nf4	0.029s
=== RUN   TestRunnerAccumulatesClipsAndCheckpoints
--- PASS: TestRunnerAccumulatesClipsAndCheckpoints (0.00s)
PASS
ok  	github.com/surya-mp/go-peft/trainer	0.029s
```

### Benchmark publication output

Command: `go test -run '^$' -tags cuda -bench 'Benchmark(LoRA|QLoRA)Forward$' -benchmem ./backends/cuda`

```text
goos: linux
goarch: amd64
pkg: github.com/surya-mp/go-peft/backends/cuda
cpu: 13th Gen Intel(R) Core(TM) i5-13450HX
BenchmarkLoRAForward-16     	    9553	    131399 ns/op	   65536 B/op	       1 allocs/op
BenchmarkQLoRAForward-16    	    3969	    267842 ns/op	   65536 B/op	       1 allocs/op
PASS
ok  	github.com/surya-mp/go-peft/backends/cuda	4.091s
```
