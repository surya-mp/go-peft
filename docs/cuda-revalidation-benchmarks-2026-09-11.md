# Native CUDA revalidation and benchmarks — 2026-09-11

> **Resolved by the follow-up investigation:** see [CUDA context fix](cuda-context-fix-2026-09-11.md).
> The default benchmark now passes. The earlier thread-pinning overlay did not apply
> because its path casing differed from Go's package path; the corrected experiment passed.
> The results below are retained as the historical pre-fix record.

## Outcome

The updated worktree passed the native build, all six native CUDA tests, the complete
CUDA-tagged suite (61 top-level tests plus three fuzz seed suites across 14 tested
packages), and both GPU examples. There were no test skips or numerical failures.

**The default documented benchmark command failed intermittently in LoRA with
`cuBLAS status 14`. This is not an unconditional benchmark validation pass.**
Three repetitions of both benchmarks passed with `CUDA_MODULE_LOADING=EAGER`.
No source or benchmark implementation changes were made during this rerun.

This run validates the modified worktree based on commit
`b1ec47ad36406babc9d8f48a62816aefa3667a95`, not a clean or final merged commit.
The earlier [validation report](cuda-validation-2026-09-11.md) is retained.

## Hardware and environment

- GPU 0: NVIDIA GeForce RTX 3050 6GB Laptop GPU, 6144 MiB.
- Windows NVIDIA driver: 591.74; WSL NVIDIA-SMI: 590.52.01.
- CUDA toolkit used: 13.2, nvcc V13.2.86; cuBLAS trace version: 13.4.1.
- Go: go1.26.5 linux/amd64; GCC: Ubuntu 15.2.0-16ubuntu1.
- Host CPU reported by Go benchmarks: 13th Gen Intel Core i5-13450HX.
- Ubuntu 26.04 on WSL2; native artifacts compiled for `sm_86`.

The driver reports CUDA capability 13.1. As in the earlier run, compilation used the
separate user-local CUDA 13.2 toolkit because system CUDA 13.1 headers conflict with
this machine's glibc. No installation changes were made in this rerun.

Source `/home/surya-mp/.local/lib/go-peft-toolchain/cuda-env.sh` on this machine.
The environment file contains:

```bash
export CUDA_HOME=/home/surya-mp/.local/lib/go-peft-toolchain/cuda-13.2-packages/root/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:/home/surya-mp/.local/lib/go-peft-toolchain/go/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CGO_ENABLED=1
export CGO_LDFLAGS="-L$CUDA_HOME/lib64"
export GOFLAGS=-mod=readonly
export NVCC_PREPEND_FLAGS=-arch=sm_86
```

Adjust the installation paths for another host. These are validation environment
settings, not new repository defaults. Tests and benchmarks ran with real GPU access
outside the read-only sandbox. `GOFLAGS=-mod=readonly` preserved module files.

## Commands and results

| Command | Result |
| --- | --- |
| `git status --short` | Captured before work; existing changes preserved |
| `nvidia-smi`, `nvcc --version`, `go version` | Succeeded; versions above |
| `sh scripts/build_cuda.sh` | Exit 0; rebuilt native object and archive |
| `ls`, `file`, `ar t`, `nm` on native artifacts | Verified ELF object, archive contents and CUDA/cuBLAS symbols |
| `go env GOOS GOARCH CGO_ENABLED CC` | `linux`, `amd64`, `1`, `gcc` |
| `go list -tags cuda ... ./backends/cuda` | Selected `cuda.go` and `train.go`; excluded both stubs |
| `go test -count=1 -v -tags cuda ./backends/cuda` | Exit 0; six passes, no skips |
| `go test -count=1 -v -tags cuda ./...` | Exit 0; 61 tests and three fuzz seed suites passed; no skips |
| `go run -tags cuda ./examples/lora/cuda` | Exit 0; `[1 2]` |
| `go run -tags cuda ./examples/qlora/cuda` | Exit 0; `[5.637332 14.675482]` |
| CUDA benchmark command from `docs/benchmarks.md` | Exit 1; LoRA failed, QLoRA measured 271,375 ns/op |

The exact documented benchmark command was:

```sh
go test -run '^$' -tags cuda -bench 'Benchmark(LoRA|QLoRA)Forward$' -benchmem ./backends/cuda
```

## Benchmark investigation and measurements

The initial complete error was:

```text
BenchmarkLoRAForward-16     	--- FAIL: BenchmarkLoRAForward-16
    benchmark_test.go:33: lora: base linear: cuda: gemm: cuBLAS status 14
```

[NVIDIA documents status 14 as an internal cuBLAS error](https://docs.nvidia.com/cuda/cublas/).
The error alone does not establish its cause. Diagnostics produced these results:

1. LoRA with `-benchtime=1x` and cuBLAS information/error logging passed. This single
   iteration is diagnostic evidence, not a comparable steady-state baseline.
2. Both benchmarks with `-count=3` and cuBLAS error logging reproduced a LoRA failure.
   No separate error-log file was emitted; the Go error output is captured below.
3. Using installed system CUDA 13.1 runtime libraries also produced a LoRA failure,
   this time during the A projection. That command reported exit 0 and final `PASS`
   despite an embedded `--- FAIL`; this report treats that run as failed.
4. A temporary Go overlay pinned each benchmark goroutine to its OS thread. LoRA
   still failed. The overlay existed only outside the repository and was not used
   for the final measurements.
5. `CUDA_MODULE_LOADING=EAGER` with three repetitions passed both benchmarks:

```sh
CUDA_MODULE_LOADING=EAGER go test -count=3 -run '^$' -tags cuda \
  -bench 'Benchmark(LoRA|QLoRA)Forward$' -benchmem ./backends/cuda
```

These observations suggest sensitivity to CUDA module loading, but do not prove the
root cause or guarantee that eager loading resolves every occurrence. No speculative
backend fix was applied. The default benchmark failure remains an open limitation.

The following measurements are **with eager module loading**, not the default command:

| Benchmark | Range (ns/op), 3 runs | Median (ns/op) | B/op, each run | allocs/op |
| --- | ---: | ---: | --- | ---: |
| LoRA | 119,014–122,744 | 120,130 | 65537, 65536, 65536 | 1 |
| QLoRA | 265,189–274,965 | 266,769 | 65536, 65536, 65538 | 1 |

Both use the existing benchmark implementation: `32 x 512` input, `512 x 512` base
weight, rank 16, alpha 32. QLoRA uses NF4, double quantization, block size 64 and
scale block size 256. Inputs and base weights are zero-filled by the fixture.
The implementation warms up once, reuses output/rank workspaces, calls `ForwardInto`,
and copies output to the host with `EncodeFloat32` on each measured iteration.
Thus timings include GPU completion and transfer; Go allocation counts do not measure
CUDA device allocations. Logging was not enabled for the final eager-loading run.
These are local measurements, not a cross-machine comparison with the Apple M1 rows.
`docs/benchmarks.md` remains unchanged because its publication note requests results
from the final merged worktree, whereas this worktree has existing modifications.

## Evidence of native execution and numerical checks

`go list` selected `cuda.go`/`train.go`, with `cuda_stub.go` and `train_stub.go` ignored.
The archive contains `peft_cuda_new`, `peft_cuda_gemm`, and
`peft_cuda_quantized_linear`, and references `__cudaLaunchKernel` and `cublasSgemm_v2`.

Both examples initialize through `cuda.New` / `NewDevice(0)`, which calls
`cudaGetDeviceCount`, `cudaSetDevice`, and `cublasCreate`. Their cuBLAS traces show
handle creation and SGEMM calls on GPU 0: three SGEMMs in LoRA, two in QLoRA.
The QLoRA base projection calls the native `quantized_linear<<<...>>>` kernel;
its result is copied from device memory with `cudaMemcpyDeviceToHost`.
The tests also exercise int4/NF4, double quantization, training, and dropout kernels.

Examples stop on initialization failure and supply the CUDA engine to the runtime;
`DisableCUDA` is not set. No CPU fallback was used. CPU reference calculations and
host-side setup/loss aggregation are part of the implementation, not a replacement
for CUDA matrix products. Native forward comparisons passed the existing `1e-4`
tolerance and training checks passed. The zero-input benchmarks do not independently
validate numerical accuracy; their GPU execution and transfer did succeed in the
reported eager-loading run.

## Worktree preservation

SHA256 comparison against the start-of-rerun baseline verified every pre-existing
tracked file unchanged, including `go.mod`, `go.sum`, and `docs/benchmarks.md`.
The rerun rebuilt `backends/cuda/gopeft_cuda.o` and `backends/cuda/libgopeftcuda.a`
and added this report. No reset, clean, source fix, or unrelated refactor was performed.

## Captured output

The following logs are embedded to avoid dependencies on temporary files.
Example runs enabled `CUBLAS_LOGINFO_DBG=1` and wrote `CUBLAS_LOGDEST_DBG` to their
respective trace files. The build emitted no output and exited successfully.

### environment.log

```text
NVIDIA GeForce RTX 3050 6GB Laptop GPU, 591.74, 6144 MiB
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Fri_May_08_10:53:34_AM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.86
Build cuda_13.2.r13.2/compiler.37953736_0
go version go1.26.5 linux/amd64
gcc (Ubuntu 15.2.0-16ubuntu1) 15.2.0
Copyright (C) 2025 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

b1ec47ad36406babc9d8f48a62816aefa3667a95
-rwxrwxrwx 1 surya-mp surya-mp 76K Sep 11 18:36 backends/cuda/gopeft_cuda.o
-rwxrwxrwx 1 surya-mp surya-mp 78K Sep 11 18:36 backends/cuda/libgopeftcuda.a
backends/cuda/gopeft_cuda.o: ELF 64-bit LSB relocatable, x86-64, version 1 (SYSV), not stripped
gopeft_cuda.o
                 U __cudaLaunchKernel
                 U cublasSgemm_v2
0000000000000c60 T peft_cuda_gemm
00000000000008c0 T peft_cuda_new
00000000000018c0 T peft_cuda_quantized_linear
linux
amd64
1
gcc
GoFiles=[train.go] CgoFiles=[cuda.go] IgnoredGoFiles=[cuda_stub.go train_stub.go] TestGoFiles=[benchmark_test.go cuda_test.go]
```

### build.log

No output; exit 0.

### native-tests.log

```text
=== RUN   TestLoRAForward
--- PASS: TestLoRAForward (0.48s)
=== RUN   TestQLoRAQuantizedLinear
--- PASS: TestQLoRAQuantizedLinear (0.00s)
=== RUN   TestAdapterTraining
--- PASS: TestAdapterTraining (0.01s)
=== RUN   TestAdapterTrainingDropout
--- PASS: TestAdapterTrainingDropout (0.01s)
=== RUN   TestQLoRAAdapterTraining
--- PASS: TestQLoRAAdapterTraining (0.01s)
=== RUN   TestQLoRAAdapterTrainingDropout
--- PASS: TestQLoRAAdapterTrainingDropout (0.00s)
PASS
ok  	github.com/surya-mp/go-peft/backends/cuda	0.758s
```

### full-suite.log

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
ok  	github.com/surya-mp/go-peft/backend	0.024s
=== RUN   TestLoRAForward
--- PASS: TestLoRAForward (0.38s)
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
ok  	github.com/surya-mp/go-peft/backends/cuda	0.516s
=== RUN   TestTargetsAndPlan
--- PASS: TestTargetsAndPlan (0.00s)
=== RUN   TestInspectCommandsRequirePaths
--- PASS: TestInspectCommandsRequirePaths (0.00s)
PASS
ok  	github.com/surya-mp/go-peft/cmd/go-peft	0.011s
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
ok  	github.com/surya-mp/go-peft/format/gguf	0.016s
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
ok  	github.com/surya-mp/go-peft/format/huggingface	0.024s
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
ok  	github.com/surya-mp/go-peft/format/huggingface/model	0.014s
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
ok  	github.com/surya-mp/go-peft/format/safetensors	0.016s
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
ok  	github.com/surya-mp/go-peft/lora	0.009s
=== RUN   TestParameterOwnsShape
--- PASS: TestParameterOwnsShape (0.00s)
=== RUN   TestParameterClone
--- PASS: TestParameterClone (0.00s)
=== RUN   TestParameterCount
--- PASS: TestParameterCount (0.00s)
PASS
ok  	github.com/surya-mp/go-peft/peft	0.009s
=== RUN   TestPlanLlamaAllLinear
--- PASS: TestPlanLlamaAllLinear (0.00s)
=== RUN   TestPlanPhi3AndFailures
--- PASS: TestPlanPhi3AndFailures (0.00s)
PASS
ok  	github.com/surya-mp/go-peft/profiles	0.009s
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
ok  	github.com/surya-mp/go-peft/qlora	0.008s
=== RUN   TestQuantize
--- PASS: TestQuantize (0.00s)
=== RUN   TestQuantizationErrorIsBounded
--- PASS: TestQuantizationErrorIsBounded (0.00s)
=== RUN   TestQuantizedStorageCopiesBuffers
--- PASS: TestQuantizedStorageCopiesBuffers (0.00s)
PASS
ok  	github.com/surya-mp/go-peft/quantization/int4	0.017s
=== RUN   TestQuantizeCodebookAndDot
--- PASS: TestQuantizeCodebookAndDot (0.00s)
=== RUN   TestDoubleQuantizationReducesScaleStorage
--- PASS: TestDoubleQuantizationReducesScaleStorage (0.00s)
=== RUN   TestQuantizedStorageCopiesDoubleQuantBuffers
--- PASS: TestQuantizedStorageCopiesDoubleQuantBuffers (0.00s)
PASS
ok  	github.com/surya-mp/go-peft/quantization/nf4	0.010s
=== RUN   TestRunnerAccumulatesClipsAndCheckpoints
--- PASS: TestRunnerAccumulatesClipsAndCheckpoints (0.00s)
PASS
ok  	github.com/surya-mp/go-peft/trainer	0.010s
```

### lora-example.log

```text
[1 2]
```

### qlora-example.log

```text
[5.637332 14.675482]
```

### benchmarks.log

```text
goos: linux
goarch: amd64
pkg: github.com/surya-mp/go-peft/backends/cuda
cpu: 13th Gen Intel(R) Core(TM) i5-13450HX
BenchmarkLoRAForward-16     	--- FAIL: BenchmarkLoRAForward-16
    benchmark_test.go:33: lora: base linear: cuda: gemm: cuBLAS status 14
BenchmarkQLoRAForward-16    	    3861	    271375 ns/op	   65538 B/op	       1 allocs/op
FAIL
exit status 1
FAIL	github.com/surya-mp/go-peft/backends/cuda	1.547s
FAIL
```

### benchmark-diagnostic.log

```text
goos: linux
goarch: amd64
pkg: github.com/surya-mp/go-peft/backends/cuda
cpu: 13th Gen Intel(R) Core(TM) i5-13450HX
BenchmarkLoRAForward-16    	       1	    845730 ns/op	   65536 B/op	       1 allocs/op
PASS
ok  	github.com/surya-mp/go-peft/backends/cuda	0.755s
```

### benchmarks-repeat.log

```text
goos: linux
goarch: amd64
pkg: github.com/surya-mp/go-peft/backends/cuda
cpu: 13th Gen Intel(R) Core(TM) i5-13450HX
BenchmarkLoRAForward-16     	--- FAIL: BenchmarkLoRAForward-16
    benchmark_test.go:33: lora: base linear: cuda: gemm: cuBLAS status 14
BenchmarkLoRAForward-16     	   10368	    116096 ns/op	   65536 B/op	       1 allocs/op
BenchmarkLoRAForward-16     	   10503	    116689 ns/op	   65536 B/op	       1 allocs/op
BenchmarkQLoRAForward-16    	    3901	    271377 ns/op	   65536 B/op	       1 allocs/op
BenchmarkQLoRAForward-16    	    3723	    270344 ns/op	   65536 B/op	       1 allocs/op
BenchmarkQLoRAForward-16    	    4245	    271304 ns/op	   65538 B/op	       1 allocs/op
FAIL
exit status 1
FAIL	github.com/surya-mp/go-peft/backends/cuda	8.045s
FAIL
```

### benchmarks-system-libraries.log

```text
goos: linux
goarch: amd64
pkg: github.com/surya-mp/go-peft/backends/cuda
cpu: 13th Gen Intel(R) Core(TM) i5-13450HX
BenchmarkLoRAForward-16     	   10198	    118543 ns/op	   65536 B/op	       1 allocs/op
BenchmarkLoRAForward-16     	--- FAIL: BenchmarkLoRAForward-16
    benchmark_test.go:33: lora: A projection: cuda: gemm: cuBLAS status 14
BenchmarkLoRAForward-16     	   10308	    117413 ns/op	   65536 B/op	       1 allocs/op
BenchmarkQLoRAForward-16    	    3968	    270031 ns/op	   65536 B/op	       1 allocs/op
BenchmarkQLoRAForward-16    	    4233	    269836 ns/op	   65536 B/op	       1 allocs/op
BenchmarkQLoRAForward-16    	    3710	    270694 ns/op	   65536 B/op	       1 allocs/op
PASS
ok  	github.com/surya-mp/go-peft/backends/cuda	8.316s
```

### benchmarks-thread-diagnostic.log

```text
goos: linux
goarch: amd64
pkg: github.com/surya-mp/go-peft/backends/cuda
cpu: 13th Gen Intel(R) Core(TM) i5-13450HX
BenchmarkLoRAForward-16     	--- FAIL: BenchmarkLoRAForward-16
    benchmark_test.go:33: lora: base linear: cuda: gemm: cuBLAS status 14
BenchmarkLoRAForward-16     	   10000	    119183 ns/op	   65536 B/op	       1 allocs/op
BenchmarkLoRAForward-16     	   10000	    115529 ns/op	   65536 B/op	       1 allocs/op
BenchmarkQLoRAForward-16    	    4015	    270791 ns/op	   65536 B/op	       1 allocs/op
BenchmarkQLoRAForward-16    	    4364	    269988 ns/op	   65536 B/op	       1 allocs/op
BenchmarkQLoRAForward-16    	    4263	    265587 ns/op	   65536 B/op	       1 allocs/op
FAIL
exit status 1
FAIL	github.com/surya-mp/go-peft/backends/cuda	6.913s
FAIL
```

### benchmarks-eager-loading.log

```text
goos: linux
goarch: amd64
pkg: github.com/surya-mp/go-peft/backends/cuda
cpu: 13th Gen Intel(R) Core(TM) i5-13450HX
BenchmarkLoRAForward-16     	    8455	    122744 ns/op	   65537 B/op	       1 allocs/op
BenchmarkLoRAForward-16     	    9398	    120130 ns/op	   65536 B/op	       1 allocs/op
BenchmarkLoRAForward-16     	    8792	    119014 ns/op	   65536 B/op	       1 allocs/op
BenchmarkQLoRAForward-16    	    4051	    274965 ns/op	   65536 B/op	       1 allocs/op
BenchmarkQLoRAForward-16    	    4358	    266769 ns/op	   65536 B/op	       1 allocs/op
BenchmarkQLoRAForward-16    	    4359	    265189 ns/op	   65538 B/op	       1 allocs/op
PASS
ok  	github.com/surya-mp/go-peft/backends/cuda	10.032s
```

### lora-cublas.log

```text
I! cuBLAS (v13.4.1) function cublasStatus_t cublasCreate_v2(cublasContext**) called:
i!  handle: type=cublasHandle_t; val=POINTER (IN HEX:0x0x2d3af900)
i! Time: 2026-09-11T18:38:49 elapsed from start 0.016667 minutes or 1.000000 seconds
i!Process=14102; Thread=129943171284992; GPU=0; Handle=POINTER (IN HEX:0x(nil))
i! COMPILED WITH: GNU GCC/G++ / 8.5.0 20210514 (Red Hat 8.5.0-26)
I! cuBLAS (v13.4.1) function cublasStatus_t cublasSgemm_v2(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int) called:
i!  handle: type=cublasHandle_t; val=POINTER (IN HEX:0x0x2d3c1100)
i!  transa: type=cublasOperation_t; val=CUBLAS_OP_T(1)
i!  transb: type=cublasOperation_t; val=CUBLAS_OP_N(0)
i!  m: type=int; val=2
i!  n: type=int; val=1
i!  k: type=int; val=2
i!  alpha: type=float; val=POINTER (IN HEX:0x0x7ffd937acafc)
i!  A: type=float; val=POINTER (IN HEX:0x0x504e20400)
i!  lda: type=int; val=2
i!  B: type=float; val=POINTER (IN HEX:0x0x504e20600)
i!  ldb: type=int; val=2
i!  beta: type=float; val=POINTER (IN HEX:0x0x7ffd937acaf8)
i!  C: type=float; val=POINTER (IN HEX:0x0x504e20c00)
i!  ldc: type=int; val=2
i! Time: 2026-09-11T18:38:49 elapsed from start 0.016667 minutes or 1.000000 seconds
i!Process=14102; Thread=129943171284992; GPU=0; Handle=POINTER (IN HEX:0x0x2d3c1100); StreamId=POINTER (IN HEX:0x(nil)) (defaultStream); MathMode=CUBLAS_DEFAULT_MATH
i! COMPILED WITH: GNU GCC/G++ / 8.5.0 20210514 (Red Hat 8.5.0-26)
I! cuBLAS (v13.4.1) function cublasStatus_t cublasSgemm_v2(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int) called:
i!  handle: type=cublasHandle_t; val=POINTER (IN HEX:0x0x2d3c1100)
i!  transa: type=cublasOperation_t; val=CUBLAS_OP_T(1)
i!  transb: type=cublasOperation_t; val=CUBLAS_OP_N(0)
i!  m: type=int; val=1
i!  n: type=int; val=1
i!  k: type=int; val=2
i!  alpha: type=float; val=POINTER (IN HEX:0x0x7ffd937acafc)
i!  A: type=float; val=POINTER (IN HEX:0x0x504e20800)
i!  lda: type=int; val=2
i!  B: type=float; val=POINTER (IN HEX:0x0x504e20600)
i!  ldb: type=int; val=2
i!  beta: type=float; val=POINTER (IN HEX:0x0x7ffd937acaf8)
i!  C: type=float; val=POINTER (IN HEX:0x0x504e20e00)
i!  ldc: type=int; val=1
i! Time: 2026-09-11T18:38:49 elapsed from start 0.016667 minutes or 1.000000 seconds
i!Process=14102; Thread=129943171284992; GPU=0; Handle=POINTER (IN HEX:0x0x2d3c1100); StreamId=POINTER (IN HEX:0x(nil)) (defaultStream); MathMode=CUBLAS_DEFAULT_MATH
i! COMPILED WITH: GNU GCC/G++ / 8.5.0 20210514 (Red Hat 8.5.0-26)
I! cuBLAS (v13.4.1) function cublasStatus_t cublasSgemm_v2(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int) called:
i!  handle: type=cublasHandle_t; val=POINTER (IN HEX:0x0x2d3c1100)
i!  transa: type=cublasOperation_t; val=CUBLAS_OP_T(1)
i!  transb: type=cublasOperation_t; val=CUBLAS_OP_N(0)
i!  m: type=int; val=2
i!  n: type=int; val=1
i!  k: type=int; val=1
i!  alpha: type=float; val=POINTER (IN HEX:0x0x7ffd937acafc)
i!  A: type=float; val=POINTER (IN HEX:0x0x504e20a00)
i!  lda: type=int; val=1
i!  B: type=float; val=POINTER (IN HEX:0x0x504e20e00)
i!  ldb: type=int; val=1
i!  beta: type=float; val=POINTER (IN HEX:0x0x7ffd937acaf8)
i!  C: type=float; val=POINTER (IN HEX:0x0x504e20c00)
i!  ldc: type=int; val=2
i! Time: 2026-09-11T18:38:49 elapsed from start 0.016667 minutes or 1.000000 seconds
i!Process=14102; Thread=129943171284992; GPU=0; Handle=POINTER (IN HEX:0x0x2d3c1100); StreamId=POINTER (IN HEX:0x(nil)) (defaultStream); MathMode=CUBLAS_DEFAULT_MATH
i! COMPILED WITH: GNU GCC/G++ / 8.5.0 20210514 (Red Hat 8.5.0-26)
I! cuBLAS (v13.4.1) function cublasStatus_t cublasDestroy_v2(cublasHandle_t) called:
i!  handle: type=cublasHandle_t; val=POINTER (IN HEX:0x0x2d3c1100)
i! Time: 2026-09-11T18:38:49 elapsed from start 0.016667 minutes or 1.000000 seconds
i!Process=14102; Thread=129943171284992; GPU=0; Handle=POINTER (IN HEX:0x0x2d3c1100); StreamId=POINTER (IN HEX:0x(nil)) (defaultStream); MathMode=CUBLAS_DEFAULT_MATH
i! COMPILED WITH: GNU GCC/G++ / 8.5.0 20210514 (Red Hat 8.5.0-26)
```

### qlora-cublas.log

```text
I! cuBLAS (v13.4.1) function cublasStatus_t cublasCreate_v2(cublasContext**) called:
i!  handle: type=cublasHandle_t; val=POINTER (IN HEX:0x0x3eab1a10)
i! Time: 2026-09-11T18:38:49 elapsed from start 0.000000 minutes or 0.000000 seconds
i!Process=14151; Thread=123598179266560; GPU=0; Handle=POINTER (IN HEX:0x(nil))
i! COMPILED WITH: GNU GCC/G++ / 8.5.0 20210514 (Red Hat 8.5.0-26)
I! cuBLAS (v13.4.1) function cublasStatus_t cublasSgemm_v2(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int) called:
i!  handle: type=cublasHandle_t; val=POINTER (IN HEX:0x0x3eac3210)
i!  transa: type=cublasOperation_t; val=CUBLAS_OP_T(1)
i!  transb: type=cublasOperation_t; val=CUBLAS_OP_N(0)
i!  m: type=int; val=2
i!  n: type=int; val=1
i!  k: type=int; val=3
i!  alpha: type=float; val=POINTER (IN HEX:0x0x7ffc48330a5c)
i!  A: type=float; val=POINTER (IN HEX:0x0x504e20800)
i!  lda: type=int; val=3
i!  B: type=float; val=POINTER (IN HEX:0x0x504e20600)
i!  ldb: type=int; val=3
i!  beta: type=float; val=POINTER (IN HEX:0x0x7ffc48330a58)
i!  C: type=float; val=POINTER (IN HEX:0x0x504e20e00)
i!  ldc: type=int; val=2
i! Time: 2026-09-11T18:38:49 elapsed from start 0.000000 minutes or 0.000000 seconds
i!Process=14151; Thread=123598179266560; GPU=0; Handle=POINTER (IN HEX:0x0x3eac3210); StreamId=POINTER (IN HEX:0x(nil)) (defaultStream); MathMode=CUBLAS_DEFAULT_MATH
i! COMPILED WITH: GNU GCC/G++ / 8.5.0 20210514 (Red Hat 8.5.0-26)
I! cuBLAS (v13.4.1) function cublasStatus_t cublasSgemm_v2(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int) called:
i!  handle: type=cublasHandle_t; val=POINTER (IN HEX:0x0x3eac3210)
i!  transa: type=cublasOperation_t; val=CUBLAS_OP_T(1)
i!  transb: type=cublasOperation_t; val=CUBLAS_OP_N(0)
i!  m: type=int; val=2
i!  n: type=int; val=1
i!  k: type=int; val=2
i!  alpha: type=float; val=POINTER (IN HEX:0x0x7ffc48330a5c)
i!  A: type=float; val=POINTER (IN HEX:0x0x504e20a00)
i!  lda: type=int; val=2
i!  B: type=float; val=POINTER (IN HEX:0x0x504e20e00)
i!  ldb: type=int; val=2
i!  beta: type=float; val=POINTER (IN HEX:0x0x7ffc48330a58)
i!  C: type=float; val=POINTER (IN HEX:0x0x504e20c00)
i!  ldc: type=int; val=2
i! Time: 2026-09-11T18:38:49 elapsed from start 0.000000 minutes or 0.000000 seconds
i!Process=14151; Thread=123598179266560; GPU=0; Handle=POINTER (IN HEX:0x0x3eac3210); StreamId=POINTER (IN HEX:0x(nil)) (defaultStream); MathMode=CUBLAS_DEFAULT_MATH
i! COMPILED WITH: GNU GCC/G++ / 8.5.0 20210514 (Red Hat 8.5.0-26)
I! cuBLAS (v13.4.1) function cublasStatus_t cublasDestroy_v2(cublasHandle_t) called:
i!  handle: type=cublasHandle_t; val=POINTER (IN HEX:0x0x3eac3210)
i! Time: 2026-09-11T18:38:49 elapsed from start 0.000000 minutes or 0.000000 seconds
i!Process=14151; Thread=123596288540672; GPU=0; Handle=POINTER (IN HEX:0x0x3eac3210); StreamId=POINTER (IN HEX:0x(nil)) (defaultStream); MathMode=CUBLAS_DEFAULT_MATH
i! COMPILED WITH: GNU GCC/G++ / 8.5.0 20210514 (Red Hat 8.5.0-26)
```

### benchmark-failure-cublas.log

```text
I! cuBLAS (v13.4.1) function cublasStatus_t cublasCreate_v2(cublasContext**) called:
i!  handle: type=cublasHandle_t; val=POINTER (IN HEX:0x0x38319070)
i! Time: 2026-09-11T18:39:31 elapsed from start 0.000000 minutes or 0.000000 seconds
i!Process=14477; Thread=137882905268224; GPU=0; Handle=POINTER (IN HEX:0x(nil))
i! COMPILED WITH: GNU GCC/G++ / 8.5.0 20210514 (Red Hat 8.5.0-26)
I! cuBLAS (v13.4.1) function cublasStatus_t cublasSgemm_v2(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int) called:
i!  handle: type=cublasHandle_t; val=POINTER (IN HEX:0x0x3832a870)
i!  transa: type=cublasOperation_t; val=CUBLAS_OP_T(1)
i!  transb: type=cublasOperation_t; val=CUBLAS_OP_N(0)
i!  m: type=int; val=512
i!  n: type=int; val=32
i!  k: type=int; val=512
i!  alpha: type=float; val=POINTER (IN HEX:0x0x7ffc26506b9c)
i!  A: type=float; val=POINTER (IN HEX:0x0x504e20400)
i!  lda: type=int; val=512
i!  B: type=float; val=POINTER (IN HEX:0x0x504f20400)
i!  ldb: type=int; val=512
i!  beta: type=float; val=POINTER (IN HEX:0x0x7ffc26506b98)
i!  C: type=float; val=POINTER (IN HEX:0x0x504f40400)
i!  ldc: type=int; val=512
i! Time: 2026-09-11T18:39:31 elapsed from start 0.000000 minutes or 0.000000 seconds
i!Process=14477; Thread=137882905268224; GPU=0; Handle=POINTER (IN HEX:0x0x3832a870); StreamId=POINTER (IN HEX:0x(nil)) (defaultStream); MathMode=CUBLAS_DEFAULT_MATH
i! COMPILED WITH: GNU GCC/G++ / 8.5.0 20210514 (Red Hat 8.5.0-26)
I! cuBLAS (v13.4.1) function cublasStatus_t cublasSgemm_v2(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int) called:
i!  handle: type=cublasHandle_t; val=POINTER (IN HEX:0x0x3832a870)
i!  transa: type=cublasOperation_t; val=CUBLAS_OP_T(1)
i!  transb: type=cublasOperation_t; val=CUBLAS_OP_N(0)
i!  m: type=int; val=16
i!  n: type=int; val=32
i!  k: type=int; val=512
i!  alpha: type=float; val=POINTER (IN HEX:0x0x7ffc26506b9c)
i!  A: type=float; val=POINTER (IN HEX:0x0x504f30400)
i!  lda: type=int; val=512
i!  B: type=float; val=POINTER (IN HEX:0x0x504f20400)
i!  ldb: type=int; val=512
i!  beta: type=float; val=POINTER (IN HEX:0x0x7ffc26506b98)
i!  C: type=float; val=POINTER (IN HEX:0x0x504f50400)
i!  ldc: type=int; val=16
i! Time: 2026-09-11T18:39:31 elapsed from start 0.000000 minutes or 0.000000 seconds
i!Process=14477; Thread=137882905268224; GPU=0; Handle=POINTER (IN HEX:0x0x3832a870); StreamId=POINTER (IN HEX:0x(nil)) (defaultStream); MathMode=CUBLAS_DEFAULT_MATH
i! COMPILED WITH: GNU GCC/G++ / 8.5.0 20210514 (Red Hat 8.5.0-26)
I! cuBLAS (v13.4.1) function cublasStatus_t cublasSgemm_v2(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int) called:
i!  handle: type=cublasHandle_t; val=POINTER (IN HEX:0x0x3832a870)
i!  transa: type=cublasOperation_t; val=CUBLAS_OP_T(1)
i!  transb: type=cublasOperation_t; val=CUBLAS_OP_N(0)
i!  m: type=int; val=512
i!  n: type=int; val=32
i!  k: type=int; val=16
i!  alpha: type=float; val=POINTER (IN HEX:0x0x7ffc26506b9c)
i!  A: type=float; val=POINTER (IN HEX:0x0x504f38400)
i!  lda: type=int; val=16
i!  B: type=float; val=POINTER (IN HEX:0x0x504f50400)
i!  ldb: type=int; val=16
i!  beta: type=float; val=POINTER (IN HEX:0x0x7ffc26506b98)
i!  C: type=float; val=POINTER (IN HEX:0x0x504f40400)
i!  ldc: type=int; val=512
i! Time: 2026-09-11T18:39:31 elapsed from start 0.000000 minutes or 0.000000 seconds
i!Process=14477; Thread=137882905268224; GPU=0; Handle=POINTER (IN HEX:0x0x3832a870); StreamId=POINTER (IN HEX:0x(nil)) (defaultStream); MathMode=CUBLAS_DEFAULT_MATH
i! COMPILED WITH: GNU GCC/G++ / 8.5.0 20210514 (Red Hat 8.5.0-26)
I! cuBLAS (v13.4.1) function cublasStatus_t cublasSgemm_v2(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int) called:
i!  handle: type=cublasHandle_t; val=POINTER (IN HEX:0x0x3832a870)
i!  transa: type=cublasOperation_t; val=CUBLAS_OP_T(1)
i!  transb: type=cublasOperation_t; val=CUBLAS_OP_N(0)
i!  m: type=int; val=512
i!  n: type=int; val=32
i!  k: type=int; val=512
i!  alpha: type=float; val=POINTER (IN HEX:0x0x7ffc26506b9c)
i!  A: type=float; val=POINTER (IN HEX:0x0x504e20400)
i!  lda: type=int; val=512
i!  B: type=float; val=POINTER (IN HEX:0x0x504f20400)
i!  ldb: type=int; val=512
i!  beta: type=float; val=POINTER (IN HEX:0x0x7ffc26506b98)
i!  C: type=float; val=POINTER (IN HEX:0x0x504f40400)
i!  ldc: type=int; val=512
i! Time: 2026-09-11T18:39:31 elapsed from start 0.000000 minutes or 0.000000 seconds
i!Process=14477; Thread=137882905268224; GPU=0; Handle=POINTER (IN HEX:0x0x3832a870); StreamId=POINTER (IN HEX:0x(nil)) (defaultStream); MathMode=CUBLAS_DEFAULT_MATH
i! COMPILED WITH: GNU GCC/G++ / 8.5.0 20210514 (Red Hat 8.5.0-26)
I! cuBLAS (v13.4.1) function cublasStatus_t cublasSgemm_v2(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int) called:
i!  handle: type=cublasHandle_t; val=POINTER (IN HEX:0x0x3832a870)
i!  transa: type=cublasOperation_t; val=CUBLAS_OP_T(1)
i!  transb: type=cublasOperation_t; val=CUBLAS_OP_N(0)
i!  m: type=int; val=16
i!  n: type=int; val=32
i!  k: type=int; val=512
i!  alpha: type=float; val=POINTER (IN HEX:0x0x7ffc26506b9c)
i!  A: type=float; val=POINTER (IN HEX:0x0x504f30400)
i!  lda: type=int; val=512
i!  B: type=float; val=POINTER (IN HEX:0x0x504f20400)
i!  ldb: type=int; val=512
i!  beta: type=float; val=POINTER (IN HEX:0x0x7ffc26506b98)
i!  C: type=float; val=POINTER (IN HEX:0x0x504f50400)
i!  ldc: type=int; val=16
i! Time: 2026-09-11T18:39:31 elapsed from start 0.000000 minutes or 0.000000 seconds
i!Process=14477; Thread=137882905268224; GPU=0; Handle=POINTER (IN HEX:0x0x3832a870); StreamId=POINTER (IN HEX:0x(nil)) (defaultStream); MathMode=CUBLAS_DEFAULT_MATH
i! COMPILED WITH: GNU GCC/G++ / 8.5.0 20210514 (Red Hat 8.5.0-26)
I! cuBLAS (v13.4.1) function cublasStatus_t cublasSgemm_v2(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int) called:
i!  handle: type=cublasHandle_t; val=POINTER (IN HEX:0x0x3832a870)
i!  transa: type=cublasOperation_t; val=CUBLAS_OP_T(1)
i!  transb: type=cublasOperation_t; val=CUBLAS_OP_N(0)
i!  m: type=int; val=512
i!  n: type=int; val=32
i!  k: type=int; val=16
i!  alpha: type=float; val=POINTER (IN HEX:0x0x7ffc26506b9c)
i!  A: type=float; val=POINTER (IN HEX:0x0x504f38400)
i!  lda: type=int; val=16
i!  B: type=float; val=POINTER (IN HEX:0x0x504f50400)
i!  ldb: type=int; val=16
i!  beta: type=float; val=POINTER (IN HEX:0x0x7ffc26506b98)
i!  C: type=float; val=POINTER (IN HEX:0x0x504f40400)
i!  ldc: type=int; val=512
i! Time: 2026-09-11T18:39:31 elapsed from start 0.000000 minutes or 0.000000 seconds
i!Process=14477; Thread=137882905268224; GPU=0; Handle=POINTER (IN HEX:0x0x3832a870); StreamId=POINTER (IN HEX:0x(nil)) (defaultStream); MathMode=CUBLAS_DEFAULT_MATH
i! COMPILED WITH: GNU GCC/G++ / 8.5.0 20210514 (Red Hat 8.5.0-26)
I! cuBLAS (v13.4.1) function cublasStatus_t cublasDestroy_v2(cublasHandle_t) called:
i!  handle: type=cublasHandle_t; val=POINTER (IN HEX:0x0x3832a870)
i! Time: 2026-09-11T18:39:31 elapsed from start 0.000000 minutes or 0.000000 seconds
i!Process=14477; Thread=137882905268224; GPU=0; Handle=POINTER (IN HEX:0x0x3832a870); StreamId=POINTER (IN HEX:0x(nil)) (defaultStream); MathMode=CUBLAS_DEFAULT_MATH
i! COMPILED WITH: GNU GCC/G++ / 8.5.0 20210514 (Red Hat 8.5.0-26)
```

### Initial worktree status

```text
 M .github/workflows/cuda.yml
 M .github/workflows/mlx.yml
 M .github/workflows/test.yml
 M .gitignore
 M .golangci-lint-version
 M .golangci.yml
 M LICENSE
 M README.md
 M backend/backend.go
 M backend/cpu.go
 M backend/cpu_test.go
 M backend/errors.go
 M backend/runtime.go
 M backend/runtime_test.go
 M backends/cuda/benchmark_test.go
 M backends/cuda/cuda.go
 M backends/cuda/cuda_stub.go
 M backends/cuda/cuda_test.go
 M backends/cuda/gopeft_cuda.cu
 M backends/cuda/gopeft_cuda.h
 M backends/cuda/train.go
 M backends/cuda/train_stub.go
 M backends/gomlx/adapter.go
 M backends/gomlx/integration_test.go
 M backends/gomlx/lora.go
 M backends/gomlx/qlora.go
 M backends/gomlx/serialization.go
 M backends/mlx/benchmark_test.go
 M backends/mlx/integration_test.go
 M backends/mlx/lora.go
 M backends/mlx/mlx.go
 M backends/mlx/qlora.go
 M backends/mlx/qlora_test.go
 M backends/mlx/serialization.go
 M backends/mlx/train.go
 M backends/mlx/train_test.go
 M cmd/go-peft/main.go
 M cmd/go-peft/main_test.go
 M docs/base-models.md
 M docs/benchmarks.md
 M docs/compatibility.md
 M docs/gguf.md
 M docs/peft-parity.md
 M docs/training.md
 M examples/README.md
 M examples/lora/cpu/main.go
 M examples/lora/cuda/main.go
 M examples/lora/gomlx/main.go
 M examples/lora/huggingface/main.go
 M examples/lora/injection/main.go
 M examples/lora/mlx/main.go
 M examples/qlora/cpu/main.go
 M examples/qlora/cuda/main.go
 M examples/qlora/gomlx/main.go
 M examples/qlora/huggingface/main.go
 M examples/qlora/injection/main.go
 M examples/qlora/mlx/main.go
 M format/gguf/fuzz_test.go
 M format/gguf/gguf.go
 M format/gguf/gguf_test.go
 M format/huggingface/golden_fixture_test.go
 M format/huggingface/huggingface.go
 M format/huggingface/integration_test.go
 M format/huggingface/model/fuzz_test.go
 M format/huggingface/model/model.go
 M format/huggingface/model/model_test.go
 M format/huggingface/peft_parity_test.go
 M format/huggingface/testdata/peft_lora/adapter_config.json
 M format/huggingface/testdata/peft_lora/adapter_model.safetensors.base64
 M format/safetensors/fuzz_test.go
 M format/safetensors/safetensors.go
 M format/safetensors/safetensors_test.go
 M go.mod
 M go.sum
 M lora/adapter.go
 M lora/config.go
 M lora/config_test.go
 M lora/errors.go
 M lora/injection.go
 M lora/injection_test.go
 M lora/linear.go
 M lora/linear_benchmark_test.go
 M lora/linear_test.go
 M lora/peft_parity_test.go
 M peft/adapter.go
 M peft/config.go
 M peft/parameter.go
 M peft/parameter_test.go
 M plan.md
 M profiles/profiles.go
 M profiles/profiles_test.go
 M qlora/adapter.go
 M qlora/config.go
 M qlora/config_test.go
 M qlora/errors.go
 M qlora/injection.go
 M qlora/injection_test.go
 M qlora/linear.go
 M qlora/linear_benchmark_test.go
 M qlora/linear_test.go
 M qlora/peft_parity_test.go
 M quantization/int4/int4.go
 M quantization/int4/int4_test.go
 M quantization/nf4/nf4.go
 M quantization/nf4/nf4_test.go
 M trainer/trainer.go
 M trainer/trainer_test.go
```

## Latest follow-up

The [fresh native CUDA and complete-suite output](cuda-context-fix-2026-09-11.md#fresh-validation-after-benchmark-publication) records the post-fix rerun after benchmark publication. Earlier results above remain a historical record.
