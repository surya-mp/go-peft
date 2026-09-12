# Test Report

Date: 2026-09-12
Host: Windows, PowerShell
GPU: NVIDIA GeForce RTX 3050 Laptop GPU detected by `nvidia-smi`; CUDA version reported by driver: 13.1.
CUDA toolkit: `nvcc` was not available on `PATH`.

## Commands Run

```powershell
$env:GOCACHE=(Join-Path (Get-Location) '.gocache')
$env:GOMODCACHE=(Join-Path (Get-Location) '.gomodcache')
go test ./...
go test -tags cuda ./backends/cuda
go run -tags cuda ./examples/lora/cuda
go run -tags cuda ./examples/qlora/cuda
```

## Standard Test Result

Pass.

Validated packages included:

- `github.com/surya-mp/go-peft/backend`
- `github.com/surya-mp/go-peft/cmd/go-peft`
- `github.com/surya-mp/go-peft/format/gguf`
- `github.com/surya-mp/go-peft/format/huggingface`
- `github.com/surya-mp/go-peft/format/huggingface/model`
- `github.com/surya-mp/go-peft/format/safetensors`
- `github.com/surya-mp/go-peft/lora`
- `github.com/surya-mp/go-peft/peft`
- `github.com/surya-mp/go-peft/profiles`
- `github.com/surya-mp/go-peft/qlora`
- `github.com/surya-mp/go-peft/quantization/int4`
- `github.com/surya-mp/go-peft/quantization/nf4`
- `github.com/surya-mp/go-peft/trainer`

## CUDA/GPU Result

Not executed on native Windows.

`go test -tags cuda ./backends/cuda` completed with:

```text
?    github.com/surya-mp/go-peft/backends/cuda    [no test files]
```

The CUDA test file and implementation are guarded by `//go:build cuda && linux && cgo`. On this Windows host, Go selected the stub implementation guarded by `//go:build !cuda || !linux || !cgo`.

The CUDA examples both exited with:

```text
backend: CUDA is unavailable
exit status 1
```

To execute the real CUDA tests, run this repo in WSL2 Ubuntu or another Linux NVIDIA environment with cgo and the CUDA toolkit available, then run:

```sh
sh scripts/build_cuda.sh
go test -tags cuda ./backends/cuda
go run -tags cuda ./examples/lora/cuda
go run -tags cuda ./examples/qlora/cuda
```
