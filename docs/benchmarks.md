# Benchmarks

Run the reproducible CPU baseline:

```sh
go test -run '^$' -bench BenchmarkLinearForwardInto -benchmem ./lora ./qlora
```

Run the MLX GPU baseline on Apple Silicon:

```sh
go test -run '^$' -tags mlx -bench 'Benchmark(LoRA|QLoRA)Forward$' -benchmem ./backends/mlx
```

Run the CUDA GPU baseline on Linux with an NVIDIA GPU:

```sh
sh scripts/build_cuda.sh
go test -run '^$' -tags cuda -bench 'Benchmark(LoRA|QLoRA)Forward$' -benchmem ./backends/cuda
```

Each baseline uses a `32 x 512` input, a `512 x 512` base weight, and rank 16.
CPU reuses output and rank workspaces; MLX uses the public `Forward` API.

| Backend | Configuration | ns/op | Go allocations |
| --- | --- | ---: | ---: |
| CPU | LoRA, F32 | 10,926,938 | 0 B / 0 allocs |
| CPU | QLoRA, NF4 + double quant | 17,380,661 | 0 B / 0 allocs |
| MLX GPU | LoRA, F32 | 386,206 | 280 B / 17 allocs |
| MLX GPU | QLoRA, affine int4 | 484,166 | 248 B / 15 allocs |
| CUDA GPU | LoRA, F32 | 131,399 | 65,536 B / 1 allocs |
| CUDA GPU | QLoRA, NF4 + double quant | 267,842 | 65,536 B / 1 allocs |

CPU and MLX captured on Apple M1, 8 GB RAM, Go 1.27.1, darwin/arm64. These are a local
baseline, not a cross-machine comparison. Run the same command on release
hardware and retain its output with the release notes.

CUDA captured on 2026-09-11 on NVIDIA GeForce RTX 3050 6GB Laptop GPU,
Windows driver 591.74, Ubuntu 26.04 on WSL2, CUDA 13.2 (nvcc 13.2.86),
Go 1.26.5 linux/amd64, and Intel Core i5-13450HX. These measurements are from
the current worktree, with native code compiled for `sm_86`; no eager-loading
workaround was used. CUDA reuses output and rank workspaces.

MLX and CUDA rows include device evaluation and output transfer.

Captured CUDA command output:

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

An MLX context pins its creating goroutine to an OS thread until `Close`; use
and close it from that goroutine.
