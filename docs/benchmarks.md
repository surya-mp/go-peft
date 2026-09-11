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
| CUDA GPU | LoRA, F32 | Pending publication | — |
| CUDA GPU | QLoRA, NF4 + double quant | Pending publication | — |

Captured on Apple M1, 8 GB RAM, Go 1.27.1, darwin/arm64. These are a local
baseline, not a cross-machine comparison. Run the same command on release
hardware and retain its output with the release notes.

MLX and CUDA rows include device evaluation and output transfer. CUDA kernel
validation passed on the NVIDIA runner; record its benchmark output here after
the final merged worktree is measured.

An MLX context pins its creating goroutine to an OS thread until `Close`; use
and close it from that goroutine.
