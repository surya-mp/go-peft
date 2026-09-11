# go-peft

Backend-independent parameter-efficient fine-tuning for Go.

The core includes LoRA, SafeTensors, Hugging Face PEFT files, a CPU backend,
and opt-in GoMLX/MLX integrations. QLoRA follows the quantized-weight layer.

## Status

Early development. The public API may change before v1.0.

## Install

```sh
go get github.com/surya-mp/go-peft
```

## Verify

```sh
go test ./...
go vet ./...
go test -bench=. -benchmem ./lora
```

## Minimal CPU example

```go
import (
	"math/rand"

	"github.com/surya-mp/go-peft/backend"
	"github.com/surya-mp/go-peft/lora"
)

engine := backend.NewCPU()
weight, _ := backend.NewDense(2, 3, []float32{1, 2, 3, 4, 5, 6})
input, _ := backend.NewDense(1, 3, []float32{1, 1, 1})
config := lora.Config{Rank: 2, Alpha: 4, TargetModules: []string{"q_proj"}}
layer, _ := lora.NewLinear("q_proj", engine, weight, nil, config, rand.New(rand.NewSource(1)))
output, _ := layer.Forward(input)
```

Use `ForwardInto` with reusable output and rank workspace tensors in hot paths.

## Package layout

`backend` is singular because it is the small, framework-neutral contract:
tensor operations, the CPU reference engine, quantized-weight interfaces, and
GPU-first runtime selection. LoRA and QLoRA depend only on this package.

`backends` is plural because each child is an opt-in framework bridge:
`backends/gomlx` maps the contract to GoMLX graphs; `backends/mlx` maps it to
MLX C arrays. Bridges can add native kernels without coupling the core API to a
specific framework.

| Capability | LoRA | QLoRA |
| --- | --- | --- |
| A/B adapters, dropout, workspaces, injection, PEFT files | Yes | Yes |
| CPU, CUDA-selection, GoMLX, MLX examples | Yes | Yes |
| Base weight | Frozen F32 | Frozen int4 or NF4 |
| NF4 and double quantization | Not applicable | Yes |
| Merge / unmerge | Yes | Returns `ErrMergeUnsupported` to avoid lossy int4 writes |

## GoMLX training

Build with `-tags gomlx`. Register the host model's native variables, then use
the replaced native layers in the normal GoMLX training graph.

```go
adapter, err := gomlx.Inject("train", modelRegistry, config, rand.New(rand.NewSource(1)))
if err != nil { /* handle */ }
trainable := adapter.TrainableVariables() // A, B, and configured bias
err = adapter.Save("adapter", huggingface.Metadata{BaseModelNameOrPath: "base"})
```

The base variables are frozen during injection, so GoMLX optimizers select only
adapter variables. `adapter.Load` validates configuration, names, and shapes
before changing native values.

Run the minimal native training graph with:

```sh
go run -tags gomlx ./examples/lora/gomlx
go run -tags gomlx ./examples/qlora/gomlx
```

## QLoRA

```go
config := qlora.Config{
    LoRA: lora.Config{Rank: 16, Alpha: 32, TargetModules: []string{"q_proj"}},
    BlockSize: 64,
    Quantization: qlora.QuantizationNF4,
    DoubleQuant: true,
    ScaleBlockSize: 256,
}
layer, _ := qlora.QuantizeLinear("q_proj", engine, weight, nil, config, rand.New(rand.NewSource(1)))
```

QLoRA stores frozen base weights as signed int4 or NF4 and keeps A/B in F32.
NF4 can double-quantize its block scales to 8-bit values. The default remains
signed int4 for compatibility.
`huggingface.SaveQLoRA` and `LoadQLoRAInto` use the same PEFT adapter format.

## MLX QLoRA

The opt-in MLX C bridge runs its packed affine-int4 base projection and LoRA
update on MLX's default GPU stream. `TrainStep` uses MLX C value-and-gradient
for A/B, and adapters save/load through the same Hugging Face PEFT files:

```sh
go test -tags mlx ./backends/mlx
go run -tags mlx ./examples/qlora/mlx
```

Use `mlx.NewCPU()` only when CPU execution is explicitly intended. MLX's
native affine representation is not NF4 and does not expose this library's
double-quantized scale format, so the MLX bridge rejects those configurations.

## Hardware fallback

```go
runtime, err := backend.NewRuntime(backend.RuntimeOptions{
    CUDA: cudaEngine, // optional accelerator backend
    Warn: func(err error) { log.Print(err) },
})
if err != nil { log.Fatal(err) }
engine := runtime.Engine
```

CUDA is required by default. If unavailable, initialization warns and returns an
error without starting work. Opt into CPU explicitly with `DisableCUDA: true`.

## CUDA

The opt-in CUDA bridge uses cuBLAS for LoRA matrix products and a packed int4/
NF4 base-projection kernel for QLoRA, including NF4 double quantization.
`cuda.TrainLoRA` and `cuda.TrainQLoRA` update only A/B with cuBLAS and apply
device-side training dropout. On a Linux
NVIDIA host with the CUDA toolkit:

```sh
sh scripts/build_cuda.sh
go test -tags cuda ./backends/cuda
go run -tags cuda ./examples/lora/cuda
go run -tags cuda ./examples/qlora/cuda
```

Set `CGO_LDFLAGS=-L$CUDA_HOME/lib64` when CUDA is not at `/usr/local/cuda`.
Regular builds never link CUDA. The native suite needs an NVIDIA runner; it is
not runnable on this Apple Silicon development machine.

For a Windows NVIDIA host, run this Linux-only bridge in WSL2 Ubuntu with GPU
passthrough. Native Windows CUDA builds are not supported yet.

## Integrations

| Integration | Use | Status |
| --- | --- | --- |
| SafeTensors | `format/safetensors` | Read/write F32 |
| Hugging Face PEFT | `format/huggingface` | LoRA and QLoRA adapter layout |
| GoMLX | `go test -tags gomlx ./backends/gomlx` | LoRA and native NF4 QLoRA bridge |
| MLX C | `go test -tags mlx ./backends/mlx` | Apple Silicon LoRA + dropout training and affine-int4 QLoRA |
| CUDA | `go test -tags cuda ./backends/cuda` | Linux NVIDIA cuBLAS LoRA and int4/NF4 QLoRA bridge |

On Apple Silicon, install MLX C with `brew install mlx-c`. It is opt-in through
`-tags mlx`; regular builds never require cgo or Metal. Source installations
outside Homebrew's default prefix must provide matching `CGO_CFLAGS` and
`CGO_LDFLAGS`. The manual `mlx` workflow runs native tests and examples on a
macOS ARM runner.

Hugging Face uses `adapter_config.json` and `adapter_model.safetensors` with
standard `base_model.model.<module>.lora_A/B.weight` names.

The functional PEFT-parity suite is written in Go. See
[PEFT parity](docs/peft-parity.md) for its supported-behavior matrix.

## Roadmap

1. NVIDIA-runner validation and CUDA kernel tuning
2. More Hugging Face base-model fixtures and PEFT methods

## Release quality

CI checks formatting, vet, tests, race tests, GoMLX tests, SafeTensors
compatibility, and pinned `golangci-lint`. See [compatibility](docs/compatibility.md),
[benchmarks](docs/benchmarks.md), [contributing](CONTRIBUTING.md), and [releasing](RELEASING.md).

## Examples

Runnable commands and hardware requirements are listed in
[examples/README.md](examples/README.md).
