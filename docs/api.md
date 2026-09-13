# API reference

The exported Go source is the authoritative API reference. Pin a release tag
in production; `v0.x` APIs may change before v1.

```sh
go get github.com/surya-mp/go-peft@latest
```

The generated reference is on [pkg.go.dev](https://pkg.go.dev/github.com/surya-mp/go-peft/lora).
It is also available from the installed source:

```sh
go doc github.com/surya-mp/go-peft/lora
go doc github.com/surya-mp/go-peft/qlora
```

## Package map

| Package | Use | Contract |
| --- | --- | --- |
| [`backend`](https://pkg.go.dev/github.com/surya-mp/go-peft/backend) | Core tensor interfaces and CPU engine | Implement `EagerEngine` for an eager framework bridge. |
| [`lora`](https://pkg.go.dev/github.com/surya-mp/go-peft/lora) | F32 frozen-base LoRA | `A` and `B` are the trainable parameters. |
| [`qlora`](https://pkg.go.dev/github.com/surya-mp/go-peft/qlora) | 4-bit frozen-base LoRA | Base weights are int4 or NF4; only `A` and `B` train. |
| [`quantization/int4`](https://pkg.go.dev/github.com/surya-mp/go-peft/quantization/int4) | Symmetric int4 weights | Portable packed base-weight storage. |
| [`quantization/nf4`](https://pkg.go.dev/github.com/surya-mp/go-peft/quantization/nf4) | NF4 weights | Optional 8-bit scale double quantization. |
| [`format/huggingface`](https://pkg.go.dev/github.com/surya-mp/go-peft/format/huggingface) | PEFT adapter files | Reads/writes `adapter_config.json` + `adapter_model.safetensors`. |
| [`format/huggingface/model`](https://pkg.go.dev/github.com/surya-mp/go-peft/format/huggingface/model) | Base-model loader | Streams F32/F16/BF16 SafeTensors one tensor at a time. |
| [`format/safetensors`](https://pkg.go.dev/github.com/surya-mp/go-peft/format/safetensors) | SafeTensors I/O | `Write` emits F32; `Visit` reads F32/F16/BF16. |
| [`format/gguf`](https://pkg.go.dev/github.com/surya-mp/go-peft/format/gguf) | GGUF inspection | Reads v3 metadata and index; never loads or executes GGML tensors. |
| [`profiles`](https://pkg.go.dev/github.com/surya-mp/go-peft/profiles) | Target-module presets | Plan and validate model targets before injection. |
| [`trainer`](https://pkg.go.dev/github.com/surya-mp/go-peft/trainer) | Training-loop coordination | Host owns autograd, batches, and optimizer implementation; `AdapterUpdater` limits updates to PEFT variables. |

Framework bridges are opt-in: [`backends/gomlx`](https://pkg.go.dev/github.com/surya-mp/go-peft/backends/gomlx), [`backends/mlx`](https://pkg.go.dev/github.com/surya-mp/go-peft/backends/mlx), and [`backends/cuda`](https://pkg.go.dev/github.com/surya-mp/go-peft/backends/cuda).

## Adapter-only updates

`trainer.AdapterVariables` returns a defensive copy of an adapter's trainable
variables. `trainer.AdapterUpdater` passes only that set to a caller-provided
optimizer. It does not implement autograd or select an optimizer.

## LoRA

`lora.Config` is valid only when `Rank > 0`, `Alpha >= 0`, `0 <= Dropout < 1`,
and `TargetModules` is non-empty. The update is:

```text
y = xWᵀ + bias + (alpha / rank) × xAᵀBᵀ
```

`NewLinear` expects a base weight shaped `[outFeatures, inFeatures]`, and an
optional bias shaped `[1, outFeatures]`. It initializes `A` with Kaiming-uniform
values and `B` with zeroes, so the initial output equals the base projection.

The runnable `lora.ExampleNewLinear` is rendered with the package on
pkg.go.dev and verified by `go test`.

Use `ForwardTraining` to apply configured dropout. Use `ForwardInto` with an
output and rank workspace from `NewWorkspace` in allocation-sensitive paths.
`Merge` and `Unmerge` mutate the F32 base weight; do not call them while another
goroutine reads or writes the same layer.

`Inject` is the host-model integration point. A host implements `lora.Model`:

```go
type Model interface {
	LinearModules() ([]lora.Module, error)
	ReplaceLinearModules([]lora.Replacement) error
}
```

`Inject` discovers all modules, builds all replacements, then calls the host's
replacement method once. A failure before replacement leaves the host unchanged.
`lora.Matches` accepts an exact target or a dot-separated module suffix.

## QLoRA

`qlora.Config` embeds `lora.Config` and adds a frozen 4-bit representation.
Use `QuantizeLinear` for a F32 source weight, or `NewLinear` if a bridge has
already created a `backend.QuantizedWeight`.

The runnable `qlora.ExampleQuantizeLinear` is rendered with the package on
pkg.go.dev and verified by `go test`.

NF4 double quantization requires `QuantizationNF4` and a positive
`ScaleBlockSize`. QLoRA deliberately rejects `Merge`/`Unmerge`: writing the
update into a 4-bit base would be lossy. `qlora.Model`, `Inject`, `Adapter`,
forward methods, and parameter metadata mirror the LoRA API.

## Serialization and model loading

Use a backend that implements `backend.Float32Codec` for portable adapter I/O.
Adapters must be unmerged before saving.

```go
err := huggingface.Save("adapter", adapter, engine, huggingface.Metadata{
	BaseModelNameOrPath: "org/model", Revision: "main", TaskType: "CAUSAL_LM",
})
metadata, err := huggingface.LoadInto("adapter", adapter, engine)
```

Use `SaveQLoRA` and `LoadQLoRAInto` for QLoRA adapters. Adapter files contain
LoRA A/B tensors, not the base model or QLoRA quantized base weights. `BiasAll`
cannot be exported because those base-model biases are not portable.

For a Hugging Face base model, use `model.Inspect` for the config and shard
layout, or `model.Load` to stream tensors directly into a framework bridge:

Pass `model.Options.OnTensor` to allocate each decoded tensor in the target
framework. The loader deliberately does not choose or retain framework tensors.

## Runtime and bridge selection

`backend.NewRuntime` requires a supplied CUDA engine by default. If CUDA is
unavailable it reports `backend.ErrCUDAUnavailable`, calls `Warn`, and does not
start CPU work. Set `DisableCUDA: true` for an explicit CPU choice.

| Bridge | Build | Notes |
| --- | --- | --- |
| CPU | default | `backend.NewCPU()` is the portable eager reference engine. |
| GoMLX | `-tags gomlx` | Native graph LoRA and NF4 QLoRA. |
| MLX C | `-tags mlx` on macOS arm64 | GPU default; affine int4 only for QLoRA. |
| CUDA | `-tags cuda` on Linux/NVIDIA | cuBLAS LoRA and int4/NF4 QLoRA kernels. |

The CUDA, MLX, and GoMLX packages are optional imports. Normal builds do not
require CUDA, MLX C, cgo, or a GPU.

## Errors, ownership, and concurrency

- Validate configuration before building a model with `Config.Validate`.
- Use `errors.Is` with exported sentinel errors such as `lora.ErrInputShape`,
  `qlora.ErrMergeUnsupported`, and `backend.ErrCUDAUnavailable`.
- Returned tensor data, parameter slices, target slices, and packed storage are
  copied when documented as a copy. Framework tensors remain owned by their
  backend.
- Layers, adapters, CPU tensors, and native contexts are mutable and are not
  safe for concurrent mutation. Synchronize host access.
- An MLX `Context` is goroutine-confined until `Close`; close child arrays before
  its context.

## Supported boundary

This module supports linear LoRA/QLoRA adapters, int4/NF4 base weights, NF4
double quantization, and the PEFT LoRA file layout. It does not implement
multi-adapter switching, DoRA, AdaLoRA, prompt methods, convolution layers, or
a bitsandbytes runtime. See [compatibility](compatibility.md) and
[PEFT parity](peft-parity.md) for the tested matrix.
