# Hugging Face base models

`format/huggingface/model` reads `config.json` and streams `model.safetensors`
or Hugging Face's `model.safetensors.index.json` shards. It decodes F32, F16,
and BF16 one tensor at a time.

```go
info, err := model.Load("./model", model.Options{
    Filter: func(name string) bool { return strings.HasSuffix(name, "q_proj.weight") },
    OnTensor: func(tensor model.Tensor) error {
        return bridge.LoadBaseTensor(tensor.Name, tensor.Shape, tensor.Data)
    },
})
```

Use `model.Inspect` when only model metadata and shard layout are needed. Its
tensor count is `-1` for an unindexed file until `Load` has streamed it.

Dense Qwen2 and Qwen3 checkpoints use the standard `q_proj`, `k_proj`,
`v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj` LoRA suffixes.
Qwen3-MoE uses the same projections inside experts plus router `gate`; use the
explicit `qwen3-moe` profile for all-linear work.
The loader validates shard paths and every tensor in an index. Pass streamed
tensors to the GoMLX, MLX, CUDA, or another bridge.
