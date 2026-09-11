# Training integration

`trainer.Runner` owns accumulation, clipping hooks, reporting, and checkpoint
cadence. The host loop owns batches, autograd, mixed precision, and optimizers.

```go
runner := trainer.Runner{
    Config: trainer.Config{Steps: 100, AccumulationSteps: 4, GradientClip: 1},
    Loop: hostLoop,
    Checkpointer: adapterCheckpoint,
}
err := runner.Run(ctx)
```

`profiles` provides tested target suffixes for Llama, Mistral, Qwen2, Gemma,
and Phi-3. Run `go-peft plan` before injection to confirm actual module names.

```sh
go run ./cmd/go-peft targets --family llama --mode all-linear
go run ./cmd/go-peft plan --family llama --modules modules.txt
go run ./cmd/go-peft validate --adapter adapter-dir
```
