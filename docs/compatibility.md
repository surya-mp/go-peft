# Compatibility

| Feature | Status |
| --- | --- |
| LoRA / QLoRA CPU | Tested |
| NF4 / double quantization | Tested on CPU; GoMLX NF4 tested |
| Hugging Face PEFT layout | Go LoRA/QLoRA SafeTensors round-trips |
| PEFT LoRA behavior | Go parity tests for injection, A/B lifecycle, merge, dropout, state, and QLoRA contract |
| GoMLX LoRA / QLoRA graphs | Tested |
| MLX LoRA / affine-int4 QLoRA | GPU forward, persistence, native-gradient training, and adapter dropout on Apple Silicon |
| CUDA execution | Validated native Linux NVIDIA bridge: LoRA, int4/NF4 QLoRA, training, and dropout |

The GoMLX dependency currently cannot run under Go's race detector.

Native CUDA validation details and the latest uncached test output are recorded in
[CUDA validation and context fix](cuda-context-fix-2026-09-11.md#fresh-validation-after-benchmark-publication).

MLX C exposes native packed affine int4 operations only. Its QLoRA bridge
therefore rejects NF4 and double quantization rather than silently dequantizing
the base model. NF4/double quant are available on CPU, GoMLX, and CUDA.
An MLX context is goroutine-confined until it is closed.
