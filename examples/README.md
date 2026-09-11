# Examples

Every integration has matching LoRA and QLoRA examples.

| Integration | LoRA | QLoRA |
| --- | --- | --- |
| CPU inference | `go run ./examples/lora/cpu` | `go run ./examples/qlora/cpu` |
| CUDA GPU | `go run -tags cuda ./examples/lora/cuda` | `go run -tags cuda ./examples/qlora/cuda` |
| GoMLX sentiment-model training | `go run -tags gomlx ./examples/lora/gomlx` | `go run -tags gomlx ./examples/qlora/gomlx` |
| MLX GPU adapter training | `go run -tags mlx ./examples/lora/mlx` | `go run -tags mlx ./examples/qlora/mlx` |
| Hugging Face PEFT files | `go run ./examples/lora/huggingface -out adapter` | `go run ./examples/qlora/huggingface -out adapter` |
| Host-model injection | `go run ./examples/lora/injection` | `go run ./examples/qlora/injection` |

The MLX QLoRA example uses MLX affine int4. MLX C has no native NF4/double
quantized matrix operation, so those configurations are rejected. CPU, GoMLX,
and CUDA cover NF4; CPU, GoMLX, and CUDA also cover double quantization.

The GoMLX examples train frozen-base sentiment classifiers and report held-out
classification accuracy. They use manually vectorized inputs so no tokenizer or
dataset package is required.

CUDA examples require `sh scripts/build_cuda.sh`, Linux, and an NVIDIA GPU.
On Windows, use WSL2 Ubuntu with GPU passthrough. Without CUDA, they report
CUDA unavailable and do not run on CPU.
