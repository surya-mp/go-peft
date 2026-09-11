# PEFT parity

The Go suite ports the supported LoRA behavior from Hugging Face PEFT's
low-level, LoRA-layer, and adapter-state tests. It does not claim coverage for
PEFT methods this module does not implement.

| PEFT behavior | Go coverage |
| --- | --- |
| Inject named linear targets | `lora/TestInject`, `qlora/TestInject` |
| Freeze base; train A/B | `lora/TestPEFTParityLoRALifecycle`, `qlora/TestPEFTParityQLoRAContract` |
| A/B initialization and scaling | `lora/TestPEFTParityLoRALifecycle` |
| Active adapter forward | `lora/TestPEFTParityLoRALifecycle` |
| Merge and restore | `lora/TestPEFTParityLoRALifecycle` |
| Dropout and bias modes | `lora/TestPEFTParityZeroDropout`, `TestPEFTParityTrainingDropoutAndBiasModes` |
| Adapter state round-trip and strict rejection | `format/huggingface/TestLoRAAdapterRoundTrip`, `TestQLoRAAdapterRoundTrip`, `TestPEFTParityAdapterStateValidation` |
| Portable `lora_only` bias state | `format/huggingface/TestPEFTParityPortableBiasState` |
| NF4 and double quantization | `qlora/TestNF4DoubleQuantForward`, `TestPEFTParityQLoRAContract` |
| Quantized merge rejection | `qlora/TestMergeIsRejected`, `TestPEFTParityQLoRAContract` |

Unsupported PEFT features include multi-adapter switching, DoRA, AdaLoRA,
prompt methods, token adapters, convolution layers, and bitsandbytes runtime
integration.

## Backend conformance

| Backend | Verified Go coverage |
| --- | --- |
| CPU | Full parity matrix, quantization, serialization, and race tests |
| GoMLX | Native LoRA/QLoRA graph and training tests |
| MLX | Native GPU forward, persistence, adapter-training, and dropout-training tests |
| CUDA | Native source and tests present; NVIDIA execution pending |
