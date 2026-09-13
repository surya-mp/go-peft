# Changelog

## Unreleased

### Added

- Adapter-only optimizer-variable selection and update helpers.
- Go-style API field documentation and executable LoRA/QLoRA package examples.
- LoRA and QLoRA core APIs, quantization, framework bridges, and examples.
- SafeTensors and Hugging Face PEFT adapter serialization.
- Opt-in CUDA cuBLAS LoRA, packed int4/NF4 QLoRA, and adapter-update bridge.
- Native MLX and CUDA training dropout, plus CPU/MLX forward benchmark baselines.
- CI formatting, vet, race, compatibility, and lint checks.

### Fixed

- CLI usage write-error propagation and CUDA example cleanup on early exit.
- GGUF malformed-fixture test bounds checking.
