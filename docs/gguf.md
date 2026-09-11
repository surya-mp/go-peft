# GGUF

`format/gguf` reads little-endian GGUF v3 metadata and tensor indexes without
loading quantized tensor payloads. It reports the architecture, alignment,
quantization version, tensor names, shapes, GGML types, and absolute offsets.

```sh
go run ./cmd/go-peft inspect-gguf --model ./model.gguf
```

Use this to inspect local GGUF models and match their tensor names to LoRA
targets. A backend must explicitly implement a GGML quantization type before it
can execute or train against that tensor.
