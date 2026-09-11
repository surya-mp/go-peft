//go:build !cuda || !linux || !cgo

package cuda

import (
	"github.com/surya-mp/go-peft/backend"
	"github.com/surya-mp/go-peft/lora"
	"github.com/surya-mp/go-peft/qlora"
)

// TrainLoRA reports that this binary was not built with CUDA support.
func TrainLoRA(*Engine, *lora.Linear, backend.Tensor, backend.Tensor, float32) (float32, error) {
	return 0, backend.ErrCUDAUnavailable
}

// TrainQLoRA reports that this binary was not built with CUDA support.
func TrainQLoRA(*Engine, *qlora.Linear, backend.Tensor, backend.Tensor, float32) (float32, error) {
	return 0, backend.ErrCUDAUnavailable
}
