//go:build !cuda || !linux || !cgo

// Package cuda provides the opt-in NVIDIA CUDA backend.
package cuda

import (
	"math/rand"

	"github.com/surya-mp/go-peft/backend"
)

// Engine is unavailable without Linux, cgo, and the cuda build tag.
type Engine struct{}

// New reports that this binary was not built with CUDA support.
func New() (*Engine, error) { return nil, backend.ErrCUDAUnavailable }

// NewDevice reports that this binary was not built with CUDA support.
func NewDevice(int) (*Engine, error) { return nil, backend.ErrCUDAUnavailable }

func (*Engine) Close() error { return nil }

func (*Engine) New(int, int) (backend.Tensor, error) { return nil, backend.ErrCUDAUnavailable }

func (*Engine) Clone(backend.Tensor) (backend.Tensor, error) { return nil, backend.ErrCUDAUnavailable }

func (*Engine) Copy(backend.Tensor, backend.Tensor) error { return backend.ErrCUDAUnavailable }

func (*Engine) Shape(backend.Tensor) (int, int, error) { return 0, 0, backend.ErrCUDAUnavailable }

func (*Engine) EncodeFloat32(backend.Tensor) ([]float32, int, int, error) {
	return nil, 0, 0, backend.ErrCUDAUnavailable
}

func (*Engine) DecodeFloat32(int, int, []float32) (backend.Tensor, error) {
	return nil, backend.ErrCUDAUnavailable
}

func (*Engine) Zero(backend.Tensor) error { return backend.ErrCUDAUnavailable }

func (*Engine) Uniform(backend.Tensor, float32, float32, *rand.Rand) error {
	return backend.ErrCUDAUnavailable
}

func (*Engine) Dropout(backend.Tensor, float32, *rand.Rand) error { return backend.ErrCUDAUnavailable }

func (*Engine) AddRow(backend.Tensor, backend.Tensor) error { return backend.ErrCUDAUnavailable }

func (*Engine) Gemm(backend.Tensor, backend.Tensor, bool, backend.Tensor, bool, float32, float32) error {
	return backend.ErrCUDAUnavailable
}

func (*Engine) QuantizedLinear(backend.Tensor, backend.Tensor, backend.QuantizedWeight, float32, float32) error {
	return backend.ErrCUDAUnavailable
}

var (
	_ backend.EagerEngine           = (*Engine)(nil)
	_ backend.Float32Codec          = (*Engine)(nil)
	_ backend.QuantizedLinearEngine = (*Engine)(nil)
)
