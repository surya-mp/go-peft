//go:build darwin && arm64 && mlx

package mlx

import (
	"errors"
	"math"
	"math/rand"

	"github.com/surya-mp/go-peft/lora"
)

// LoRALinear is an MLX-native LoRA projection.
type LoRALinear struct {
	weight  *Array
	bias    *Array
	a       *Array
	b       *Array
	rank    int
	scaling float32
	dropout float32
	merged  bool
}

// NewLoRALinear initializes a trainable MLX LoRA projection around weight.
func NewLoRALinear(weight, bias *Array, config lora.Config, rng *rand.Rand) (*LoRALinear, error) {
	if weight == nil || rng == nil {
		return nil, errors.New("mlx: weight and random source are required")
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	shape, err := weight.Shape()
	if err != nil {
		return nil, err
	}
	if len(shape) != 2 || shape[0] <= 0 || shape[1] <= 0 {
		return nil, ErrShape
	}
	out, in := shape[0], shape[1]
	if bias != nil {
		biasShape, err := bias.Shape()
		if err != nil {
			return nil, err
		}
		if len(biasShape) != 1 || biasShape[0] != out {
			return nil, ErrShape
		}
	}
	aData := make([]float32, config.Rank*in)
	bound := float32(1 / math.Sqrt(float64(in)))
	for index := range aData {
		aData[index] = (2*rng.Float32() - 1) * bound
	}
	a, err := weight.context.Float32([]int{config.Rank, in}, aData)
	if err != nil {
		return nil, err
	}
	b, err := weight.context.Float32([]int{out, config.Rank}, make([]float32, out*config.Rank))
	if err != nil {
		_ = a.Close()
		return nil, err
	}
	return &LoRALinear{
		weight: weight, bias: bias, a: a, b: b, rank: config.Rank, scaling: config.Alpha / float32(config.Rank), dropout: config.Dropout,
	}, nil
}

// A returns the rank-by-input MLX adapter matrix.
func (l *LoRALinear) A() *Array { return l.a }

// B returns the output-by-rank MLX adapter matrix.
func (l *LoRALinear) B() *Array { return l.b }

// Forward computes the inference projection without allocating Go buffers.
func (l *LoRALinear) Forward(input *Array) (*Array, error) {
	return l.forward(input, nil)
}

// ForwardTraining computes a training projection with inverted adapter dropout.
func (l *LoRALinear) ForwardTraining(input *Array) (*Array, error) {
	mask, err := l.trainingMask(input)
	if err != nil {
		return nil, err
	}
	defer mask.Close()
	return l.forward(input, mask)
}

func (l *LoRALinear) forward(input, mask *Array) (*Array, error) {
	if input == nil || input.context != l.weight.context {
		return nil, ErrContext
	}
	weightT, err := l.weight.Transpose()
	if err != nil {
		return nil, err
	}
	defer weightT.Close()
	output, err := input.MatMul(weightT)
	if err != nil {
		return nil, err
	}
	if l.bias != nil {
		withBias, err := output.Add(l.bias)
		_ = output.Close()
		if err != nil {
			return nil, err
		}
		output = withBias
	}
	if l.merged {
		return output, nil
	}
	aT, err := l.a.Transpose()
	if err != nil {
		_ = output.Close()
		return nil, err
	}
	defer aT.Close()
	projected, err := input.MatMul(aT)
	if err != nil {
		_ = output.Close()
		return nil, err
	}
	if mask != nil {
		dropped, err := projected.Mul(mask)
		_ = projected.Close()
		if err != nil {
			_ = output.Close()
			return nil, err
		}
		projected = dropped
	}
	bT, err := l.b.Transpose()
	if err != nil {
		_ = projected.Close()
		_ = output.Close()
		return nil, err
	}
	defer bT.Close()
	delta, err := projected.MatMul(bT)
	_ = projected.Close()
	if err != nil {
		_ = output.Close()
		return nil, err
	}
	scaled, err := delta.Scale(l.scaling)
	_ = delta.Close()
	if err != nil {
		_ = output.Close()
		return nil, err
	}
	result, err := output.Add(scaled)
	_ = output.Close()
	_ = scaled.Close()
	return result, err
}

func (l *LoRALinear) trainingMask(input *Array) (*Array, error) {
	return adapterDropoutMask(input, l.rank, l.dropout)
}
