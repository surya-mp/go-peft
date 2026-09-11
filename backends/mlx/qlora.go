//go:build darwin && arm64 && mlx

package mlx

import (
	"errors"
	"math"
	"math/rand"

	"github.com/surya-mp/go-peft/qlora"
)

var ErrQLoRAQuantization = errors.New("mlx: native QLoRA requires affine int4 without double quantization")

// QLoRALinear applies F32 LoRA matrices to a frozen MLX-packed int4 weight.
type QLoRALinear struct {
	weight    *Array
	scales    *Array
	biases    *Array
	bias      *Array
	a         *Array
	b         *Array
	rank      int
	scaling   float32
	dropout   float32
	groupSize int
}

// NewQLoRALinear quantizes weight once using MLX's native affine 4-bit format.
func NewQLoRALinear(weight, bias *Array, config qlora.Config, rng *rand.Rand) (*QLoRALinear, error) {
	if weight == nil || rng == nil {
		return nil, errors.New("mlx: weight and random source are required")
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	if config.Quantization != qlora.QuantizationInt4 || config.DoubleQuant {
		return nil, ErrQLoRAQuantization
	}
	shape, err := weight.Shape()
	if err != nil {
		return nil, err
	}
	if len(shape) != 2 || shape[0] <= 0 || shape[1] <= 0 || shape[1]%config.BlockSize != 0 {
		return nil, ErrShape
	}
	out, in := shape[0], shape[1]
	if bias != nil {
		biasShape, err := bias.Shape()
		if err != nil {
			return nil, err
		}
		if bias.context != weight.context || len(biasShape) != 1 || biasShape[0] != out {
			return nil, ErrShape
		}
	}
	packed, scales, biases, err := weight.Quantize(config.BlockSize, 4)
	if err != nil {
		return nil, err
	}
	cleanup := func() {
		_ = packed.Close()
		_ = scales.Close()
		_ = biases.Close()
	}
	aData := make([]float32, config.LoRA.Rank*in)
	bound := float32(1 / math.Sqrt(float64(in)))
	for index := range aData {
		aData[index] = (2*rng.Float32() - 1) * bound
	}
	a, err := weight.context.Float32([]int{config.LoRA.Rank, in}, aData)
	if err != nil {
		cleanup()
		return nil, err
	}
	b, err := weight.context.Float32([]int{out, config.LoRA.Rank}, make([]float32, out*config.LoRA.Rank))
	if err != nil {
		_ = a.Close()
		cleanup()
		return nil, err
	}
	return &QLoRALinear{
		weight: packed, scales: scales, biases: biases, bias: bias, a: a, b: b,
		rank: config.LoRA.Rank, scaling: config.LoRA.Alpha / float32(config.LoRA.Rank), dropout: config.LoRA.Dropout, groupSize: config.BlockSize,
	}, nil
}

// A returns the rank-by-input trainable MLX adapter matrix.
func (l *QLoRALinear) A() *Array { return l.a }

// B returns the output-by-rank trainable MLX adapter matrix.
func (l *QLoRALinear) B() *Array { return l.b }

// Forward computes a quantized base projection plus its LoRA update on MLX.
func (l *QLoRALinear) Forward(input *Array) (*Array, error) {
	return l.forward(input, nil)
}

// ForwardTraining computes a training projection with inverted adapter dropout.
func (l *QLoRALinear) ForwardTraining(input *Array) (*Array, error) {
	mask, err := l.trainingMask(input)
	if err != nil {
		return nil, err
	}
	defer mask.Close()
	return l.forward(input, mask)
}

// Close releases the packed weight and trainable matrices. It does not close bias.
func (l *QLoRALinear) Close() error {
	if l == nil {
		return nil
	}
	var first error
	for _, array := range []*Array{l.b, l.a, l.biases, l.scales, l.weight} {
		if err := array.Close(); err != nil && first == nil {
			first = err
		}
	}
	return first
}

func (l *QLoRALinear) forward(input, mask *Array) (*Array, error) {
	if l == nil || input == nil || input.context != l.weight.context {
		return nil, ErrContext
	}
	output, err := input.QuantizedMatMul(l.weight, l.scales, l.biases, true, l.groupSize, 4)
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

func (l *QLoRALinear) trainingMask(input *Array) (*Array, error) {
	return adapterDropoutMask(input, l.rank, l.dropout)
}

func adapterDropoutMask(input *Array, rank int, dropout float32) (*Array, error) {
	if input == nil || rank <= 0 || dropout < 0 || dropout >= 1 {
		return nil, ErrShape
	}
	shape, err := input.Shape()
	if err != nil {
		return nil, err
	}
	if len(shape) < 2 {
		return nil, ErrShape
	}
	shape[len(shape)-1] = rank
	if dropout == 0 {
		values := make([]float32, product(shape))
		for index := range values {
			values[index] = 1
		}
		return input.context.Float32(shape, values)
	}
	mask, err := input.context.Bernoulli(shape, 1-dropout)
	if err != nil {
		return nil, err
	}
	scaled, err := mask.Scale(1 / (1 - dropout))
	_ = mask.Close()
	return scaled, err
}

func product(shape []int) int {
	value := 1
	for _, dimension := range shape {
		value *= dimension
	}
	return value
}
