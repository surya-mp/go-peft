//go:build gomlx

// Package gomlx builds LoRA layers in GoMLX computation graphs.
package gomlx

import (
	"errors"
	"fmt"
	"math"
	"math/rand"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/surya-mp/go-peft/lora"
)

// Linear adds LoRA graph nodes around a GoMLX linear weight variable.
type Linear struct {
	name    string
	weight  *model.Variable
	bias    *model.Variable
	a       *model.Variable
	b       *model.Variable
	scaling float32
	dropout float32
}

// NewLinear creates trainable A/B variables under scope/lora.
func NewLinear(scope *model.Scope, weight, bias *model.Variable, config lora.Config, rng *rand.Rand) (*Linear, error) {
	return NewNamedLinear("", scope, weight, bias, config, rng)
}

// NewNamedLinear creates a native layer for one model module.
func NewNamedLinear(name string, scope *model.Scope, weight, bias *model.Variable, config lora.Config, rng *rand.Rand) (*Linear, error) {
	if scope == nil || weight == nil || rng == nil {
		return nil, errors.New("gomlx: scope, weight, and random source are required")
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	shape := weight.Shape()
	if shape.Rank() != 2 || !supportedDType(shape.DType) || shape.Dimensions[0] <= 0 || shape.Dimensions[1] <= 0 {
		return nil, errors.New("gomlx: weight must be a non-empty floating-point matrix")
	}
	out, in := shape.Dimensions[0], shape.Dimensions[1]
	if bias != nil {
		biasShape := bias.Shape()
		if biasShape.Rank() != 1 || biasShape.Dimensions[0] != out {
			return nil, errors.New("gomlx: bias shape must be [out_features]")
		}
	}

	aValues := make([]float32, config.Rank*in)
	bound := float32(1 / math.Sqrt(float64(in)))
	for index := range aValues {
		aValues[index] = (2*rng.Float32() - 1) * bound
	}
	bValues := make([]float32, out*config.Rank)
	aInitial, err := matrixValue(shape.DType, config.Rank, in, aValues)
	if err != nil {
		return nil, err
	}
	bInitial, err := matrixValue(shape.DType, out, config.Rank, bValues)
	if err != nil {
		return nil, err
	}

	loraScope := scope.In("lora")
	a := loraScope.VariableWithValue("A", aInitial).SetTrainable(true)
	b := loraScope.VariableWithValue("B", bInitial).SetTrainable(true)
	weight.SetTrainable(false)
	if bias != nil {
		bias.SetTrainable(config.Bias != lora.BiasNone)
	}
	return &Linear{name: name, weight: weight, bias: bias, a: a, b: b, scaling: config.Alpha / float32(config.Rank), dropout: config.Dropout}, nil
}

// Apply appends the LoRA projection to input's graph.
func (l *Linear) Apply(scope *model.Scope, input *graph.Node) *graph.Node {
	weight := l.weight.NodeValue(input)
	output := graph.MatMul(input, graph.Transpose(weight, -2, -1))
	if l.bias != nil {
		output = graph.Add(output, l.bias.NodeValue(input))
	}
	projected := graph.MatMul(input, graph.Transpose(l.a.NodeValue(input), -2, -1))
	projected = layers.DropoutStatic(scope, projected, float64(l.dropout))
	projected = graph.MatMul(projected, graph.Transpose(l.b.NodeValue(input), -2, -1))
	return graph.Add(output, graph.MulScalar(projected, l.scaling))
}

// A returns GoMLX's trainable rank-by-input variable.
func (l *Linear) A() *model.Variable { return l.a }

// B returns GoMLX's trainable output-by-rank variable.
func (l *Linear) B() *model.Variable { return l.b }

// Weight returns the frozen base variable.
func (l *Linear) Weight() *model.Variable { return l.weight }

// Bias returns the optional base bias variable.
func (l *Linear) Bias() *model.Variable { return l.bias }

// Name returns the model module name supplied during injection.
func (l *Linear) Name() string { return l.name }

func (l *Linear) String() string { return fmt.Sprintf("gomlx.LoRA(scale=%g)", l.scaling) }

func supportedDType(dtype dtypes.DType) bool {
	return dtype == dtypes.Float32 || dtype == dtypes.Float16 || dtype == dtypes.BFloat16
}

func matrixValue(dtype dtypes.DType, rows, cols int, values []float32) (any, error) {
	switch dtype {
	case dtypes.Float32:
		return tensors.FromFlatDataAndDimensions(values, rows, cols), nil
	case dtypes.Float16:
		converted := make([]float16.Float16, len(values))
		for index, value := range values {
			converted[index] = float16.FromFloat32(value)
		}
		return tensors.FromFlatDataAndDimensions(converted, rows, cols), nil
	case dtypes.BFloat16:
		converted := make([]bfloat16.BFloat16, len(values))
		for index, value := range values {
			converted[index] = bfloat16.FromFloat32(value)
		}
		return tensors.FromFlatDataAndDimensions(converted, rows, cols), nil
	default:
		return nil, errors.New("gomlx: only float32, float16, and bfloat16 weights are supported")
	}
}
