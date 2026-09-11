package qlora

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/surya-mp/go-peft/backend"
	"github.com/surya-mp/go-peft/lora"
	"github.com/surya-mp/go-peft/peft"
	"github.com/surya-mp/go-peft/quantization/int4"
	"github.com/surya-mp/go-peft/quantization/nf4"
)

// Linear applies trainable LoRA matrices to a frozen int4 base weight.
type Linear struct {
	name      string
	engine    backend.Engine
	quantized backend.QuantizedLinearEngine
	weight    backend.QuantizedWeight
	bias      backend.Tensor
	a         backend.Tensor
	b         backend.Tensor
	in        int
	out       int
	rank      int
	scaling   float32
	dropout   float32
	biasMode  lora.BiasMode
}

// QuantizeLinear quantizes a float32 base weight and creates a QLoRA layer.
func QuantizeLinear(name string, engine backend.Engine, weight, bias backend.Tensor, config Config, rng *rand.Rand) (*Linear, error) {
	if err := config.Validate(); err != nil {
		return nil, err
	}
	codec, ok := engine.(backend.Float32Codec)
	if !ok {
		return nil, ErrUnsupportedEngine
	}
	values, rows, cols, err := codec.EncodeFloat32(weight)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidWeight, err)
	}
	quantized, err := quantize(rows, cols, values, config)
	if err != nil {
		return nil, err
	}
	return NewLinear(name, engine, quantized, bias, config, rng)
}

func quantize(rows, cols int, values []float32, config Config) (backend.QuantizedWeight, error) {
	if config.Quantization == QuantizationNF4 {
		scaleBlockSize := 0
		if config.DoubleQuant {
			scaleBlockSize = config.ScaleBlockSize
		}
		return nf4.Quantize(rows, cols, config.BlockSize, scaleBlockSize, values)
	}
	return int4.Quantize(rows, cols, config.BlockSize, values)
}

// NewLinear creates a layer around an existing quantized weight.
func NewLinear(name string, engine backend.Engine, weight backend.QuantizedWeight, bias backend.Tensor, config Config, rng *rand.Rand) (*Linear, error) {
	if name == "" {
		return nil, ErrInvalidName
	}
	if engine == nil || weight == nil || weight.Rows() <= 0 || weight.Cols() <= 0 {
		return nil, ErrInvalidWeight
	}
	quantized, ok := engine.(backend.QuantizedLinearEngine)
	if !ok {
		return nil, ErrUnsupportedEngine
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	if rng == nil {
		return nil, ErrRandomSource
	}
	out, in := weight.Rows(), weight.Cols()
	if bias != nil {
		rows, cols, err := engine.Shape(bias)
		if err != nil || rows != 1 || cols != out {
			return nil, ErrInvalidBiasShape
		}
	}
	a, err := engine.New(config.LoRA.Rank, in)
	if err != nil {
		return nil, fmt.Errorf("qlora: allocate A: %w", err)
	}
	b, err := engine.New(out, config.LoRA.Rank)
	if err != nil {
		return nil, fmt.Errorf("qlora: allocate B: %w", err)
	}
	bound := float32(1 / math.Sqrt(float64(in)))
	if err := engine.Uniform(a, -bound, bound, rng); err != nil {
		return nil, err
	}
	if err := engine.Zero(b); err != nil {
		return nil, err
	}
	return &Linear{
		name: name, engine: engine, quantized: quantized, weight: weight, bias: bias, a: a, b: b,
		in: in, out: out, rank: config.LoRA.Rank, scaling: config.LoRA.Alpha / float32(config.LoRA.Rank),
		dropout: config.LoRA.Dropout, biasMode: config.LoRA.Bias,
	}, nil
}

// Name returns the model module name.
func (l *Linear) Name() string { return l.name }

// Weight returns the frozen int4 base weight.
func (l *Linear) Weight() backend.QuantizedWeight { return l.weight }

// QuantizationMetadata returns storage metadata when the weight provides it.
func (l *Linear) QuantizationMetadata() (backend.QuantizationMetadata, bool) {
	metadata, ok := l.weight.(backend.QuantizedWeightMetadata)
	if !ok {
		return backend.QuantizationMetadata{}, false
	}
	return metadata.QuantizationMetadata(), true
}

// A returns the rank-by-input trainable matrix.
func (l *Linear) A() backend.Tensor { return l.a }

// B returns the output-by-rank trainable matrix.
func (l *Linear) B() backend.Tensor { return l.b }

// Bias returns the optional base bias.
func (l *Linear) Bias() backend.Tensor { return l.bias }

// Scaling returns alpha / rank.
func (l *Linear) Scaling() float32 { return l.scaling }

// Dropout returns the configured LoRA-branch dropout probability.
func (l *Linear) Dropout() float32 { return l.dropout }

// InFeatures returns the input feature count.
func (l *Linear) InFeatures() int { return l.in }

// OutFeatures returns the output feature count.
func (l *Linear) OutFeatures() int { return l.out }

// Rank returns the LoRA rank.
func (l *Linear) Rank() int { return l.rank }

// Merged is always false because QLoRA keeps its base weight quantized.
func (l *Linear) Merged() bool { return false }

// NewWorkspace allocates output and rank workspace for input.
func (l *Linear) NewWorkspace(input backend.Tensor) (output, workspace backend.Tensor, err error) {
	rows, cols, err := l.engine.Shape(input)
	if err != nil {
		return nil, nil, err
	}
	if cols != l.in {
		return nil, nil, ErrInputShape
	}
	output, err = l.engine.New(rows, l.out)
	if err != nil {
		return nil, nil, err
	}
	workspace, err = l.engine.New(rows, l.rank)
	if err != nil {
		return nil, nil, err
	}
	return output, workspace, nil
}

// Forward allocates output and workspace. Reuse ForwardInto in hot paths.
func (l *Linear) Forward(input backend.Tensor) (backend.Tensor, error) {
	output, workspace, err := l.NewWorkspace(input)
	if err != nil {
		return nil, err
	}
	if err := l.ForwardInto(output, workspace, input, false, nil); err != nil {
		return nil, err
	}
	return output, nil
}

// ForwardTraining applies configured dropout to the LoRA branch.
func (l *Linear) ForwardTraining(input backend.Tensor, rng *rand.Rand) (backend.Tensor, error) {
	output, workspace, err := l.NewWorkspace(input)
	if err != nil {
		return nil, err
	}
	if err := l.ForwardInto(output, workspace, input, true, rng); err != nil {
		return nil, err
	}
	return output, nil
}

// ForwardInto computes with packed int4 base weights and caller-owned tensors.
func (l *Linear) ForwardInto(output, workspace, input backend.Tensor, training bool, rng *rand.Rand) error {
	rows, cols, err := l.engine.Shape(input)
	if err != nil {
		return err
	}
	if cols != l.in {
		return ErrInputShape
	}
	outputRows, outputCols, err := l.engine.Shape(output)
	if err != nil {
		return err
	}
	workspaceRows, workspaceCols, err := l.engine.Shape(workspace)
	if err != nil {
		return err
	}
	if outputRows != rows || outputCols != l.out || workspaceRows != rows || workspaceCols != l.rank {
		return ErrWorkspaceShape
	}
	if err := l.quantized.QuantizedLinear(output, input, l.weight, 1, 0); err != nil {
		return fmt.Errorf("qlora: base linear: %w", err)
	}
	if l.bias != nil {
		if err := l.engine.AddRow(output, l.bias); err != nil {
			return err
		}
	}
	if err := l.engine.Gemm(workspace, input, false, l.a, true, 1, 0); err != nil {
		return fmt.Errorf("qlora: A projection: %w", err)
	}
	if training {
		if err := l.engine.Dropout(workspace, l.dropout, rng); err != nil {
			return fmt.Errorf("qlora: dropout: %w", err)
		}
	}
	if err := l.engine.Gemm(output, workspace, false, l.b, true, l.scaling, 1); err != nil {
		return fmt.Errorf("qlora: B projection: %w", err)
	}
	return nil
}

// Merge always fails because merging into int4 storage is lossy.
func (l *Linear) Merge() error { return ErrMergeUnsupported }

// Unmerge always fails because QLoRA never merges into its quantized base.
func (l *Linear) Unmerge() error { return ErrMergeUnsupported }

// Parameters returns base and adapter parameter metadata.
func (l *Linear) Parameters() []peft.Parameter {
	parameters := make([]peft.Parameter, 0, 4)
	parameters = append(parameters, peft.NewParameter(l.name+".weight", []int{l.out, l.in}, false))
	if l.bias != nil {
		parameters = append(parameters, peft.NewParameter(l.name+".bias", []int{l.out}, l.biasMode != lora.BiasNone))
	}
	parameters = append(parameters,
		peft.NewParameter(l.name+".lora_A.weight", []int{l.rank, l.in}, true),
		peft.NewParameter(l.name+".lora_B.weight", []int{l.out, l.rank}, true),
	)
	return parameters
}

// AdapterParameters returns only trainable metadata.
func (l *Linear) AdapterParameters() []peft.Parameter {
	parameters := make([]peft.Parameter, 0, 3)
	if l.bias != nil && l.biasMode != lora.BiasNone {
		parameters = append(parameters, peft.NewParameter(l.name+".bias", []int{l.out}, true))
	}
	parameters = append(parameters,
		peft.NewParameter(l.name+".lora_A.weight", []int{l.rank, l.in}, true),
		peft.NewParameter(l.name+".lora_B.weight", []int{l.out, l.rank}, true),
	)
	return parameters
}
