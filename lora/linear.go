package lora

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/surya-mp/go-peft/backend"
	"github.com/surya-mp/go-peft/peft"
)

// Linear adds trainable low-rank matrices to a frozen linear weight.
type Linear struct {
	name     string
	engine   backend.Engine
	weight   backend.Tensor
	bias     backend.Tensor
	a        backend.Tensor
	b        backend.Tensor
	in       int
	out      int
	rank     int
	scaling  float32
	dropout  float32
	biasMode BiasMode
	merged   bool
}

// NewLinear initializes A with Kaiming-uniform values and B with zeros.
func NewLinear(name string, engine backend.Engine, weight, bias backend.Tensor, config Config, rng *rand.Rand) (*Linear, error) {
	if name == "" {
		return nil, ErrInvalidName
	}
	if engine == nil {
		return nil, fmt.Errorf("%w: engine", ErrInvalidWeight)
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	if rng == nil {
		return nil, ErrRandomSource
	}

	out, in, err := engine.Shape(weight)
	if err != nil || out <= 0 || in <= 0 {
		return nil, fmt.Errorf("%w: %v", ErrInvalidWeight, err)
	}
	if bias != nil {
		biasRows, biasCols, shapeErr := engine.Shape(bias)
		if shapeErr != nil || biasRows != 1 || biasCols != out {
			return nil, ErrInvalidBiasShape
		}
	}

	a, err := engine.New(config.Rank, in)
	if err != nil {
		return nil, fmt.Errorf("lora: allocate A: %w", err)
	}
	b, err := engine.New(out, config.Rank)
	if err != nil {
		return nil, fmt.Errorf("lora: allocate B: %w", err)
	}
	bound := float32(1 / math.Sqrt(float64(in)))
	if err := engine.Uniform(a, -bound, bound, rng); err != nil {
		return nil, fmt.Errorf("lora: initialize A: %w", err)
	}
	if err := engine.Zero(b); err != nil {
		return nil, fmt.Errorf("lora: initialize B: %w", err)
	}

	return &Linear{
		name: name, engine: engine, weight: weight, bias: bias, a: a, b: b,
		in: in, out: out, rank: config.Rank, scaling: config.Alpha / float32(config.Rank), dropout: config.Dropout, biasMode: config.Bias,
	}, nil
}

// Name returns the model module name.
func (l *Linear) Name() string { return l.name }

// A returns the rank-by-input trainable matrix.
func (l *Linear) A() backend.Tensor { return l.a }

// B returns the output-by-rank trainable matrix.
func (l *Linear) B() backend.Tensor { return l.b }

// Weight returns the frozen base weight, or its merged equivalent.
func (l *Linear) Weight() backend.Tensor { return l.weight }

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

// Merged reports whether the LoRA update is in the base weight.
func (l *Linear) Merged() bool { return l.merged }

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

// Forward allocates its output and workspace. Reuse ForwardInto in hot paths.
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

// ForwardInto computes into caller-owned output and workspace tensors.
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

	if err := l.engine.Gemm(output, input, false, l.weight, true, 1, 0); err != nil {
		return fmt.Errorf("lora: base linear: %w", err)
	}
	if l.bias != nil {
		if err := l.addBias(output); err != nil {
			return err
		}
	}
	if l.merged {
		return nil
	}
	if err := l.engine.Gemm(workspace, input, false, l.a, true, 1, 0); err != nil {
		return fmt.Errorf("lora: A projection: %w", err)
	}
	if training {
		if err := l.engine.Dropout(workspace, l.dropout, rng); err != nil {
			return fmt.Errorf("lora: dropout: %w", err)
		}
	}
	if err := l.engine.Gemm(output, workspace, false, l.b, true, l.scaling, 1); err != nil {
		return fmt.Errorf("lora: B projection: %w", err)
	}
	return nil
}

// Merge adds scaling * B * A to the base weight.
func (l *Linear) Merge() error {
	if l.merged {
		return ErrAlreadyMerged
	}
	if err := l.engine.Gemm(l.weight, l.b, false, l.a, false, l.scaling, 1); err != nil {
		return fmt.Errorf("lora: merge %q: %w", l.name, err)
	}
	l.merged = true
	return nil
}

// Unmerge removes scaling * B * A from the base weight.
func (l *Linear) Unmerge() error {
	if !l.merged {
		return ErrNotMerged
	}
	if err := l.engine.Gemm(l.weight, l.b, false, l.a, false, -l.scaling, 1); err != nil {
		return fmt.Errorf("lora: unmerge %q: %w", l.name, err)
	}
	l.merged = false
	return nil
}

// Parameters returns base and adapter parameter metadata.
func (l *Linear) Parameters() []peft.Parameter {
	parameters := make([]peft.Parameter, 0, 4)
	parameters = append(parameters, peft.NewParameter(l.name+".weight", []int{l.out, l.in}, false))
	if l.bias != nil {
		parameters = append(parameters, peft.NewParameter(l.name+".bias", []int{l.out}, l.biasMode != BiasNone))
	}
	parameters = append(parameters,
		peft.NewParameter(l.name+".lora_A.weight", []int{l.rank, l.in}, true),
		peft.NewParameter(l.name+".lora_B.weight", []int{l.out, l.rank}, true),
	)
	return parameters
}

// AdapterParameters returns only parameters trainable through this adapter.
func (l *Linear) AdapterParameters() []peft.Parameter {
	parameters := make([]peft.Parameter, 0, 3)
	if l.bias != nil && l.biasMode != BiasNone {
		parameters = append(parameters, peft.NewParameter(l.name+".bias", []int{l.out}, true))
	}
	parameters = append(parameters,
		peft.NewParameter(l.name+".lora_A.weight", []int{l.rank, l.in}, true),
		peft.NewParameter(l.name+".lora_B.weight", []int{l.out, l.rank}, true),
	)
	return parameters
}

func (l *Linear) addBias(output backend.Tensor) error {
	return l.engine.AddRow(output, l.bias)
}
