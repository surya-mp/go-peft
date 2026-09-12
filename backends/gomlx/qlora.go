//go:build gomlx

package gomlx

import (
	"errors"
	"math"
	"math/rand"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/nn"
	"github.com/surya-mp/go-peft/lora"
	"github.com/surya-mp/go-peft/qlora"
	"github.com/surya-mp/go-peft/quantization/nf4"
)

// NF4Weight stores [input, output] NF4 codes and per-input block scales.
type NF4Weight struct {
	// InputFeatures is the number of input columns.
	InputFeatures int
	// OutputFeatures is the number of output columns.
	OutputFeatures int
	// BlockSize is the number of output values sharing one scale.
	BlockSize int
	// Packed contains two NF4 codes per byte in [input, output] order.
	Packed []byte
	// Scales contains F32 block scales without double quantization.
	Scales []float32
	// ScaleCodes contains 8-bit block scales with double quantization.
	ScaleCodes []byte
	// ScaleScales contains F32 second-level scales.
	ScaleScales []float32
	// ScaleBlockSize is zero without double quantization.
	ScaleBlockSize int
}

// QuantizeNF4Weight creates GoMLX-compatible NF4 storage from [input, output] values.
func QuantizeNF4Weight(inputFeatures, outputFeatures, blockSize int, values []float32) (*NF4Weight, error) {
	return quantizeNF4Weight(inputFeatures, outputFeatures, blockSize, 0, values)
}

// QuantizeNF4WeightDouble creates NF4 storage with 8-bit double-quantized scales.
func QuantizeNF4WeightDouble(inputFeatures, outputFeatures, blockSize, scaleBlockSize int, values []float32) (*NF4Weight, error) {
	return quantizeNF4Weight(inputFeatures, outputFeatures, blockSize, scaleBlockSize, values)
}

func quantizeNF4Weight(inputFeatures, outputFeatures, blockSize, scaleBlockSize int, values []float32) (*NF4Weight, error) {
	if inputFeatures <= 0 || outputFeatures <= 0 || blockSize <= 0 || inputFeatures > math.MaxInt/outputFeatures || len(values) != inputFeatures*outputFeatures {
		return nil, errors.New("gomlx: invalid NF4 weight")
	}
	if scaleBlockSize < 0 {
		return nil, errors.New("gomlx: invalid NF4 scale block")
	}
	blocks := (outputFeatures + blockSize - 1) / blockSize
	weight := &NF4Weight{
		InputFeatures: inputFeatures, OutputFeatures: outputFeatures, BlockSize: blockSize,
		Packed: make([]byte, inputFeatures*((outputFeatures+1)/2)), Scales: make([]float32, inputFeatures*blocks),
	}
	for input := 0; input < inputFeatures; input++ {
		row := values[input*outputFeatures : (input+1)*outputFeatures]
		for block := 0; block < blocks; block++ {
			start, end := block*blockSize, min((block+1)*blockSize, outputFeatures)
			var scale float32
			for _, value := range row[start:end] {
				absolute := float32(math.Abs(float64(value)))
				if absolute > scale {
					scale = absolute
				}
			}
			weight.Scales[input*blocks+block] = scale
			if scale == 0 {
				continue
			}
			for output := start; output < end; output++ {
				weight.setCode(input, output, nf4.NearestCode(row[output]/scale))
			}
		}
	}
	if scaleBlockSize != 0 {
		weight.doubleQuantizeScales(scaleBlockSize)
	}
	return weight, nil
}

func (w *NF4Weight) validate() error {
	if w == nil || w.InputFeatures <= 0 || w.OutputFeatures <= 0 || w.BlockSize <= 0 {
		return errors.New("gomlx: invalid NF4 weight")
	}
	packed := w.InputFeatures * ((w.OutputFeatures + 1) / 2)
	blocks := (w.OutputFeatures + w.BlockSize - 1) / w.BlockSize
	scales := w.InputFeatures * blocks
	if len(w.Packed) != packed || (len(w.Scales) != scales && len(w.ScaleCodes) != scales) {
		return errors.New("gomlx: invalid NF4 storage")
	}
	if len(w.ScaleCodes) > 0 && (w.ScaleBlockSize <= 0 || len(w.ScaleScales) != w.InputFeatures*((blocks+w.ScaleBlockSize-1)/w.ScaleBlockSize)) {
		return errors.New("gomlx: invalid NF4 storage")
	}
	return nil
}

func (w *NF4Weight) doubleQuantizeScales(scaleBlockSize int) {
	blocks := (w.OutputFeatures + w.BlockSize - 1) / w.BlockSize
	w.ScaleBlockSize = scaleBlockSize
	w.ScaleCodes = make([]byte, len(w.Scales))
	w.ScaleScales = make([]float32, w.InputFeatures*((blocks+scaleBlockSize-1)/scaleBlockSize))
	for input := 0; input < w.InputFeatures; input++ {
		for group := 0; group*scaleBlockSize < blocks; group++ {
			start, end := group*scaleBlockSize, min((group+1)*scaleBlockSize, blocks)
			var maximum float32
			for _, scale := range w.Scales[input*blocks+start : input*blocks+end] {
				if scale > maximum {
					maximum = scale
				}
			}
			groupScale := maximum / 255
			w.ScaleScales[input*((blocks+scaleBlockSize-1)/scaleBlockSize)+group] = groupScale
			if groupScale == 0 {
				continue
			}
			for block := start; block < end; block++ {
				code := int(math.Round(float64(w.Scales[input*blocks+block] / groupScale)))
				if code > 255 {
					code = 255
				}
				w.ScaleCodes[input*blocks+block] = byte(code)
			}
		}
	}
	w.Scales = nil
}

func (w *NF4Weight) setCode(input, output int, code byte) {
	index := input*((w.OutputFeatures+1)/2) + output/2
	if output&1 == 0 {
		w.Packed[index] = w.Packed[index]&0xf0 | code
		return
	}
	w.Packed[index] = w.Packed[index]&0x0f | code<<4
}

// NF4Linear combines a frozen native NF4 projection with trainable GoMLX LoRA variables.
type NF4Linear struct {
	name        string
	packed      *model.Variable
	scales      *model.Variable
	scaleCodes  *model.Variable
	scaleScales *model.Variable
	bias        *model.Variable
	a           *model.Variable
	b           *model.Variable
	in          int
	out         int
	block       int
	scaling     float32
	dropout     float32
}

// NF4BaseLinear is a frozen NF4 projection without LoRA parameters.
// It is used to keep non-adapted base projections quantized during QLoRA.
type NF4BaseLinear struct {
	packed      *model.Variable
	scales      *model.Variable
	scaleCodes  *model.Variable
	scaleScales *model.Variable
	bias        *model.Variable
	in          int
	out         int
	block       int
}

// NF4Module is one host-discovered quantized linear module.
type NF4Module struct {
	// Name is the host model's fully qualified module name.
	Name string
	// Scope owns variables created for the replacement layer.
	Scope *model.Scope
	// Weight is the frozen NF4 base weight.
	Weight *NF4Weight
	// Bias is the optional host bias variable.
	Bias *model.Variable
}

// NF4Replacement associates a host module with its native QLoRA layer.
type NF4Replacement struct {
	// Name identifies the host module to replace.
	Name string
	// Layer is the native GoMLX NF4 LoRA layer for Name.
	Layer *NF4Linear
}

// NF4Model connects host traversal to native QLoRA replacement.
type NF4Model interface {
	NF4LinearModules() ([]NF4Module, error)
	ReplaceNF4LinearModules([]NF4Replacement) error
}

// NF4Adapter owns injected native QLoRA layers.
type NF4Adapter struct {
	name   string
	config qlora.Config
	layers []*NF4Linear
}

// InjectNF4 atomically injects NF4 QLoRA layers into matching host modules.
func InjectNF4(name string, host NF4Model, config qlora.Config, rng *rand.Rand) (*NF4Adapter, error) {
	if name == "" || host == nil || rng == nil {
		return nil, errors.New("gomlx: name, model, and random source are required")
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	if config.Quantization != qlora.QuantizationNF4 {
		return nil, errors.New("gomlx: native QLoRA requires NF4")
	}
	modules, err := host.NF4LinearModules()
	if err != nil {
		return nil, err
	}
	replacements := make([]NF4Replacement, 0, len(modules))
	seen := make(map[string]struct{}, len(modules))
	for _, module := range modules {
		if !lora.Matches(module.Name, config.LoRA.TargetModules) {
			continue
		}
		if module.Name == "" || module.Scope == nil || module.Weight == nil {
			return nil, errors.New("gomlx: invalid NF4 module")
		}
		if _, exists := seen[module.Name]; exists {
			return nil, errors.New("gomlx: duplicate NF4 module")
		}
		layer, err := NewNamedNF4Linear(module.Name, module.Scope, module.Weight, module.Bias, config.LoRA, rng)
		if err != nil {
			return nil, err
		}
		replacements = append(replacements, NF4Replacement{Name: module.Name, Layer: layer})
		seen[module.Name] = struct{}{}
	}
	if len(replacements) == 0 {
		return nil, ErrNoTargetModules
	}
	if err := host.ReplaceNF4LinearModules(replacements); err != nil {
		return nil, err
	}
	layers := make([]*NF4Linear, len(replacements))
	for index, replacement := range replacements {
		layers[index] = replacement.Layer
	}
	config.LoRA.TargetModules = append([]string(nil), config.LoRA.TargetModules...)
	return &NF4Adapter{name: name, config: config, layers: layers}, nil
}

// Name returns the adapter name.
func (a *NF4Adapter) Name() string { return a.name }

// Config returns a detached QLoRA configuration copy.
func (a *NF4Adapter) Config() qlora.Config {
	config := a.config
	config.LoRA.TargetModules = append([]string(nil), config.LoRA.TargetModules...)
	return config
}

// Layers returns native QLoRA layers in injection order.
func (a *NF4Adapter) Layers() []*NF4Linear { return append([]*NF4Linear(nil), a.layers...) }

// TrainableVariables returns the exact variables selected for optimization.
func (a *NF4Adapter) TrainableVariables() []*model.Variable {
	variables := make([]*model.Variable, 0, len(a.layers)*2)
	for _, layer := range a.layers {
		variables = append(variables, layer.TrainableVariables()...)
	}
	return variables
}

// NewNF4Linear creates an NF4 base projection with float32 LoRA parameters.
func NewNF4Linear(scope *model.Scope, weight *NF4Weight, bias *model.Variable, config lora.Config, rng *rand.Rand) (*NF4Linear, error) {
	return NewNamedNF4Linear("", scope, weight, bias, config, rng)
}

// NewNamedNF4Linear creates a named NF4 base projection with float32 LoRA parameters.
func NewNamedNF4Linear(name string, scope *model.Scope, weight *NF4Weight, bias *model.Variable, config lora.Config, rng *rand.Rand) (*NF4Linear, error) {
	if scope == nil || rng == nil {
		return nil, errors.New("gomlx: scope and random source are required")
	}
	if err := weight.validate(); err != nil {
		return nil, err
	}
	if weight.OutputFeatures&1 != 0 {
		return nil, errors.New("gomlx: NF4 output features must be even")
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	if bias != nil && (bias.Shape().Rank() != 1 || bias.Shape().Dimensions[0] != weight.OutputFeatures) {
		return nil, errors.New("gomlx: bias shape must be [out_features]")
	}
	aValues := make([]float32, config.Rank*weight.InputFeatures)
	bound := float32(1 / math.Sqrt(float64(weight.InputFeatures)))
	for index := range aValues {
		aValues[index] = (2*rng.Float32() - 1) * bound
	}
	qScope := scope.In("qlora")
	packed := qScope.VariableWithValue("nf4_packed", tensors.FromFlatDataAndDimensions(weight.Packed, weight.InputFeatures, (weight.OutputFeatures+1)/2)).SetTrainable(false)
	blocks := (weight.OutputFeatures + weight.BlockSize - 1) / weight.BlockSize
	var scales, scaleCodes, scaleScales *model.Variable
	if len(weight.ScaleCodes) == 0 {
		scales = qScope.VariableWithValue("nf4_scales", tensors.FromFlatDataAndDimensions(weight.Scales, weight.InputFeatures, blocks)).SetTrainable(false)
	} else {
		scaleCodes = qScope.VariableWithValue("nf4_scale_codes", tensors.FromFlatDataAndDimensions(weight.ScaleCodes, weight.InputFeatures, blocks)).SetTrainable(false)
		scaleScales = qScope.VariableWithValue("nf4_scale_scales", tensors.FromFlatDataAndDimensions(weight.ScaleScales, weight.InputFeatures, (blocks+weight.ScaleBlockSize-1)/weight.ScaleBlockSize)).SetTrainable(false)
	}
	a := qScope.VariableWithValue("A", tensors.FromFlatDataAndDimensions(aValues, config.Rank, weight.InputFeatures)).SetTrainable(true)
	b := qScope.VariableWithValue("B", tensors.FromFlatDataAndDimensions(make([]float32, weight.OutputFeatures*config.Rank), weight.OutputFeatures, config.Rank)).SetTrainable(true)
	if bias != nil {
		bias.SetTrainable(config.Bias != lora.BiasNone)
	}
	return &NF4Linear{name: name, packed: packed, scales: scales, scaleCodes: scaleCodes, scaleScales: scaleScales, bias: bias, a: a, b: b, in: weight.InputFeatures, out: weight.OutputFeatures, block: weight.BlockSize, scaling: config.Alpha / float32(config.Rank), dropout: config.Dropout}, nil
}

// NewNF4BaseLinear creates a frozen NF4 projection without adapter variables.
func NewNF4BaseLinear(scope *model.Scope, weight *NF4Weight, bias *model.Variable) (*NF4BaseLinear, error) {
	if scope == nil {
		return nil, errors.New("gomlx: scope is required")
	}
	if err := weight.validate(); err != nil {
		return nil, err
	}
	if weight.OutputFeatures&1 != 0 {
		return nil, errors.New("gomlx: NF4 output features must be even")
	}
	if bias != nil && (bias.Shape().Rank() != 1 || bias.Shape().Dimensions[0] != weight.OutputFeatures) {
		return nil, errors.New("gomlx: bias shape must be [out_features]")
	}
	qScope := scope.In("nf4_base")
	packed := qScope.VariableWithValue("packed", tensors.FromFlatDataAndDimensions(weight.Packed, weight.InputFeatures, (weight.OutputFeatures+1)/2)).SetTrainable(false)
	blocks := (weight.OutputFeatures + weight.BlockSize - 1) / weight.BlockSize
	base := &NF4BaseLinear{packed: packed, bias: bias, in: weight.InputFeatures, out: weight.OutputFeatures, block: weight.BlockSize}
	if len(weight.ScaleCodes) == 0 {
		base.scales = qScope.VariableWithValue("scales", tensors.FromFlatDataAndDimensions(weight.Scales, weight.InputFeatures, blocks)).SetTrainable(false)
	} else {
		base.scaleCodes = qScope.VariableWithValue("scale_codes", tensors.FromFlatDataAndDimensions(weight.ScaleCodes, weight.InputFeatures, blocks)).SetTrainable(false)
		base.scaleScales = qScope.VariableWithValue("scale_scales", tensors.FromFlatDataAndDimensions(weight.ScaleScales, weight.InputFeatures, (blocks+weight.ScaleBlockSize-1)/weight.ScaleBlockSize)).SetTrainable(false)
	}
	if bias != nil {
		bias.SetTrainable(false)
	}
	return base, nil
}

// Apply appends the frozen NF4 projection to input's graph.
func (l *NF4BaseLinear) Apply(_ *model.Scope, input *graph.Node) *graph.Node {
	packed := graph.Bitcast(l.packed.NodeValue(input), dtypes.Uint4)
	weights := graph.Reshape(packed, l.in, l.out)
	return nn.QuantizedDense(input, weights, &graph.Quantization{
		Scheme: compute.QuantNF4, Scale: l.quantizationScales(input), BlockAxis: 1, BlockSize: l.block,
	}, nodeValue(l.bias, input))
}

// Apply appends fused-or-decomposed NF4 dense and the LoRA branch to the graph.
func (l *NF4Linear) Apply(scope *model.Scope, input *graph.Node) *graph.Node {
	packed := graph.Bitcast(l.packed.NodeValue(input), dtypes.Uint4)
	weights := graph.Reshape(packed, l.in, l.out)
	base := nn.QuantizedDense(input, weights, &graph.Quantization{
		Scheme: compute.QuantNF4, Scale: l.quantizationScales(input), BlockAxis: 1, BlockSize: l.block,
	}, nodeValue(l.bias, input))
	projected := graph.MatMul(input, graph.Transpose(l.a.NodeValue(input), -2, -1))
	projected = layers.DropoutStatic(scope, projected, float64(l.dropout))
	projected = graph.MatMul(projected, graph.Transpose(l.b.NodeValue(input), -2, -1))
	return graph.Add(base, graph.MulScalar(projected, l.scaling))
}

func (l *NF4Linear) quantizationScales(input *graph.Node) *graph.Node {
	if l.scales != nil {
		return l.scales.NodeValue(input)
	}
	blocks := (l.out + l.block - 1) / l.block
	groups := l.scaleScales.Shape().Dimensions[1]
	groupSize := (blocks + groups - 1) / groups
	indices := make([]int32, blocks)
	for index := range indices {
		indices[index] = int32(index / groupSize)
	}
	group := graph.Reshape(graph.Const(input.Graph(), indices), blocks, 1)
	expanded := graph.Gather(graph.Transpose(l.scaleScales.NodeValue(input), 0, 1), group)
	expanded = graph.Transpose(expanded, 0, 1)
	return graph.Mul(graph.ConvertDType(l.scaleCodes.NodeValue(input), dtypes.Float32), expanded)
}

func (l *NF4BaseLinear) quantizationScales(input *graph.Node) *graph.Node {
	if l.scales != nil {
		return l.scales.NodeValue(input)
	}
	blocks := (l.out + l.block - 1) / l.block
	groups := l.scaleScales.Shape().Dimensions[1]
	groupSize := (blocks + groups - 1) / groups
	indices := make([]int32, blocks)
	for index := range indices {
		indices[index] = int32(index / groupSize)
	}
	group := graph.Reshape(graph.Const(input.Graph(), indices), blocks, 1)
	expanded := graph.Gather(graph.Transpose(l.scaleScales.NodeValue(input), 0, 1), group)
	expanded = graph.Transpose(expanded, 0, 1)
	return graph.Mul(graph.ConvertDType(l.scaleCodes.NodeValue(input), dtypes.Float32), expanded)
}

// TrainableVariables returns A, B, and the configured bias.
func (l *NF4Linear) TrainableVariables() []*model.Variable {
	variables := []*model.Variable{l.a, l.b}
	if l.bias != nil && l.bias.Trainable {
		variables = append(variables, l.bias)
	}
	return variables
}

// Name returns the host module name.
func (l *NF4Linear) Name() string { return l.name }

// A returns the rank-by-input adapter variable.
func (l *NF4Linear) A() *model.Variable { return l.a }

// B returns the output-by-rank adapter variable.
func (l *NF4Linear) B() *model.Variable { return l.b }

// Bias returns the optional model bias.
func (l *NF4Linear) Bias() *model.Variable { return l.bias }

func nodeValue(variable *model.Variable, input *graph.Node) *graph.Node {
	if variable == nil {
		return nil
	}
	return variable.NodeValue(input)
}
