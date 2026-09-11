//go:build gomlx

package gomlx

import (
	"errors"
	"fmt"
	"math/rand"

	"github.com/gomlx/gomlx/ml/model"
	"github.com/surya-mp/go-peft/lora"
	"github.com/surya-mp/go-peft/peft"
)

var ErrNoTargetModules = errors.New("gomlx: no target modules matched")

// Module is one host-discovered GoMLX linear module.
type Module struct {
	Name   string
	Scope  *model.Scope
	Weight *model.Variable
	Bias   *model.Variable
}

// Replacement associates a model path with its native LoRA layer.
type Replacement struct {
	Name  string
	Layer *Linear
}

// Model connects a framework-specific model registry to native injection.
type Model interface {
	LinearModules() ([]Module, error)
	ReplaceLoRALinearModules([]Replacement) error
}

// Adapter owns native GoMLX LoRA layers and their optimizer variables.
type Adapter struct {
	name   string
	config lora.Config
	layers []*Linear
	index  map[string]*Linear
}

// Inject atomically replaces matching host modules with native GoMLX LoRA layers.
func Inject(name string, host Model, config lora.Config, rng *rand.Rand) (*Adapter, error) {
	if name == "" || host == nil || rng == nil {
		return nil, errors.New("gomlx: name, model, and random source are required")
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	modules, err := host.LinearModules()
	if err != nil {
		return nil, fmt.Errorf("gomlx: discover modules: %w", err)
	}
	replacements := make([]Replacement, 0, len(modules))
	index := make(map[string]*Linear, len(modules))
	for _, module := range modules {
		if !lora.Matches(module.Name, config.TargetModules) {
			continue
		}
		if module.Name == "" || module.Scope == nil || module.Weight == nil {
			return nil, fmt.Errorf("gomlx: invalid module %q", module.Name)
		}
		if _, exists := index[module.Name]; exists {
			return nil, fmt.Errorf("gomlx: duplicate module %q", module.Name)
		}
		layer, err := NewNamedLinear(module.Name, module.Scope, module.Weight, module.Bias, config, rng)
		if err != nil {
			return nil, fmt.Errorf("gomlx: inject %q: %w", module.Name, err)
		}
		replacements = append(replacements, Replacement{Name: module.Name, Layer: layer})
		index[module.Name] = layer
	}
	if len(replacements) == 0 {
		return nil, ErrNoTargetModules
	}
	if err := host.ReplaceLoRALinearModules(replacements); err != nil {
		return nil, fmt.Errorf("gomlx: replace modules: %w", err)
	}
	return &Adapter{name: name, config: cloneConfig(config), layers: replacementLayers(replacements), index: index}, nil
}

// Name returns the adapter name.
func (a *Adapter) Name() string { return a.name }

// Config returns a detached configuration copy.
func (a *Adapter) Config() lora.Config { return cloneConfig(a.config) }

// Layer returns a layer by model module name.
func (a *Adapter) Layer(name string) (*Linear, bool) { layer, ok := a.index[name]; return layer, ok }

// Layers returns native layers in injection order.
func (a *Adapter) Layers() []*Linear { return append([]*Linear(nil), a.layers...) }

// TrainableVariables returns the exact variables to pass to a GoMLX optimizer.
func (a *Adapter) TrainableVariables() []*model.Variable {
	count := len(a.layers) * 2
	if a.config.Bias != lora.BiasNone {
		count += len(a.layers)
	}
	variables := make([]*model.Variable, 0, count)
	for _, layer := range a.layers {
		variables = append(variables, layer.A(), layer.B())
		if a.config.Bias != lora.BiasNone && layer.Bias() != nil {
			variables = append(variables, layer.Bias())
		}
	}
	return variables
}

// Parameters reports lightweight trainability metadata.
func (a *Adapter) Parameters() []peft.Parameter {
	variables := a.TrainableVariables()
	parameters := make([]peft.Parameter, 0, len(variables))
	for _, variable := range variables {
		parameters = append(parameters, peft.NewParameter(variable.Path(), variable.Shape().Dimensions, variable.Trainable))
	}
	return parameters
}

func replacementLayers(replacements []Replacement) []*Linear {
	result := make([]*Linear, len(replacements))
	for index, replacement := range replacements {
		result[index] = replacement.Layer
	}
	return result
}

func cloneConfig(config lora.Config) lora.Config {
	config.TargetModules = append([]string(nil), config.TargetModules...)
	return config
}
