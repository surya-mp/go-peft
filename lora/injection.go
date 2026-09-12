package lora

import (
	"fmt"
	"math/rand"
	"strings"

	"github.com/surya-mp/go-peft/backend"
)

// Module is one replaceable base linear layer.
type Module struct {
	// Name is the host model's fully qualified module name.
	Name string
	// Weight is the frozen [outFeatures, inFeatures] base weight.
	Weight backend.Tensor
	// Bias is an optional [1, outFeatures] host bias.
	Bias backend.Tensor
}

// Replacement associates a model module with its LoRA wrapper.
type Replacement struct {
	// Name identifies the host module to replace.
	Name string
	// Layer is the LoRA wrapper for Name.
	Layer *Linear
}

// Model supplies discovery and atomic replacement for linear modules.
type Model interface {
	LinearModules() ([]Module, error)
	ReplaceLinearModules([]Replacement) error
}

// Inject discovers configured modules, creates all layers, then replaces them.
func Inject(name string, engine backend.Engine, model Model, config Config, rng *rand.Rand) (*Adapter, error) {
	adapter, err := NewAdapter(name, config)
	if err != nil {
		return nil, err
	}
	if model == nil {
		return nil, fmt.Errorf("lora: model is required")
	}
	modules, err := model.LinearModules()
	if err != nil {
		return nil, fmt.Errorf("lora: discover modules: %w", err)
	}

	replacements := make([]Replacement, 0, len(modules))
	for _, module := range modules {
		if !Matches(module.Name, config.TargetModules) {
			continue
		}
		layer, err := NewLinear(module.Name, engine, module.Weight, module.Bias, config, rng)
		if err != nil {
			return nil, fmt.Errorf("lora: inject %q: %w", module.Name, err)
		}
		replacements = append(replacements, Replacement{Name: module.Name, Layer: layer})
	}
	if len(replacements) == 0 {
		return nil, ErrNoTargetModules
	}
	if err := model.ReplaceLinearModules(replacements); err != nil {
		return nil, fmt.Errorf("lora: replace modules: %w", err)
	}
	for _, replacement := range replacements {
		if err := adapter.Add(replacement.Layer); err != nil {
			return nil, err
		}
	}
	return adapter, nil
}

// Matches accepts exact module names and dot-separated suffixes.
func Matches(module string, targets []string) bool {
	for _, target := range targets {
		if module == target || strings.HasSuffix(module, "."+target) {
			return true
		}
	}
	return false
}
