package qlora

import (
	"errors"
	"fmt"
	"math/rand"
	"strings"

	"github.com/surya-mp/go-peft/backend"
)

// Module is one replaceable float32 base linear layer.
type Module struct {
	// Name is the host model's fully qualified module name.
	Name string
	// Weight is the F32 [outFeatures, inFeatures] base weight to quantize.
	Weight backend.Tensor
	// Bias is an optional [1, outFeatures] host bias.
	Bias backend.Tensor
}

// Replacement associates a model module with its QLoRA wrapper.
type Replacement struct {
	// Name identifies the host module to replace.
	Name string
	// Layer is the QLoRA wrapper for Name.
	Layer *Linear
}

// Model supplies discovery and atomic replacement for linear modules.
type Model interface {
	LinearModules() ([]Module, error)
	ReplaceQLoRALinearModules([]Replacement) error
}

// Inject quantizes all configured modules before atomically replacing them.
func Inject(name string, engine backend.Engine, model Model, config Config, rng *rand.Rand) (*Adapter, error) {
	adapter, err := NewAdapter(name, config)
	if err != nil {
		return nil, err
	}
	if model == nil {
		return nil, errModelRequired
	}
	modules, err := model.LinearModules()
	if err != nil {
		return nil, fmt.Errorf("qlora: discover modules: %w", err)
	}
	replacements := make([]Replacement, 0, len(modules))
	for _, module := range modules {
		if !Matches(module.Name, config.LoRA.TargetModules) {
			continue
		}
		layer, err := QuantizeLinear(module.Name, engine, module.Weight, module.Bias, config, rng)
		if err != nil {
			return nil, fmt.Errorf("qlora: inject %q: %w", module.Name, err)
		}
		replacements = append(replacements, Replacement{Name: module.Name, Layer: layer})
	}
	if len(replacements) == 0 {
		return nil, ErrNoTargetModules
	}
	if err := model.ReplaceQLoRALinearModules(replacements); err != nil {
		return nil, fmt.Errorf("qlora: replace modules: %w", err)
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

var errModelRequired = errors.New("qlora: model is required")
