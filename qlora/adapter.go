package qlora

import (
	"fmt"

	"github.com/surya-mp/go-peft/peft"
)

// Adapter owns QLoRA layers for one named configuration.
type Adapter struct {
	name   string
	config Config
	layers []*Linear
	index  map[string]*Linear
}

// NewAdapter validates and creates an empty QLoRA adapter.
func NewAdapter(name string, config Config) (*Adapter, error) {
	if name == "" {
		return nil, ErrInvalidName
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	config.LoRA.TargetModules = append([]string(nil), config.LoRA.TargetModules...)
	return &Adapter{name: name, config: config, index: make(map[string]*Linear)}, nil
}

// Name returns the adapter identifier.
func (a *Adapter) Name() string { return a.name }

// Config returns an independent configuration value.
func (a *Adapter) Config() Config {
	config := a.config
	config.LoRA.TargetModules = append([]string(nil), config.LoRA.TargetModules...)
	return config
}

// Add registers an injected QLoRA layer.
func (a *Adapter) Add(layer *Linear) error {
	if layer == nil {
		return ErrInvalidWeight
	}
	if _, exists := a.index[layer.name]; exists {
		return fmt.Errorf("%w: %s", ErrDuplicateLayer, layer.name)
	}
	a.index[layer.name] = layer
	a.layers = append(a.layers, layer)
	return nil
}

// Layer returns a registered QLoRA layer.
func (a *Adapter) Layer(name string) (*Linear, bool) {
	layer, ok := a.index[name]
	return layer, ok
}

// Layers returns registered layers in injection order.
func (a *Adapter) Layers() []*Linear { return append([]*Linear(nil), a.layers...) }

// Parameters returns trainable adapter metadata.
func (a *Adapter) Parameters() []peft.Parameter {
	count := len(a.layers) * 2
	if a.config.LoRA.Bias != 0 {
		count += len(a.layers)
	}
	parameters := make([]peft.Parameter, 0, count)
	for _, layer := range a.layers {
		parameters = append(parameters, layer.AdapterParameters()...)
	}
	return parameters
}

// Merge always fails because the base weights are int4.
func (a *Adapter) Merge() error { return ErrMergeUnsupported }

// Unmerge always fails because QLoRA never merges into quantized base weights.
func (a *Adapter) Unmerge() error { return ErrMergeUnsupported }
