package lora

import (
	"fmt"

	"github.com/surya-mp/go-peft/peft"
)

// Adapter owns LoRA layers for one named configuration.
type Adapter struct {
	name   string
	config Config
	layers []*Linear
	index  map[string]*Linear
}

// NewAdapter validates and creates an empty adapter.
func NewAdapter(name string, config Config) (*Adapter, error) {
	if name == "" {
		return nil, ErrInvalidName
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	config.TargetModules = append([]string(nil), config.TargetModules...)
	return &Adapter{name: name, config: config, index: make(map[string]*Linear)}, nil
}

// Name returns the adapter identifier.
func (a *Adapter) Name() string { return a.name }

// Config returns an independent configuration value.
func (a *Adapter) Config() Config {
	config := a.config
	config.TargetModules = append([]string(nil), config.TargetModules...)
	return config
}

// Add registers a layer after it has been safely attached to a model.
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

// Layer returns a registered LoRA layer.
func (a *Adapter) Layer(name string) (*Linear, bool) {
	layer, ok := a.index[name]
	return layer, ok
}

// Layers returns registered layers in injection order.
func (a *Adapter) Layers() []*Linear { return append([]*Linear(nil), a.layers...) }

// Parameters returns the adapter's trainable parameter metadata.
func (a *Adapter) Parameters() []peft.Parameter {
	count := len(a.layers) * 2
	if a.config.Bias != BiasNone {
		count += len(a.layers)
	}
	parameters := make([]peft.Parameter, 0, count)
	for _, layer := range a.layers {
		parameters = append(parameters, layer.AdapterParameters()...)
	}
	return parameters
}

// Merge merges every layer after checking that none is already merged.
func (a *Adapter) Merge() error {
	for _, layer := range a.layers {
		if layer.merged {
			return fmt.Errorf("%w: %s", ErrAlreadyMerged, layer.name)
		}
	}
	for index, layer := range a.layers {
		if err := layer.Merge(); err != nil {
			for rollback := index - 1; rollback >= 0; rollback-- {
				_ = a.layers[rollback].Unmerge()
			}
			return err
		}
	}
	return nil
}

// Unmerge removes every layer update after checking their state.
func (a *Adapter) Unmerge() error {
	for _, layer := range a.layers {
		if !layer.merged {
			return fmt.Errorf("%w: %s", ErrNotMerged, layer.name)
		}
	}
	for index, layer := range a.layers {
		if err := layer.Unmerge(); err != nil {
			for rollback := index - 1; rollback >= 0; rollback-- {
				_ = a.layers[rollback].Merge()
			}
			return err
		}
	}
	return nil
}
