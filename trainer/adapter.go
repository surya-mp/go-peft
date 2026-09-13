package trainer

import (
	"context"
	"errors"
)

// ErrInvalidAdapter reports a nil or empty adapter variable provider.
var ErrInvalidAdapter = errors.New("trainer: invalid adapter")

// TrainableAdapter exposes the backend variables owned by a PEFT adapter.
type TrainableAdapter[V any] interface {
	TrainableVariables() []V
}

// AdapterOptimizer applies accumulated gradients to a selected variable set.
type AdapterOptimizer[V any] interface {
	ApplyGradients(context.Context, []V) error
}

// AdapterVariables returns a defensive copy of the variables selected for PEFT
// training. Optimizers should update only this returned set.
func AdapterVariables[V any](adapter TrainableAdapter[V]) ([]V, error) {
	if adapter == nil {
		return nil, ErrInvalidAdapter
	}
	variables := adapter.TrainableVariables()
	if len(variables) == 0 {
		return nil, ErrInvalidAdapter
	}
	return append([]V(nil), variables...), nil
}

// AdapterUpdater applies accumulated gradients only to adapter variables.
type AdapterUpdater[V any] struct {
	Adapter   TrainableAdapter[V]
	Optimizer AdapterOptimizer[V]
}

// ApplyGradients applies one optimizer update to the adapter-owned variables.
func (u AdapterUpdater[V]) ApplyGradients(ctx context.Context) error {
	if u.Optimizer == nil {
		return ErrInvalidConfig
	}
	variables, err := AdapterVariables(u.Adapter)
	if err != nil {
		return err
	}
	return u.Optimizer.ApplyGradients(ctx, variables)
}
