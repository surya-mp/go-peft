package trainer

import (
	"context"
	"errors"
	"reflect"
	"testing"
)

type variableAdapter struct {
	variables []string
}

func (a *variableAdapter) TrainableVariables() []string {
	return a.variables
}

type variableOptimizer struct {
	variables []string
}

func (o *variableOptimizer) ApplyGradients(_ context.Context, variables []string) error {
	o.variables = append([]string(nil), variables...)
	return nil
}

func TestAdapterVariablesReturnsDefensiveCopy(t *testing.T) {
	adapter := &variableAdapter{variables: []string{"A", "B"}}
	variables, err := AdapterVariables[string](adapter)
	if err != nil {
		t.Fatal(err)
	}
	variables[0] = "changed"
	if adapter.variables[0] != "A" {
		t.Fatalf("adapter variables mutated: %#v", adapter.variables)
	}
}

func TestAdapterUpdaterAppliesOnlyAdapterVariables(t *testing.T) {
	optimizer := &variableOptimizer{}
	updater := AdapterUpdater[string]{
		Adapter:   &variableAdapter{variables: []string{"adapter.A", "adapter.B"}},
		Optimizer: optimizer,
	}
	if err := updater.ApplyGradients(context.Background()); err != nil {
		t.Fatal(err)
	}
	want := []string{"adapter.A", "adapter.B"}
	if !reflect.DeepEqual(optimizer.variables, want) {
		t.Fatalf("variables = %#v, want %#v", optimizer.variables, want)
	}
}

func TestAdapterUpdaterRejectsInvalidAdapter(t *testing.T) {
	err := (AdapterUpdater[string]{Optimizer: &variableOptimizer{}}).ApplyGradients(context.Background())
	if !errors.Is(err, ErrInvalidAdapter) {
		t.Fatalf("err = %v", err)
	}
	err = (AdapterUpdater[string]{Adapter: &variableAdapter{}}).ApplyGradients(context.Background())
	if !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("err = %v", err)
	}
}
