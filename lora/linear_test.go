package lora

import (
	"math/rand"
	"reflect"
	"testing"

	"github.com/surya-mp/go-peft/backend"
)

func TestLinearForwardMergeAndUnmerge(t *testing.T) {
	engine := backend.NewCPU()
	weight, _ := backend.NewDense(2, 3, []float32{1, 2, 3, 4, 5, 6})
	input, _ := backend.NewDense(1, 3, []float32{1, 1, 1})
	layer := newTestLinear(t, engine, weight)
	setAdapterWeights(t, engine, layer)

	output, err := layer.Forward(input)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{12, 29}
	if got := output.(*backend.Dense).Values(); !reflect.DeepEqual(got, want) {
		t.Fatalf("Forward() = %v, want %v", got, want)
	}

	if err := layer.Merge(); err != nil {
		t.Fatal(err)
	}
	merged, err := layer.Forward(input)
	if err != nil {
		t.Fatal(err)
	}
	if got := merged.(*backend.Dense).Values(); !reflect.DeepEqual(got, want) {
		t.Fatalf("merged Forward() = %v, want %v", got, want)
	}
	if err := layer.Unmerge(); err != nil {
		t.Fatal(err)
	}
	if got, wantWeight := weight.Values(), []float32{1, 2, 3, 4, 5, 6}; !reflect.DeepEqual(got, wantWeight) {
		t.Fatalf("Unmerge() weight = %v, want %v", got, wantWeight)
	}
}

func TestLinearForwardIntoMatchesForward(t *testing.T) {
	engine := backend.NewCPU()
	weight, _ := backend.NewDense(2, 3, []float32{1, 2, 3, 4, 5, 6})
	input, _ := backend.NewDense(2, 3, []float32{1, 1, 1, 2, 3, 4})
	layer := newTestLinear(t, engine, weight)
	setAdapterWeights(t, engine, layer)

	want, err := layer.Forward(input)
	if err != nil {
		t.Fatal(err)
	}
	output, workspace, err := layer.NewWorkspace(input)
	if err != nil {
		t.Fatal(err)
	}
	if err := layer.ForwardInto(output, workspace, input, false, nil); err != nil {
		t.Fatal(err)
	}
	if got := output.(*backend.Dense).Values(); !reflect.DeepEqual(got, want.(*backend.Dense).Values()) {
		t.Fatalf("ForwardInto() = %v, want %v", got, want.(*backend.Dense).Values())
	}
}

func TestLinearBias(t *testing.T) {
	engine := backend.NewCPU()
	weight, _ := backend.NewDense(2, 2, []float32{1, 0, 0, 1})
	bias, _ := backend.NewDense(1, 2, []float32{3, 5})
	input, _ := backend.NewDense(1, 2, []float32{2, 4})
	config := Config{Rank: 1, Alpha: 1, TargetModules: []string{"linear"}}
	layer, err := NewLinear("linear", engine, weight, bias, config, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	output, err := layer.Forward(input)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := output.(*backend.Dense).Values(), []float32{5, 9}; !reflect.DeepEqual(got, want) {
		t.Fatalf("Forward() = %v, want %v", got, want)
	}
}

func newTestLinear(t *testing.T, engine backend.CPU, weight *backend.Dense) *Linear {
	t.Helper()
	config := Config{Rank: 2, Alpha: 4, TargetModules: []string{"linear"}}
	layer, err := NewLinear("linear", engine, weight, nil, config, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	return layer
}

func setAdapterWeights(t *testing.T, engine backend.CPU, layer *Linear) {
	t.Helper()
	a, _ := backend.NewDense(2, 3, []float32{1, 0, 0, 0, 1, 0})
	b, _ := backend.NewDense(2, 2, []float32{1, 2, 3, 4})
	if err := engine.Copy(layer.A(), a); err != nil {
		t.Fatal(err)
	}
	if err := engine.Copy(layer.B(), b); err != nil {
		t.Fatal(err)
	}
}
