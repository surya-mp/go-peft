package lora

import (
	"math/rand"
	"reflect"
	"testing"

	"github.com/surya-mp/go-peft/backend"
)

func TestPEFTParityLoRALifecycle(t *testing.T) {
	engine := backend.NewCPU()
	weight, _ := backend.NewDense(2, 3, []float32{1, -2, 0.5, 0.25, 1.5, -1})
	input, _ := backend.NewDense(2, 3, []float32{2, -1, 0.5, 0, 1, -2})
	config := Config{Rank: 2, Alpha: 4, TargetModules: []string{"q_proj"}}
	layer, err := NewLinear("q_proj", engine, weight, nil, config, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	base, err := layer.Forward(input)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := base.(*backend.Dense).Values(), []float32{4.25, -1.5, -3, 3.5}; !reflect.DeepEqual(got, want) {
		t.Fatalf("zero-B output = %v, want %v", got, want)
	}
	a, _ := backend.NewDense(2, 3, []float32{0.5, -1, 0.25, -0.75, 0.5, 1})
	b, _ := backend.NewDense(2, 2, []float32{1, -0.5, 0.25, 0.75})
	if err := engine.Copy(layer.A(), a); err != nil {
		t.Fatal(err)
	}
	if err := engine.Copy(layer.B(), b); err != nil {
		t.Fatal(err)
	}
	active, err := layer.Forward(input)
	if err != nil {
		t.Fatal(err)
	}
	adapter, err := NewAdapter("default", config)
	if err != nil {
		t.Fatal(err)
	}
	if err := adapter.Add(layer); err != nil {
		t.Fatal(err)
	}
	if err := adapter.Merge(); err != nil {
		t.Fatal(err)
	}
	merged, err := layer.Forward(input)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := merged.(*backend.Dense).Values(), active.(*backend.Dense).Values(); !reflect.DeepEqual(got, want) {
		t.Fatalf("merged output = %v, want %v", got, want)
	}
	if err := adapter.Unmerge(); err != nil {
		t.Fatal(err)
	}
	restored, err := layer.Forward(input)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := restored.(*backend.Dense).Values(), active.(*backend.Dense).Values(); !reflect.DeepEqual(got, want) {
		t.Fatalf("unmerged output = %v, want %v", got, want)
	}
	for _, parameter := range layer.Parameters()[:1] {
		if parameter.Trainable {
			t.Fatalf("base parameter = %#v", parameter)
		}
	}
	for _, parameter := range adapter.Parameters() {
		if !parameter.Trainable {
			t.Fatalf("adapter parameter = %#v", parameter)
		}
	}
}

func TestPEFTParityZeroDropout(t *testing.T) {
	engine := backend.NewCPU()
	weight, _ := backend.NewDense(1, 2, []float32{1, 2})
	input, _ := backend.NewDense(1, 2, []float32{3, 4})
	layer, err := NewLinear("q_proj", engine, weight, nil, Config{Rank: 1, Alpha: 1, TargetModules: []string{"q_proj"}}, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	inference, err := layer.Forward(input)
	if err != nil {
		t.Fatal(err)
	}
	training, err := layer.ForwardTraining(input, rand.New(rand.NewSource(2)))
	if err != nil {
		t.Fatal(err)
	}
	if got, want := training.(*backend.Dense).Values(), inference.(*backend.Dense).Values(); !reflect.DeepEqual(got, want) {
		t.Fatalf("zero-dropout training output = %v, want %v", got, want)
	}
}

func TestPEFTParityTrainingDropoutAndBiasModes(t *testing.T) {
	engine := backend.NewCPU()
	weight, _ := backend.NewDense(1, 2, []float32{0, 0})
	input, _ := backend.NewDense(1, 2, []float32{1, 1})
	config := Config{Rank: 1, Alpha: 1, Dropout: 0.5, TargetModules: []string{"q_proj"}}
	layer, err := NewLinear("q_proj", engine, weight, nil, config, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	a, _ := backend.NewDense(1, 2, []float32{1, 1})
	b, _ := backend.NewDense(1, 1, []float32{1})
	if err := engine.Copy(layer.A(), a); err != nil {
		t.Fatal(err)
	}
	if err := engine.Copy(layer.B(), b); err != nil {
		t.Fatal(err)
	}
	seed := int64(4)
	training, err := layer.ForwardTraining(input, rand.New(rand.NewSource(seed)))
	if err != nil {
		t.Fatal(err)
	}
	want := float32(4)
	if rand.New(rand.NewSource(seed)).Float32() < 0.5 {
		want = 0
	}
	if got := training.(*backend.Dense).Values(); !reflect.DeepEqual(got, []float32{want}) {
		t.Fatalf("dropout output = %v, want %v", got, want)
	}
	bias, _ := backend.NewDense(1, 1, []float32{3})
	for _, item := range []struct {
		mode      BiasMode
		trainable bool
	}{
		{BiasNone, false},
		{BiasLoRAOnly, true},
		{BiasAll, true},
	} {
		biasLayer, err := NewLinear("q_proj", engine, weight, bias, Config{Rank: 1, Alpha: 1, TargetModules: []string{"q_proj"}, Bias: item.mode}, rand.New(rand.NewSource(1)))
		if err != nil {
			t.Fatal(err)
		}
		parameters := biasLayer.Parameters()
		if parameters[1].Trainable != item.trainable {
			t.Fatalf("bias mode %d trainable = %v", item.mode, parameters[1].Trainable)
		}
	}
}
