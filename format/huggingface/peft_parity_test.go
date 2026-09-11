package huggingface

import (
	"errors"
	"math/rand"
	"testing"

	"github.com/surya-mp/go-peft/backend"
	"github.com/surya-mp/go-peft/format/safetensors"
	"github.com/surya-mp/go-peft/lora"
)

func TestPEFTParityAdapterStateValidation(t *testing.T) {
	engine, config, adapter, layer, tensors := parityAdapter(t)
	for _, item := range []struct {
		name    string
		tensors map[string]safetensors.Tensor
		want    error
	}{
		{
			name: "missing", tensors: map[string]safetensors.Tensor{
				key(layer.Name(), "lora_A.weight"): tensors[key(layer.Name(), "lora_A.weight")],
			}, want: ErrMissingTensor,
		},
		{
			name: "unexpected", tensors: map[string]safetensors.Tensor{
				key(layer.Name(), "lora_A.weight"): tensors[key(layer.Name(), "lora_A.weight")],
				key(layer.Name(), "lora_B.weight"): tensors[key(layer.Name(), "lora_B.weight")],
				"base_model.model.extra.weight":    {Shape: []int{1, 1}, Data: []float32{1}},
			}, want: ErrUnexpectedTensor,
		},
		{
			name: "shape", tensors: map[string]safetensors.Tensor{
				key(layer.Name(), "lora_A.weight"): tensors[key(layer.Name(), "lora_A.weight")],
				key(layer.Name(), "lora_B.weight"): {Shape: []int{1, 2}, Data: []float32{1, 2}},
			}, want: ErrConfigMismatch,
		},
	} {
		t.Run(item.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := SaveTensors(dir, config, item.tensors, Metadata{}); err != nil {
				t.Fatal(err)
			}
			if _, err := LoadInto(dir, adapter, engine); !errors.Is(err, item.want) {
				t.Fatalf("LoadInto() = %v, want %v", err, item.want)
			}
		})
	}
}

func TestPEFTParityPortableBiasState(t *testing.T) {
	engine := backend.NewCPU()
	weight, _ := backend.NewDense(1, 2, []float32{1, 2})
	bias, _ := backend.NewDense(1, 1, []float32{3})
	config := lora.Config{Rank: 1, Alpha: 1, TargetModules: []string{"q_proj"}, Bias: lora.BiasLoRAOnly}
	layer, err := lora.NewLinear("q_proj", engine, weight, bias, config, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	adapter, err := lora.NewAdapter("default", config)
	if err != nil {
		t.Fatal(err)
	}
	if err := adapter.Add(layer); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	if err := Save(dir, adapter, engine, Metadata{}); err != nil {
		t.Fatal(err)
	}
	if err := engine.Zero(layer.Bias()); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadInto(dir, adapter, engine); err != nil {
		t.Fatal(err)
	}
	if got := layer.Bias().(*backend.Dense).Values(); got[0] != 3 {
		t.Fatalf("bias = %v, want 3", got)
	}
	all := config
	all.Bias = lora.BiasAll
	allAdapter, err := lora.NewAdapter("all", all)
	if err != nil {
		t.Fatal(err)
	}
	if err := allAdapter.Add(layer); err != nil {
		t.Fatal(err)
	}
	if err := Save(t.TempDir(), allAdapter, engine, Metadata{}); !errors.Is(err, ErrUnsupportedBias) {
		t.Fatalf("Save() = %v, want %v", err, ErrUnsupportedBias)
	}
}

func parityAdapter(t *testing.T) (backend.CPU, lora.Config, *lora.Adapter, *lora.Linear, map[string]safetensors.Tensor) {
	t.Helper()
	engine := backend.NewCPU()
	weight, _ := backend.NewDense(2, 2, []float32{1, 0, 0, 1})
	config := lora.Config{Rank: 1, Alpha: 1, TargetModules: []string{"q_proj"}}
	layer, err := lora.NewLinear("q_proj", engine, weight, nil, config, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	adapter, err := lora.NewAdapter("default", config)
	if err != nil {
		t.Fatal(err)
	}
	if err := adapter.Add(layer); err != nil {
		t.Fatal(err)
	}
	a, _ := backend.NewDense(1, 2, []float32{1, 2})
	b, _ := backend.NewDense(2, 1, []float32{3, 4})
	if err := engine.Copy(layer.A(), a); err != nil {
		t.Fatal(err)
	}
	if err := engine.Copy(layer.B(), b); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	if err := Save(dir, adapter, engine, Metadata{}); err != nil {
		t.Fatal(err)
	}
	tensors, _, err := safetensors.ReadFile(dir + "/" + ModelFile)
	if err != nil {
		t.Fatal(err)
	}
	return engine, config, adapter, layer, tensors
}
