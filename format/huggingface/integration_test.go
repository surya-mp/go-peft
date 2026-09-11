package huggingface

import (
	"math/rand"
	"reflect"
	"testing"

	"github.com/surya-mp/go-peft/backend"
	"github.com/surya-mp/go-peft/format/safetensors"
	"github.com/surya-mp/go-peft/lora"
	"github.com/surya-mp/go-peft/qlora"
)

func TestLoRAAdapterRoundTrip(t *testing.T) {
	engine := backend.NewCPU()
	weight, _ := backend.NewDense(2, 3, []float32{1, 2, 3, 4, 5, 6})
	config := lora.Config{Rank: 2, Alpha: 4, Dropout: 0.1, TargetModules: []string{"q_proj"}}
	adapter, err := lora.NewAdapter("default", config)
	if err != nil {
		t.Fatal(err)
	}
	layer, err := lora.NewLinear("layers.0.q_proj", engine, weight, nil, config, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	if err := adapter.Add(layer); err != nil {
		t.Fatal(err)
	}
	a, _ := backend.NewDense(2, 3, []float32{1, 2, 3, 4, 5, 6})
	b, _ := backend.NewDense(2, 2, []float32{7, 8, 9, 10})
	if err := engine.Copy(layer.A(), a); err != nil {
		t.Fatal(err)
	}
	if err := engine.Copy(layer.B(), b); err != nil {
		t.Fatal(err)
	}

	dir := t.TempDir()
	metadata := Metadata{BaseModelNameOrPath: "example/model", Revision: "main", TaskType: "CAUSAL_LM"}
	if err := Save(dir, adapter, engine, metadata); err != nil {
		t.Fatal(err)
	}
	loadedConfig, loadedMetadata, err := ReadConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !sameConfig(loadedConfig, config) || loadedMetadata != metadata {
		t.Fatalf("ReadConfig() = %#v, %#v", loadedConfig, loadedMetadata)
	}
	tensors, _, err := safetensors.ReadFile(dir + "/" + ModelFile)
	if err != nil {
		t.Fatal(err)
	}
	if len(tensors) != 2 {
		t.Fatalf("tensor count = %d", len(tensors))
	}
	if _, ok := tensors[key(layer.Name(), "lora_A.weight")]; !ok {
		t.Fatal("missing Hugging Face A key")
	}

	if err := engine.Zero(layer.A()); err != nil {
		t.Fatal(err)
	}
	if err := engine.Zero(layer.B()); err != nil {
		t.Fatal(err)
	}
	gotMetadata, err := LoadInto(dir, adapter, engine)
	if err != nil {
		t.Fatal(err)
	}
	if gotMetadata != metadata {
		t.Fatalf("LoadInto() metadata = %#v", gotMetadata)
	}
	if got, want := layer.A().(*backend.Dense).Values(), a.Values(); !reflect.DeepEqual(got, want) {
		t.Fatalf("A = %v, want %v", got, want)
	}
	if got, want := layer.B().(*backend.Dense).Values(), b.Values(); !reflect.DeepEqual(got, want) {
		t.Fatalf("B = %v, want %v", got, want)
	}
}

func TestQLoRAAdapterRoundTrip(t *testing.T) {
	engine := backend.NewCPU()
	weight, _ := backend.NewDense(2, 2, []float32{1, 2, 3, 4})
	config := qlora.Config{LoRA: lora.Config{Rank: 1, Alpha: 2, TargetModules: []string{"q_proj"}}, BlockSize: 64}
	adapter, err := qlora.NewAdapter("default", config)
	if err != nil {
		t.Fatal(err)
	}
	layer, err := qlora.QuantizeLinear("q_proj", engine, weight, nil, config, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	if err := adapter.Add(layer); err != nil {
		t.Fatal(err)
	}
	a, _ := backend.NewDense(1, 2, []float32{5, 6})
	b, _ := backend.NewDense(2, 1, []float32{7, 8})
	if err := engine.Copy(layer.A(), a); err != nil {
		t.Fatal(err)
	}
	if err := engine.Copy(layer.B(), b); err != nil {
		t.Fatal(err)
	}

	dir := t.TempDir()
	if err := SaveQLoRA(dir, adapter, engine, Metadata{}); err != nil {
		t.Fatal(err)
	}
	if err := engine.Zero(layer.A()); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadQLoRAInto(dir, adapter, engine); err != nil {
		t.Fatal(err)
	}
	if got, want := layer.A().(*backend.Dense).Values(), a.Values(); !reflect.DeepEqual(got, want) {
		t.Fatalf("A = %v, want %v", got, want)
	}
}
