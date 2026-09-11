package qlora

import (
	"math/rand"
	"testing"

	"github.com/surya-mp/go-peft/backend"
	"github.com/surya-mp/go-peft/lora"
)

func TestInject(t *testing.T) {
	engine := backend.NewCPU()
	weight, _ := backend.NewDense(2, 2, []float32{1, 0, 0, 1})
	model := &testModel{modules: []Module{{Name: "layers.0.q_proj", Weight: weight}}}
	config := Config{LoRA: lora.Config{Rank: 1, Alpha: 1, TargetModules: []string{"q_proj"}}, BlockSize: 64}
	adapter, err := Inject("adapter", engine, model, config, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	if model.replaceCalls != 1 || len(adapter.Layers()) != 1 || len(adapter.Parameters()) != 2 {
		t.Fatalf("injection = %#v, layers = %d, parameters = %d", model, len(adapter.Layers()), len(adapter.Parameters()))
	}
}

type testModel struct {
	modules      []Module
	replacements []Replacement
	replaceCalls int
}

func (m *testModel) LinearModules() ([]Module, error) { return m.modules, nil }

func (m *testModel) ReplaceQLoRALinearModules(replacements []Replacement) error {
	m.replaceCalls++
	m.replacements = replacements
	return nil
}
