package lora

import (
	"errors"
	"math/rand"
	"testing"

	"github.com/surya-mp/go-peft/backend"
)

func TestMatches(t *testing.T) {
	if !Matches("layers.0.self_attn.q_proj", []string{"q_proj"}) {
		t.Fatal("suffix target did not match")
	}
	if Matches("layers.0.self_attn.k_proj", []string{"q_proj"}) {
		t.Fatal("unexpected target match")
	}
}

func TestInjectDoesNotReplaceOnInvalidModule(t *testing.T) {
	engine := backend.NewCPU()
	weight, _ := backend.NewDense(2, 2, []float32{1, 0, 0, 1})
	model := &testModel{modules: []Module{
		{Name: "q_proj", Weight: weight},
		{Name: "v_proj", Weight: nil},
	}}
	config := Config{Rank: 1, Alpha: 1, TargetModules: []string{"q_proj", "v_proj"}}
	_, err := Inject("adapter", engine, model, config, rand.New(rand.NewSource(1)))
	if !errors.Is(err, ErrInvalidWeight) {
		t.Fatalf("Inject() = %v, want invalid weight", err)
	}
	if model.replaceCalls != 0 {
		t.Fatalf("replacement ran %d times", model.replaceCalls)
	}
}

func TestInject(t *testing.T) {
	engine := backend.NewCPU()
	weight, _ := backend.NewDense(2, 2, []float32{1, 0, 0, 1})
	model := &testModel{modules: []Module{{Name: "layers.0.q_proj", Weight: weight}}}
	config := Config{Rank: 1, Alpha: 1, TargetModules: []string{"q_proj"}}
	adapter, err := Inject("adapter", engine, model, config, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	if model.replaceCalls != 1 || len(model.replacements) != 1 {
		t.Fatalf("replacement = %#v", model)
	}
	if len(adapter.Parameters()) != 2 {
		t.Fatalf("parameters = %v", adapter.Parameters())
	}
}

type testModel struct {
	modules      []Module
	replacements []Replacement
	replaceCalls int
}

func (m *testModel) LinearModules() ([]Module, error) { return m.modules, nil }

func (m *testModel) ReplaceLinearModules(replacements []Replacement) error {
	m.replaceCalls++
	m.replacements = replacements
	return nil
}
