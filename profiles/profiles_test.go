package profiles

import (
	"errors"
	"reflect"
	"testing"
)

func TestPlanLlamaAllLinear(t *testing.T) {
	plan, err := Plan(Llama, AllLinear, []string{
		"model.layers.0.self_attn.q_proj", "model.layers.0.mlp.down_proj", "model.layers.0.norm",
	})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(plan.Matched, []string{"model.layers.0.mlp.down_proj", "model.layers.0.self_attn.q_proj"}) {
		t.Fatalf("matched = %v", plan.Matched)
	}
	if err := plan.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestPlanPhi3AndFailures(t *testing.T) {
	plan, err := Plan(Phi3, Attention, []string{"layers.0.self_attn.qkv_proj"})
	if err != nil || !reflect.DeepEqual(plan.Matched, []string{"layers.0.self_attn.qkv_proj"}) {
		t.Fatalf("Plan() = %#v, %v", plan, err)
	}
	if _, err := Plan("unknown", Attention, nil); !errors.Is(err, ErrUnknownFamily) {
		t.Fatalf("unknown family = %v", err)
	}
	if _, err := Plan(Llama, "other", nil); !errors.Is(err, ErrUnknownMode) {
		t.Fatalf("unknown mode = %v", err)
	}
	if err := (InjectionPlan{}).Validate(); !errors.Is(err, ErrNoMatch) {
		t.Fatalf("empty plan = %v", err)
	}
}
