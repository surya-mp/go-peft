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

func TestQwen3UsesDenseQwenProjectionNames(t *testing.T) {
	profile, err := Resolve(Qwen3)
	if err != nil {
		t.Fatal(err)
	}
	targets, err := profile.Targets(AllLinear)
	if err != nil || !reflect.DeepEqual(targets, []string{"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}) {
		t.Fatalf("targets = %v, err = %v", targets, err)
	}
	plan, err := Plan(Qwen3, Attention, []string{"model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.v_proj"})
	if err != nil || len(plan.Matched) != 2 || plan.Family != Qwen3 {
		t.Fatalf("plan = %#v, err = %v", plan, err)
	}
}

func TestQwen3MoEIncludesExpertAndRouterTargets(t *testing.T) {
	profile, err := Resolve(Qwen3MoE)
	if err != nil {
		t.Fatal(err)
	}
	targets, err := profile.Targets(AllLinear)
	if err != nil || !reflect.DeepEqual(targets, []string{"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "gate"}) {
		t.Fatalf("targets = %v, err = %v", targets, err)
	}
	plan, err := Plan(Qwen3MoE, AllLinear, []string{
		"model.layers.0.mlp.experts.3.gate_proj",
		"model.layers.0.mlp.gate",
	})
	if err != nil || !reflect.DeepEqual(plan.Matched, []string{
		"model.layers.0.mlp.experts.3.gate_proj",
		"model.layers.0.mlp.gate",
	}) {
		t.Fatalf("plan = %#v, err = %v", plan, err)
	}
}
