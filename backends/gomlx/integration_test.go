//go:build gomlx

package gomlx

import (
	"math"
	"math/rand"
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/gobackend"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/surya-mp/go-peft/format/huggingface"
	"github.com/surya-mp/go-peft/lora"
	"github.com/surya-mp/go-peft/qlora"
)

type testModel struct {
	modules      []Module
	replacements []Replacement
}

type nf4TestModel struct {
	modules      []NF4Module
	replacements []NF4Replacement
}

func (m *nf4TestModel) NF4LinearModules() ([]NF4Module, error) { return m.modules, nil }

func (m *nf4TestModel) ReplaceNF4LinearModules(replacements []NF4Replacement) error {
	m.replacements = append([]NF4Replacement(nil), replacements...)
	return nil
}

func (m *testModel) LinearModules() ([]Module, error) { return m.modules, nil }

func (m *testModel) ReplaceLoRALinearModules(replacements []Replacement) error {
	m.replacements = append([]Replacement(nil), replacements...)
	return nil
}

func TestNativeInjectionForwardAndPersistence(t *testing.T) {
	store := model.NewStore()
	scope := store.RootScope().In("layers").In("0").In("q_proj")
	weight := scope.VariableWithValue("weight", [][]float32{{1, 2}, {3, 4}})
	host := &testModel{modules: []Module{{Name: "layers.0.q_proj", Scope: scope, Weight: weight}}}
	config := lora.Config{Rank: 1, Alpha: 2, TargetModules: []string{"q_proj"}}
	adapter, err := Inject("example", host, config, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	if len(host.replacements) != 1 || weight.Trainable || len(adapter.TrainableVariables()) != 2 {
		t.Fatal("injection did not freeze the base or expose exactly A/B")
	}
	layer := host.replacements[0].Layer
	exec, err := model.NewExec(gobackend.GetBackend(), store, func(scope *model.Scope, input *graph.Node) *graph.Node {
		return layer.Apply(scope, input)
	})
	if err != nil {
		t.Fatal(err)
	}
	base := call(t, exec)
	if !closeSlice(base, []float32{5, 11}) {
		t.Fatalf("base forward = %v", base)
	}
	if err := layer.B().SetValue(tensors.FromFlatDataAndDimensions([]float32{1, 1}, 2, 1)); err != nil {
		t.Fatal(err)
	}
	trained := call(t, exec)
	if closeSlice(trained, base) {
		t.Fatalf("LoRA update did not affect forward: %v", trained)
	}
	dir := t.TempDir()
	metadata := huggingface.Metadata{BaseModelNameOrPath: "base", Revision: "main", TaskType: "CAUSAL_LM"}
	if err := adapter.Save(dir, metadata); err != nil {
		t.Fatal(err)
	}
	if err := layer.B().SetValue(tensors.FromFlatDataAndDimensions([]float32{0, 0}, 2, 1)); err != nil {
		t.Fatal(err)
	}
	if zeroed := call(t, exec); !closeSlice(zeroed, base) {
		t.Fatalf("zeroed adapter forward = %v, want %v", zeroed, base)
	}
	loaded, err := adapter.Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	if loaded != metadata {
		t.Fatalf("metadata = %#v, want %#v", loaded, metadata)
	}
	if restored := call(t, exec); !closeSlice(restored, trained) {
		t.Fatalf("restored forward = %v, want %v", restored, trained)
	}
}

func TestNativeInjectionExposesOnlyAdapterGradients(t *testing.T) {
	store := model.NewStore()
	scope := store.RootScope().In("q_proj")
	weight := scope.VariableWithValue("weight", [][]float32{{1, 2}, {3, 4}})
	host := &testModel{modules: []Module{{Name: "q_proj", Scope: scope, Weight: weight}}}
	adapter, err := Inject("example", host, lora.Config{Rank: 1, Alpha: 1, TargetModules: []string{"q_proj"}}, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	layer, _ := adapter.Layer("q_proj")
	exec, err := model.NewExec(gobackend.GetBackend(), store, func(scope *model.Scope, input *graph.Node) []*graph.Node {
		loss := graph.ReduceAllSum(layer.Apply(scope, input))
		return scope.BuildTrainableVariablesGradientsGraph(loss)
	})
	if err != nil {
		t.Fatal(err)
	}
	gradients, err := exec.Call([][]float32{{1, 2}})
	if err != nil {
		t.Fatal(err)
	}
	if len(gradients) != 2 {
		t.Fatalf("gradient count = %d, want 2 adapter variables", len(gradients))
	}
	for _, gradient := range gradients {
		if gradient.Shape().Size() == 0 {
			t.Fatal("empty adapter gradient")
		}
	}
}

func TestNativeLinearRetainsFloat16(t *testing.T) {
	store := model.NewStore()
	scope := store.RootScope().In("q_proj")
	weight := scope.VariableWithValue("weight", tensors.FromFlatDataAndDimensions([]float16.Float16{
		float16.FromFloat32(1), float16.FromFloat32(2), float16.FromFloat32(3), float16.FromFloat32(4),
	}, 2, 2))
	layer, err := NewLinear(scope, weight, nil, lora.Config{Rank: 1, Alpha: 1, TargetModules: []string{"q_proj"}}, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	if layer.A().DType() != dtypes.Float16 || layer.B().DType() != dtypes.Float16 {
		t.Fatalf("adapter dtypes = %s, %s", layer.A().DType(), layer.B().DType())
	}
}

func TestNativeNF4Linear(t *testing.T) {
	store := model.NewStore()
	scope := store.RootScope().In("q_proj")
	weight, err := QuantizeNF4Weight(2, 2, 2, []float32{1, -1, -1, 1})
	if err != nil {
		t.Fatal(err)
	}
	layer, err := NewNF4Linear(scope, weight, nil, lora.Config{Rank: 1, Alpha: 1, TargetModules: []string{"q_proj"}}, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	exec, err := model.NewExec(gobackend.GetBackend(), store, func(scope *model.Scope, input *graph.Node) *graph.Node {
		return layer.Apply(scope, input)
	})
	if err != nil {
		t.Fatal(err)
	}
	if output := call(t, exec); !closeSlice(output, []float32{-1, 1}) {
		t.Fatalf("NF4 forward = %v", output)
	}
	if len(layer.TrainableVariables()) != 2 {
		t.Fatal("NF4 base variables must be frozen")
	}
}

func TestNF4BaseLinear(t *testing.T) {
	store := model.NewStore()
	scope := store.RootScope().In("k_proj")
	weight, err := QuantizeNF4WeightDouble(2, 2, 2, 2, []float32{1, -1, -1, 1})
	if err != nil {
		t.Fatal(err)
	}
	layer, err := NewNF4BaseLinear(scope, weight, nil)
	if err != nil {
		t.Fatal(err)
	}
	exec, err := model.NewExec(gobackend.GetBackend(), store, func(scope *model.Scope, input *graph.Node) *graph.Node {
		return layer.Apply(scope, input)
	})
	if err != nil {
		t.Fatal(err)
	}
	if output := call(t, exec); !closeSlice(output, []float32{-1, 1}) {
		t.Fatalf("NF4 base forward = %v", output)
	}
}

func TestNF4BaseLinearOddOutput(t *testing.T) {
	store := model.NewStore()
	scope := store.RootScope().In("lm_head")
	weight, err := QuantizeNF4Weight(2, 3, 3, []float32{1, -1, 0.5, -1, 1, -0.5})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := NewNF4BaseLinear(scope, weight, nil); err == nil {
		t.Fatal("odd NF4 output width must be rejected")
	}
}

func TestNativeNF4DoubleQuantLinear(t *testing.T) {
	store := model.NewStore()
	scope := store.RootScope().In("q_proj")
	weight, err := QuantizeNF4WeightDouble(2, 4, 2, 2, []float32{1, -1, 0.5, -0.5, -1, 1, -0.5, 0.5})
	if err != nil {
		t.Fatal(err)
	}
	if len(weight.Scales) != 0 || len(weight.ScaleCodes) == 0 {
		t.Fatal("weight was not double quantized")
	}
	layer, err := NewNF4Linear(scope, weight, nil, lora.Config{Rank: 1, Alpha: 1, TargetModules: []string{"q_proj"}}, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	exec, err := model.NewExec(gobackend.GetBackend(), store, func(scope *model.Scope, input *graph.Node) *graph.Node {
		return layer.Apply(scope, input)
	})
	if err != nil {
		t.Fatal(err)
	}
	output := call(t, exec)
	for index, want := range []float32{-1, 1, -0.5, 0.5} {
		if math.Abs(float64(output[index]-want)) > 0.01 {
			t.Fatalf("double quant NF4 forward = %v", output)
		}
	}
}

func TestNativeNF4Injection(t *testing.T) {
	store := model.NewStore()
	scope := store.RootScope().In("q_proj")
	weight, err := QuantizeNF4Weight(2, 2, 2, []float32{1, -1, -1, 1})
	if err != nil {
		t.Fatal(err)
	}
	host := &nf4TestModel{modules: []NF4Module{{Name: "q_proj", Scope: scope, Weight: weight}}}
	config := qlora.Config{LoRA: lora.Config{Rank: 1, Alpha: 1, TargetModules: []string{"q_proj"}}, BlockSize: 2, Quantization: qlora.QuantizationNF4}
	adapter, err := InjectNF4("nf4", host, config, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	if len(host.replacements) != 1 || len(adapter.TrainableVariables()) != 2 {
		t.Fatal("NF4 injection did not expose adapter variables")
	}
	layer := host.replacements[0].Layer
	exec, err := model.NewExec(gobackend.GetBackend(), store, func(scope *model.Scope, input *graph.Node) *graph.Node {
		return layer.Apply(scope, input)
	})
	if err != nil {
		t.Fatal(err)
	}
	_ = call(t, exec)
	if err := layer.B().SetValue(tensors.FromFlatDataAndDimensions([]float32{1, 1}, 2, 1)); err != nil {
		t.Fatal(err)
	}
	trained := call(t, exec)
	dir := t.TempDir()
	if err := adapter.Save(dir, huggingface.Metadata{BaseModelNameOrPath: "base"}); err != nil {
		t.Fatal(err)
	}
	if err := layer.B().SetValue(tensors.FromFlatDataAndDimensions([]float32{0, 0}, 2, 1)); err != nil {
		t.Fatal(err)
	}
	if _, err := adapter.Load(dir); err != nil {
		t.Fatal(err)
	}
	if restored := call(t, exec); !closeSlice(restored, trained) {
		t.Fatalf("restored QLoRA forward = %v, want %v", restored, trained)
	}
}

func call(t *testing.T, exec *model.Exec) []float32 {
	t.Helper()
	outputs, err := exec.Call([][]float32{{1, 2}})
	if err != nil {
		t.Fatal(err)
	}
	values, err := tensors.CopyFlatData[float32](outputs[0])
	if err != nil {
		t.Fatal(err)
	}
	return values
}

func closeSlice(left, right []float32) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if math.Abs(float64(left[index]-right[index])) > 1e-5 {
			return false
		}
	}
	return true
}
