//go:build darwin && arm64 && mlx

package mlx

import (
	"math"
	"math/rand"
	"testing"

	"github.com/surya-mp/go-peft/format/huggingface"
	"github.com/surya-mp/go-peft/lora"
	"github.com/surya-mp/go-peft/qlora"
)

func TestGPUAdapterRoundTrip(t *testing.T) {
	context, err := New()
	if err != nil {
		t.Fatal(err)
	}
	defer context.Close()
	weight, err := context.Float32([]int{2, 2}, []float32{1, 0, 0, 1})
	if err != nil {
		t.Fatal(err)
	}
	defer weight.Close()
	config := lora.Config{Rank: 1, Alpha: 1, TargetModules: []string{"q_proj"}}
	layer, err := NewLoRALinear(weight, nil, config, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		_ = layer.a.Close()
		_ = layer.b.Close()
	}()
	trained, err := context.Float32([]int{2, 1}, []float32{1, 1})
	if err != nil {
		t.Fatal(err)
	}
	old := layer.b
	layer.b = trained
	_ = old.Close()
	input, err := context.Float32([]int{1, 2}, []float32{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	defer input.Close()
	before := forwardValues(t, layer, input)
	dir := t.TempDir()
	if err := SaveAdapter(dir, config, map[string]*LoRALinear{"q_proj": layer}, huggingface.Metadata{BaseModelNameOrPath: "base"}); err != nil {
		t.Fatal(err)
	}
	zero, err := context.Float32([]int{2, 1}, []float32{0, 0})
	if err != nil {
		t.Fatal(err)
	}
	old = layer.b
	layer.b = zero
	_ = old.Close()
	if _, err := LoadAdapter(dir, config, map[string]*LoRALinear{"q_proj": layer}); err != nil {
		t.Fatal(err)
	}
	after := forwardValues(t, layer, input)
	for index := range before {
		if math.Abs(float64(before[index]-after[index])) > 1e-5 {
			t.Fatalf("output = %v, want %v", after, before)
		}
	}
}

func TestGPUQLoRAAdapterRoundTrip(t *testing.T) {
	context, err := New()
	if err != nil {
		t.Fatal(err)
	}
	defer context.Close()
	weight, err := context.Float32([]int{2, 32}, make([]float32, 64))
	if err != nil {
		t.Fatal(err)
	}
	defer weight.Close()
	config := qlora.Config{
		LoRA:      lora.Config{Rank: 1, Alpha: 1, TargetModules: []string{"q_proj"}},
		BlockSize: 32, Quantization: qlora.QuantizationInt4,
	}
	layer, err := NewQLoRALinear(weight, nil, config, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	defer layer.Close()
	trained, err := context.Float32([]int{2, 1}, []float32{1, 1})
	if err != nil {
		t.Fatal(err)
	}
	old := layer.b
	layer.b = trained
	_ = old.Close()
	inputData := make([]float32, 32)
	for index := range inputData {
		inputData[index] = 1
	}
	input, err := context.Float32([]int{1, 32}, inputData)
	if err != nil {
		t.Fatal(err)
	}
	defer input.Close()
	before := qloraForwardValues(t, layer, input)
	dir := t.TempDir()
	if err := SaveQLoRAAdapter(dir, config, map[string]*QLoRALinear{"q_proj": layer}, huggingface.Metadata{BaseModelNameOrPath: "base"}); err != nil {
		t.Fatal(err)
	}
	zero, err := context.Float32([]int{2, 1}, []float32{0, 0})
	if err != nil {
		t.Fatal(err)
	}
	old = layer.b
	layer.b = zero
	_ = old.Close()
	if _, err := LoadQLoRAAdapter(dir, config, map[string]*QLoRALinear{"q_proj": layer}); err != nil {
		t.Fatal(err)
	}
	after := qloraForwardValues(t, layer, input)
	for index := range before {
		if math.Abs(float64(before[index]-after[index])) > 1e-5 {
			t.Fatalf("output = %v, want %v", after, before)
		}
	}
}

func forwardValues(t *testing.T, layer *LoRALinear, input *Array) []float32 {
	t.Helper()
	output, err := layer.Forward(input)
	if err != nil {
		t.Fatal(err)
	}
	defer output.Close()
	values, err := output.Float32Values()
	if err != nil {
		t.Fatal(err)
	}
	return values
}

func qloraForwardValues(t *testing.T, layer *QLoRALinear, input *Array) []float32 {
	t.Helper()
	output, err := layer.Forward(input)
	if err != nil {
		t.Fatal(err)
	}
	defer output.Close()
	values, err := output.Float32Values()
	if err != nil {
		t.Fatal(err)
	}
	return values
}
