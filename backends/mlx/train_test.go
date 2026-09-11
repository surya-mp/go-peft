//go:build darwin && arm64 && mlx

package mlx

import (
	"math"
	"math/rand"
	"testing"

	"github.com/surya-mp/go-peft/lora"
	"github.com/surya-mp/go-peft/qlora"
)

func TestLoRATrainStep(t *testing.T) {
	context, err := New()
	if err != nil {
		t.Fatal(err)
	}
	defer context.Close()
	weight, err := context.Float32([]int{1, 2}, []float32{0, 0})
	if err != nil {
		t.Fatal(err)
	}
	defer weight.Close()
	layer, err := NewLoRALinear(weight, nil, lora.Config{Rank: 1, Alpha: 1, TargetModules: []string{"q_proj"}}, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	defer layer.a.Close()
	defer layer.b.Close()
	input, err := context.Float32([]int{1, 2}, []float32{1, 1})
	if err != nil {
		t.Fatal(err)
	}
	defer input.Close()
	target, err := context.Float32([]int{1, 1}, []float32{1})
	if err != nil {
		t.Fatal(err)
	}
	defer target.Close()
	first, err := layer.TrainStep(input, target, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	last := first
	for range 39 {
		last, err = layer.TrainStep(input, target, 0.1)
		if err != nil {
			t.Fatal(err)
		}
	}
	if last >= first {
		t.Fatalf("loss = %v, want below %v", last, first)
	}
}

func TestQLoRATrainStep(t *testing.T) {
	context, err := New()
	if err != nil {
		t.Fatal(err)
	}
	defer context.Close()
	weight, err := context.Float32([]int{1, 32}, make([]float32, 32))
	if err != nil {
		t.Fatal(err)
	}
	defer weight.Close()
	layer, err := NewQLoRALinear(weight, nil, qlora.Config{
		LoRA:         lora.Config{Rank: 1, Alpha: 1, TargetModules: []string{"q_proj"}},
		BlockSize:    32,
		Quantization: qlora.QuantizationInt4,
	}, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	defer layer.Close()
	inputValues := make([]float32, 32)
	for index := range inputValues {
		inputValues[index] = 1
	}
	input, err := context.Float32([]int{1, 32}, inputValues)
	if err != nil {
		t.Fatal(err)
	}
	defer input.Close()
	target, err := context.Float32([]int{1, 1}, []float32{1})
	if err != nil {
		t.Fatal(err)
	}
	defer target.Close()
	first, err := layer.TrainStep(input, target, 0.02)
	if err != nil {
		t.Fatal(err)
	}
	last := first
	for range 79 {
		last, err = layer.TrainStep(input, target, 0.02)
		if err != nil {
			t.Fatal(err)
		}
	}
	if last >= first {
		t.Fatalf("loss = %v, want below %v", last, first)
	}
}

func TestAdapterTrainingDropout(t *testing.T) {
	context, err := New()
	if err != nil {
		t.Fatal(err)
	}
	defer context.Close()
	weight, err := context.Float32([]int{1, 2}, []float32{0, 0})
	if err != nil {
		t.Fatal(err)
	}
	defer weight.Close()
	layer, err := NewLoRALinear(weight, nil, lora.Config{Rank: 1, Alpha: 1, Dropout: 0.5, TargetModules: []string{"q_proj"}}, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	defer layer.a.Close()
	defer layer.b.Close()
	inputData := make([]float32, 256)
	targetData := make([]float32, 128)
	for index := range inputData {
		inputData[index] = 1
	}
	for index := range targetData {
		targetData[index] = 1
	}
	input, err := context.Float32([]int{128, 2}, inputData)
	if err != nil {
		t.Fatal(err)
	}
	defer input.Close()
	target, err := context.Float32([]int{128, 1}, targetData)
	if err != nil {
		t.Fatal(err)
	}
	defer target.Close()
	loss, err := layer.TrainStep(input, target, 0.01)
	if err != nil || math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
		t.Fatalf("dropout TrainStep() = %v, %v", loss, err)
	}
	values, err := layer.b.Float32Values()
	if err != nil {
		t.Fatal(err)
	}
	if values[0] == 0 {
		t.Fatal("dropout training did not update B")
	}
}
