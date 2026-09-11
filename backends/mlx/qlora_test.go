//go:build darwin && arm64 && mlx

package mlx

import (
	"math"
	"math/rand"
	"testing"

	"github.com/surya-mp/go-peft/lora"
	"github.com/surya-mp/go-peft/qlora"
)

func TestQLoRALinearForward(t *testing.T) {
	context, err := NewCPU()
	if err != nil {
		t.Fatal(err)
	}
	defer context.Close()
	weightData := make([]float32, 64)
	for index := range weightData {
		weightData[index] = float32(index%11-5) / 5
	}
	weight, err := context.Float32([]int{2, 32}, weightData)
	if err != nil {
		t.Fatal(err)
	}
	defer weight.Close()
	layer, err := NewQLoRALinear(weight, nil, qlora.Config{
		LoRA:      lora.Config{Rank: 1, Alpha: 1, TargetModules: []string{"q_proj"}},
		BlockSize: 32, Quantization: qlora.QuantizationInt4,
	}, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	defer layer.Close()
	input, err := context.Float32([]int{1, 32}, make([]float32, 32))
	if err != nil {
		t.Fatal(err)
	}
	defer input.Close()
	output, err := layer.Forward(input)
	if err != nil {
		t.Fatal(err)
	}
	defer output.Close()
	shape, err := output.Shape()
	if err != nil {
		t.Fatal(err)
	}
	if !sameShape(shape, []int{1, 2}) {
		t.Fatalf("output shape = %v, want [1 2]", shape)
	}
}

func TestQLoRATrainingDropoutForward(t *testing.T) {
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
	layer, err := NewQLoRALinear(weight, nil, qlora.Config{
		LoRA: lora.Config{Rank: 1, Alpha: 1, Dropout: 0.5, TargetModules: []string{"q_proj"}}, BlockSize: 32,
	}, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	defer layer.Close()
	inputData := make([]float32, 128)
	for index := range inputData {
		inputData[index] = 1
	}
	input, err := context.Float32([]int{4, 32}, inputData)
	if err != nil {
		t.Fatal(err)
	}
	defer input.Close()
	output, err := layer.ForwardTraining(input)
	if err != nil {
		t.Fatal(err)
	}
	defer output.Close()
	shape, err := output.Shape()
	if err != nil {
		t.Fatal(err)
	}
	if !sameShape(shape, []int{4, 2}) {
		t.Fatalf("output shape = %v, want [4 2]", shape)
	}
	target, err := context.Float32([]int{4, 2}, []float32{1, 1, 1, 1, 1, 1, 1, 1})
	if err != nil {
		t.Fatal(err)
	}
	defer target.Close()
	loss, err := layer.TrainStep(input, target, 0.01)
	if err != nil || math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
		t.Fatalf("dropout TrainStep() = %v, %v", loss, err)
	}
}
