//go:build cuda && linux && cgo

package cuda

import (
	"errors"
	"math"
	"math/rand"
	"testing"

	"github.com/surya-mp/go-peft/backend"
	"github.com/surya-mp/go-peft/lora"
	"github.com/surya-mp/go-peft/qlora"
)

func TestLoRAForward(t *testing.T) {
	engine := testEngine(t)
	weight, err := engine.DecodeFloat32(2, 2, []float32{1, 0, 0, 1})
	if err != nil {
		t.Fatal(err)
	}
	defer weight.(*Tensor).Close()
	input, err := engine.DecodeFloat32(1, 2, []float32{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	defer input.(*Tensor).Close()
	layer, err := lora.NewLinear("q_proj", engine, weight, nil, lora.Config{Rank: 1, Alpha: 1, TargetModules: []string{"q_proj"}}, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	trainedB, err := engine.DecodeFloat32(2, 1, []float32{1, -1})
	if err != nil {
		t.Fatal(err)
	}
	defer trainedB.(*Tensor).Close()
	if err := engine.Copy(layer.B(), trainedB); err != nil {
		t.Fatal(err)
	}
	defer layer.A().(*Tensor).Close()
	defer layer.B().(*Tensor).Close()
	output, err := layer.Forward(input)
	if err != nil {
		t.Fatal(err)
	}
	defer output.(*Tensor).Close()
	values, _, _, err := engine.EncodeFloat32(output)
	if err != nil {
		t.Fatal(err)
	}
	aValues, _, _, err := engine.EncodeFloat32(layer.A())
	if err != nil {
		t.Fatal(err)
	}
	cpu := backend.NewCPU()
	cpuWeight, _ := backend.NewDense(2, 2, []float32{1, 0, 0, 1})
	cpuInput, _ := backend.NewDense(1, 2, []float32{1, 2})
	cpuLayer, err := lora.NewLinear("q_proj", cpu, cpuWeight, nil, lora.Config{Rank: 1, Alpha: 1, TargetModules: []string{"q_proj"}}, rand.New(rand.NewSource(2)))
	if err != nil {
		t.Fatal(err)
	}
	cpuA, _ := backend.NewDense(1, 2, aValues)
	cpuB, _ := backend.NewDense(2, 1, []float32{1, -1})
	if err := cpu.Copy(cpuLayer.A(), cpuA); err != nil {
		t.Fatal(err)
	}
	if err := cpu.Copy(cpuLayer.B(), cpuB); err != nil {
		t.Fatal(err)
	}
	wantTensor, err := cpuLayer.Forward(cpuInput)
	if err != nil {
		t.Fatal(err)
	}
	if !same(values, wantTensor.(*backend.Dense).Values()) {
		t.Fatalf("got %v, want %v", values, wantTensor.(*backend.Dense).Values())
	}
}

func TestQLoRAQuantizedLinear(t *testing.T) {
	engine := testEngine(t)
	for _, config := range []qlora.Config{
		{LoRA: lora.Config{Rank: 1, Alpha: 1, TargetModules: []string{"q_proj"}}, BlockSize: 32},
		{LoRA: lora.Config{Rank: 1, Alpha: 1, TargetModules: []string{"q_proj"}}, BlockSize: 32, Quantization: qlora.QuantizationNF4, DoubleQuant: true, ScaleBlockSize: 2},
	} {
		weightValues := make([]float32, 64)
		for index := range weightValues {
			weightValues[index] = float32(index%13-6) / 6
		}
		inputValues := make([]float32, 32)
		for index := range inputValues {
			inputValues[index] = float32(index%5 - 2)
		}
		weight, err := engine.DecodeFloat32(2, 32, weightValues)
		if err != nil {
			t.Fatal(err)
		}
		input, err := engine.DecodeFloat32(1, 32, inputValues)
		if err != nil {
			t.Fatal(err)
		}
		layer, err := qlora.QuantizeLinear("q_proj", engine, weight, nil, config, rand.New(rand.NewSource(1)))
		if err != nil {
			t.Fatal(err)
		}
		trainedB, err := engine.DecodeFloat32(2, 1, []float32{0.25, -0.5})
		if err != nil {
			t.Fatal(err)
		}
		if err := engine.Copy(layer.B(), trainedB); err != nil {
			t.Fatal(err)
		}
		output, err := layer.Forward(input)
		if err != nil {
			t.Fatal(err)
		}
		got, _, _, err := engine.EncodeFloat32(output)
		if err != nil {
			t.Fatal(err)
		}
		cpu := backend.NewCPU()
		cpuWeight, _ := backend.NewDense(2, 32, weightValues)
		cpuInput, _ := backend.NewDense(1, 32, inputValues)
		expectedLayer, err := qlora.QuantizeLinear("q_proj", cpu, cpuWeight, nil, config, rand.New(rand.NewSource(1)))
		if err != nil {
			t.Fatal(err)
		}
		aValues, _, _, err := engine.EncodeFloat32(layer.A())
		if err != nil {
			t.Fatal(err)
		}
		cpuA, _ := backend.NewDense(1, 32, aValues)
		cpuB, _ := backend.NewDense(2, 1, []float32{0.25, -0.5})
		if err := cpu.Copy(expectedLayer.A(), cpuA); err != nil {
			t.Fatal(err)
		}
		if err := cpu.Copy(expectedLayer.B(), cpuB); err != nil {
			t.Fatal(err)
		}
		expectedOutput, err := expectedLayer.Forward(cpuInput)
		if err != nil {
			t.Fatal(err)
		}
		want := expectedOutput.(*backend.Dense).Values()
		if !same(got, want) {
			t.Fatalf("config %#v: got %v, want %v", config, got, want)
		}
		_ = weight.(*Tensor).Close()
		_ = input.(*Tensor).Close()
		_ = layer.A().(*Tensor).Close()
		_ = layer.B().(*Tensor).Close()
		_ = trainedB.(*Tensor).Close()
		_ = output.(*Tensor).Close()
	}
}

func TestAdapterTraining(t *testing.T) {
	engine := testEngine(t)
	weight, err := engine.DecodeFloat32(1, 2, []float32{0, 0})
	if err != nil {
		t.Fatal(err)
	}
	input, err := engine.DecodeFloat32(1, 2, []float32{1, 1})
	if err != nil {
		t.Fatal(err)
	}
	target, err := engine.DecodeFloat32(1, 1, []float32{1})
	if err != nil {
		t.Fatal(err)
	}
	layer, err := lora.NewLinear("q_proj", engine, weight, nil, lora.Config{Rank: 1, Alpha: 1, TargetModules: []string{"q_proj"}}, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	first, err := TrainLoRA(engine, layer, input, target, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	last := first
	for range 39 {
		last, err = TrainLoRA(engine, layer, input, target, 0.1)
		if err != nil {
			t.Fatal(err)
		}
	}
	if last >= first {
		t.Fatalf("loss = %v, want below %v", last, first)
	}
	for _, value := range []backend.Tensor{weight, input, target, layer.A(), layer.B()} {
		_ = value.(*Tensor).Close()
	}
}

func TestAdapterTrainingDropout(t *testing.T) {
	engine := testEngine(t)
	weight, err := engine.DecodeFloat32(1, 2, []float32{0, 0})
	if err != nil {
		t.Fatal(err)
	}
	inputValues := make([]float32, 256)
	targetValues := make([]float32, 128)
	for index := range inputValues {
		inputValues[index] = 1
	}
	for index := range targetValues {
		targetValues[index] = 1
	}
	input, err := engine.DecodeFloat32(128, 2, inputValues)
	if err != nil {
		t.Fatal(err)
	}
	target, err := engine.DecodeFloat32(128, 1, targetValues)
	if err != nil {
		t.Fatal(err)
	}
	layer, err := lora.NewLinear("q_proj", engine, weight, nil, lora.Config{Rank: 1, Alpha: 1, Dropout: 0.5, TargetModules: []string{"q_proj"}}, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	loss, err := TrainLoRA(engine, layer, input, target, 0.01)
	if err != nil || math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
		t.Fatalf("dropout TrainLoRA() = %v, %v", loss, err)
	}
	values, _, _, err := engine.EncodeFloat32(layer.B())
	if err != nil {
		t.Fatal(err)
	}
	if values[0] == 0 {
		t.Fatal("dropout training did not update B")
	}
	for _, value := range []backend.Tensor{weight, input, target, layer.A(), layer.B()} {
		_ = value.(*Tensor).Close()
	}
}

func TestQLoRAAdapterTraining(t *testing.T) {
	engine := testEngine(t)
	weight, err := engine.DecodeFloat32(1, 32, make([]float32, 32))
	if err != nil {
		t.Fatal(err)
	}
	inputValues := make([]float32, 32)
	for index := range inputValues {
		inputValues[index] = 1
	}
	input, err := engine.DecodeFloat32(1, 32, inputValues)
	if err != nil {
		t.Fatal(err)
	}
	target, err := engine.DecodeFloat32(1, 1, []float32{1})
	if err != nil {
		t.Fatal(err)
	}
	layer, err := qlora.QuantizeLinear("q_proj", engine, weight, nil, qlora.Config{
		LoRA: lora.Config{Rank: 1, Alpha: 1, TargetModules: []string{"q_proj"}}, BlockSize: 32,
	}, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	first, err := TrainQLoRA(engine, layer, input, target, 0.02)
	if err != nil {
		t.Fatal(err)
	}
	last := first
	for range 79 {
		last, err = TrainQLoRA(engine, layer, input, target, 0.02)
		if err != nil {
			t.Fatal(err)
		}
	}
	if last >= first {
		t.Fatalf("loss = %v, want below %v", last, first)
	}
	for _, value := range []backend.Tensor{weight, input, target, layer.A(), layer.B()} {
		_ = value.(*Tensor).Close()
	}
}

func TestQLoRAAdapterTrainingDropout(t *testing.T) {
	engine := testEngine(t)
	weight, err := engine.DecodeFloat32(1, 32, make([]float32, 32))
	if err != nil {
		t.Fatal(err)
	}
	inputValues := make([]float32, 128*32)
	targetValues := make([]float32, 128)
	for index := range inputValues {
		inputValues[index] = 1
	}
	for index := range targetValues {
		targetValues[index] = 1
	}
	input, err := engine.DecodeFloat32(128, 32, inputValues)
	if err != nil {
		t.Fatal(err)
	}
	target, err := engine.DecodeFloat32(128, 1, targetValues)
	if err != nil {
		t.Fatal(err)
	}
	layer, err := qlora.QuantizeLinear("q_proj", engine, weight, nil, qlora.Config{
		LoRA: lora.Config{Rank: 1, Alpha: 1, Dropout: 0.5, TargetModules: []string{"q_proj"}}, BlockSize: 32,
	}, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	loss, err := TrainQLoRA(engine, layer, input, target, 0.01)
	if err != nil || math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
		t.Fatalf("dropout TrainQLoRA() = %v, %v", loss, err)
	}
	values, _, _, err := engine.EncodeFloat32(layer.B())
	if err != nil {
		t.Fatal(err)
	}
	if values[0] == 0 {
		t.Fatal("dropout training did not update B")
	}
	for _, value := range []backend.Tensor{weight, input, target, layer.A(), layer.B()} {
		_ = value.(*Tensor).Close()
	}
}

func testEngine(t *testing.T) *Engine {
	t.Helper()
	engine, err := New()
	if errors.Is(err, backend.ErrCUDAUnavailable) {
		t.Skip(err)
	}
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = engine.Close() })
	return engine
}

func same(got, want []float32) bool {
	if len(got) != len(want) {
		return false
	}
	for index := range got {
		if math.Abs(float64(got[index]-want[index])) > 1e-4 {
			return false
		}
	}
	return true
}
