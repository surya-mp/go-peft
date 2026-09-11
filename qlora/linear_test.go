package qlora

import (
	"errors"
	"math"
	"math/rand"
	"testing"

	"github.com/surya-mp/go-peft/backend"
	"github.com/surya-mp/go-peft/lora"
	"github.com/surya-mp/go-peft/quantization/int4"
	"github.com/surya-mp/go-peft/quantization/nf4"
)

func TestForwardMatchesQuantizedReference(t *testing.T) {
	engine := backend.NewCPU()
	weight, _ := backend.NewDense(2, 3, []float32{1, -2, 3, -4, 5, -6})
	input, _ := backend.NewDense(1, 3, []float32{2, -1, 0.5})
	config := Config{LoRA: lora.Config{Rank: 2, Alpha: 4, TargetModules: []string{"linear"}}, BlockSize: 64}
	layer, err := QuantizeLinear("linear", engine, weight, nil, config, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	a, _ := backend.NewDense(2, 3, []float32{1, 0, 0, 0, 1, 0})
	b, _ := backend.NewDense(2, 2, []float32{1, 2, 3, 4})
	if err := engine.Copy(layer.A(), a); err != nil {
		t.Fatal(err)
	}
	if err := engine.Copy(layer.B(), b); err != nil {
		t.Fatal(err)
	}

	output, err := layer.Forward(input)
	if err != nil {
		t.Fatal(err)
	}
	values := output.(*backend.Dense).Values()
	for row := 0; row < 2; row++ {
		base := layer.Weight().(*int4.Matrix).DotRow([]float32{2, -1, 0.5}, row)
		delta := float32(2) * ([]float32{2, -1}[0]*b.At(row, 0) + []float32{2, -1}[1]*b.At(row, 1))
		if math.Abs(float64(values[row]-(base+delta))) > 1e-5 {
			t.Fatalf("output[%d] = %v, want %v", row, values[row], base+delta)
		}
	}
}

func TestMergeIsRejected(t *testing.T) {
	engine := backend.NewCPU()
	weight, _ := backend.NewDense(1, 1, []float32{1})
	config := Config{LoRA: lora.Config{Rank: 1, Alpha: 1, TargetModules: []string{"linear"}}, BlockSize: 1}
	layer, err := QuantizeLinear("linear", engine, weight, nil, config, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	if !errors.Is(layer.Merge(), ErrMergeUnsupported) {
		t.Fatalf("Merge() = %v", layer.Merge())
	}
	if !errors.Is(layer.Unmerge(), ErrMergeUnsupported) {
		t.Fatalf("Unmerge() = %v", layer.Unmerge())
	}
	if layer.Merged() || layer.InFeatures() != 1 || layer.OutFeatures() != 1 || layer.Rank() != 1 {
		t.Fatalf("metadata = merged:%v in:%d out:%d rank:%d", layer.Merged(), layer.InFeatures(), layer.OutFeatures(), layer.Rank())
	}
}

func TestNF4DoubleQuantForward(t *testing.T) {
	engine := backend.NewCPU()
	weight, _ := backend.NewDense(2, 3, []float32{1, -2, 3, -4, 5, -6})
	input, _ := backend.NewDense(1, 3, []float32{2, -1, 0.5})
	config := Config{
		LoRA: lora.Config{Rank: 1, Alpha: 1, TargetModules: []string{"linear"}}, BlockSize: 2,
		Quantization: QuantizationNF4, DoubleQuant: true, ScaleBlockSize: 2,
	}
	layer, err := QuantizeLinear("linear", engine, weight, nil, config, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	if matrix, ok := layer.Weight().(*nf4.Matrix); !ok || !matrix.DoubleQuantized() {
		t.Fatalf("weight = %T", layer.Weight())
	}
	metadata, ok := layer.QuantizationMetadata()
	if !ok || metadata.Scheme != "nf4" || !metadata.DoubleQuant {
		t.Fatalf("metadata = %#v, ok=%v", metadata, ok)
	}
	output, err := layer.Forward(input)
	if err != nil {
		t.Fatal(err)
	}
	for _, value := range output.(*backend.Dense).Values() {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("invalid output %v", value)
		}
	}
}
