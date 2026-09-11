package qlora

import (
	"errors"
	"math/rand"
	"testing"

	"github.com/surya-mp/go-peft/backend"
	"github.com/surya-mp/go-peft/lora"
)

func TestPEFTParityQLoRAContract(t *testing.T) {
	engine := backend.NewCPU()
	weight, _ := backend.NewDense(2, 3, []float32{1, -2, 0.5, 0.25, 1.5, -1})
	config := Config{
		LoRA: lora.Config{Rank: 2, Alpha: 4, TargetModules: []string{"q_proj"}}, BlockSize: 3,
		Quantization: QuantizationNF4, DoubleQuant: true, ScaleBlockSize: 2,
	}
	layer, err := QuantizeLinear("q_proj", engine, weight, nil, config, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	metadata, ok := layer.QuantizationMetadata()
	if !ok || metadata.Scheme != "nf4" || !metadata.DoubleQuant {
		t.Fatalf("metadata = %#v, ok=%v", metadata, ok)
	}
	for _, parameter := range layer.Parameters()[:1] {
		if parameter.Trainable {
			t.Fatalf("quantized base parameter = %#v", parameter)
		}
	}
	for _, parameter := range layer.AdapterParameters() {
		if !parameter.Trainable {
			t.Fatalf("adapter parameter = %#v", parameter)
		}
	}
	if !errors.Is(layer.Merge(), ErrMergeUnsupported) || !errors.Is(layer.Unmerge(), ErrMergeUnsupported) {
		t.Fatal("QLoRA merge must be rejected")
	}
}
