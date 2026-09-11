package qlora

import (
	"math/rand"
	"testing"

	"github.com/surya-mp/go-peft/backend"
	"github.com/surya-mp/go-peft/lora"
)

func BenchmarkLinearForwardInto(b *testing.B) {
	engine := backend.NewCPU()
	weight, _ := engine.New(512, 512)
	input, _ := engine.New(32, 512)
	config := Config{
		LoRA: lora.Config{Rank: 16, Alpha: 32, TargetModules: []string{"projection"}}, BlockSize: 64,
		Quantization: QuantizationNF4, DoubleQuant: true, ScaleBlockSize: 256,
	}
	layer, err := QuantizeLinear("projection", engine, weight, nil, config, rand.New(rand.NewSource(1)))
	if err != nil {
		b.Fatal(err)
	}
	output, workspace, err := layer.NewWorkspace(input)
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for index := 0; index < b.N; index++ {
		if err := layer.ForwardInto(output, workspace, input, false, nil); err != nil {
			b.Fatal(err)
		}
	}
}
