package qlora_test

import (
	"fmt"
	"math/rand"

	"github.com/surya-mp/go-peft/backend"
	"github.com/surya-mp/go-peft/lora"
	"github.com/surya-mp/go-peft/qlora"
)

func ExampleQuantizeLinear() {
	engine := backend.NewCPU()
	weight, err := backend.NewDense(2, 3, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		panic(err)
	}
	layer, err := qlora.QuantizeLinear("q_proj", engine, weight, nil, qlora.Config{
		LoRA:           lora.Config{Rank: 2, Alpha: 4, TargetModules: []string{"q_proj"}},
		BlockSize:      64,
		Quantization:   qlora.QuantizationNF4,
		DoubleQuant:    true,
		ScaleBlockSize: 256,
	}, rand.New(rand.NewSource(1)))
	if err != nil {
		panic(err)
	}
	fmt.Println(layer.InFeatures(), layer.OutFeatures(), layer.Rank())

	// Output:
	// 3 2 2
}
