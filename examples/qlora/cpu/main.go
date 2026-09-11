package main

import (
	"fmt"
	"math/rand"

	"github.com/surya-mp/go-peft/backend"
	"github.com/surya-mp/go-peft/lora"
	"github.com/surya-mp/go-peft/qlora"
)

func main() {
	engine := backend.NewCPU()
	weight, _ := backend.NewDense(2, 3, []float32{1, 2, 3, 4, 5, 6})
	input, _ := backend.NewDense(1, 3, []float32{1, 1, 1})
	config := qlora.Config{
		LoRA: lora.Config{Rank: 2, Alpha: 4, TargetModules: []string{"q_proj"}}, BlockSize: 64,
		Quantization: qlora.QuantizationNF4, DoubleQuant: true, ScaleBlockSize: 256,
	}
	layer, err := qlora.QuantizeLinear("q_proj", engine, weight, nil, config, rand.New(rand.NewSource(1)))
	if err != nil {
		panic(err)
	}
	output, err := layer.Forward(input)
	if err != nil {
		panic(err)
	}
	fmt.Println(output.(*backend.Dense).Values())
}
