//go:build darwin && arm64 && mlx

package main

import (
	"fmt"
	"math/rand"

	"github.com/surya-mp/go-peft/backends/mlx"
	"github.com/surya-mp/go-peft/lora"
	"github.com/surya-mp/go-peft/qlora"
)

func main() {
	context, err := mlx.New()
	if err != nil {
		panic(err)
	}
	defer context.Close()
	weightData := make([]float32, 64)
	for index := range weightData {
		weightData[index] = float32(index%11-5) / 5
	}
	weight, err := context.Float32([]int{2, 32}, weightData)
	if err != nil {
		panic(err)
	}
	defer weight.Close()
	inputData := make([]float32, 32)
	for index := range inputData {
		inputData[index] = float32(index) / 32
	}
	input, err := context.Float32([]int{1, 32}, inputData)
	if err != nil {
		panic(err)
	}
	defer input.Close()
	target, err := context.Float32([]int{1, 2}, []float32{1, -1})
	if err != nil {
		panic(err)
	}
	defer target.Close()
	layer, err := mlx.NewQLoRALinear(weight, nil, qlora.Config{
		LoRA:      lora.Config{Rank: 4, Alpha: 8, TargetModules: []string{"q_proj"}},
		BlockSize: 32, Quantization: qlora.QuantizationInt4,
	}, rand.New(rand.NewSource(1)))
	if err != nil {
		panic(err)
	}
	defer layer.Close()
	var loss float32
	for range 80 {
		loss, err = layer.TrainStep(input, target, 0.01)
		if err != nil {
			panic(err)
		}
	}
	output, err := layer.Forward(input)
	if err != nil {
		panic(err)
	}
	defer output.Close()
	values, err := output.Float32Values()
	if err != nil {
		panic(err)
	}
	fmt.Println("loss", loss, "output", values)
}
