//go:build darwin && arm64 && mlx

package main

import (
	"fmt"
	"math/rand"

	"github.com/surya-mp/go-peft/backends/mlx"
	"github.com/surya-mp/go-peft/lora"
)

func main() {
	context, err := mlx.New()
	if err != nil {
		panic(err)
	}
	defer context.Close()
	weight, _ := context.Float32([]int{2, 2}, []float32{1, 0, 0, 1})
	defer weight.Close()
	input, _ := context.Float32([]int{1, 2}, []float32{1, 2})
	defer input.Close()
	target, _ := context.Float32([]int{1, 2}, []float32{2, 4})
	defer target.Close()
	layer, err := mlx.NewLoRALinear(weight, nil, lora.Config{Rank: 1, Alpha: 1, TargetModules: []string{"q_proj"}}, rand.New(rand.NewSource(1)))
	if err != nil {
		panic(err)
	}
	defer layer.A().Close()
	defer layer.B().Close()
	var loss float32
	for range 40 {
		loss, err = layer.TrainStep(input, target, 0.05)
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
