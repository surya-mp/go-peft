package lora_test

import (
	"fmt"
	"math/rand"

	"github.com/surya-mp/go-peft/backend"
	"github.com/surya-mp/go-peft/lora"
)

func ExampleNewLinear() {
	engine := backend.NewCPU()
	weight, err := backend.NewDense(2, 3, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		panic(err)
	}
	input, err := backend.NewDense(1, 3, []float32{1, 1, 1})
	if err != nil {
		panic(err)
	}
	layer, err := lora.NewLinear("q_proj", engine, weight, nil, lora.Config{
		Rank: 2, Alpha: 4, TargetModules: []string{"q_proj"},
	}, rand.New(rand.NewSource(1)))
	if err != nil {
		panic(err)
	}
	output, err := layer.Forward(input)
	if err != nil {
		panic(err)
	}
	fmt.Println(output.(*backend.Dense).Values())

	// Output:
	// [6 15]
}
