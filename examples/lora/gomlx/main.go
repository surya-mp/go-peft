//go:build gomlx

package main

import (
	"fmt"
	"math/rand"

	"github.com/gomlx/compute/gobackend"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/surya-mp/go-peft/backends/gomlx"
	"github.com/surya-mp/go-peft/lora"
)

type sample struct {
	input  []float32
	target []float32
}

var trainingReviews = []sample{
	{[]float32{1, 0, 1, 0, 1, 0}, []float32{1, 0}},
	{[]float32{1, 0, 0, 0, 1, 1}, []float32{1, 0}},
	{[]float32{0, 1, 0, 1, 1, 0}, []float32{0, 1}},
	{[]float32{0, 0, 0, 1, 0, 1}, []float32{0, 1}},
}

var heldOutReviews = []sample{
	{[]float32{1, 0, 1, 0, 0, 1}, []float32{1, 0}},
	{[]float32{0, 1, 0, 1, 0, 1}, []float32{0, 1}},
}

func main() {
	store := model.NewStore()
	scope := store.RootScope().In("sentiment").In("classifier")
	weight := scope.VariableWithValue("weight", [][]float32{
		{0.1, 0.1, 0.1, 0.1, 0.1, 0.1},
		{-0.1, -0.1, -0.1, -0.1, -0.1, -0.1},
	})
	config := lora.Config{Rank: 2, Alpha: 4, TargetModules: []string{"classifier"}}
	layer, err := gomlx.NewLinear(scope, weight, nil, config, rand.New(rand.NewSource(1)))
	if err != nil {
		panic(err)
	}
	trainer, err := model.NewExec(gobackend.GetBackend(), store, func(scope *model.Scope, input, target *graph.Node) *graph.Node {
		loss := graph.ReduceAllSum(graph.Square(graph.Sub(layer.Apply(scope, input), target)))
		a, b := layer.A().NodeValue(input), layer.B().NodeValue(input)
		gradients := graph.Gradient(loss, a, b)
		layer.A().SetNodeValue(graph.Sub(a, graph.MulScalar(gradients[0], 0.05)))
		layer.B().SetNodeValue(graph.Sub(b, graph.MulScalar(gradients[1], 0.05)))
		return loss
	})
	if err != nil {
		panic(err)
	}
	predict, err := model.NewExec(gobackend.GetBackend(), store, func(scope *model.Scope, input *graph.Node) *graph.Node {
		return layer.Apply(scope, input)
	})
	if err != nil {
		panic(err)
	}
	for epoch := 0; epoch < 80; epoch++ {
		var loss float32
		for _, review := range trainingReviews {
			outputs, err := trainer.Call([][]float32{review.input}, [][]float32{review.target})
			if err != nil {
				panic(err)
			}
			values, err := tensors.CopyFlatData[float32](outputs[0])
			if err != nil {
				panic(err)
			}
			loss += values[0]
		}
		if epoch%20 == 0 {
			fmt.Println("epoch", epoch, "loss", loss)
		}
	}
	correct := 0
	for _, review := range heldOutReviews {
		outputs, err := predict.Call([][]float32{review.input})
		if err != nil {
			panic(err)
		}
		values, err := tensors.CopyFlatData[float32](outputs[0])
		if err != nil {
			panic(err)
		}
		if class(values) == class(review.target) {
			correct++
		}
	}
	fmt.Printf("held-out accuracy %d/%d; base trainable=%v\n", correct, len(heldOutReviews), weight.Trainable)
}

func class(values []float32) int {
	if values[1] > values[0] {
		return 1
	}
	return 0
}
