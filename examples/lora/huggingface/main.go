package main

import (
	"flag"
	"fmt"
	"math/rand"

	"github.com/surya-mp/go-peft/backend"
	"github.com/surya-mp/go-peft/format/huggingface"
	"github.com/surya-mp/go-peft/lora"
)

func main() {
	dir := flag.String("out", "adapter", "adapter directory")
	flag.Parse()
	engine := backend.NewCPU()
	weight, _ := backend.NewDense(2, 2, []float32{1, 2, 3, 4})
	config := lora.Config{Rank: 1, Alpha: 2, TargetModules: []string{"q_proj"}}
	layer, err := lora.NewLinear("q_proj", engine, weight, nil, config, rand.New(rand.NewSource(1)))
	if err != nil {
		panic(err)
	}
	adapter, err := lora.NewAdapter("example", config)
	if err != nil {
		panic(err)
	}
	if err := adapter.Add(layer); err != nil {
		panic(err)
	}
	metadata := huggingface.Metadata{BaseModelNameOrPath: "example/base", TaskType: "CAUSAL_LM"}
	if err := huggingface.Save(*dir, adapter, engine, metadata); err != nil {
		panic(err)
	}
	if err := engine.Zero(layer.A()); err != nil {
		panic(err)
	}
	if _, err := huggingface.LoadInto(*dir, adapter, engine); err != nil {
		panic(err)
	}
	fmt.Println("saved and restored", *dir)
}
