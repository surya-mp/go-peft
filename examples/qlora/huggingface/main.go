package main

import (
	"flag"
	"fmt"
	"math/rand"

	"github.com/surya-mp/go-peft/backend"
	"github.com/surya-mp/go-peft/format/huggingface"
	"github.com/surya-mp/go-peft/lora"
	"github.com/surya-mp/go-peft/qlora"
)

func main() {
	dir := flag.String("out", "adapter", "adapter directory")
	flag.Parse()
	engine := backend.NewCPU()
	weight, _ := backend.NewDense(2, 2, []float32{1, 2, 3, 4})
	config := qlora.Config{
		LoRA: lora.Config{Rank: 1, Alpha: 2, TargetModules: []string{"q_proj"}}, BlockSize: 64,
		Quantization: qlora.QuantizationNF4, DoubleQuant: true, ScaleBlockSize: 256,
	}
	layer, err := qlora.QuantizeLinear("q_proj", engine, weight, nil, config, rand.New(rand.NewSource(1)))
	if err != nil {
		panic(err)
	}
	adapter, err := qlora.NewAdapter("example", config)
	if err != nil {
		panic(err)
	}
	if err := adapter.Add(layer); err != nil {
		panic(err)
	}
	if err := huggingface.SaveQLoRA(*dir, adapter, engine, huggingface.Metadata{BaseModelNameOrPath: "example/base"}); err != nil {
		panic(err)
	}
	if err := engine.Zero(layer.A()); err != nil {
		panic(err)
	}
	if _, err := huggingface.LoadQLoRAInto(*dir, adapter, engine); err != nil {
		panic(err)
	}
	fmt.Println("saved and restored", *dir)
}
