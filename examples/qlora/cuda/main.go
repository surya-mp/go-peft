package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/surya-mp/go-peft/backend"
	"github.com/surya-mp/go-peft/backends/cuda"
	"github.com/surya-mp/go-peft/lora"
	"github.com/surya-mp/go-peft/qlora"
)

func main() {
	engine, err := cuda.New()
	if err != nil {
		log.Fatal(err)
	}
	defer engine.Close()
	runtime, err := backend.NewRuntime(backend.RuntimeOptions{CUDA: engine, Warn: func(err error) { log.Print(err) }})
	if err != nil {
		log.Fatal("CUDA is required by default; set DisableCUDA: true to opt into CPU")
	}
	codec, ok := runtime.Engine.(backend.Float32Codec)
	if !ok {
		log.Fatal("selected engine does not support float32 transfer")
	}
	weight, err := codec.DecodeFloat32(2, 3, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		log.Fatal(err)
	}
	input, err := codec.DecodeFloat32(1, 3, []float32{1, 1, 1})
	if err != nil {
		log.Fatal(err)
	}
	config := qlora.Config{
		LoRA: lora.Config{Rank: 2, Alpha: 4, TargetModules: []string{"q_proj"}}, BlockSize: 64,
		Quantization: qlora.QuantizationNF4, DoubleQuant: true, ScaleBlockSize: 256,
	}
	layer, err := qlora.QuantizeLinear("q_proj", runtime.Engine, weight, nil, config, rand.New(rand.NewSource(1)))
	if err != nil {
		log.Fatal(err)
	}
	output, err := layer.Forward(input)
	if err != nil {
		log.Fatal(err)
	}
	values, _, _, err := codec.EncodeFloat32(output)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(values)
}
