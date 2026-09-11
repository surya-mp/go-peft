package main

import (
	"fmt"
	"math/rand"

	"github.com/surya-mp/go-peft/backend"
	"github.com/surya-mp/go-peft/lora"
	"github.com/surya-mp/go-peft/qlora"
)

type model struct {
	modules      []qlora.Module
	replacements []qlora.Replacement
}

func (m *model) LinearModules() ([]qlora.Module, error) { return m.modules, nil }

func (m *model) ReplaceQLoRALinearModules(replacements []qlora.Replacement) error {
	m.replacements = append([]qlora.Replacement(nil), replacements...)
	return nil
}

func main() {
	engine := backend.NewCPU()
	weight, _ := backend.NewDense(2, 2, []float32{1, 0, 0, 1})
	host := &model{modules: []qlora.Module{{Name: "layers.0.q_proj", Weight: weight}}}
	config := qlora.Config{
		LoRA: lora.Config{Rank: 1, Alpha: 1, TargetModules: []string{"q_proj"}}, BlockSize: 64,
		Quantization: qlora.QuantizationNF4,
	}
	adapter, err := qlora.Inject("example", engine, host, config, rand.New(rand.NewSource(1)))
	if err != nil {
		panic(err)
	}
	input, _ := backend.NewDense(1, 2, []float32{1, 2})
	output, err := host.replacements[0].Layer.Forward(input)
	if err != nil {
		panic(err)
	}
	fmt.Println(adapter.Name(), output.(*backend.Dense).Values())
}
