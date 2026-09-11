package main

import (
	"fmt"
	"math/rand"

	"github.com/surya-mp/go-peft/backend"
	"github.com/surya-mp/go-peft/lora"
)

type model struct {
	modules      []lora.Module
	replacements []lora.Replacement
}

func (m *model) LinearModules() ([]lora.Module, error) { return m.modules, nil }

func (m *model) ReplaceLinearModules(replacements []lora.Replacement) error {
	m.replacements = append([]lora.Replacement(nil), replacements...)
	return nil
}

func main() {
	engine := backend.NewCPU()
	weight, _ := backend.NewDense(2, 2, []float32{1, 0, 0, 1})
	host := &model{modules: []lora.Module{{Name: "layers.0.q_proj", Weight: weight}}}
	config := lora.Config{Rank: 1, Alpha: 1, TargetModules: []string{"q_proj"}}
	adapter, err := lora.Inject("example", engine, host, config, rand.New(rand.NewSource(1)))
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
