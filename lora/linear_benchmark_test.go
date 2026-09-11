package lora

import (
	"math/rand"
	"testing"

	"github.com/surya-mp/go-peft/backend"
)

func BenchmarkLinearForwardInto(b *testing.B) {
	engine := backend.NewCPU()
	weight, _ := engine.New(512, 512)
	input, _ := engine.New(32, 512)
	config := Config{Rank: 16, Alpha: 32, TargetModules: []string{"projection"}}
	layer, err := NewLinear("projection", engine, weight, nil, config, rand.New(rand.NewSource(1)))
	if err != nil {
		b.Fatal(err)
	}
	output, workspace, err := layer.NewWorkspace(input)
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := layer.ForwardInto(output, workspace, input, false, nil); err != nil {
			b.Fatal(err)
		}
	}
}
