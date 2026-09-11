//go:build darwin && arm64 && mlx

package mlx

import (
	"math/rand"
	"testing"

	"github.com/surya-mp/go-peft/lora"
	"github.com/surya-mp/go-peft/qlora"
)

func BenchmarkLoRAForward(b *testing.B) {
	context, err := New()
	if err != nil {
		b.Fatal(err)
	}
	defer context.Close()
	weight, err := context.Float32([]int{512, 512}, make([]float32, 512*512))
	if err != nil {
		b.Fatal(err)
	}
	defer weight.Close()
	layer, err := NewLoRALinear(weight, nil, lora.Config{Rank: 16, Alpha: 32, TargetModules: []string{"projection"}}, rand.New(rand.NewSource(1)))
	if err != nil {
		b.Fatal(err)
	}
	defer layer.a.Close()
	defer layer.b.Close()
	input, err := context.Float32([]int{32, 512}, make([]float32, 32*512))
	if err != nil {
		b.Fatal(err)
	}
	defer input.Close()
	b.ReportAllocs()
	b.ResetTimer()
	for index := 0; index < b.N; index++ {
		output, err := layer.Forward(input)
		if err != nil {
			b.Fatal(err)
		}
		if err := output.Eval(); err != nil {
			b.Fatal(err)
		}
		_ = output.Close()
	}
}

func BenchmarkQLoRAForward(b *testing.B) {
	context, err := New()
	if err != nil {
		b.Fatal(err)
	}
	defer context.Close()
	weight, err := context.Float32([]int{512, 512}, make([]float32, 512*512))
	if err != nil {
		b.Fatal(err)
	}
	defer weight.Close()
	layer, err := NewQLoRALinear(weight, nil, qlora.Config{
		LoRA: lora.Config{Rank: 16, Alpha: 32, TargetModules: []string{"projection"}}, BlockSize: 64,
	}, rand.New(rand.NewSource(1)))
	if err != nil {
		b.Fatal(err)
	}
	defer layer.Close()
	input, err := context.Float32([]int{32, 512}, make([]float32, 32*512))
	if err != nil {
		b.Fatal(err)
	}
	defer input.Close()
	b.ReportAllocs()
	b.ResetTimer()
	for index := 0; index < b.N; index++ {
		output, err := layer.Forward(input)
		if err != nil {
			b.Fatal(err)
		}
		if err := output.Eval(); err != nil {
			b.Fatal(err)
		}
		_ = output.Close()
	}
}
