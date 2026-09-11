//go:build cuda && linux && cgo

package cuda

import (
	"math/rand"
	"testing"

	"github.com/surya-mp/go-peft/backend"
	"github.com/surya-mp/go-peft/lora"
	"github.com/surya-mp/go-peft/qlora"
)

func BenchmarkLoRAForward(b *testing.B) {
	engine := benchmarkEngine(b)
	defer engine.Close()
	weight := benchmarkTensor(b, engine, 512, 512)
	defer weight.Close()
	input := benchmarkTensor(b, engine, 32, 512)
	defer input.Close()
	layer, err := lora.NewLinear("projection", engine, weight, nil, lora.Config{Rank: 16, Alpha: 32, TargetModules: []string{"projection"}}, rand.New(rand.NewSource(1)))
	if err != nil {
		b.Fatal(err)
	}
	defer layer.A().(*Tensor).Close()
	defer layer.B().(*Tensor).Close()
	output, workspace, err := layer.NewWorkspace(input)
	if err != nil {
		b.Fatal(err)
	}
	defer output.(*Tensor).Close()
	defer workspace.(*Tensor).Close()
	benchmarkForward(b, engine, output, func() error { return layer.ForwardInto(output, workspace, input, false, nil) })
}

func BenchmarkQLoRAForward(b *testing.B) {
	engine := benchmarkEngine(b)
	defer engine.Close()
	weight := benchmarkTensor(b, engine, 512, 512)
	defer weight.Close()
	input := benchmarkTensor(b, engine, 32, 512)
	defer input.Close()
	layer, err := qlora.QuantizeLinear("projection", engine, weight, nil, qlora.Config{
		LoRA: lora.Config{Rank: 16, Alpha: 32, TargetModules: []string{"projection"}}, BlockSize: 64,
		Quantization: qlora.QuantizationNF4, DoubleQuant: true, ScaleBlockSize: 256,
	}, rand.New(rand.NewSource(1)))
	if err != nil {
		b.Fatal(err)
	}
	defer layer.A().(*Tensor).Close()
	defer layer.B().(*Tensor).Close()
	output, workspace, err := layer.NewWorkspace(input)
	if err != nil {
		b.Fatal(err)
	}
	defer output.(*Tensor).Close()
	defer workspace.(*Tensor).Close()
	benchmarkForward(b, engine, output, func() error { return layer.ForwardInto(output, workspace, input, false, nil) })
}

func benchmarkEngine(b *testing.B) *Engine {
	b.Helper()
	engine, err := New()
	if err != nil {
		b.Skipf("CUDA unavailable: %v", err)
	}
	return engine
}

func benchmarkTensor(b *testing.B, engine *Engine, rows, cols int) *Tensor {
	b.Helper()
	tensor, err := engine.DecodeFloat32(rows, cols, make([]float32, rows*cols))
	if err != nil {
		b.Fatal(err)
	}
	return tensor.(*Tensor)
}

func benchmarkForward(b *testing.B, engine *Engine, output backend.Tensor, forward func() error) {
	b.Helper()
	if err := forward(); err != nil {
		b.Fatal(err)
	}
	if _, _, _, err := engine.EncodeFloat32(output); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if err := forward(); err != nil {
			b.Fatal(err)
		}
		if _, _, _, err := engine.EncodeFloat32(output); err != nil {
			b.Fatal(err)
		}
	}
}
