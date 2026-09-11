package backend

import (
	"math/rand"
	"reflect"
	"testing"
)

func TestCPUGemm(t *testing.T) {
	engine := NewCPU()
	a, _ := NewDense(2, 3, []float32{1, 2, 3, 4, 5, 6})
	b, _ := NewDense(2, 3, []float32{1, 0, 1, 0, 1, 0})
	dst, _ := engine.New(2, 2)

	if err := engine.Gemm(dst, a, false, b, true, 1, 0); err != nil {
		t.Fatal(err)
	}
	if got, want := dst.(*Dense).Values(), []float32{4, 2, 10, 5}; !reflect.DeepEqual(got, want) {
		t.Fatalf("Gemm() = %v, want %v", got, want)
	}
}

func TestCPUDropoutIsDeterministic(t *testing.T) {
	engine := NewCPU()
	first, _ := NewDense(1, 8, []float32{1, 1, 1, 1, 1, 1, 1, 1})
	second, _ := engine.Clone(first)

	if err := engine.Dropout(first, 0.5, rand.New(rand.NewSource(7))); err != nil {
		t.Fatal(err)
	}
	if err := engine.Dropout(second, 0.5, rand.New(rand.NewSource(7))); err != nil {
		t.Fatal(err)
	}
	if got, want := first.Values(), second.(*Dense).Values(); !reflect.DeepEqual(got, want) {
		t.Fatalf("dropout differs: %v != %v", got, want)
	}
}
