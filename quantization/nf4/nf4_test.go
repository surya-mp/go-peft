package nf4

import (
	"math"
	"testing"
)

func TestQuantizeCodebookAndDot(t *testing.T) {
	values := []float32{-1, -0.6961928, 0, 0.1609302, 1}
	matrix, err := Quantize(1, len(values), len(values), 0, values)
	if err != nil {
		t.Fatal(err)
	}
	for index, want := range values {
		if got := matrix.At(0, index); math.Abs(float64(got-want)) > 1e-6 {
			t.Fatalf("At(%d) = %v, want %v", index, got, want)
		}
	}
	if got, want := matrix.DotRow([]float32{1, 2, 3, 4, 5}, 0), float32(3.2513351); math.Abs(float64(got-want)) > 1e-5 {
		t.Fatalf("DotRow() = %v, want %v", got, want)
	}
}

func TestDoubleQuantizationReducesScaleStorage(t *testing.T) {
	values := make([]float32, 1024)
	for index := range values {
		values[index] = float32(index%23-11) / 11
	}
	plain, err := Quantize(1, len(values), 8, 0, values)
	if err != nil {
		t.Fatal(err)
	}
	double, err := Quantize(1, len(values), 8, 64, values)
	if err != nil {
		t.Fatal(err)
	}
	if !double.DoubleQuantized() || double.StorageBytes() >= plain.StorageBytes() {
		t.Fatalf("double=%d, plain=%d", double.StorageBytes(), plain.StorageBytes())
	}
	for index, want := range plain.Values() {
		got := double.At(0, index)
		if math.Abs(float64(got-want)) > 0.01 {
			t.Fatalf("At(%d) = %v, plain = %v", index, got, want)
		}
	}
}

func TestQuantizedStorageCopiesDoubleQuantBuffers(t *testing.T) {
	matrix, err := Quantize(1, 8, 2, 2, []float32{-1, -0.5, 0, 0.5, 1, 0.25, -0.25, 0.75})
	if err != nil {
		t.Fatal(err)
	}
	storage := matrix.QuantizedStorage()
	if len(storage.Codes) != 4 || len(storage.Scales) != 0 || len(storage.ScaleCodes) != 4 || len(storage.ScaleScales) != 2 {
		t.Fatalf("storage = %#v", storage)
	}
	before := matrix.At(0, 0)
	storage.Codes[0] ^= 0x0f
	if matrix.At(0, 0) != before {
		t.Fatal("storage aliases matrix")
	}
}
