package int4

import (
	"math"
	"testing"
)

func TestQuantize(t *testing.T) {
	values := []float32{-7, -3, -1, 0, 1, 3, 7}
	matrix, err := Quantize(1, len(values), len(values), values)
	if err != nil {
		t.Fatal(err)
	}
	for index, want := range values {
		if got := matrix.At(0, index); got != want {
			t.Fatalf("At(%d) = %v, want %v", index, got, want)
		}
	}
	if matrix.StorageBytes() >= len(values)*4 {
		t.Fatalf("storage = %d, float32 = %d", matrix.StorageBytes(), len(values)*4)
	}
}

func TestQuantizationErrorIsBounded(t *testing.T) {
	values := []float32{-2.1, -0.7, 0.2, 0.8, 1.3, 2.1}
	matrix, err := Quantize(1, len(values), len(values), values)
	if err != nil {
		t.Fatal(err)
	}
	bound := float32(2.1/7/2) + 1e-6
	for index, want := range values {
		if error := float32(math.Abs(float64(matrix.At(0, index) - want))); error > bound {
			t.Fatalf("error = %v, bound = %v", error, bound)
		}
	}
}

func TestQuantizedStorageCopiesBuffers(t *testing.T) {
	matrix, err := Quantize(1, 4, 2, []float32{-1, 0, 1, 2})
	if err != nil {
		t.Fatal(err)
	}
	storage := matrix.QuantizedStorage()
	if len(storage.Codes) != 2 || len(storage.Scales) != 2 {
		t.Fatalf("storage = %#v", storage)
	}
	before := matrix.At(0, 0)
	storage.Codes[0] ^= 0x0f
	if matrix.At(0, 0) != before {
		t.Fatal("storage aliases matrix")
	}
}
