package safetensors

import (
	"bytes"
	"errors"
	"reflect"
	"testing"
)

func TestRoundTripIsDeterministic(t *testing.T) {
	tensors := map[string]Tensor{
		"z": {Shape: []int{1, 2}, Data: []float32{3, 4}},
		"a": {Shape: []int{2, 1}, Data: []float32{1, 2}},
	}
	metadata := map[string]string{"format": "pt"}
	var first, second bytes.Buffer
	if err := Write(&first, tensors, metadata); err != nil {
		t.Fatal(err)
	}
	if err := Write(&second, tensors, metadata); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first.Bytes(), second.Bytes()) {
		t.Fatal("output is not deterministic")
	}
	got, gotMetadata, err := Read(&first)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, tensors) || !reflect.DeepEqual(gotMetadata, metadata) {
		t.Fatalf("Read() = %#v, %#v", got, gotMetadata)
	}
}

func TestReadRejectsTrailingData(t *testing.T) {
	var data bytes.Buffer
	if err := Write(&data, map[string]Tensor{"x": {Shape: []int{1}, Data: []float32{1}}}, nil); err != nil {
		t.Fatal(err)
	}
	data.WriteByte(0)
	_, _, err := Read(&data)
	if !errors.Is(err, ErrInvalidFile) {
		t.Fatalf("Read() = %v", err)
	}
}
