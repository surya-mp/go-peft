package peft

import "testing"

func TestParameterOwnsShape(t *testing.T) {
	shape := []int{8, 4}
	parameter := NewParameter("adapter.a", shape, true)
	shape[0] = 99

	if parameter.Shape[0] != 8 {
		t.Fatalf("shape was not copied: %v", parameter.Shape)
	}
}

func TestParameterClone(t *testing.T) {
	parameter := NewParameter("adapter.b", []int{4, 8}, true)
	clone := parameter.Clone()
	clone.Shape[0] = 99

	if parameter.Shape[0] != 4 {
		t.Fatalf("clone shares shape: %v", parameter.Shape)
	}
}

func TestParameterCount(t *testing.T) {
	parameters := []Parameter{
		NewParameter("base", []int{4, 8}, false),
		NewParameter("adapter", []int{2, 8}, true),
	}
	total, trainable, err := ParameterCount(parameters)
	if err != nil {
		t.Fatal(err)
	}
	if total != 48 || trainable != 16 {
		t.Fatalf("ParameterCount() = %d, %d", total, trainable)
	}
}
