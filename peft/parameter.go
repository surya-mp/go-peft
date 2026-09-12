package peft

import (
	"errors"
	"math"
)

var ErrInvalidParameterShape = errors.New("peft: parameter dimensions must be positive")

// Parameter describes a model or adapter parameter without owning its tensor.
type Parameter struct {
	// Name is the stable framework-neutral parameter identifier.
	Name string
	// Shape is a copy of the parameter dimensions.
	Shape []int
	// Trainable reports whether an optimizer should update this parameter.
	Trainable bool
}

// NewParameter copies shape so callers retain ownership of their slice.
func NewParameter(name string, shape []int, trainable bool) Parameter {
	return Parameter{Name: name, Shape: cloneShape(shape), Trainable: trainable}
}

// Clone returns an independent metadata value.
func (p Parameter) Clone() Parameter {
	p.Shape = cloneShape(p.Shape)
	return p
}

func cloneShape(shape []int) []int {
	if len(shape) == 0 {
		return nil
	}

	return append([]int(nil), shape...)
}

// ParameterCount reports total and trainable scalar values.
func ParameterCount(parameters []Parameter) (total, trainable uint64, err error) {
	for _, parameter := range parameters {
		count := uint64(1)
		for _, dimension := range parameter.Shape {
			if dimension <= 0 {
				return 0, 0, ErrInvalidParameterShape
			}
			if count > math.MaxUint64/uint64(dimension) {
				return 0, 0, ErrInvalidParameterShape
			}
			count *= uint64(dimension)
		}
		if total > math.MaxUint64-count {
			return 0, 0, ErrInvalidParameterShape
		}
		total += count
		if parameter.Trainable {
			if trainable > math.MaxUint64-count {
				return 0, 0, ErrInvalidParameterShape
			}
			trainable += count
		}
	}
	return total, trainable, nil
}
