// Package peft contains small abstractions shared by PEFT algorithms.
package peft

// Adapter exposes the parameters owned by one PEFT method.
type Adapter interface {
	Name() string
	Parameters() []Parameter
}
