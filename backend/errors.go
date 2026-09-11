package backend

import "errors"

var (
	ErrInvalidShape  = errors.New("backend: dimensions must be positive")
	ErrInvalidTensor = errors.New("backend: unsupported tensor")
	ErrShapeMismatch = errors.New("backend: tensor shapes do not match")
	ErrInvalidRandom = errors.New("backend: random source is required")
)
