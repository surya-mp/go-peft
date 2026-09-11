package qlora

import "errors"

var (
	ErrInvalidName       = errors.New("qlora: name is required")
	ErrUnsupportedEngine = errors.New("qlora: engine does not support quantized linear")
	ErrInvalidWeight     = errors.New("qlora: invalid quantized weight")
	ErrInvalidBiasShape  = errors.New("qlora: bias must have shape [1, out_features]")
	ErrInputShape        = errors.New("qlora: input feature size does not match weight")
	ErrWorkspaceShape    = errors.New("qlora: output or workspace has an invalid shape")
	ErrRandomSource      = errors.New("qlora: random source is required")
	ErrMergeUnsupported  = errors.New("qlora: quantized weights cannot be merged losslessly")
	ErrDuplicateLayer    = errors.New("qlora: layer is already registered")
	ErrNoTargetModules   = errors.New("qlora: no target modules found")
)
