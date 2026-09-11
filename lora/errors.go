// Package lora provides LoRA configuration and, later, adapter primitives.
package lora

import "errors"

var (
	ErrInvalidRank          = errors.New("lora: rank must be greater than zero")
	ErrInvalidAlpha         = errors.New("lora: alpha must be finite and non-negative")
	ErrInvalidDropout       = errors.New("lora: dropout must be finite and in [0, 1)")
	ErrMissingTargetModules = errors.New("lora: at least one target module is required")
	ErrInvalidBias          = errors.New("lora: invalid bias mode")
	ErrInvalidName          = errors.New("lora: name is required")
	ErrInvalidWeight        = errors.New("lora: weight must be a non-empty matrix")
	ErrInvalidBiasShape     = errors.New("lora: bias must have shape [1, out_features]")
	ErrInputShape           = errors.New("lora: input feature size does not match weight")
	ErrWorkspaceShape       = errors.New("lora: output or workspace has an invalid shape")
	ErrRandomSource         = errors.New("lora: random source is required")
	ErrAlreadyMerged        = errors.New("lora: adapter is already merged")
	ErrNotMerged            = errors.New("lora: adapter is not merged")
	ErrDuplicateLayer       = errors.New("lora: layer is already registered")
	ErrNoTargetModules      = errors.New("lora: no target modules found")
)
