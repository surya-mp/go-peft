package lora

import "math"

// BiasMode selects which bias parameters the host should train.
type BiasMode uint8

const (
	BiasNone BiasMode = iota
	BiasAll
	BiasLoRAOnly
)

// Config configures one LoRA adapter.
type Config struct {
	Rank          int
	Alpha         float32
	Dropout       float32
	TargetModules []string
	Bias          BiasMode
}

// Validate reports configuration errors before injection mutates a model.
func (c Config) Validate() error {
	if c.Rank <= 0 {
		return ErrInvalidRank
	}
	if !isFinite(c.Alpha) || c.Alpha < 0 {
		return ErrInvalidAlpha
	}
	if !isFinite(c.Dropout) || c.Dropout < 0 || c.Dropout >= 1 {
		return ErrInvalidDropout
	}
	if len(c.TargetModules) == 0 {
		return ErrMissingTargetModules
	}
	for _, target := range c.TargetModules {
		if target == "" {
			return ErrMissingTargetModules
		}
	}
	if c.Bias > BiasLoRAOnly {
		return ErrInvalidBias
	}

	return nil
}

func isFinite(value float32) bool {
	return !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0)
}
