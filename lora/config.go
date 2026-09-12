package lora

import "math"

// BiasMode selects which bias parameters the host should train.
type BiasMode uint8

// BiasMode values select trainable host biases.
const (
	BiasNone BiasMode = iota
	BiasAll
	BiasLoRAOnly
)

// Config configures one LoRA adapter.
type Config struct {
	// Rank is the inner dimension of A and B. It must be positive.
	Rank int
	// Alpha scales the update by Alpha / Rank. It must be finite and non-negative.
	Alpha float32
	// Dropout is applied to the LoRA branch only during training.
	// It must be in [0, 1).
	Dropout float32
	// TargetModules contains exact module names or dot-separated suffixes to replace.
	TargetModules []string
	// Bias selects which host bias parameters remain trainable.
	Bias BiasMode
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
