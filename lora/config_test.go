package lora

import (
	"errors"
	"math"
	"testing"
)

func TestConfigValidate(t *testing.T) {
	valid := Config{
		Rank:          8,
		Alpha:         16,
		Dropout:       0.1,
		TargetModules: []string{"q_proj", "v_proj"},
		Bias:          BiasNone,
	}

	tests := []struct {
		name string
		edit func(*Config)
		want error
	}{
		{"rank", func(c *Config) { c.Rank = 0 }, ErrInvalidRank},
		{"negative alpha", func(c *Config) { c.Alpha = -1 }, ErrInvalidAlpha},
		{"nan alpha", func(c *Config) { c.Alpha = float32(math.NaN()) }, ErrInvalidAlpha},
		{"infinite alpha", func(c *Config) { c.Alpha = float32(math.Inf(1)) }, ErrInvalidAlpha},
		{"negative dropout", func(c *Config) { c.Dropout = -0.1 }, ErrInvalidDropout},
		{"unit dropout", func(c *Config) { c.Dropout = 1 }, ErrInvalidDropout},
		{"nan dropout", func(c *Config) { c.Dropout = float32(math.NaN()) }, ErrInvalidDropout},
		{"missing targets", func(c *Config) { c.TargetModules = nil }, ErrMissingTargetModules},
		{"empty target", func(c *Config) { c.TargetModules = []string{""} }, ErrMissingTargetModules},
		{"bias", func(c *Config) { c.Bias = BiasMode(99) }, ErrInvalidBias},
	}

	if err := valid.Validate(); err != nil {
		t.Fatalf("valid config: %v", err)
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			config := valid
			test.edit(&config)
			if !errors.Is(config.Validate(), test.want) {
				t.Fatalf("Validate() = %v, want %v", config.Validate(), test.want)
			}
		})
	}
}
