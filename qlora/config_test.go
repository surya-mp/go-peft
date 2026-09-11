package qlora

import (
	"errors"
	"testing"

	"github.com/surya-mp/go-peft/lora"
)

func TestConfigValidate(t *testing.T) {
	valid := Config{
		LoRA:      lora.Config{Rank: 8, Alpha: 16, Dropout: 0.1, TargetModules: []string{"q_proj"}},
		BlockSize: 64, Quantization: QuantizationNF4, DoubleQuant: true, ScaleBlockSize: 256,
	}
	tests := []struct {
		name string
		edit func(*Config)
		want error
	}{
		{"lora", func(c *Config) { c.LoRA.Rank = 0 }, lora.ErrInvalidRank},
		{"block size", func(c *Config) { c.BlockSize = 0 }, ErrInvalidBlockSize},
		{"quantization", func(c *Config) { c.Quantization = Quantization(2) }, ErrInvalidQuantization},
		{"int4 double quant", func(c *Config) { c.Quantization = QuantizationInt4 }, ErrDoubleQuantRequiresNF4},
		{"scale block size", func(c *Config) { c.ScaleBlockSize = 0 }, ErrInvalidScaleBlockSize},
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

func TestConfigAllowsInt4WithoutDoubleQuant(t *testing.T) {
	config := Config{
		LoRA:      lora.Config{Rank: 1, Alpha: 1, TargetModules: []string{"q_proj"}},
		BlockSize: 64, Quantization: QuantizationInt4,
	}
	if err := config.Validate(); err != nil {
		t.Fatal(err)
	}
}
