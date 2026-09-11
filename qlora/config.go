// Package qlora provides LoRA over frozen quantized base weights.
package qlora

import (
	"errors"

	"github.com/surya-mp/go-peft/lora"
)

var (
	ErrInvalidBlockSize       = errors.New("qlora: block size must be positive")
	ErrInvalidQuantization    = errors.New("qlora: invalid quantization")
	ErrDoubleQuantRequiresNF4 = errors.New("qlora: double quantization requires NF4")
	ErrInvalidScaleBlockSize  = errors.New("qlora: scale block size must be positive with double quantization")
)

// Quantization selects the frozen 4-bit weight representation.
type Quantization uint8

const (
	// QuantizationInt4 preserves the original symmetric signed-int4 format.
	QuantizationInt4 Quantization = iota
	// QuantizationNF4 uses the NormalFloat4 codebook used by QLoRA.
	QuantizationNF4
)

// Config combines LoRA settings with a hardware-neutral 4-bit base weight.
type Config struct {
	LoRA           lora.Config
	BlockSize      int
	Quantization   Quantization
	DoubleQuant    bool
	ScaleBlockSize int
}

// Validate reports configuration errors before quantization.
func (c Config) Validate() error {
	if err := c.LoRA.Validate(); err != nil {
		return err
	}
	if c.BlockSize <= 0 {
		return ErrInvalidBlockSize
	}
	if c.Quantization > QuantizationNF4 {
		return ErrInvalidQuantization
	}
	if c.DoubleQuant && c.Quantization != QuantizationNF4 {
		return ErrDoubleQuantRequiresNF4
	}
	if c.DoubleQuant && c.ScaleBlockSize <= 0 {
		return ErrInvalidScaleBlockSize
	}
	return nil
}
