// Package qlora implements LoRA over frozen int4 or NF4 linear weights.
//
// QLoRA layers preserve trainable float32 A and B matrices but do not support
// merging their update into the quantized base matrix.
package qlora

import (
	"errors"

	"github.com/surya-mp/go-peft/lora"
)

// Errors returned by QLoRA configuration validation.
// Callers can test them with errors.Is.
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
	// LoRA configures the trainable adapter matrices.
	LoRA lora.Config
	// BlockSize is the number of base weights sharing one primary scale.
	BlockSize int
	// Quantization selects int4 or NF4 for the frozen base weight.
	Quantization Quantization
	// DoubleQuant enables 8-bit quantization of NF4 block scales.
	// It requires QuantizationNF4.
	DoubleQuant bool
	// ScaleBlockSize is the number of NF4 block scales sharing one scale.
	// It must be positive when DoubleQuant is true.
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
