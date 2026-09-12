// Package backend defines the tensor operations required by PEFT algorithms.
package backend

import "math/rand"

// Tensor is owned by a concrete backend.
type Tensor interface{}

// EagerEngine supplies the reference eager-tensor path used by the CPU backend.
// Graph and lazy frameworks use their native bridges instead.
type EagerEngine interface {
	New(rows, cols int) (Tensor, error)
	Clone(Tensor) (Tensor, error)
	Copy(dst, src Tensor) error
	Shape(Tensor) (rows, cols int, err error)
	Zero(Tensor) error
	Uniform(t Tensor, low, high float32, rng *rand.Rand) error
	Dropout(t Tensor, probability float32, rng *rand.Rand) error
	AddRow(dst, row Tensor) error
	Gemm(dst, a Tensor, transposeA bool, b Tensor, transposeB bool, alpha, beta float32) error
}

// Engine is kept for source compatibility. New eager integrations use EagerEngine.
type Engine = EagerEngine

// Float32Codec transfers dense float32 tensors for portable formats.
type Float32Codec interface {
	EncodeFloat32(Tensor) (data []float32, rows, cols int, err error)
	DecodeFloat32(rows, cols int, data []float32) (Tensor, error)
}

// QuantizedWeight is a frozen matrix stored in a backend-neutral format.
type QuantizedWeight interface {
	Rows() int
	Cols() int
}

// QuantizationMetadata describes immutable quantized-weight storage.
type QuantizationMetadata struct {
	// Scheme identifies the quantizer, such as "int4" or "nf4".
	Scheme string
	// Bits is the number of bits used for each weight code.
	Bits int
	// BlockSize is the number of weights sharing one scale.
	BlockSize int
	// DoubleQuant reports whether block scales are quantized.
	DoubleQuant bool
	// ScaleBlockSize is the number of scales sharing one second-level scale.
	// It is zero when DoubleQuant is false.
	ScaleBlockSize int
}

// QuantizedWeightMetadata exposes portable quantization metadata when available.
type QuantizedWeightMetadata interface {
	QuantizedWeight
	QuantizationMetadata() QuantizationMetadata
}

// QuantizedStorage is the portable packed representation used by native kernels.
type QuantizedStorage struct {
	// Codes contains packed primary quantization codes.
	Codes []byte
	// Scales contains F32 block scales when double quantization is disabled.
	Scales []float32
	// ScaleCodes contains quantized block scales when double quantization is enabled.
	ScaleCodes []byte
	// ScaleScales contains F32 second-level scale values.
	ScaleScales []float32
}

// QuantizedStorageCodec exposes copied packed buffers for device upload.
type QuantizedStorageCodec interface {
	QuantizedWeightMetadata
	QuantizedStorage() QuantizedStorage
}

// QuantizedLinearEngine evaluates a quantized base linear projection.
type QuantizedLinearEngine interface {
	QuantizedLinear(dst, input Tensor, weight QuantizedWeight, alpha, beta float32) error
}
