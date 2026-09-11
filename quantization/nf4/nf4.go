// Package nf4 provides blockwise NormalFloat4 matrices for QLoRA.
package nf4

import (
	"errors"
	"math"

	"github.com/surya-mp/go-peft/backend"
)

var (
	ErrInvalidShape     = errors.New("nf4: invalid matrix shape")
	ErrInvalidBlockSize = errors.New("nf4: block sizes must be positive")
)

// Matrix stores two NF4 codes per byte and block absolute-max scales.
type Matrix struct {
	rows           int
	cols           int
	blockSize      int
	scaleBlockSize int
	packed         []byte
	scales         []float32
	scaleCodes     []byte
	scaleScales    []float32
}

var codebook = [...]float32{
	-1, -0.6961928, -0.52507305, -0.3949175, -0.28444138, -0.18477343, -0.09105004, 0,
	0.0795803, 0.1609302, 0.2461123, 0.33791524, 0.44070983, 0.562617, 0.72295684, 1,
}

// Quantize creates an NF4 matrix. A positive scale block enables double quantization.
func Quantize(rows, cols, blockSize, scaleBlockSize int, values []float32) (*Matrix, error) {
	if rows <= 0 || cols <= 0 || rows > math.MaxInt/cols || len(values) != rows*cols {
		return nil, ErrInvalidShape
	}
	if blockSize <= 0 || scaleBlockSize < 0 {
		return nil, ErrInvalidBlockSize
	}
	count := len(values)
	blocks := (count + blockSize - 1) / blockSize
	matrix := &Matrix{rows: rows, cols: cols, blockSize: blockSize, scaleBlockSize: scaleBlockSize, packed: make([]byte, (count+1)/2)}
	blockScales := make([]float32, blocks)
	for block := range blockScales {
		start, end := block*blockSize, min((block+1)*blockSize, count)
		var scale float32
		for _, value := range values[start:end] {
			absolute := float32(math.Abs(float64(value)))
			if absolute > scale {
				scale = absolute
			}
		}
		blockScales[block] = scale
		if scale == 0 {
			continue
		}
		for index := start; index < end; index++ {
			matrix.setCode(index, nearest(values[index]/scale))
		}
	}
	if scaleBlockSize == 0 {
		matrix.scales = blockScales
		return matrix, nil
	}
	matrix.scaleCodes = make([]byte, blocks)
	matrix.scaleScales = make([]float32, (blocks+scaleBlockSize-1)/scaleBlockSize)
	for group := range matrix.scaleScales {
		start, end := group*scaleBlockSize, min((group+1)*scaleBlockSize, blocks)
		var maximum float32
		for _, scale := range blockScales[start:end] {
			if scale > maximum {
				maximum = scale
			}
		}
		groupScale := maximum / 255
		matrix.scaleScales[group] = groupScale
		if groupScale == 0 {
			continue
		}
		for index := start; index < end; index++ {
			code := int(math.Round(float64(blockScales[index] / groupScale)))
			if code > 255 {
				code = 255
			}
			matrix.scaleCodes[index] = byte(code)
		}
	}
	return matrix, nil
}

// Rows returns the output feature count.
func (m *Matrix) Rows() int { return m.rows }

// Cols returns the input feature count.
func (m *Matrix) Cols() int { return m.cols }

// BlockSize returns the NF4 value block size.
func (m *Matrix) BlockSize() int { return m.blockSize }

// DoubleQuantized reports whether block scales use an 8-bit second quantizer.
func (m *Matrix) DoubleQuantized() bool { return m.scaleBlockSize != 0 }

// ScaleBlockSize returns zero when double quantization is disabled.
func (m *Matrix) ScaleBlockSize() int { return m.scaleBlockSize }

// QuantizationMetadata describes this NF4 representation.
func (m *Matrix) QuantizationMetadata() backend.QuantizationMetadata {
	return backend.QuantizationMetadata{
		Scheme: "nf4", Bits: 4, BlockSize: m.blockSize,
		DoubleQuant: m.DoubleQuantized(), ScaleBlockSize: m.scaleBlockSize,
	}
}

// QuantizedStorage returns copies of the packed buffers for native backends.
func (m *Matrix) QuantizedStorage() backend.QuantizedStorage {
	return backend.QuantizedStorage{
		Codes: append([]byte(nil), m.packed...), Scales: append([]float32(nil), m.scales...),
		ScaleCodes: append([]byte(nil), m.scaleCodes...), ScaleScales: append([]float32(nil), m.scaleScales...),
	}
}

// StorageBytes returns packed codes and scale storage only.
func (m *Matrix) StorageBytes() int {
	return len(m.packed) + len(m.scales)*4 + len(m.scaleCodes) + len(m.scaleScales)*4
}

// At dequantizes one value.
func (m *Matrix) At(row, col int) float32 {
	index := row*m.cols + col
	return codebook[m.code(index)] * m.scale(index/m.blockSize)
}

// DotRow computes input · dequantized(weight[row]) without allocating.
func (m *Matrix) DotRow(input []float32, row int) float32 {
	start := row * m.cols
	var sum float32
	for col := 0; col < m.cols; col++ {
		index := start + col
		sum += input[col] * codebook[m.code(index)] * m.scale(index/m.blockSize)
	}
	return sum
}

// Values returns a dequantized row-major copy.
func (m *Matrix) Values() []float32 {
	values := make([]float32, m.rows*m.cols)
	for index := range values {
		values[index] = codebook[m.code(index)] * m.scale(index/m.blockSize)
	}
	return values
}

func (m *Matrix) scale(block int) float32 {
	if m.scaleBlockSize == 0 {
		return m.scales[block]
	}
	return float32(m.scaleCodes[block]) * m.scaleScales[block/m.scaleBlockSize]
}

func (m *Matrix) code(index int) int {
	value := m.packed[index/2]
	if index&1 == 0 {
		return int(value & 0x0f)
	}
	return int(value >> 4)
}

func (m *Matrix) setCode(index, code int) {
	if index&1 == 0 {
		m.packed[index/2] = m.packed[index/2]&0xf0 | byte(code)
		return
	}
	m.packed[index/2] = m.packed[index/2]&0x0f | byte(code)<<4
}

func nearest(value float32) int {
	best, distance := 0, float32(math.MaxFloat32)
	for index, code := range codebook {
		candidate := float32(math.Abs(float64(value - code)))
		if candidate < distance {
			best, distance = index, candidate
		}
	}
	return best
}

// NearestCode returns the NF4 code closest to a normalized value.
func NearestCode(value float32) byte { return byte(nearest(value)) }

func min(left, right int) int {
	if left < right {
		return left
	}
	return right
}
