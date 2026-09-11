// Package int4 provides blockwise symmetric signed 4-bit matrices.
package int4

import (
	"errors"
	"math"

	"github.com/surya-mp/go-peft/backend"
)

var (
	ErrInvalidShape     = errors.New("int4: invalid matrix shape")
	ErrInvalidBlockSize = errors.New("int4: block size must be positive")
)

// Matrix stores two signed 4-bit values per byte with one scale per block.
type Matrix struct {
	rows      int
	cols      int
	blockSize int
	packed    []byte
	scales    []float32
}

// Quantize creates a blockwise symmetric int4 matrix.
func Quantize(rows, cols, blockSize int, values []float32) (*Matrix, error) {
	if rows <= 0 || cols <= 0 || rows > math.MaxInt/cols || len(values) != rows*cols {
		return nil, ErrInvalidShape
	}
	if blockSize <= 0 {
		return nil, ErrInvalidBlockSize
	}
	count := len(values)
	blocks := (count + blockSize - 1) / blockSize
	matrix := &Matrix{
		rows: rows, cols: cols, blockSize: blockSize,
		packed: make([]byte, (count+1)/2), scales: make([]float32, blocks),
	}
	for block := 0; block < blocks; block++ {
		start := block * blockSize
		end := min(start+blockSize, count)
		var maximum float32
		for _, value := range values[start:end] {
			absolute := float32(math.Abs(float64(value)))
			if absolute > maximum {
				maximum = absolute
			}
		}
		scale := maximum / 7
		matrix.scales[block] = scale
		if scale == 0 {
			continue
		}
		for index := start; index < end; index++ {
			quantized := int(math.Round(float64(values[index] / scale)))
			if quantized < -8 {
				quantized = -8
			} else if quantized > 7 {
				quantized = 7
			}
			matrix.setCode(index, int8(quantized))
		}
	}
	return matrix, nil
}

// Rows returns the output feature count.
func (m *Matrix) Rows() int { return m.rows }

// Cols returns the input feature count.
func (m *Matrix) Cols() int { return m.cols }

// BlockSize returns the number of values sharing one scale.
func (m *Matrix) BlockSize() int { return m.blockSize }

// QuantizationMetadata describes this signed int4 representation.
func (m *Matrix) QuantizationMetadata() backend.QuantizationMetadata {
	return backend.QuantizationMetadata{Scheme: "int4", Bits: 4, BlockSize: m.blockSize}
}

// QuantizedStorage returns copies of the packed buffers for native backends.
func (m *Matrix) QuantizedStorage() backend.QuantizedStorage {
	return backend.QuantizedStorage{Codes: append([]byte(nil), m.packed...), Scales: append([]float32(nil), m.scales...)}
}

// StorageBytes returns the packed-code and scale storage size.
func (m *Matrix) StorageBytes() int { return len(m.packed) + len(m.scales)*4 }

// At dequantizes one value.
func (m *Matrix) At(row, col int) float32 {
	index := row*m.cols + col
	return float32(m.code(index)) * m.scales[index/m.blockSize]
}

// DotRow computes input · dequantized(weight[row]) without allocating.
func (m *Matrix) DotRow(input []float32, row int) float32 {
	start := row * m.cols
	var sum float32
	for col, value := range input {
		index := start + col
		sum += value * float32(m.code(index)) * m.scales[index/m.blockSize]
	}
	return sum
}

// Values returns a dequantized row-major copy.
func (m *Matrix) Values() []float32 {
	values := make([]float32, m.rows*m.cols)
	for row := 0; row < m.rows; row++ {
		for col := 0; col < m.cols; col++ {
			values[row*m.cols+col] = m.At(row, col)
		}
	}
	return values
}

func (m *Matrix) code(index int) int8 {
	value := m.packed[index/2]
	if index&1 != 0 {
		value >>= 4
	} else {
		value &= 0x0f
	}
	if value&0x08 != 0 {
		return int8(value | 0xf0)
	}
	return int8(value)
}

func (m *Matrix) setCode(index int, value int8) {
	encoded := byte(value) & 0x0f
	if index&1 == 0 {
		m.packed[index/2] = m.packed[index/2]&0xf0 | encoded
	} else {
		m.packed[index/2] = m.packed[index/2]&0x0f | encoded<<4
	}
}

func min(left, right int) int {
	if left < right {
		return left
	}
	return right
}
