package backend

import (
	"fmt"
	"math/rand"
)

// CPU is a dense float32 engine for tests, examples, and small workloads.
type CPU struct{}

// Dense is a contiguous row-major float32 matrix.
type Dense struct {
	rows int
	cols int
	data []float32
}

// NewCPU returns a stateless dense CPU engine.
func NewCPU() CPU { return CPU{} }

// NewDense copies data into a dense matrix.
func NewDense(rows, cols int, data []float32) (*Dense, error) {
	if rows <= 0 || cols <= 0 {
		return nil, ErrInvalidShape
	}
	if len(data) != rows*cols {
		return nil, fmt.Errorf("%w: got %d values for %dx%d", ErrShapeMismatch, len(data), rows, cols)
	}

	return &Dense{rows: rows, cols: cols, data: append([]float32(nil), data...)}, nil
}

// At returns one element.
func (m *Dense) At(row, col int) float32 { return m.data[row*m.cols+col] }

// Values returns a copy of the matrix data in row-major order.
func (m *Dense) Values() []float32 { return append([]float32(nil), m.data...) }

func (CPU) New(rows, cols int) (Tensor, error) {
	if rows <= 0 || cols <= 0 {
		return nil, ErrInvalidShape
	}

	return &Dense{rows: rows, cols: cols, data: make([]float32, rows*cols)}, nil
}

func (CPU) Clone(t Tensor) (Tensor, error) {
	m, err := dense(t)
	if err != nil {
		return nil, err
	}

	return &Dense{rows: m.rows, cols: m.cols, data: append([]float32(nil), m.data...)}, nil
}

func (CPU) Copy(dst, src Tensor) error {
	d, err := dense(dst)
	if err != nil {
		return err
	}
	s, err := dense(src)
	if err != nil {
		return err
	}
	if d.rows != s.rows || d.cols != s.cols {
		return ErrShapeMismatch
	}

	copy(d.data, s.data)
	return nil
}

func (CPU) Shape(t Tensor) (rows, cols int, err error) {
	m, err := dense(t)
	if err != nil {
		return 0, 0, err
	}

	return m.rows, m.cols, nil
}

func (CPU) EncodeFloat32(t Tensor) (data []float32, rows, cols int, err error) {
	m, err := dense(t)
	if err != nil {
		return nil, 0, 0, err
	}
	return append([]float32(nil), m.data...), m.rows, m.cols, nil
}

func (CPU) DecodeFloat32(rows, cols int, data []float32) (Tensor, error) {
	return NewDense(rows, cols, data)
}

func (CPU) Zero(t Tensor) error {
	m, err := dense(t)
	if err != nil {
		return err
	}
	for i := range m.data {
		m.data[i] = 0
	}
	return nil
}

func (CPU) Uniform(t Tensor, low, high float32, rng *rand.Rand) error {
	m, err := dense(t)
	if err != nil {
		return err
	}
	if rng == nil {
		return ErrInvalidRandom
	}

	width := high - low
	for i := range m.data {
		m.data[i] = low + width*rng.Float32()
	}
	return nil
}

func (CPU) Dropout(t Tensor, probability float32, rng *rand.Rand) error {
	m, err := dense(t)
	if err != nil {
		return err
	}
	if probability == 0 {
		return nil
	}
	if probability < 0 || probability >= 1 {
		return fmt.Errorf("%w: dropout probability", ErrShapeMismatch)
	}
	if rng == nil {
		return ErrInvalidRandom
	}

	scale := 1 / (1 - probability)
	for i, value := range m.data {
		if rng.Float32() < probability {
			m.data[i] = 0
		} else {
			m.data[i] = value * scale
		}
	}
	return nil
}

func (CPU) AddRow(dst, row Tensor) error {
	d, err := dense(dst)
	if err != nil {
		return err
	}
	r, err := dense(row)
	if err != nil {
		return err
	}
	if r.rows != 1 || r.cols != d.cols {
		return ErrShapeMismatch
	}
	for offset := 0; offset < len(d.data); offset += d.cols {
		for col, value := range r.data {
			d.data[offset+col] += value
		}
	}
	return nil
}

// Gemm computes dst = alpha * op(a) * op(b) + beta * dst without allocations.
func (CPU) Gemm(dst, a Tensor, transposeA bool, b Tensor, transposeB bool, alpha, beta float32) error {
	d, err := dense(dst)
	if err != nil {
		return err
	}
	left, err := dense(a)
	if err != nil {
		return err
	}
	right, err := dense(b)
	if err != nil {
		return err
	}

	m, k := matrixShape(left, transposeA)
	bRows, n := matrixShape(right, transposeB)
	if k != bRows || d.rows != m || d.cols != n {
		return fmt.Errorf("%w: dst=%dx%d, left=%dx%d, right=%dx%d", ErrShapeMismatch, d.rows, d.cols, m, k, bRows, n)
	}

	if transposeB {
		for row := 0; row < m; row++ {
			dstRow := d.data[row*n : (row+1)*n]
			for col := 0; col < n; col++ {
				var sum float32
				for inner := 0; inner < k; inner++ {
					sum += matrixAt(left, transposeA, row, inner) * right.data[col*right.cols+inner]
				}
				dstRow[col] = alpha*sum + beta*dstRow[col]
			}
		}
		return nil
	}

	for row := 0; row < m; row++ {
		dstRow := d.data[row*n : (row+1)*n]
		if beta == 0 {
			for col := range dstRow {
				dstRow[col] = 0
			}
		} else if beta != 1 {
			for col := range dstRow {
				dstRow[col] *= beta
			}
		}
		for inner := 0; inner < k; inner++ {
			value := alpha * matrixAt(left, transposeA, row, inner)
			if value == 0 {
				continue
			}
			for col := 0; col < n; col++ {
				dstRow[col] += value * right.data[inner*right.cols+col]
			}
		}
	}

	return nil
}

func (CPU) QuantizedLinear(dst, input Tensor, weight QuantizedWeight, alpha, beta float32) error {
	d, err := dense(dst)
	if err != nil {
		return err
	}
	x, err := dense(input)
	if err != nil {
		return err
	}
	quantized, ok := weight.(interface {
		DotRow(input []float32, row int) float32
	})
	if !ok || weight.Rows() != d.cols || weight.Cols() != x.cols || d.rows != x.rows {
		return ErrShapeMismatch
	}
	for row := 0; row < x.rows; row++ {
		inputRow := x.data[row*x.cols : (row+1)*x.cols]
		outputRow := d.data[row*d.cols : (row+1)*d.cols]
		for col := range outputRow {
			outputRow[col] = alpha*quantized.DotRow(inputRow, col) + beta*outputRow[col]
		}
	}
	return nil
}

func dense(t Tensor) (*Dense, error) {
	m, ok := t.(*Dense)
	if !ok || m == nil {
		return nil, ErrInvalidTensor
	}
	return m, nil
}

func matrixShape(m *Dense, transpose bool) (rows, cols int) {
	if transpose {
		return m.cols, m.rows
	}
	return m.rows, m.cols
}

func matrixAt(m *Dense, transpose bool, row, col int) float32 {
	if transpose {
		return m.data[col*m.cols+row]
	}
	return m.data[row*m.cols+col]
}
