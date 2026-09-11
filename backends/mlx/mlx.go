//go:build darwin && arm64 && mlx

// Package mlx is an opt-in Go binding for MLX C on Apple Silicon.
package mlx

/*
#cgo CFLAGS: -I/opt/homebrew/opt/mlx-c/include
#cgo LDFLAGS: -L/opt/homebrew/opt/mlx-c/lib -lmlxc -Wl,-rpath,/opt/homebrew/opt/mlx-c/lib
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mlx/c/mlx.h"

static _Thread_local char mlx_go_error_buffer[4096];

static void mlx_go_error_handler(const char* message, void* data) {
	(void)data;
	snprintf(mlx_go_error_buffer, sizeof(mlx_go_error_buffer), "%s", message);
}

static void mlx_go_install_error_handler(void) {
	mlx_set_error_handler(mlx_go_error_handler, NULL, NULL);
}

static void mlx_go_reset_error(void) {
	mlx_go_error_buffer[0] = '\0';
}

static const char* mlx_go_last_error(void) {
	return mlx_go_error_buffer;
}
*/
import "C"

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"
)

var (
	ErrClosed      = errors.New("mlx: value is closed")
	ErrShape       = errors.New("mlx: invalid shape")
	ErrContext     = errors.New("mlx: arrays belong to different contexts")
	ErrUnsupported = errors.New("mlx: unsupported array dtype")
	installHandler sync.Once
)

// Context owns an MLX stream and pins its creating goroutine until Close.
// Use and close a context from the goroutine that created it.
type Context struct {
	stream C.mlx_stream
	closed atomic.Bool
	locked bool
}

// Array owns one MLX array handle.
type Array struct {
	context *Context
	value   C.mlx_array
	closed  atomic.Bool
}

// New creates a context on MLX's default GPU stream.
func New() (*Context, error) {
	installHandler.Do(func() { C.mlx_go_install_error_handler() })
	runtime.LockOSThread()
	C.mlx_go_reset_error()
	context := &Context{stream: C.mlx_default_gpu_stream_new(), locked: true}
	if context.stream.ctx == nil {
		runtime.UnlockOSThread()
		return nil, nativeError("create GPU stream", 1)
	}
	runtime.SetFinalizer(context, (*Context).finalize)
	return context, nil
}

// NewCPU creates a context on MLX's default CPU stream.
func NewCPU() (*Context, error) {
	installHandler.Do(func() { C.mlx_go_install_error_handler() })
	runtime.LockOSThread()
	C.mlx_go_reset_error()
	context := &Context{stream: C.mlx_default_cpu_stream_new(), locked: true}
	if context.stream.ctx == nil {
		runtime.UnlockOSThread()
		return nil, nativeError("create CPU stream", 1)
	}
	runtime.SetFinalizer(context, (*Context).finalize)
	return context, nil
}

// Close releases the stream. Close all arrays before closing their context.
func (c *Context) Close() error {
	if c == nil || c.closed.Swap(true) {
		return nil
	}
	runtime.SetFinalizer(c, nil)
	C.mlx_go_reset_error()
	err := nativeError("free stream", C.mlx_stream_free(c.stream))
	if c.locked {
		c.locked = false
		runtime.UnlockOSThread()
	}
	return err
}

// Float32 copies row-major data into an MLX float32 array.
func (c *Context) Float32(shape []int, data []float32) (*Array, error) {
	if err := c.check(); err != nil {
		return nil, err
	}
	if err := validShape(shape, len(data)); err != nil {
		return nil, err
	}
	dimensions := make([]C.int, len(shape))
	for index, dimension := range shape {
		dimensions[index] = C.int(dimension)
	}
	C.mlx_go_reset_error()
	value := C.mlx_array_new_data(
		unsafe.Pointer(unsafe.SliceData(data)),
		(*C.int)(unsafe.Pointer(unsafe.SliceData(dimensions))),
		C.int(len(dimensions)),
		C.MLX_FLOAT32,
	)
	if value.ctx == nil {
		return nil, nativeError("create float32 array", 1)
	}
	return newArray(c, value), nil
}

// Bernoulli creates a device-resident 0/1 mask with the given keep probability.
func (c *Context) Bernoulli(shape []int, probability float32) (*Array, error) {
	if err := c.check(); err != nil {
		return nil, err
	}
	if probability < 0 || probability > 1 || len(shape) == 0 {
		return nil, ErrShape
	}
	count := 1
	dimensions := make([]C.int, len(shape))
	for index, dimension := range shape {
		if dimension <= 0 {
			return nil, ErrShape
		}
		count *= dimension
		dimensions[index] = C.int(dimension)
	}
	if count <= 0 {
		return nil, ErrShape
	}
	C.mlx_go_reset_error()
	p := C.mlx_array_new_float32(C.float(probability))
	if p.ctx == nil {
		return nil, nativeError("create Bernoulli probability", 1)
	}
	defer C.mlx_array_free(p)
	result := C.mlx_array_new()
	if err := nativeError("create Bernoulli mask", C.mlx_random_bernoulli(&result, p, (*C.int)(unsafe.Pointer(unsafe.SliceData(dimensions))), C.size_t(len(dimensions)), C.mlx_array{}, c.stream)); err != nil {
		_ = C.mlx_array_free(result)
		return nil, err
	}
	return newArray(c, result), nil
}

// Shape returns a copy of the array shape.
func (a *Array) Shape() ([]int, error) {
	if err := a.check(); err != nil {
		return nil, err
	}
	dimensions := int(C.mlx_array_ndim(a.value))
	shape := make([]int, dimensions)
	nativeShape := C.mlx_array_shape(a.value)
	for index := range shape {
		shape[index] = int(unsafe.Slice(nativeShape, dimensions)[index])
	}
	return shape, nil
}

// Eval materializes the array and its dependencies.
func (a *Array) Eval() error {
	if err := a.check(); err != nil {
		return err
	}
	C.mlx_go_reset_error()
	return nativeError("evaluate array", C.mlx_array_eval(a.value))
}

// Float32Values evaluates and copies a contiguous float32 array to Go memory.
func (a *Array) Float32Values() ([]float32, error) {
	if err := a.check(); err != nil {
		return nil, err
	}
	if C.mlx_array_dtype(a.value) != C.MLX_FLOAT32 {
		return nil, ErrUnsupported
	}
	contiguous, err := a.contiguous()
	if err != nil {
		return nil, err
	}
	defer contiguous.Close()
	if err := contiguous.Eval(); err != nil {
		return nil, err
	}
	count := int(C.mlx_array_size(contiguous.value))
	data := make([]float32, count)
	if count > 0 {
		copy(data, unsafe.Slice((*float32)(unsafe.Pointer(C.mlx_array_data_float32(contiguous.value))), count))
	}
	return data, nil
}

// Add returns a + b.
func (a *Array) Add(b *Array) (*Array, error) {
	return a.binary("add", b, func(result *C.mlx_array, left, right C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_add(result, left, right, stream)
	})
}

// Mul returns a * b.
func (a *Array) Mul(b *Array) (*Array, error) {
	return a.binary("multiply", b, func(result *C.mlx_array, left, right C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_multiply(result, left, right, stream)
	})
}

// MatMul returns a @ b.
func (a *Array) MatMul(b *Array) (*Array, error) {
	return a.binary("matmul", b, func(result *C.mlx_array, left, right C.mlx_array, stream C.mlx_stream) C.int {
		return C.mlx_matmul(result, left, right, stream)
	})
}

// Transpose reverses array axes. For matrices, it returns the matrix transpose.
func (a *Array) Transpose() (*Array, error) {
	if err := a.check(); err != nil {
		return nil, err
	}
	return a.unary("transpose", func(result *C.mlx_array) C.int {
		return C.mlx_transpose(result, a.value, a.context.stream)
	})
}

// Scale returns a * scalar.
func (a *Array) Scale(scalar float32) (*Array, error) {
	if err := a.check(); err != nil {
		return nil, err
	}
	C.mlx_go_reset_error()
	value := C.mlx_array_new_float32(C.float(scalar))
	if value.ctx == nil {
		return nil, nativeError("create scalar", 1)
	}
	temporary := newArray(a.context, value)
	defer temporary.Close()
	return a.Mul(temporary)
}

// Quantize stores a matrix in MLX's native packed affine format.
func (a *Array) Quantize(groupSize, bits int) (weight, scales, biases *Array, err error) {
	if err = a.check(); err != nil {
		return nil, nil, nil, err
	}
	if groupSize <= 0 || bits <= 0 || bits > 8 {
		return nil, nil, nil, ErrShape
	}
	result := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(result)
	group := C.mlx_optional_int{value: C.int(groupSize), has_value: true}
	width := C.mlx_optional_int{value: C.int(bits), has_value: true}
	mode := C.CString("affine")
	defer C.free(unsafe.Pointer(mode))
	C.mlx_go_reset_error()
	if err = nativeError("quantize", C.mlx_quantize(&result, a.value, group, width, mode, C.mlx_array{}, a.context.stream)); err != nil {
		return nil, nil, nil, err
	}
	if C.mlx_vector_array_size(result) != 3 {
		return nil, nil, nil, ErrUnsupported
	}
	arrays := make([]*Array, 3)
	for index := range arrays {
		var value C.mlx_array
		if err = nativeError("quantize result", C.mlx_vector_array_get(&value, result, C.size_t(index))); err != nil {
			for _, array := range arrays {
				_ = array.Close()
			}
			return nil, nil, nil, err
		}
		arrays[index] = newArray(a.context, value)
	}
	return arrays[0], arrays[1], arrays[2], nil
}

// QuantizedMatMul multiplies input by a native packed affine weight matrix.
func (a *Array) QuantizedMatMul(weight, scales, biases *Array, transpose bool, groupSize, bits int) (*Array, error) {
	if err := a.check(); err != nil {
		return nil, err
	}
	if err := weight.check(); err != nil {
		return nil, err
	}
	if err := scales.check(); err != nil {
		return nil, err
	}
	if a.context != weight.context || a.context != scales.context || (biases != nil && biases.context != a.context) {
		return nil, ErrContext
	}
	if groupSize <= 0 || bits <= 0 || bits > 8 {
		return nil, ErrShape
	}
	var bias C.mlx_array
	if biases != nil {
		if err := biases.check(); err != nil {
			return nil, err
		}
		bias = biases.value
	}
	group := C.mlx_optional_int{value: C.int(groupSize), has_value: true}
	width := C.mlx_optional_int{value: C.int(bits), has_value: true}
	mode := C.CString("affine")
	defer C.free(unsafe.Pointer(mode))
	result := C.mlx_array_new()
	C.mlx_go_reset_error()
	if err := nativeError("quantized matmul", C.mlx_quantized_matmul(&result, a.value, weight.value, scales.value, bias, C.bool(transpose), group, width, mode, a.context.stream)); err != nil {
		_ = C.mlx_array_free(result)
		return nil, err
	}
	return newArray(a.context, result), nil
}

// Close releases the native MLX array.
func (a *Array) Close() error {
	if a == nil || a.closed.Swap(true) {
		return nil
	}
	runtime.SetFinalizer(a, nil)
	C.mlx_go_reset_error()
	return nativeError("free array", C.mlx_array_free(a.value))
}

type binaryOperation func(*C.mlx_array, C.mlx_array, C.mlx_array, C.mlx_stream) C.int

func (a *Array) binary(name string, b *Array, operation binaryOperation) (*Array, error) {
	if err := a.check(); err != nil {
		return nil, err
	}
	if err := b.check(); err != nil {
		return nil, err
	}
	if a.context != b.context {
		return nil, ErrContext
	}
	return a.unary(name, func(result *C.mlx_array) C.int {
		return operation(result, a.value, b.value, a.context.stream)
	})
}

func (a *Array) unary(name string, operation func(*C.mlx_array) C.int) (*Array, error) {
	result := C.mlx_array_new()
	C.mlx_go_reset_error()
	if err := nativeError(name, operation(&result)); err != nil {
		_ = C.mlx_array_free(result)
		return nil, err
	}
	return newArray(a.context, result), nil
}

func (a *Array) contiguous() (*Array, error) {
	if err := a.check(); err != nil {
		return nil, err
	}
	return a.unary("make contiguous", func(result *C.mlx_array) C.int {
		return C.mlx_contiguous(result, a.value, false, a.context.stream)
	})
}

func newArray(context *Context, value C.mlx_array) *Array {
	array := &Array{context: context, value: value}
	runtime.SetFinalizer(array, (*Array).finalize)
	return array
}

func (a *Array) finalize()   { _ = a.Close() }
func (c *Context) finalize() { _ = c.Close() }

func (a *Array) check() error {
	if a == nil || a.closed.Load() {
		return ErrClosed
	}
	return a.context.check()
}

func (c *Context) check() error {
	if c == nil || c.closed.Load() {
		return ErrClosed
	}
	return nil
}

func validShape(shape []int, dataLength int) error {
	count := uint64(1)
	for _, dimension := range shape {
		if dimension < 0 || dimension > math.MaxInt32 || count > math.MaxUint64/uint64(max(1, dimension)) {
			return ErrShape
		}
		count *= uint64(dimension)
	}
	if count != uint64(dataLength) {
		return ErrShape
	}
	return nil
}

func nativeError(operation string, code C.int) error {
	if code == 0 {
		return nil
	}
	if message := C.GoString(C.mlx_go_last_error()); message != "" {
		return fmt.Errorf("mlx: %s: %s", operation, message)
	}
	return fmt.Errorf("mlx: %s failed", operation)
}

func max(left, right int) int {
	if left > right {
		return left
	}
	return right
}
