//go:build cuda && linux && cgo

// Package cuda provides a native CUDA backend for LoRA and QLoRA.
package cuda

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -L${SRCDIR} -L/usr/local/cuda/lib64 -lgopeftcuda -lcudart -lcublas -lstdc++
#include "gopeft_cuda.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"math/rand"
	"reflect"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/surya-mp/go-peft/backend"
)

var ErrClosed = errors.New("cuda: value is closed")

// Engine owns one CUDA device and cuBLAS handle.
type Engine struct {
	context unsafe.Pointer
	mu      sync.Mutex
	closed  bool
	weights map[uintptr]*quantizedWeight
	seed    atomic.Uint64
}

// Tensor owns a row-major float32 device buffer.
type Tensor struct {
	engine  *Engine
	pointer unsafe.Pointer
	rows    int
	cols    int
	closed  bool
}

type quantizedWeight struct {
	codes, scales, scaleCodes, scaleScales unsafe.Pointer
	metadata                               backend.QuantizationMetadata
}

// New selects CUDA device zero.
func New() (*Engine, error) { return NewDevice(0) }

// NewDevice selects a CUDA device and creates its cuBLAS handle.
func NewDevice(device int) (*Engine, error) {
	var count C.int
	if err := native("query devices", C.peft_cuda_device_count(&count)); err != nil || count == 0 {
		if err != nil {
			return nil, fmt.Errorf("%w: %v", backend.ErrCUDAUnavailable, err)
		}
		return nil, backend.ErrCUDAUnavailable
	}
	if device < 0 || device >= int(count) {
		return nil, backend.ErrCUDAUnavailable
	}
	engine := &Engine{weights: make(map[uintptr]*quantizedWeight)}
	if err := native("create context", C.peft_cuda_new((*unsafe.Pointer)(unsafe.Pointer(&engine.context)), C.int(device))); err != nil {
		return nil, err
	}
	runtime.SetFinalizer(engine, (*Engine).finalize)
	return engine, nil
}

// Close releases cached quantized weights and the cuBLAS handle.
func (e *Engine) Close() error {
	if e == nil {
		return nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil
	}
	e.closed = true
	runtime.SetFinalizer(e, nil)
	var first error
	for _, weight := range e.weights {
		for _, pointer := range []unsafe.Pointer{weight.scaleScales, weight.scaleCodes, weight.scales, weight.codes} {
			if pointer != nil {
				if err := native("free quantized weight", C.peft_cuda_free(pointer)); err != nil && first == nil {
					first = err
				}
			}
		}
	}
	if err := native("free context", C.peft_cuda_free_context(e.context)); err != nil && first == nil {
		first = err
	}
	return first
}

// New allocates a zeroed device matrix.
func (e *Engine) New(rows, cols int) (backend.Tensor, error) {
	if err := e.check(); err != nil {
		return nil, err
	}
	if rows <= 0 || cols <= 0 {
		return nil, backend.ErrInvalidShape
	}
	tensor := &Tensor{engine: e, rows: rows, cols: cols}
	if err := native("allocate tensor", C.peft_cuda_alloc((*unsafe.Pointer)(unsafe.Pointer(&tensor.pointer)), C.size_t(rows*cols*4))); err != nil {
		return nil, err
	}
	if err := native("zero tensor", C.peft_cuda_zero(tensor.pointer, C.size_t(rows*cols*4))); err != nil {
		_ = tensor.Close()
		return nil, err
	}
	runtime.SetFinalizer(tensor, (*Tensor).finalize)
	return tensor, nil
}

// Clone copies a device matrix.
func (e *Engine) Clone(value backend.Tensor) (backend.Tensor, error) {
	source, err := e.tensor(value)
	if err != nil {
		return nil, err
	}
	clone, err := e.New(source.rows, source.cols)
	if err != nil {
		return nil, err
	}
	if err := e.Copy(clone, source); err != nil {
		_ = clone.(*Tensor).Close()
		return nil, err
	}
	return clone, nil
}

// Copy copies a same-shaped device matrix.
func (e *Engine) Copy(dst, src backend.Tensor) error {
	destination, err := e.tensor(dst)
	if err != nil {
		return err
	}
	source, err := e.tensor(src)
	if err != nil {
		return err
	}
	if destination.rows != source.rows || destination.cols != source.cols {
		return backend.ErrShapeMismatch
	}
	return native("copy tensor", C.peft_cuda_d2d(destination.pointer, source.pointer, C.size_t(destination.rows*destination.cols*4)))
}

// Shape returns matrix dimensions.
func (e *Engine) Shape(value backend.Tensor) (int, int, error) {
	tensor, err := e.tensor(value)
	if err != nil {
		return 0, 0, err
	}
	return tensor.rows, tensor.cols, nil
}

// EncodeFloat32 copies a device matrix to row-major host memory.
func (e *Engine) EncodeFloat32(value backend.Tensor) ([]float32, int, int, error) {
	tensor, err := e.tensor(value)
	if err != nil {
		return nil, 0, 0, err
	}
	data := make([]float32, tensor.rows*tensor.cols)
	if len(data) != 0 {
		if err := native("copy tensor to host", C.peft_cuda_d2h(unsafe.Pointer(unsafe.SliceData(data)), tensor.pointer, C.size_t(len(data)*4))); err != nil {
			return nil, 0, 0, err
		}
	}
	return data, tensor.rows, tensor.cols, nil
}

// DecodeFloat32 copies row-major host data to a device matrix.
func (e *Engine) DecodeFloat32(rows, cols int, data []float32) (backend.Tensor, error) {
	if rows <= 0 || cols <= 0 || len(data) != rows*cols {
		return nil, backend.ErrInvalidShape
	}
	value, err := e.New(rows, cols)
	if err != nil {
		return nil, err
	}
	tensor := value.(*Tensor)
	if err := native("copy tensor to device", C.peft_cuda_h2d(tensor.pointer, unsafe.Pointer(unsafe.SliceData(data)), C.size_t(len(data)*4))); err != nil {
		_ = tensor.Close()
		return nil, err
	}
	return tensor, nil
}

// Zero clears a device matrix.
func (e *Engine) Zero(value backend.Tensor) error {
	tensor, err := e.tensor(value)
	if err != nil {
		return err
	}
	return native("zero tensor", C.peft_cuda_zero(tensor.pointer, C.size_t(tensor.rows*tensor.cols*4)))
}

// Uniform initializes a matrix deterministically from the supplied random source.
func (e *Engine) Uniform(value backend.Tensor, low, high float32, rng *rand.Rand) error {
	if rng == nil {
		return backend.ErrInvalidRandom
	}
	tensor, err := e.tensor(value)
	if err != nil {
		return err
	}
	data := make([]float32, tensor.rows*tensor.cols)
	for index := range data {
		data[index] = low + (high-low)*rng.Float32()
	}
	return native("initialize tensor", C.peft_cuda_h2d(tensor.pointer, unsafe.Pointer(unsafe.SliceData(data)), C.size_t(len(data)*4)))
}

// Dropout applies deterministic device-side dropout.
func (e *Engine) Dropout(value backend.Tensor, probability float32, rng *rand.Rand) error {
	if probability == 0 {
		return nil
	}
	if probability < 0 || probability >= 1 {
		return backend.ErrShapeMismatch
	}
	if rng == nil {
		return backend.ErrInvalidRandom
	}
	tensor, err := e.tensor(value)
	if err != nil {
		return err
	}
	return native("apply dropout", C.peft_cuda_dropout(tensor.pointer, C.int(tensor.rows*tensor.cols), C.float(probability), C.uint64_t(rng.Uint64())))
}

func (e *Engine) dropoutMask(rows, cols int, probability float32, seed uint64) (*Tensor, error) {
	if probability <= 0 || probability >= 1 {
		return nil, backend.ErrShapeMismatch
	}
	value, err := e.New(rows, cols)
	if err != nil {
		return nil, err
	}
	mask := value.(*Tensor)
	if err := native("create dropout mask", C.peft_cuda_dropout_mask(mask.pointer, C.int(rows*cols), C.float(probability), C.uint64_t(seed))); err != nil {
		_ = mask.Close()
		return nil, err
	}
	return mask, nil
}

func (e *Engine) multiply(dst, src backend.Tensor) error {
	destination, err := e.tensor(dst)
	if err != nil {
		return err
	}
	source, err := e.tensor(src)
	if err != nil {
		return err
	}
	if destination.rows != source.rows || destination.cols != source.cols {
		return backend.ErrShapeMismatch
	}
	return native("multiply", C.peft_cuda_multiply(destination.pointer, source.pointer, C.int(destination.rows*destination.cols)))
}

func (e *Engine) nextDropoutSeed() int64 { return int64(e.seed.Add(1)) }

// AddRow adds one row vector to every row of dst.
func (e *Engine) AddRow(dst, row backend.Tensor) error {
	destination, err := e.tensor(dst)
	if err != nil {
		return err
	}
	bias, err := e.tensor(row)
	if err != nil {
		return err
	}
	if bias.rows != 1 || bias.cols != destination.cols {
		return backend.ErrShapeMismatch
	}
	return native("add row", C.peft_cuda_add_row(destination.pointer, bias.pointer, C.int(destination.rows), C.int(destination.cols)))
}

func (e *Engine) axpy(dst, src backend.Tensor, alpha float32) error {
	destination, err := e.tensor(dst)
	if err != nil {
		return err
	}
	source, err := e.tensor(src)
	if err != nil {
		return err
	}
	if destination.rows != source.rows || destination.cols != source.cols {
		return backend.ErrShapeMismatch
	}
	return native("axpy", C.peft_cuda_axpy(e.context, destination.pointer, source.pointer, C.int(destination.rows*destination.cols), C.float(alpha)))
}

// Gemm computes dst = alpha * op(a) * op(b) + beta * dst with cuBLAS.
func (e *Engine) Gemm(dst, a backend.Tensor, transposeA bool, b backend.Tensor, transposeB bool, alpha, beta float32) error {
	destination, err := e.tensor(dst)
	if err != nil {
		return err
	}
	left, err := e.tensor(a)
	if err != nil {
		return err
	}
	right, err := e.tensor(b)
	if err != nil {
		return err
	}
	m, k := matrixShape(left, transposeA)
	bRows, n := matrixShape(right, transposeB)
	if k != bRows || destination.rows != m || destination.cols != n {
		return backend.ErrShapeMismatch
	}
	return native("gemm", C.peft_cuda_gemm(e.context, destination.pointer, left.pointer, right.pointer, C.int(m), C.int(n), C.int(k), C.int(left.cols), C.int(right.cols), C.int(boolInt(transposeA)), C.int(boolInt(transposeB)), C.float(alpha), C.float(beta)))
}

// QuantizedLinear runs the fused int4/NF4 base projection kernel.
func (e *Engine) QuantizedLinear(dst, input backend.Tensor, weight backend.QuantizedWeight, alpha, beta float32) error {
	destination, err := e.tensor(dst)
	if err != nil {
		return err
	}
	x, err := e.tensor(input)
	if err != nil {
		return err
	}
	if destination.rows != x.rows || destination.cols != weight.Rows() || x.cols != weight.Cols() {
		return backend.ErrShapeMismatch
	}
	deviceWeight, err := e.uploadWeight(weight)
	if err != nil {
		return err
	}
	scheme := 0
	switch deviceWeight.metadata.Scheme {
	case "int4":
	case "nf4":
		scheme = 1
	default:
		return backend.ErrInvalidTensor
	}
	return native("quantized linear", C.peft_cuda_quantized_linear(e.context, destination.pointer, x.pointer, deviceWeight.codes, deviceWeight.scales, deviceWeight.scaleCodes, deviceWeight.scaleScales, C.int(x.rows), C.int(weight.Rows()), C.int(weight.Cols()), C.int(deviceWeight.metadata.BlockSize), C.int(deviceWeight.metadata.ScaleBlockSize), C.int(scheme), C.float(alpha), C.float(beta)))
}

// Close releases a device tensor.
func (t *Tensor) Close() error {
	if t == nil || t.closed {
		return nil
	}
	t.closed = true
	runtime.SetFinalizer(t, nil)
	return native("free tensor", C.peft_cuda_free(t.pointer))
}

func (e *Engine) uploadWeight(weight backend.QuantizedWeight) (*quantizedWeight, error) {
	storage, ok := weight.(backend.QuantizedStorageCodec)
	if !ok {
		return nil, backend.ErrInvalidTensor
	}
	key, cacheable := pointerKey(weight)
	e.mu.Lock()
	if cacheable {
		if cached := e.weights[key]; cached != nil {
			e.mu.Unlock()
			return cached, nil
		}
	}
	e.mu.Unlock()
	buffers := storage.QuantizedStorage()
	deviceWeight := &quantizedWeight{metadata: storage.QuantizationMetadata()}
	var err error
	if deviceWeight.codes, err = uploadBytes(buffers.Codes); err != nil {
		return nil, err
	}
	if deviceWeight.scales, err = uploadFloats(buffers.Scales); err != nil {
		_ = freeWeight(deviceWeight)
		return nil, err
	}
	if deviceWeight.scaleCodes, err = uploadBytes(buffers.ScaleCodes); err != nil {
		_ = freeWeight(deviceWeight)
		return nil, err
	}
	if deviceWeight.scaleScales, err = uploadFloats(buffers.ScaleScales); err != nil {
		_ = freeWeight(deviceWeight)
		return nil, err
	}
	if !cacheable {
		return deviceWeight, nil
	}
	e.mu.Lock()
	if cached := e.weights[key]; cached != nil {
		e.mu.Unlock()
		_ = freeWeight(deviceWeight)
		return cached, nil
	}
	e.weights[key] = deviceWeight
	e.mu.Unlock()
	return deviceWeight, nil
}

func uploadBytes(data []byte) (unsafe.Pointer, error) {
	if len(data) == 0 {
		return nil, nil
	}
	var pointer unsafe.Pointer
	if err := native("allocate quantized buffer", C.peft_cuda_alloc((*unsafe.Pointer)(unsafe.Pointer(&pointer)), C.size_t(len(data)))); err != nil {
		return nil, err
	}
	if err := native("copy quantized buffer", C.peft_cuda_h2d(pointer, unsafe.Pointer(unsafe.SliceData(data)), C.size_t(len(data)))); err != nil {
		_ = native("free quantized buffer", C.peft_cuda_free(pointer))
		return nil, err
	}
	return pointer, nil
}

func uploadFloats(data []float32) (unsafe.Pointer, error) {
	if len(data) == 0 {
		return nil, nil
	}
	var pointer unsafe.Pointer
	if err := native("allocate quantized scales", C.peft_cuda_alloc((*unsafe.Pointer)(unsafe.Pointer(&pointer)), C.size_t(len(data)*4))); err != nil {
		return nil, err
	}
	if err := native("copy quantized scales", C.peft_cuda_h2d(pointer, unsafe.Pointer(unsafe.SliceData(data)), C.size_t(len(data)*4))); err != nil {
		_ = native("free quantized scales", C.peft_cuda_free(pointer))
		return nil, err
	}
	return pointer, nil
}

func freeWeight(weight *quantizedWeight) error {
	for _, pointer := range []unsafe.Pointer{weight.scaleScales, weight.scaleCodes, weight.scales, weight.codes} {
		if pointer != nil {
			if err := native("free quantized weight", C.peft_cuda_free(pointer)); err != nil {
				return err
			}
		}
	}
	return nil
}

func (e *Engine) tensor(value backend.Tensor) (*Tensor, error) {
	if err := e.check(); err != nil {
		return nil, err
	}
	tensor, ok := value.(*Tensor)
	if !ok || tensor == nil || tensor.engine != e || tensor.closed {
		return nil, backend.ErrInvalidTensor
	}
	return tensor, nil
}

func (e *Engine) check() error {
	if e == nil || e.closed {
		return ErrClosed
	}
	return nil
}

func (e *Engine) finalize() { _ = e.Close() }
func (t *Tensor) finalize() { _ = t.Close() }

func native(operation string, code C.int) error {
	if code == 0 {
		return nil
	}
	return fmt.Errorf("cuda: %s: %s", operation, C.GoString(C.peft_cuda_last_error()))
}

func matrixShape(tensor *Tensor, transpose bool) (int, int) {
	if transpose {
		return tensor.cols, tensor.rows
	}
	return tensor.rows, tensor.cols
}

func boolInt(value bool) int {
	if value {
		return 1
	}
	return 0
}

func pointerKey(value any) (uintptr, bool) {
	v := reflect.ValueOf(value)
	if v.Kind() != reflect.Pointer || v.IsNil() {
		return 0, false
	}
	return v.Pointer(), true
}

var (
	_ backend.EagerEngine           = (*Engine)(nil)
	_ backend.Float32Codec          = (*Engine)(nil)
	_ backend.QuantizedLinearEngine = (*Engine)(nil)
)
