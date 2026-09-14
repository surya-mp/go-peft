// Package safetensors reads and writes SafeTensors files.
package safetensors

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
)

const maxHeaderSize = 100 << 20

// Errors returned while parsing or encoding SafeTensors data.
// Callers can test them with errors.Is.
var (
	ErrInvalidFile     = errors.New("safetensors: invalid file")
	ErrUnsupportedType = errors.New("safetensors: unsupported dtype")
	ErrInvalidTensor   = errors.New("safetensors: invalid tensor")
)

// Tensor is a row-major float32 tensor.
type Tensor struct {
	// Shape contains row-major dimensions.
	Shape []int
	// Data contains row-major F32 values.
	Data []float32
}

// DecodedTensor is one decoded SafeTensors value.
// Visit keeps only this tensor's data in memory at once.
type DecodedTensor struct {
	// Name is the tensor key in the SafeTensors file.
	Name string
	// DType is the source SafeTensors dtype: F32, F16, or BF16.
	DType string
	// Shape contains row-major dimensions.
	Shape []int
	// Data contains decoded F32 values.
	Data []float32
}

// Visitor receives a decoded tensor while streaming a file.
type Visitor func(DecodedTensor) error

// VisitOptions controls SafeTensors streaming diagnostics.
type VisitOptions struct {
	// Progress receives human-readable status messages while the file is read.
	Progress func(string)
	// ProgressEvery reports selected tensor decode progress every N tensors.
	// When zero, only file/header milestones are reported.
	ProgressEvery int
}

type headerTensor struct {
	DType       string    `json:"dtype"`
	Shape       []int     `json:"shape"`
	DataOffsets [2]uint64 `json:"data_offsets"`
}

// Write serializes tensors in deterministic name order.
func Write(w io.Writer, tensors map[string]Tensor, metadata map[string]string) error {
	if len(tensors) == 0 {
		return fmt.Errorf("%w: no tensors", ErrInvalidTensor)
	}
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	sort.Strings(names)

	header := make(map[string]any, len(tensors)+1)
	if len(metadata) > 0 {
		header["__metadata__"] = metadata
	}
	var offset uint64
	for _, name := range names {
		tensor := tensors[name]
		bytes, err := tensorBytes(tensor)
		if err != nil {
			return fmt.Errorf("%w %q: %w", ErrInvalidTensor, name, err)
		}
		header[name] = headerTensor{DType: "F32", Shape: tensor.Shape, DataOffsets: [2]uint64{offset, offset + bytes}}
		offset += bytes
	}
	rawHeader, err := json.Marshal(header)
	if err != nil {
		return err
	}
	if len(rawHeader) == 0 || rawHeader[0] != '{' || len(rawHeader) > maxHeaderSize {
		return ErrInvalidFile
	}

	buffered := bufio.NewWriter(w)
	var length [8]byte
	binary.LittleEndian.PutUint64(length[:], uint64(len(rawHeader)))
	if _, err := buffered.Write(length[:]); err != nil {
		return err
	}
	if _, err := buffered.Write(rawHeader); err != nil {
		return err
	}
	var value [4]byte
	for _, name := range names {
		for _, number := range tensors[name].Data {
			binary.LittleEndian.PutUint32(value[:], math.Float32bits(number))
			if _, err := buffered.Write(value[:]); err != nil {
				return err
			}
		}
	}
	return buffered.Flush()
}

// Read validates and decodes a SafeTensors float32 file.
func Read(r io.Reader) (map[string]Tensor, map[string]string, error) {
	var length [8]byte
	if _, err := io.ReadFull(r, length[:]); err != nil {
		return nil, nil, fmt.Errorf("%w: header length: %v", ErrInvalidFile, err)
	}
	headerSize := binary.LittleEndian.Uint64(length[:])
	if headerSize == 0 || headerSize > maxHeaderSize {
		return nil, nil, ErrInvalidFile
	}
	rawHeader := make([]byte, headerSize)
	if _, err := io.ReadFull(r, rawHeader); err != nil {
		return nil, nil, fmt.Errorf("%w: header: %v", ErrInvalidFile, err)
	}
	if rawHeader[0] != '{' {
		return nil, nil, ErrInvalidFile
	}
	header, metadata, err := parseHeader(rawHeader)
	if err != nil {
		return nil, nil, err
	}

	type entry struct {
		name   string
		tensor headerTensor
		bytes  uint64
	}
	entries := make([]entry, 0, len(header))
	for name, tensor := range header {
		if tensor.DType != "F32" {
			return nil, nil, fmt.Errorf("%w: %s", ErrUnsupportedType, tensor.DType)
		}
		bytes, err := shapeBytes(tensor.Shape)
		if err != nil || tensor.DataOffsets[1] < tensor.DataOffsets[0] || tensor.DataOffsets[1]-tensor.DataOffsets[0] != bytes {
			return nil, nil, fmt.Errorf("%w: %s", ErrInvalidTensor, name)
		}
		entries = append(entries, entry{name: name, tensor: tensor, bytes: bytes})
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].tensor.DataOffsets[0] < entries[j].tensor.DataOffsets[0] })
	var expected uint64
	for _, entry := range entries {
		if entry.tensor.DataOffsets[0] != expected {
			return nil, nil, ErrInvalidFile
		}
		expected += entry.bytes
	}

	result := make(map[string]Tensor, len(entries))
	var encoded [4]byte
	for _, entry := range entries {
		count := entry.bytes / 4
		data := make([]float32, count)
		for index := range data {
			if _, err := io.ReadFull(r, encoded[:]); err != nil {
				return nil, nil, fmt.Errorf("%w: tensor data: %v", ErrInvalidFile, err)
			}
			data[index] = math.Float32frombits(binary.LittleEndian.Uint32(encoded[:]))
		}
		result[entry.name] = Tensor{Shape: append([]int(nil), entry.tensor.Shape...), Data: data}
	}
	if err := requireEOF(r); err != nil {
		return nil, nil, err
	}
	return result, metadata, nil
}

// WriteFile writes a SafeTensors file.
func WriteFile(path string, tensors map[string]Tensor, metadata map[string]string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := Write(file, tensors, metadata); err != nil {
		_ = file.Close()
		return err
	}
	return file.Close()
}

// ReadFile reads a SafeTensors file.
func ReadFile(path string) (map[string]Tensor, map[string]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer func() { _ = file.Close() }()
	return Read(file)
}

// Visit streams F32, F16, and BF16 tensors in file-offset order.
// keep may be nil; a false result skips decoding that tensor.
func Visit(r io.Reader, keep func(name string) bool, visit Visitor) (map[string]string, error) {
	return VisitWithOptions(r, keep, visit, VisitOptions{})
}

// VisitWithOptions streams F32, F16, and BF16 tensors in file-offset order with
// optional progress reporting.
func VisitWithOptions(r io.Reader, keep func(name string) bool, visit Visitor, options VisitOptions) (map[string]string, error) {
	if visit == nil {
		return nil, fmt.Errorf("%w: visitor is required", ErrInvalidTensor)
	}
	var length [8]byte
	progressf(options.Progress, "safetensors: reading header length")
	if _, err := io.ReadFull(r, length[:]); err != nil {
		return nil, fmt.Errorf("%w: header length: %v", ErrInvalidFile, err)
	}
	headerSize := binary.LittleEndian.Uint64(length[:])
	if headerSize == 0 || headerSize > maxHeaderSize {
		return nil, ErrInvalidFile
	}
	progressf(options.Progress, "safetensors: reading header bytes=%d", headerSize)
	rawHeader := make([]byte, headerSize)
	if _, err := io.ReadFull(r, rawHeader); err != nil {
		return nil, fmt.Errorf("%w: header: %v", ErrInvalidFile, err)
	}
	if rawHeader[0] != '{' {
		return nil, ErrInvalidFile
	}
	header, metadata, err := parseHeader(rawHeader)
	if err != nil {
		return nil, err
	}
	progressf(options.Progress, "safetensors: parsed header tensors=%d", len(header))

	type entry struct {
		name     string
		tensor   headerTensor
		bytes    uint64
		selected bool
	}
	entries := make([]entry, 0, len(header))
	for name, tensor := range header {
		width, ok := dtypeWidth(tensor.DType)
		if !ok {
			return nil, fmt.Errorf("%w: %s", ErrUnsupportedType, tensor.DType)
		}
		count, err := shapeCount(tensor.Shape)
		if err != nil || count > math.MaxUint64/width {
			return nil, fmt.Errorf("%w: %s", ErrInvalidTensor, name)
		}
		bytes := count * width
		if tensor.DataOffsets[1] < tensor.DataOffsets[0] || tensor.DataOffsets[1]-tensor.DataOffsets[0] != bytes {
			return nil, fmt.Errorf("%w: %s", ErrInvalidTensor, name)
		}
		entries = append(entries, entry{name: name, tensor: tensor, bytes: bytes})
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].tensor.DataOffsets[0] < entries[j].tensor.DataOffsets[0] })
	progressf(options.Progress, "safetensors: streaming tensors=%d", len(entries))
	var expected uint64
	for _, entry := range entries {
		if entry.tensor.DataOffsets[0] != expected || entry.bytes > math.MaxInt64 {
			return nil, ErrInvalidFile
		}
		expected += entry.bytes
	}
	selectedTotal := 0
	for index := range entries {
		entries[index].selected = keep == nil || keep(entries[index].name)
		if entries[index].selected {
			selectedTotal++
		}
	}

	selectedIndex := 0
	for _, entry := range entries {
		if !entry.selected {
			if _, err := io.CopyN(io.Discard, r, int64(entry.bytes)); err != nil {
				return nil, fmt.Errorf("%w: tensor data: %v", ErrInvalidFile, err)
			}
			continue
		}
		selectedIndex++
		if shouldReportTensor(selectedIndex, selectedTotal, options.ProgressEvery) {
			progressf(options.Progress, "safetensors: decoding tensor %d/%d %s shape=%v dtype=%s bytes=%d", selectedIndex, selectedTotal, entry.name, entry.tensor.Shape, entry.tensor.DType, entry.bytes)
		}
		data, err := decode(r, entry.tensor.DType, entry.tensor.Shape)
		if err != nil {
			return nil, fmt.Errorf("%w: %s: %v", ErrInvalidFile, entry.name, err)
		}
		if err := visit(DecodedTensor{Name: entry.name, DType: entry.tensor.DType, Shape: append([]int(nil), entry.tensor.Shape...), Data: data}); err != nil {
			return nil, err
		}
	}
	if err := requireEOF(r); err != nil {
		return nil, err
	}
	progressf(options.Progress, "safetensors: finished file")
	return metadata, nil
}

// VisitFile opens and streams a SafeTensors file.
func VisitFile(path string, keep func(name string) bool, visit Visitor) (map[string]string, error) {
	return VisitFileWithOptions(path, keep, visit, VisitOptions{})
}

// VisitFileWithOptions opens and streams a SafeTensors file with optional
// progress reporting.
func VisitFileWithOptions(path string, keep func(name string) bool, visit Visitor, options VisitOptions) (map[string]string, error) {
	progressf(options.Progress, "safetensors: opening file %s", path)
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = file.Close() }()
	metadata, err := VisitWithOptions(file, keep, visit, options)
	if err != nil {
		return nil, err
	}
	progressf(options.Progress, "safetensors: closed file %s", path)
	return metadata, nil
}

func progressf(progress func(string), format string, args ...any) {
	if progress != nil {
		progress(fmt.Sprintf(format, args...))
	}
}

func shouldReportTensor(index, total, every int) bool {
	return every > 0 && (index == 1 || index == total || index%every == 0)
}

func parseHeader(raw []byte) (map[string]headerTensor, map[string]string, error) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	token, err := decoder.Token()
	if err != nil || token != json.Delim('{') {
		return nil, nil, ErrInvalidFile
	}
	header := make(map[string]headerTensor)
	metadata := map[string]string(nil)
	for decoder.More() {
		token, err := decoder.Token()
		name, ok := token.(string)
		if err != nil || !ok || name == "" {
			return nil, nil, ErrInvalidFile
		}
		if _, exists := header[name]; exists || name == "__metadata__" && metadata != nil {
			return nil, nil, ErrInvalidFile
		}
		if name == "__metadata__" {
			if err := decoder.Decode(&metadata); err != nil || metadata == nil {
				return nil, nil, ErrInvalidFile
			}
			continue
		}
		var tensor headerTensor
		if err := decoder.Decode(&tensor); err != nil {
			return nil, nil, ErrInvalidFile
		}
		header[name] = tensor
	}
	if token, err := decoder.Token(); err != nil || token != json.Delim('}') || decoder.More() {
		return nil, nil, ErrInvalidFile
	}
	if _, err := decoder.Token(); err != io.EOF {
		return nil, nil, ErrInvalidFile
	}
	if len(header) == 0 {
		return nil, nil, ErrInvalidFile
	}
	return header, metadata, nil
}

func tensorBytes(tensor Tensor) (uint64, error) {
	bytes, err := shapeBytes(tensor.Shape)
	if err != nil {
		return 0, err
	}
	if uint64(len(tensor.Data)) != bytes/4 {
		return 0, ErrInvalidTensor
	}
	return bytes, nil
}

func shapeBytes(shape []int) (uint64, error) {
	count, err := shapeCount(shape)
	if err != nil || count > math.MaxUint64/4 {
		return 0, ErrInvalidTensor
	}
	return count * 4, nil
}

func shapeCount(shape []int) (uint64, error) {
	count := uint64(1)
	for _, dimension := range shape {
		if dimension < 0 || count > math.MaxUint64/uint64(max(1, dimension)) {
			return 0, ErrInvalidTensor
		}
		count *= uint64(dimension)
	}
	return count, nil
}

func dtypeWidth(dtype string) (uint64, bool) {
	switch dtype {
	case "F32":
		return 4, true
	case "F16", "BF16":
		return 2, true
	default:
		return 0, false
	}
}

func decode(r io.Reader, dtype string, shape []int) ([]float32, error) {
	count, err := shapeCount(shape)
	if err != nil || count > uint64(^uint(0)>>1) {
		return nil, ErrInvalidTensor
	}
	data := make([]float32, int(count))
	switch dtype {
	case "F32":
		var encoded [4]byte
		for index := range data {
			if _, err := io.ReadFull(r, encoded[:]); err != nil {
				return nil, err
			}
			data[index] = math.Float32frombits(binary.LittleEndian.Uint32(encoded[:]))
		}
	case "F16", "BF16":
		var encoded [2]byte
		for index := range data {
			if _, err := io.ReadFull(r, encoded[:]); err != nil {
				return nil, err
			}
			bits := binary.LittleEndian.Uint16(encoded[:])
			if dtype == "F16" {
				data[index] = float16(bits)
			} else {
				data[index] = math.Float32frombits(uint32(bits) << 16)
			}
		}
	default:
		return nil, ErrUnsupportedType
	}
	return data, nil
}

func float16(bits uint16) float32 {
	sign := uint32(bits&0x8000) << 16
	exponent := int32(bits>>10) & 0x1f
	fraction := uint32(bits & 0x03ff)
	switch exponent {
	case 0:
		if fraction == 0 {
			return math.Float32frombits(sign)
		}
		exponent = -14
		for fraction&0x0400 == 0 {
			fraction <<= 1
			exponent--
		}
		fraction &= 0x03ff
	case 31:
		return math.Float32frombits(sign | 0x7f800000 | fraction<<13)
	default:
		exponent -= 15
	}
	return math.Float32frombits(sign | uint32(exponent+127)<<23 | fraction<<13)
}

func requireEOF(r io.Reader) error {
	var byte [1]byte
	for {
		n, err := r.Read(byte[:])
		if n != 0 {
			return ErrInvalidFile
		}
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
	}
}

func max(left, right int) int {
	if left > right {
		return left
	}
	return right
}
