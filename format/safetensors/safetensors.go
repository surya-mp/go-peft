// Package safetensors reads and writes the SafeTensors float32 format.
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

var (
	ErrInvalidFile     = errors.New("safetensors: invalid file")
	ErrUnsupportedType = errors.New("safetensors: unsupported dtype")
	ErrInvalidTensor   = errors.New("safetensors: invalid tensor")
)

// Tensor is a row-major float32 tensor.
type Tensor struct {
	Shape []int
	Data  []float32
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
	count := uint64(1)
	for _, dimension := range shape {
		if dimension < 0 || count > math.MaxUint64/uint64(max(1, dimension)) {
			return 0, ErrInvalidTensor
		}
		count *= uint64(dimension)
	}
	if count > math.MaxUint64/4 {
		return 0, ErrInvalidTensor
	}
	return count * 4, nil
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
