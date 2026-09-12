// Package gguf inspects GGUF v3 model files without loading tensor data.
package gguf

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
)

// Version is the supported GGUF container version.
const (
	Version = 3

	defaultAlignment = 32
	maxEntries       = 1_000_000
	maxStringSize    = 16 << 20
)

// Errors returned while inspecting GGUF files.
// Callers can test them with errors.Is.
var (
	ErrInvalidFile         = errors.New("gguf: invalid file")
	ErrUnsupported         = errors.New("gguf: unsupported version")
	ErrMissingArchitecture = errors.New("gguf: missing general.architecture")
)

// Type identifies a GGML tensor encoding. It does not decode quantized data.
type Type uint32

const (
	F32  Type = 0
	F16  Type = 1
	Q4_0 Type = 2
	Q4_1 Type = 3
	Q5_0 Type = 6
	Q5_1 Type = 7
	Q8_0 Type = 8
	Q8_1 Type = 9
	Q2K  Type = 10
	Q3K  Type = 11
	Q4K  Type = 12
	Q5K  Type = 13
	Q6K  Type = 14
	Q8K  Type = 15
	BF16 Type = 30
)

// String returns the GGML encoding name when it is known.
func (t Type) String() string {
	if name, ok := typeNames[t]; ok {
		return name
	}
	return fmt.Sprintf("UNKNOWN_%d", t)
}

var typeNames = map[Type]string{
	F32: "F32", F16: "F16", Q4_0: "Q4_0", Q4_1: "Q4_1", Q5_0: "Q5_0", Q5_1: "Q5_1", Q8_0: "Q8_0", Q8_1: "Q8_1",
	Q2K: "Q2_K", Q3K: "Q3_K", Q4K: "Q4_K", Q5K: "Q5_K", Q6K: "Q6_K", Q8K: "Q8_K", BF16: "BF16",
}

// Tensor identifies a GGUF tensor. Offset is absolute from the start of file.
type Tensor struct {
	// Name is the GGUF tensor name.
	Name string `json:"name"`
	// Shape contains GGML dimensions in file order.
	Shape []uint64 `json:"shape"`
	// Type identifies the GGML tensor encoding.
	Type Type `json:"type"`
	// Encoding is Type's human-readable name.
	Encoding string `json:"encoding"`
	// Offset is the absolute tensor-data offset from the start of the file.
	Offset uint64 `json:"offset"`
}

// Info is the read-only GGUF model index.
type Info struct {
	// Version is the GGUF format version.
	Version uint32 `json:"version"`
	// Architecture is the general.architecture metadata value.
	Architecture string `json:"architecture"`
	// Name is the optional general.name metadata value.
	Name string `json:"name,omitempty"`
	// Alignment is the tensor-data alignment in bytes.
	Alignment uint32 `json:"alignment"`
	// QuantizationVersion is the optional general.quantization_version value.
	QuantizationVersion uint32 `json:"quantization_version,omitempty"`
	// FileType is the optional general.file_type value.
	FileType uint32 `json:"file_type,omitempty"`
	// DataOffset is the start of the aligned tensor-data section.
	DataOffset uint64 `json:"data_offset"`
	// Tensors is the tensor index in ascending data-offset order.
	Tensors []Tensor `json:"tensors"`
}

// Read parses a GGUF v3 file's metadata and tensor index. Tensor payloads stay
// on disk for a runtime that explicitly supports their GGML encoding.
func Read(path string) (Info, error) {
	file, err := os.Open(path)
	if err != nil {
		return Info{}, err
	}
	defer func() { _ = file.Close() }()
	stat, err := file.Stat()
	if err != nil {
		return Info{}, err
	}
	if stat.Size() < 0 {
		return Info{}, ErrInvalidFile
	}
	reader := &binaryReader{r: file, limit: uint64(stat.Size())}
	var magic [4]byte
	if err := reader.read(magic[:]); err != nil || string(magic[:]) != "GGUF" {
		return Info{}, ErrInvalidFile
	}
	version, err := reader.u32()
	if err != nil {
		return Info{}, ErrInvalidFile
	}
	if version != Version {
		return Info{}, fmt.Errorf("%w: %d", ErrUnsupported, version)
	}
	tensorCount, err := reader.u64()
	if err != nil || tensorCount > maxEntries {
		return Info{}, ErrInvalidFile
	}
	metadataCount, err := reader.u64()
	if err != nil || metadataCount > maxEntries {
		return Info{}, ErrInvalidFile
	}
	info := Info{Version: version, Alignment: defaultAlignment}
	for range metadataCount {
		key, err := reader.string()
		if err != nil {
			return Info{}, ErrInvalidFile
		}
		kind, err := reader.u32()
		if err != nil {
			return Info{}, ErrInvalidFile
		}
		if err := readMetadata(reader, key, kind, &info); err != nil {
			return Info{}, err
		}
	}
	if info.Architecture == "" {
		return Info{}, ErrMissingArchitecture
	}
	if info.Alignment == 0 || info.Alignment%8 != 0 || info.Alignment > math.MaxInt32 {
		return Info{}, ErrInvalidFile
	}
	info.Tensors = make([]Tensor, 0, tensorCount)
	names := make(map[string]struct{}, tensorCount)
	for range tensorCount {
		name, err := reader.string()
		if err != nil || name == "" {
			return Info{}, ErrInvalidFile
		}
		if _, duplicate := names[name]; duplicate {
			return Info{}, ErrInvalidFile
		}
		names[name] = struct{}{}
		dimensions, err := reader.u32()
		if err != nil || dimensions > 4 {
			return Info{}, ErrInvalidFile
		}
		shape := make([]uint64, dimensions)
		for index := range shape {
			if shape[index], err = reader.u64(); err != nil || shape[index] == 0 {
				return Info{}, ErrInvalidFile
			}
		}
		kind, err := reader.u32()
		if err != nil {
			return Info{}, ErrInvalidFile
		}
		offset, err := reader.u64()
		if err != nil || offset%uint64(info.Alignment) != 0 {
			return Info{}, ErrInvalidFile
		}
		typeID := Type(kind)
		info.Tensors = append(info.Tensors, Tensor{Name: name, Shape: shape, Type: typeID, Encoding: typeID.String(), Offset: offset})
	}
	dataOffset := align(reader.offset, uint64(info.Alignment))
	if dataOffset < reader.offset || dataOffset > uint64(stat.Size()) {
		return Info{}, ErrInvalidFile
	}
	for index := range info.Tensors {
		tensor := &info.Tensors[index]
		if tensor.Offset > uint64(stat.Size())-dataOffset {
			return Info{}, ErrInvalidFile
		}
		tensor.Offset += dataOffset
	}
	info.DataOffset = dataOffset
	sort.Slice(info.Tensors, func(i, j int) bool { return info.Tensors[i].Offset < info.Tensors[j].Offset })
	return info, nil
}

const (
	uint8Type = iota
	int8Type
	uint16Type
	int16Type
	uint32Type
	int32Type
	float32Type
	boolType
	stringType
	arrayType
	uint64Type
	int64Type
	float64Type
)

func readMetadata(reader *binaryReader, key string, kind uint32, info *Info) error {
	switch key {
	case "general.architecture", "general.name":
		if kind != stringType {
			return ErrInvalidFile
		}
		value, err := reader.string()
		if err != nil {
			return ErrInvalidFile
		}
		if key == "general.architecture" {
			info.Architecture = value
		} else {
			info.Name = value
		}
		return nil
	case "general.alignment", "general.quantization_version", "general.file_type":
		if kind != uint32Type {
			return ErrInvalidFile
		}
		value, err := reader.u32()
		if err != nil {
			return ErrInvalidFile
		}
		switch key {
		case "general.alignment":
			info.Alignment = value
		case "general.quantization_version":
			info.QuantizationVersion = value
		case "general.file_type":
			info.FileType = value
		}
		return nil
	default:
		return skipValue(reader, kind)
	}
}

func skipValue(reader *binaryReader, kind uint32) error {
	switch kind {
	case uint8Type, int8Type, boolType:
		_, err := reader.u8()
		return err
	case uint16Type, int16Type:
		_, err := reader.u16()
		return err
	case uint32Type, int32Type, float32Type:
		_, err := reader.u32()
		return err
	case uint64Type, int64Type, float64Type:
		_, err := reader.u64()
		return err
	case stringType:
		_, err := reader.string()
		return err
	case arrayType:
		elementKind, err := reader.u32()
		if err != nil {
			return err
		}
		count, err := reader.u64()
		if err != nil || count > maxEntries {
			return ErrInvalidFile
		}
		for range count {
			if err := skipValue(reader, elementKind); err != nil {
				return err
			}
		}
		return nil
	default:
		return ErrInvalidFile
	}
}

type binaryReader struct {
	r      io.Reader
	offset uint64
	limit  uint64
}

func (reader *binaryReader) read(data []byte) error {
	length := uint64(len(data))
	if reader.offset > math.MaxUint64-length || reader.limit != 0 && (reader.offset > reader.limit || length > reader.limit-reader.offset) {
		return io.ErrUnexpectedEOF
	}
	if _, err := io.ReadFull(reader.r, data); err != nil {
		return err
	}
	reader.offset += length
	return nil
}

func (reader *binaryReader) u8() (uint8, error) {
	var data [1]byte
	if err := reader.read(data[:]); err != nil {
		return 0, err
	}
	return data[0], nil
}

func (reader *binaryReader) u16() (uint16, error) {
	var data [2]byte
	if err := reader.read(data[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint16(data[:]), nil
}

func (reader *binaryReader) u32() (uint32, error) {
	var data [4]byte
	if err := reader.read(data[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(data[:]), nil
}

func (reader *binaryReader) u64() (uint64, error) {
	var data [8]byte
	if err := reader.read(data[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(data[:]), nil
}

func (reader *binaryReader) string() (string, error) {
	length, err := reader.u64()
	if err != nil || length > maxStringSize || reader.limit != 0 && (reader.offset > reader.limit || length > reader.limit-reader.offset) {
		return "", ErrInvalidFile
	}
	data := make([]byte, length)
	if err := reader.read(data); err != nil {
		return "", err
	}
	return string(data), nil
}

func align(offset, alignment uint64) uint64 {
	return offset + (alignment-offset%alignment)%alignment
}
