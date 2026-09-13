package gguf

import (
	"bytes"
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestRead(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(path, fixture(), 0o600); err != nil {
		t.Fatal(err)
	}
	info, err := Read(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Version != Version || info.Architecture != "llama" || info.Name != "example" || info.Alignment != 32 || info.QuantizationVersion != 2 {
		t.Fatalf("info = %#v", info)
	}
	if len(info.Tensors) != 1 {
		t.Fatalf("tensors = %#v", info.Tensors)
	}
	tensor := info.Tensors[0]
	if tensor.Name != "blk.0.attn_q.weight" || tensor.Type != Q4K || tensor.Encoding != "Q4_K" || !reflect.DeepEqual(tensor.Shape, []uint64{8, 4}) || tensor.Offset != info.DataOffset {
		t.Fatalf("tensor = %#v, data offset = %d", tensor, info.DataOffset)
	}
}

func TestReadRejectsUnsupportedVersion(t *testing.T) {
	data := fixture()
	binary.LittleEndian.PutUint32(data[4:8], 2)
	path := filepath.Join(t.TempDir(), "old.gguf")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	_, err := Read(path)
	if !errors.Is(err, ErrUnsupported) {
		t.Fatalf("Read() = %v", err)
	}
}

func TestReadRejectsMissingArchitectureAndInvalidTensorOffset(t *testing.T) {
	t.Run("architecture", func(t *testing.T) {
		data := fixture()
		position := bytes.Index(data, []byte("general.architecture"))
		if position < 0 {
			t.Fatal("fixture has no architecture key")
		}
		copy(data[position:], "xeneral.architecture")
		path := filepath.Join(t.TempDir(), "missing.gguf")
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
		_, err := Read(path)
		if !errors.Is(err, ErrMissingArchitecture) {
			t.Fatalf("Read() = %v", err)
		}
	})
	t.Run("offset", func(t *testing.T) {
		data := fixture()
		position := bytes.Index(data, []byte("blk.0.attn_q.weight")) + len("blk.0.attn_q.weight") + 4 + 16 + 4
		binary.LittleEndian.PutUint64(data[position:position+8], ^uint64(0))
		path := filepath.Join(t.TempDir(), "offset.gguf")
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
		_, err := Read(path)
		if !errors.Is(err, ErrInvalidFile) {
			t.Fatalf("Read() = %v", err)
		}
	})
}

func TestSkipValueScalarsAndNestedArray(t *testing.T) {
	cases := []struct {
		kind uint32
		data []byte
	}{
		{uint8Type, []byte{1}}, {int8Type, []byte{1}}, {boolType, []byte{1}},
		{uint16Type, []byte{1, 0}}, {int16Type, []byte{1, 0}},
		{uint32Type, []byte{1, 0, 0, 0}}, {int32Type, []byte{1, 0, 0, 0}}, {float32Type, []byte{1, 0, 0, 0}},
		{uint64Type, []byte{1, 0, 0, 0, 0, 0, 0, 0}}, {int64Type, []byte{1, 0, 0, 0, 0, 0, 0, 0}}, {float64Type, []byte{1, 0, 0, 0, 0, 0, 0, 0}},
	}
	for _, test := range cases {
		reader := &binaryReader{r: bytes.NewReader(test.data)}
		if err := skipValue(reader, test.kind); err != nil || reader.offset != uint64(len(test.data)) {
			t.Fatalf("kind %d: offset %d, err %v", test.kind, reader.offset, err)
		}
	}
	var data bytes.Buffer
	writeU32(&data, arrayType)
	writeU64(&data, 1)
	writeU32(&data, uint8Type)
	writeU64(&data, 2)
	data.Write([]byte{1, 2})
	length := data.Len()
	reader := &binaryReader{r: &data}
	if err := skipValue(reader, arrayType); err != nil || reader.offset != uint64(length) {
		t.Fatalf("array: offset %d, err %v", reader.offset, err)
	}
}

func TestTypeString(t *testing.T) {
	if Q4K.String() != "Q4_K" || Type(99).String() != "UNKNOWN_99" {
		t.Fatalf("type names are incorrect")
	}
}

func fixture() []byte {
	var data bytes.Buffer
	data.WriteString("GGUF")
	writeU32(&data, Version)
	writeU64(&data, 1)
	writeU64(&data, 5)
	writeString(&data, "general.architecture")
	writeU32(&data, stringType)
	writeString(&data, "llama")
	writeString(&data, "general.name")
	writeU32(&data, stringType)
	writeString(&data, "example")
	writeString(&data, "general.alignment")
	writeU32(&data, uint32Type)
	writeU32(&data, 32)
	writeString(&data, "general.quantization_version")
	writeU32(&data, uint32Type)
	writeU32(&data, 2)
	writeString(&data, "tokenizer.ggml.tokens")
	writeU32(&data, arrayType)
	writeU32(&data, stringType)
	writeU64(&data, 2)
	writeString(&data, "a")
	writeString(&data, "b")
	writeString(&data, "blk.0.attn_q.weight")
	writeU32(&data, 2)
	writeU64(&data, 8)
	writeU64(&data, 4)
	writeU32(&data, uint32(Q4K))
	writeU64(&data, 0)
	for data.Len()%32 != 0 {
		data.WriteByte(0)
	}
	data.Write(make([]byte, 64))
	return data.Bytes()
}

func writeString(buffer *bytes.Buffer, value string) {
	writeU64(buffer, uint64(len(value)))
	buffer.WriteString(value)
}

func writeU32(buffer *bytes.Buffer, value uint32) {
	var data [4]byte
	binary.LittleEndian.PutUint32(data[:], value)
	buffer.Write(data[:])
}

func writeU64(buffer *bytes.Buffer, value uint64) {
	var data [8]byte
	binary.LittleEndian.PutUint64(data[:], value)
	buffer.Write(data[:])
}
