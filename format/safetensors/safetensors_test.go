package safetensors

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
)

func TestRoundTripIsDeterministic(t *testing.T) {
	tensors := map[string]Tensor{
		"z": {Shape: []int{1, 2}, Data: []float32{3, 4}},
		"a": {Shape: []int{2, 1}, Data: []float32{1, 2}},
	}
	metadata := map[string]string{"format": "pt"}
	var first, second bytes.Buffer
	if err := Write(&first, tensors, metadata); err != nil {
		t.Fatal(err)
	}
	if err := Write(&second, tensors, metadata); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first.Bytes(), second.Bytes()) {
		t.Fatal("output is not deterministic")
	}
	got, gotMetadata, err := Read(&first)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, tensors) || !reflect.DeepEqual(gotMetadata, metadata) {
		t.Fatalf("Read() = %#v, %#v", got, gotMetadata)
	}
}

func TestVisitDecodesFloat16AndBFloat16(t *testing.T) {
	header := map[string]headerTensor{
		"half":   {DType: "F16", Shape: []int{2}, DataOffsets: [2]uint64{0, 4}},
		"bfloat": {DType: "BF16", Shape: []int{2}, DataOffsets: [2]uint64{4, 8}},
	}
	raw, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	var file bytes.Buffer
	var length [8]byte
	binary.LittleEndian.PutUint64(length[:], uint64(len(raw)))
	file.Write(length[:])
	file.Write(raw)
	for _, value := range []uint16{0x3c00, 0xc000, 0x3f80, 0xc020} {
		var encoded [2]byte
		binary.LittleEndian.PutUint16(encoded[:], value)
		file.Write(encoded[:])
	}
	got := make(map[string][]float32)
	_, err = Visit(&file, nil, func(tensor DecodedTensor) error {
		got[tensor.Name] = tensor.Data
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got["half"], []float32{1, -2}) || !reflect.DeepEqual(got["bfloat"], []float32{1, -2.5}) {
		t.Fatalf("Visit() = %#v", got)
	}
}

func TestVisitSkipsUnwantedTensor(t *testing.T) {
	var file bytes.Buffer
	if err := Write(&file, map[string]Tensor{
		"keep": {Shape: []int{1}, Data: []float32{2}},
		"skip": {Shape: []int{1}, Data: []float32{1}},
	}, nil); err != nil {
		t.Fatal(err)
	}
	var names []string
	_, err := Visit(&file, func(name string) bool { return name == "keep" }, func(tensor DecodedTensor) error {
		names = append(names, tensor.Name)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(names, []string{"keep"}) {
		t.Fatalf("visited %v", names)
	}
}

func TestVisitWithOptionsReportsProgress(t *testing.T) {
	var file bytes.Buffer
	if err := Write(&file, map[string]Tensor{
		"keep": {Shape: []int{1}, Data: []float32{2}},
		"skip": {Shape: []int{1}, Data: []float32{1}},
	}, nil); err != nil {
		t.Fatal(err)
	}
	var messages []string
	_, err := VisitWithOptions(&file, func(name string) bool { return name == "keep" }, func(tensor DecodedTensor) error {
		return nil
	}, VisitOptions{Progress: func(message string) { messages = append(messages, message) }, ProgressEvery: 1})
	if err != nil {
		t.Fatal(err)
	}
	joined := strings.Join(messages, "\n")
	for _, want := range []string{"safetensors: reading header length", "safetensors: parsed header", "safetensors: decoding tensor 1/1 keep", "safetensors: finished file"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("progress %q missing from %v", want, messages)
		}
	}
}

func TestVisitPropagatesVisitorAndRejectsUnsupportedDType(t *testing.T) {
	var file bytes.Buffer
	if err := Write(&file, map[string]Tensor{"x": {Shape: []int{1}, Data: []float32{1}}}, nil); err != nil {
		t.Fatal(err)
	}
	want := errors.New("stop")
	_, err := Visit(&file, nil, func(DecodedTensor) error { return want })
	if !errors.Is(err, want) {
		t.Fatalf("Visit() = %v", err)
	}

	header, err := json.Marshal(map[string]headerTensor{
		"x": {DType: "F64", Shape: []int{1}, DataOffsets: [2]uint64{0, 8}},
	})
	if err != nil {
		t.Fatal(err)
	}
	file.Reset()
	var length [8]byte
	binary.LittleEndian.PutUint64(length[:], uint64(len(header)))
	file.Write(length[:])
	file.Write(header)
	file.Write(make([]byte, 8))
	_, err = Visit(&file, nil, func(DecodedTensor) error { return nil })
	if !errors.Is(err, ErrUnsupportedType) {
		t.Fatalf("Visit() = %v", err)
	}
}

func TestReadRejectsTrailingData(t *testing.T) {
	var data bytes.Buffer
	if err := Write(&data, map[string]Tensor{"x": {Shape: []int{1}, Data: []float32{1}}}, nil); err != nil {
		t.Fatal(err)
	}
	data.WriteByte(0)
	_, _, err := Read(&data)
	if !errors.Is(err, ErrInvalidFile) {
		t.Fatalf("Read() = %v", err)
	}
}
