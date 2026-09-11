package safetensors

import (
	"bytes"
	"encoding/binary"
	"testing"
)

func FuzzReadAndVisit(f *testing.F) {
	var seed bytes.Buffer
	if err := Write(&seed, map[string]Tensor{"weight": {Shape: []int{1}, Data: []float32{1}}}, nil); err != nil {
		f.Fatal(err)
	}
	f.Add(seed.Bytes())
	f.Add([]byte("invalid"))
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > 1<<20 || len(data) >= 8 && binary.LittleEndian.Uint64(data[:8]) > 1<<20 {
			return
		}
		_, _, _ = Read(bytes.NewReader(data))
		_, _ = Visit(bytes.NewReader(data), nil, func(DecodedTensor) error { return nil })
	})
}
