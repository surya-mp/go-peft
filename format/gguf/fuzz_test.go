package gguf

import (
	"os"
	"path/filepath"
	"testing"
)

func FuzzRead(f *testing.F) {
	f.Add(fixture())
	f.Add([]byte("GGUF"))
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > 1<<20 {
			return
		}
		path := filepath.Join(t.TempDir(), "model.gguf")
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
		_, _ = Read(path)
	})
}
