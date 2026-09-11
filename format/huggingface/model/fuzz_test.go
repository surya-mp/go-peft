package model

import (
	"os"
	"path/filepath"
	"testing"
)

func FuzzInspectManifest(f *testing.F) {
	f.Add([]byte(`{"weight_map":{"model.embed_tokens.weight":"model.safetensors"}}`))
	f.Add([]byte("invalid"))
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > 1<<20 {
			return
		}
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, ConfigFile), []byte(`{"model_type":"llama"}`), 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, IndexFile), data, 0o600); err != nil {
			t.Fatal(err)
		}
		_, _ = Inspect(dir)
	})
}
