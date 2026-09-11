package huggingface

import (
	"encoding/base64"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/surya-mp/go-peft/backend"
	"github.com/surya-mp/go-peft/lora"
)

func TestPEFTGoldenAdapterFixture(t *testing.T) {
	dir := t.TempDir()
	copyGoldenFixture(t, dir)
	engine := backend.NewCPU()
	weight, err := backend.NewDense(2, 2, []float32{1, 0, 0, 1})
	if err != nil {
		t.Fatal(err)
	}
	config := lora.Config{Rank: 1, Alpha: 2, TargetModules: []string{"q_proj"}}
	layer, err := lora.NewLinear("layers.0.self_attn.q_proj", engine, weight, nil, config, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	adapter, err := lora.NewAdapter("default", config)
	if err != nil {
		t.Fatal(err)
	}
	if err := adapter.Add(layer); err != nil {
		t.Fatal(err)
	}
	metadata, err := LoadInto(dir, adapter, engine)
	if err != nil {
		t.Fatal(err)
	}
	if metadata != (Metadata{BaseModelNameOrPath: "fixture/llama", Revision: "main", TaskType: "CAUSAL_LM"}) {
		t.Fatalf("metadata = %#v", metadata)
	}
	if got := layer.A().(*backend.Dense).Values(); !reflect.DeepEqual(got, []float32{1, 2}) {
		t.Fatalf("A = %v", got)
	}
	if got := layer.B().(*backend.Dense).Values(); !reflect.DeepEqual(got, []float32{3, 4}) {
		t.Fatalf("B = %v", got)
	}
}

func copyGoldenFixture(t *testing.T, dir string) {
	t.Helper()
	base := filepath.Join("testdata", "peft_lora")
	config, err := os.ReadFile(filepath.Join(base, ConfigFile))
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := os.ReadFile(filepath.Join(base, "adapter_model.safetensors.base64"))
	if err != nil {
		t.Fatal(err)
	}
	model, err := base64.StdEncoding.DecodeString(strings.TrimSpace(string(encoded)))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, ConfigFile), config, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, ModelFile), model, 0o600); err != nil {
		t.Fatal(err)
	}
}
