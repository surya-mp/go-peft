package model

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/surya-mp/go-peft/format/safetensors"
)

func TestLoadShardedModel(t *testing.T) {
	dir := t.TempDir()
	writeConfig(t, dir)
	writeShard(t, dir, "model-00001-of-00002.safetensors", map[string]safetensors.Tensor{
		"model.layers.0.self_attn.q_proj.weight": {Shape: []int{1}, Data: []float32{1}},
	})
	writeShard(t, dir, "model-00002-of-00002.safetensors", map[string]safetensors.Tensor{
		"model.layers.0.self_attn.v_proj.weight": {Shape: []int{1}, Data: []float32{2}},
	})
	index := Manifest{WeightMap: map[string]string{
		"model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
		"model.layers.0.self_attn.v_proj.weight": "model-00002-of-00002.safetensors",
	}}
	data, err := json.Marshal(index)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, IndexFile), data, 0o600); err != nil {
		t.Fatal(err)
	}

	var got []string
	info, err := Load(dir, Options{Filter: func(name string) bool { return name == "model.layers.0.self_attn.v_proj.weight" }, OnTensor: func(tensor Tensor) error {
		got = append(got, tensor.Name+":"+tensor.Shard)
		if !reflect.DeepEqual(tensor.Data, []float32{2}) {
			t.Fatalf("data = %v", tensor.Data)
		}
		return nil
	}})
	if err != nil {
		t.Fatal(err)
	}
	if info.Config.ModelType != "llama" || !info.Indexed || info.TensorCount != 2 {
		t.Fatalf("info = %#v", info)
	}
	if !reflect.DeepEqual(got, []string{"model.layers.0.self_attn.v_proj.weight:model-00002-of-00002.safetensors"}) {
		t.Fatalf("tensors = %v", got)
	}
}

func TestInspectRejectsEscapingShard(t *testing.T) {
	dir := t.TempDir()
	writeConfig(t, dir)
	data, err := json.Marshal(Manifest{WeightMap: map[string]string{"weight": "../outside.safetensors"}})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, IndexFile), data, 0o600); err != nil {
		t.Fatal(err)
	}
	_, err = Inspect(dir)
	if !errors.Is(err, ErrInvalidShard) {
		t.Fatalf("Inspect() = %v", err)
	}
}

func TestLoadSingleModel(t *testing.T) {
	dir := t.TempDir()
	writeConfig(t, dir)
	writeShard(t, dir, ModelFile, map[string]safetensors.Tensor{
		"model.embed_tokens.weight": {Shape: []int{1, 2}, Data: []float32{1, 2}},
	})
	info, err := Inspect(dir)
	if err != nil {
		t.Fatal(err)
	}
	if info.Indexed || info.TensorCount != -1 {
		t.Fatalf("inspect = %#v", info)
	}
	info, err = Load(dir, Options{OnTensor: func(tensor Tensor) error { return nil }})
	if err != nil {
		t.Fatal(err)
	}
	if info.TensorCount != 1 {
		t.Fatalf("loaded = %#v", info)
	}
}

func TestLoadRejectsMismatchedIndexAndCallbackError(t *testing.T) {
	if _, err := Load(t.TempDir(), Options{}); err == nil {
		t.Fatal("Load() accepted a nil callback")
	}
	dir := t.TempDir()
	writeConfig(t, dir)
	writeShard(t, dir, "model-00001-of-00001.safetensors", map[string]safetensors.Tensor{
		"actual": {Shape: []int{1}, Data: []float32{1}},
	})
	data, err := json.Marshal(Manifest{WeightMap: map[string]string{"expected": "model-00001-of-00001.safetensors"}})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, IndexFile), data, 0o600); err != nil {
		t.Fatal(err)
	}
	_, err = Load(dir, Options{OnTensor: func(Tensor) error { return nil }})
	if !errors.Is(err, ErrInvalidIndex) {
		t.Fatalf("Load() = %v", err)
	}

	single := t.TempDir()
	writeConfig(t, single)
	writeShard(t, single, ModelFile, map[string]safetensors.Tensor{"weight": {Shape: []int{1}, Data: []float32{1}}})
	want := errors.New("stop")
	_, err = Load(single, Options{OnTensor: func(Tensor) error { return want }})
	if !errors.Is(err, want) {
		t.Fatalf("Load() = %v", err)
	}
}

func writeConfig(t *testing.T, dir string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, ConfigFile), []byte(`{"model_type":"llama","hidden_size":16}`), 0o600); err != nil {
		t.Fatal(err)
	}
}

func writeShard(t *testing.T, dir, name string, tensors map[string]safetensors.Tensor) {
	t.Helper()
	if err := safetensors.WriteFile(filepath.Join(dir, name), tensors, nil); err != nil {
		t.Fatal(err)
	}
}
