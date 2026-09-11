// Package huggingface reads and writes Hugging Face PEFT LoRA adapters.
package huggingface

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/surya-mp/go-peft/backend"
	"github.com/surya-mp/go-peft/format/safetensors"
	"github.com/surya-mp/go-peft/lora"
	"github.com/surya-mp/go-peft/qlora"
)

const (
	ConfigFile = "adapter_config.json"
	ModelFile  = "adapter_model.safetensors"
)

var (
	ErrUnsupportedBias  = errors.New("huggingface: bias=all is not portable without the base model")
	ErrMergedAdapter    = errors.New("huggingface: merged adapters cannot be exported")
	ErrConfigMismatch   = errors.New("huggingface: adapter configuration mismatch")
	ErrUnexpectedTensor = errors.New("huggingface: unexpected tensor")
	ErrMissingTensor    = errors.New("huggingface: missing tensor")
)

// Metadata identifies the base model represented by the adapter.
type Metadata struct {
	BaseModelNameOrPath string
	Revision            string
	TaskType            string
}

type configFile struct {
	PeftType            string   `json:"peft_type"`
	BaseModelNameOrPath string   `json:"base_model_name_or_path"`
	Revision            string   `json:"revision"`
	TaskType            string   `json:"task_type"`
	Bias                string   `json:"bias"`
	FanInFanOut         bool     `json:"fan_in_fan_out"`
	InferenceMode       bool     `json:"inference_mode"`
	InitLoRAWeights     bool     `json:"init_lora_weights"`
	LoRAAlpha           float32  `json:"lora_alpha"`
	LoRADropout         float32  `json:"lora_dropout"`
	Rank                int      `json:"r"`
	TargetModules       []string `json:"target_modules"`
	ModulesToSave       any      `json:"modules_to_save"`
	UseRSLoRA           bool     `json:"use_rslora"`
}

type adapterLayer interface {
	Name() string
	A() backend.Tensor
	B() backend.Tensor
	Bias() backend.Tensor
}

// Save writes an unmerged adapter in PEFT's standard directory layout.
func Save(dir string, adapter *lora.Adapter, codec backend.Float32Codec, metadata Metadata) error {
	if adapter == nil || codec == nil {
		return errors.New("huggingface: adapter and codec are required")
	}
	config := adapter.Config()
	if config.Bias == lora.BiasAll {
		return ErrUnsupportedBias
	}
	for _, layer := range adapter.Layers() {
		if layer.Merged() {
			return ErrMergedAdapter
		}
	}
	return save(dir, config, loraLayers(adapter.Layers()), codec, metadata)
}

// SaveQLoRA writes QLoRA adapter tensors in PEFT's standard LoRA layout.
func SaveQLoRA(dir string, adapter *qlora.Adapter, codec backend.Float32Codec, metadata Metadata) error {
	if adapter == nil || codec == nil {
		return errors.New("huggingface: adapter and codec are required")
	}
	config := adapter.Config().LoRA
	if config.Bias == lora.BiasAll {
		return ErrUnsupportedBias
	}
	return save(dir, config, qloraLayers(adapter.Layers()), codec, metadata)
}

func save(dir string, config lora.Config, layers []adapterLayer, codec backend.Float32Codec, metadata Metadata) error {
	tensors := make(map[string]safetensors.Tensor, len(layers)*2)
	for _, layer := range layers {
		if err := addMatrix(tensors, key(layer.Name(), "lora_A.weight"), layer.A(), codec); err != nil {
			return err
		}
		if err := addMatrix(tensors, key(layer.Name(), "lora_B.weight"), layer.B(), codec); err != nil {
			return err
		}
		if config.Bias == lora.BiasLoRAOnly && layer.Bias() != nil {
			if err := addVector(tensors, key(layer.Name(), "bias"), layer.Bias(), codec); err != nil {
				return err
			}
		}
	}
	return SaveTensors(dir, config, tensors, metadata)
}

// SaveTensors writes pre-encoded LoRA tensors in PEFT's standard layout.
func SaveTensors(dir string, config lora.Config, tensors map[string]safetensors.Tensor, metadata Metadata) error {
	if err := config.Validate(); err != nil {
		return err
	}
	if config.Bias == lora.BiasAll {
		return ErrUnsupportedBias
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	if err := writeTensors(filepath.Join(dir, ModelFile), tensors); err != nil {
		return err
	}
	fileConfig := configFile{
		PeftType: "LORA", BaseModelNameOrPath: metadata.BaseModelNameOrPath, Revision: metadata.Revision, TaskType: metadata.TaskType,
		Bias: biasString(config.Bias), InferenceMode: true, InitLoRAWeights: true, LoRAAlpha: config.Alpha,
		LoRADropout: config.Dropout, Rank: config.Rank, TargetModules: config.TargetModules, ModulesToSave: nil,
	}
	data, err := json.MarshalIndent(fileConfig, "", "  ")
	if err != nil {
		return err
	}
	return writeFile(filepath.Join(dir, ConfigFile), data)
}

// ReadConfig reads the LoRA configuration and base-model metadata.
func ReadConfig(dir string) (lora.Config, Metadata, error) {
	data, err := os.ReadFile(filepath.Join(dir, ConfigFile))
	if err != nil {
		return lora.Config{}, Metadata{}, err
	}
	var fileConfig configFile
	if err := json.Unmarshal(data, &fileConfig); err != nil {
		return lora.Config{}, Metadata{}, err
	}
	if fileConfig.PeftType != "LORA" || fileConfig.FanInFanOut || fileConfig.UseRSLoRA {
		return lora.Config{}, Metadata{}, ErrConfigMismatch
	}
	bias, err := parseBias(fileConfig.Bias)
	if err != nil {
		return lora.Config{}, Metadata{}, err
	}
	config := lora.Config{
		Rank: fileConfig.Rank, Alpha: fileConfig.LoRAAlpha, Dropout: fileConfig.LoRADropout,
		TargetModules: append([]string(nil), fileConfig.TargetModules...), Bias: bias,
	}
	if err := config.Validate(); err != nil {
		return lora.Config{}, Metadata{}, err
	}
	return config, Metadata{
		BaseModelNameOrPath: fileConfig.BaseModelNameOrPath, Revision: fileConfig.Revision, TaskType: fileConfig.TaskType,
	}, nil
}

// LoadTensors reads a PEFT LoRA configuration and its SafeTensors payload.
func LoadTensors(dir string) (lora.Config, map[string]safetensors.Tensor, Metadata, error) {
	config, metadata, err := ReadConfig(dir)
	if err != nil {
		return lora.Config{}, nil, Metadata{}, err
	}
	tensors, _, err := safetensors.ReadFile(filepath.Join(dir, ModelFile))
	if err != nil {
		return lora.Config{}, nil, Metadata{}, err
	}
	return config, tensors, metadata, nil
}

// LoadInto validates and copies a PEFT adapter into an already injected adapter.
func LoadInto(dir string, adapter *lora.Adapter, engine backend.Engine) (Metadata, error) {
	if adapter == nil || engine == nil {
		return Metadata{}, errors.New("huggingface: adapter and engine are required")
	}
	return loadInto(dir, adapter.Config(), loraLayers(adapter.Layers()), engine)
}

// LoadQLoRAInto validates and copies PEFT LoRA tensors into a QLoRA adapter.
func LoadQLoRAInto(dir string, adapter *qlora.Adapter, engine backend.Engine) (Metadata, error) {
	if adapter == nil || engine == nil {
		return Metadata{}, errors.New("huggingface: adapter and engine are required")
	}
	return loadInto(dir, adapter.Config().LoRA, qloraLayers(adapter.Layers()), engine)
}

func loadInto(dir string, expectedConfig lora.Config, layers []adapterLayer, engine backend.Engine) (Metadata, error) {
	codec, ok := engine.(backend.Float32Codec)
	if !ok {
		return Metadata{}, errors.New("huggingface: engine does not support float32 transfer")
	}
	config, tensors, metadata, err := LoadTensors(dir)
	if err != nil {
		return Metadata{}, err
	}
	if !sameConfig(config, expectedConfig) {
		return Metadata{}, ErrConfigMismatch
	}

	type assignment struct{ dst, src backend.Tensor }
	assignments := make([]assignment, 0, len(layers)*2)
	expected := make(map[string]struct{}, len(layers)*2)
	for _, layer := range layers {
		for _, item := range []struct {
			name string
			dst  backend.Tensor
		}{
			{key(layer.Name(), "lora_A.weight"), layer.A()},
			{key(layer.Name(), "lora_B.weight"), layer.B()},
		} {
			rows, cols, err := engine.Shape(item.dst)
			if err != nil {
				return Metadata{}, err
			}
			expected[item.name] = struct{}{}
			tensor, exists := tensors[item.name]
			if !exists {
				return Metadata{}, fmt.Errorf("%w: %s", ErrMissingTensor, item.name)
			}
			if !sameShape(tensor.Shape, []int{rows, cols}) {
				return Metadata{}, fmt.Errorf("%w: %s", ErrConfigMismatch, item.name)
			}
			src, err := codec.DecodeFloat32(rows, cols, tensor.Data)
			if err != nil {
				return Metadata{}, err
			}
			assignments = append(assignments, assignment{dst: item.dst, src: src})
		}
		if config.Bias == lora.BiasLoRAOnly && layer.Bias() != nil {
			name := key(layer.Name(), "bias")
			expected[name] = struct{}{}
			tensor, exists := tensors[name]
			if !exists || len(tensor.Shape) != 1 {
				return Metadata{}, fmt.Errorf("%w: %s", ErrMissingTensor, name)
			}
			_, cols, err := engine.Shape(layer.Bias())
			if err != nil || tensor.Shape[0] != cols {
				return Metadata{}, fmt.Errorf("%w: %s", ErrConfigMismatch, name)
			}
			src, err := codec.DecodeFloat32(1, cols, tensor.Data)
			if err != nil {
				return Metadata{}, err
			}
			assignments = append(assignments, assignment{dst: layer.Bias(), src: src})
		}
	}
	for name := range tensors {
		if _, exists := expected[name]; !exists {
			return Metadata{}, fmt.Errorf("%w: %s", ErrUnexpectedTensor, name)
		}
	}
	for _, assignment := range assignments {
		if err := engine.Copy(assignment.dst, assignment.src); err != nil {
			return Metadata{}, err
		}
	}
	return metadata, nil
}

func addMatrix(tensors map[string]safetensors.Tensor, name string, value backend.Tensor, codec backend.Float32Codec) error {
	data, rows, cols, err := codec.EncodeFloat32(value)
	if err != nil {
		return err
	}
	tensors[name] = safetensors.Tensor{Shape: []int{rows, cols}, Data: data}
	return nil
}

func addVector(tensors map[string]safetensors.Tensor, name string, value backend.Tensor, codec backend.Float32Codec) error {
	data, rows, cols, err := codec.EncodeFloat32(value)
	if err != nil || rows != 1 {
		return ErrConfigMismatch
	}
	tensors[name] = safetensors.Tensor{Shape: []int{cols}, Data: data}
	return nil
}

func writeTensors(path string, tensors map[string]safetensors.Tensor) error {
	file, err := os.CreateTemp(filepath.Dir(path), ".adapter-*.safetensors")
	if err != nil {
		return err
	}
	temporary := file.Name()
	defer func() { _ = os.Remove(temporary) }()
	if err := safetensors.Write(file, tensors, map[string]string{"format": "pt"}); err != nil {
		_ = file.Close()
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	return os.Rename(temporary, path)
}

func writeFile(path string, data []byte) error {
	file, err := os.CreateTemp(filepath.Dir(path), ".adapter-*.json")
	if err != nil {
		return err
	}
	temporary := file.Name()
	defer func() { _ = os.Remove(temporary) }()
	if _, err := file.Write(data); err != nil {
		_ = file.Close()
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	return os.Rename(temporary, path)
}

func key(module, suffix string) string { return "base_model.model." + module + "." + suffix }

func biasString(bias lora.BiasMode) string {
	if bias == lora.BiasLoRAOnly {
		return "lora_only"
	}
	if bias == lora.BiasAll {
		return "all"
	}
	return "none"
}

func parseBias(value string) (lora.BiasMode, error) {
	switch value {
	case "none":
		return lora.BiasNone, nil
	case "lora_only":
		return lora.BiasLoRAOnly, nil
	case "all":
		return lora.BiasAll, nil
	default:
		return 0, ErrConfigMismatch
	}
}

func sameConfig(left, right lora.Config) bool {
	if left.Rank != right.Rank || left.Alpha != right.Alpha || left.Dropout != right.Dropout || left.Bias != right.Bias {
		return false
	}
	leftTargets := append([]string(nil), left.TargetModules...)
	rightTargets := append([]string(nil), right.TargetModules...)
	sort.Strings(leftTargets)
	sort.Strings(rightTargets)
	if len(leftTargets) != len(rightTargets) {
		return false
	}
	for index := range leftTargets {
		if leftTargets[index] != rightTargets[index] {
			return false
		}
	}
	return true
}

func sameShape(left, right []int) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

func loraLayers(layers []*lora.Linear) []adapterLayer {
	result := make([]adapterLayer, len(layers))
	for index, layer := range layers {
		result[index] = layer
	}
	return result
}

func qloraLayers(layers []*qlora.Linear) []adapterLayer {
	result := make([]adapterLayer, len(layers))
	for index, layer := range layers {
		result[index] = layer
	}
	return result
}
