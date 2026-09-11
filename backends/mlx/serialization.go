//go:build darwin && arm64 && mlx

package mlx

import (
	"errors"
	"fmt"
	"sort"

	"github.com/surya-mp/go-peft/format/huggingface"
	"github.com/surya-mp/go-peft/format/safetensors"
	"github.com/surya-mp/go-peft/lora"
	"github.com/surya-mp/go-peft/qlora"
)

type arrayReplacement struct {
	dst **Array
	new *Array
}

// SaveAdapter writes named MLX LoRA layers in Hugging Face PEFT format.
func SaveAdapter(dir string, config lora.Config, layers map[string]*LoRALinear, metadata huggingface.Metadata) error {
	if err := config.Validate(); err != nil {
		return err
	}
	if config.Bias == lora.BiasAll {
		return huggingface.ErrUnsupportedBias
	}
	values := make(map[string]safetensors.Tensor, len(layers)*2)
	for name, layer := range layers {
		if name == "" || layer == nil {
			return errors.New("mlx: invalid adapter layer")
		}
		if err := addArray(values, adapterKey(name, "lora_A.weight"), layer.a); err != nil {
			return err
		}
		if err := addArray(values, adapterKey(name, "lora_B.weight"), layer.b); err != nil {
			return err
		}
		if config.Bias == lora.BiasLoRAOnly && layer.bias != nil {
			if err := addArray(values, adapterKey(name, "bias"), layer.bias); err != nil {
				return err
			}
		}
	}
	return huggingface.SaveTensors(dir, config, values, metadata)
}

// LoadAdapter validates all tensors before swapping any MLX adapter arrays.
func LoadAdapter(dir string, config lora.Config, layers map[string]*LoRALinear) (huggingface.Metadata, error) {
	expectedConfig, values, metadata, err := huggingface.LoadTensors(dir)
	if err != nil {
		return huggingface.Metadata{}, err
	}
	if !sameLoRAConfig(config, expectedConfig) {
		return huggingface.Metadata{}, huggingface.ErrConfigMismatch
	}
	replacements := make([]arrayReplacement, 0, len(layers)*2)
	expected := make(map[string]struct{}, len(layers)*2)
	for name, layer := range layers {
		if name == "" || layer == nil {
			return huggingface.Metadata{}, errors.New("mlx: invalid adapter layer")
		}
		for _, entry := range []struct {
			name string
			dst  **Array
		}{
			{adapterKey(name, "lora_A.weight"), &layer.a},
			{adapterKey(name, "lora_B.weight"), &layer.b},
		} {
			array, err := newArrayFromTensor(values, expected, entry.name, *entry.dst)
			if err != nil {
				closeReplacements(replacements)
				return huggingface.Metadata{}, err
			}
			replacements = append(replacements, arrayReplacement{entry.dst, array})
		}
		if config.Bias == lora.BiasLoRAOnly && layer.bias != nil {
			name := adapterKey(name, "bias")
			array, err := newArrayFromTensor(values, expected, name, layer.bias)
			if err != nil {
				closeReplacements(replacements)
				return huggingface.Metadata{}, err
			}
			replacements = append(replacements, arrayReplacement{&layer.bias, array})
		}
	}
	for name := range values {
		if _, ok := expected[name]; !ok {
			closeReplacements(replacements)
			return huggingface.Metadata{}, fmt.Errorf("%w: %s", huggingface.ErrUnexpectedTensor, name)
		}
	}
	for _, replacement := range replacements {
		old := *replacement.dst
		*replacement.dst = replacement.new
		_ = old.Close()
	}
	return metadata, nil
}

// SaveQLoRAAdapter writes MLX affine-int4 QLoRA matrices in PEFT format.
func SaveQLoRAAdapter(dir string, config qlora.Config, layers map[string]*QLoRALinear, metadata huggingface.Metadata) error {
	if err := validateQLoRAConfig(config); err != nil {
		return err
	}
	if config.LoRA.Bias == lora.BiasAll {
		return huggingface.ErrUnsupportedBias
	}
	values := make(map[string]safetensors.Tensor, len(layers)*2)
	for name, layer := range layers {
		if name == "" || layer == nil {
			return errors.New("mlx: invalid adapter layer")
		}
		if err := addArray(values, adapterKey(name, "lora_A.weight"), layer.a); err != nil {
			return err
		}
		if err := addArray(values, adapterKey(name, "lora_B.weight"), layer.b); err != nil {
			return err
		}
		if config.LoRA.Bias == lora.BiasLoRAOnly && layer.bias != nil {
			if err := addArray(values, adapterKey(name, "bias"), layer.bias); err != nil {
				return err
			}
		}
	}
	return huggingface.SaveTensors(dir, config.LoRA, values, metadata)
}

// LoadQLoRAAdapter validates all tensors before changing MLX adapter matrices.
func LoadQLoRAAdapter(dir string, config qlora.Config, layers map[string]*QLoRALinear) (huggingface.Metadata, error) {
	if err := validateQLoRAConfig(config); err != nil {
		return huggingface.Metadata{}, err
	}
	expectedConfig, values, metadata, err := huggingface.LoadTensors(dir)
	if err != nil {
		return huggingface.Metadata{}, err
	}
	if !sameLoRAConfig(config.LoRA, expectedConfig) {
		return huggingface.Metadata{}, huggingface.ErrConfigMismatch
	}
	replacements := make([]arrayReplacement, 0, len(layers)*2)
	expected := make(map[string]struct{}, len(layers)*2)
	for name, layer := range layers {
		if name == "" || layer == nil {
			return huggingface.Metadata{}, errors.New("mlx: invalid adapter layer")
		}
		for _, entry := range []struct {
			name string
			dst  **Array
		}{
			{adapterKey(name, "lora_A.weight"), &layer.a},
			{adapterKey(name, "lora_B.weight"), &layer.b},
		} {
			array, err := newArrayFromTensor(values, expected, entry.name, *entry.dst)
			if err != nil {
				closeReplacements(replacements)
				return huggingface.Metadata{}, err
			}
			replacements = append(replacements, arrayReplacement{entry.dst, array})
		}
		if config.LoRA.Bias == lora.BiasLoRAOnly && layer.bias != nil {
			name := adapterKey(name, "bias")
			array, err := newArrayFromTensor(values, expected, name, layer.bias)
			if err != nil {
				closeReplacements(replacements)
				return huggingface.Metadata{}, err
			}
			replacements = append(replacements, arrayReplacement{&layer.bias, array})
		}
	}
	for name := range values {
		if _, ok := expected[name]; !ok {
			closeReplacements(replacements)
			return huggingface.Metadata{}, fmt.Errorf("%w: %s", huggingface.ErrUnexpectedTensor, name)
		}
	}
	for _, replacement := range replacements {
		old := *replacement.dst
		*replacement.dst = replacement.new
		_ = old.Close()
	}
	return metadata, nil
}

func addArray(values map[string]safetensors.Tensor, name string, array *Array) error {
	shape, err := array.Shape()
	if err != nil {
		return err
	}
	data, err := array.Float32Values()
	if err != nil {
		return err
	}
	values[name] = safetensors.Tensor{Shape: shape, Data: data}
	return nil
}

func newArrayFromTensor(values map[string]safetensors.Tensor, expected map[string]struct{}, name string, old *Array) (*Array, error) {
	expected[name] = struct{}{}
	tensor, ok := values[name]
	if !ok {
		return nil, fmt.Errorf("%w: %s", huggingface.ErrMissingTensor, name)
	}
	shape, err := old.Shape()
	if err != nil {
		return nil, err
	}
	if !sameShape(shape, tensor.Shape) {
		return nil, fmt.Errorf("%w: %s", huggingface.ErrConfigMismatch, name)
	}
	return old.context.Float32(shape, tensor.Data)
}

func closeReplacements(replacements []arrayReplacement) {
	for _, replacement := range replacements {
		_ = replacement.new.Close()
	}
}

func adapterKey(module, suffix string) string { return "base_model.model." + module + "." + suffix }

func sameLoRAConfig(left, right lora.Config) bool {
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

func validateQLoRAConfig(config qlora.Config) error {
	if err := config.Validate(); err != nil {
		return err
	}
	if config.Quantization != qlora.QuantizationInt4 || config.DoubleQuant {
		return ErrQLoRAQuantization
	}
	return nil
}
