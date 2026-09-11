//go:build gomlx

package gomlx

import (
	"errors"
	"fmt"
	"sort"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/surya-mp/go-peft/format/huggingface"
	"github.com/surya-mp/go-peft/format/safetensors"
	"github.com/surya-mp/go-peft/lora"
)

// Save writes this native adapter in Hugging Face PEFT format.
func (a *Adapter) Save(dir string, metadata huggingface.Metadata) error {
	if a == nil {
		return errors.New("gomlx: adapter is required")
	}
	if a.config.Bias == lora.BiasAll {
		return huggingface.ErrUnsupportedBias
	}
	tensorsByName := make(map[string]safetensors.Tensor, len(a.layers)*2)
	for _, layer := range a.layers {
		if err := addVariable(tensorsByName, tensorKey(layer.Name(), "lora_A.weight"), layer.A()); err != nil {
			return err
		}
		if err := addVariable(tensorsByName, tensorKey(layer.Name(), "lora_B.weight"), layer.B()); err != nil {
			return err
		}
		if a.config.Bias == lora.BiasLoRAOnly && layer.Bias() != nil {
			if err := addVariable(tensorsByName, tensorKey(layer.Name(), "bias"), layer.Bias()); err != nil {
				return err
			}
		}
	}
	return huggingface.SaveTensors(dir, a.config, tensorsByName, metadata)
}

// Load validates every tensor before replacing any native variable value.
func (a *Adapter) Load(dir string) (huggingface.Metadata, error) {
	if a == nil {
		return huggingface.Metadata{}, errors.New("gomlx: adapter is required")
	}
	config, tensorsByName, metadata, err := huggingface.LoadTensors(dir)
	if err != nil {
		return huggingface.Metadata{}, err
	}
	if !sameConfig(config, a.config) {
		return huggingface.Metadata{}, huggingface.ErrConfigMismatch
	}
	type assignment struct {
		variable *model.Variable
		value    *tensors.Tensor
	}
	assignments := make([]assignment, 0, len(a.layers)*2)
	expected := make(map[string]struct{}, len(a.layers)*2)
	for _, layer := range a.layers {
		for _, entry := range []struct {
			name     string
			variable *model.Variable
		}{
			{tensorKey(layer.Name(), "lora_A.weight"), layer.A()},
			{tensorKey(layer.Name(), "lora_B.weight"), layer.B()},
		} {
			value, err := validatedTensor(tensorsByName, expected, entry.name, entry.variable)
			if err != nil {
				return huggingface.Metadata{}, err
			}
			assignments = append(assignments, assignment{entry.variable, value})
		}
		if a.config.Bias == lora.BiasLoRAOnly && layer.Bias() != nil {
			name := tensorKey(layer.Name(), "bias")
			value, err := validatedTensor(tensorsByName, expected, name, layer.Bias())
			if err != nil {
				return huggingface.Metadata{}, err
			}
			assignments = append(assignments, assignment{layer.Bias(), value})
		}
	}
	for name := range tensorsByName {
		if _, ok := expected[name]; !ok {
			return huggingface.Metadata{}, fmt.Errorf("%w: %s", huggingface.ErrUnexpectedTensor, name)
		}
	}
	for _, assignment := range assignments {
		if err := assignment.variable.SetValue(assignment.value); err != nil {
			return huggingface.Metadata{}, err
		}
	}
	return metadata, nil
}

// Save writes a native NF4 QLoRA adapter in Hugging Face PEFT format.
func (a *NF4Adapter) Save(dir string, metadata huggingface.Metadata) error {
	if a == nil {
		return errors.New("gomlx: adapter is required")
	}
	if a.config.LoRA.Bias == lora.BiasAll {
		return huggingface.ErrUnsupportedBias
	}
	values := make(map[string]safetensors.Tensor, len(a.layers)*2)
	for _, layer := range a.layers {
		if err := addVariable(values, tensorKey(layer.Name(), "lora_A.weight"), layer.A()); err != nil {
			return err
		}
		if err := addVariable(values, tensorKey(layer.Name(), "lora_B.weight"), layer.B()); err != nil {
			return err
		}
		if a.config.LoRA.Bias == lora.BiasLoRAOnly && layer.Bias() != nil {
			if err := addVariable(values, tensorKey(layer.Name(), "bias"), layer.Bias()); err != nil {
				return err
			}
		}
	}
	return huggingface.SaveTensors(dir, a.config.LoRA, values, metadata)
}

// Load validates all Hugging Face QLoRA adapter tensors before mutation.
func (a *NF4Adapter) Load(dir string) (huggingface.Metadata, error) {
	if a == nil {
		return huggingface.Metadata{}, errors.New("gomlx: adapter is required")
	}
	config, values, metadata, err := huggingface.LoadTensors(dir)
	if err != nil {
		return huggingface.Metadata{}, err
	}
	if !sameConfig(config, a.config.LoRA) {
		return huggingface.Metadata{}, huggingface.ErrConfigMismatch
	}
	type assignment struct {
		variable *model.Variable
		value    *tensors.Tensor
	}
	assignments := make([]assignment, 0, len(a.layers)*2)
	expected := make(map[string]struct{}, len(a.layers)*2)
	for _, layer := range a.layers {
		for _, entry := range []struct {
			name     string
			variable *model.Variable
		}{
			{tensorKey(layer.Name(), "lora_A.weight"), layer.A()},
			{tensorKey(layer.Name(), "lora_B.weight"), layer.B()},
		} {
			value, err := validatedTensor(values, expected, entry.name, entry.variable)
			if err != nil {
				return huggingface.Metadata{}, err
			}
			assignments = append(assignments, assignment{entry.variable, value})
		}
		if a.config.LoRA.Bias == lora.BiasLoRAOnly && layer.Bias() != nil {
			name := tensorKey(layer.Name(), "bias")
			value, err := validatedTensor(values, expected, name, layer.Bias())
			if err != nil {
				return huggingface.Metadata{}, err
			}
			assignments = append(assignments, assignment{layer.Bias(), value})
		}
	}
	for name := range values {
		if _, ok := expected[name]; !ok {
			return huggingface.Metadata{}, fmt.Errorf("%w: %s", huggingface.ErrUnexpectedTensor, name)
		}
	}
	for _, assignment := range assignments {
		if err := assignment.variable.SetValue(assignment.value); err != nil {
			return huggingface.Metadata{}, err
		}
	}
	return metadata, nil
}

func addVariable(dst map[string]safetensors.Tensor, name string, variable *model.Variable) error {
	value, err := variable.Value()
	if err != nil {
		return fmt.Errorf("gomlx: read %s: %w", name, err)
	}
	data, err := float32Data(variable.Shape().DType, value)
	if err != nil {
		return fmt.Errorf("gomlx: encode %s: %w", name, err)
	}
	dst[name] = safetensors.Tensor{Shape: append([]int(nil), variable.Shape().Dimensions...), Data: data}
	return nil
}

func validatedTensor(values map[string]safetensors.Tensor, expected map[string]struct{}, name string, variable *model.Variable) (*tensors.Tensor, error) {
	expected[name] = struct{}{}
	encoded, ok := values[name]
	if !ok {
		return nil, fmt.Errorf("%w: %s", huggingface.ErrMissingTensor, name)
	}
	shape := variable.Shape().Dimensions
	if !sameShape(encoded.Shape, shape) {
		return nil, fmt.Errorf("%w: %s", huggingface.ErrConfigMismatch, name)
	}
	return tensorValue(variable.Shape().DType, encoded.Data, shape...)
}

func float32Data(dtype dtypes.DType, value *tensors.Tensor) ([]float32, error) {
	if dtype == dtypes.Float32 {
		return tensors.CopyFlatData[float32](value)
	}
	result := make([]float32, value.Shape().Size())
	if dtype == dtypes.Float16 {
		values, err := tensors.CopyFlatData[float16.Float16](value)
		if err != nil {
			return nil, err
		}
		for index, value := range values {
			result[index] = value.Float32()
		}
		return result, nil
	}
	if dtype == dtypes.BFloat16 {
		values, err := tensors.CopyFlatData[bfloat16.BFloat16](value)
		if err != nil {
			return nil, err
		}
		for index, value := range values {
			result[index] = value.Float32()
		}
		return result, nil
	}
	return nil, errors.New("gomlx: unsupported adapter dtype")
}

func tensorValue(dtype dtypes.DType, data []float32, shape ...int) (*tensors.Tensor, error) {
	if dtype == dtypes.Float32 {
		return tensors.FromFlatDataAndDimensions(data, shape...), nil
	}
	if dtype == dtypes.Float16 {
		values := make([]float16.Float16, len(data))
		for index, value := range data {
			values[index] = float16.FromFloat32(value)
		}
		return tensors.FromFlatDataAndDimensions(values, shape...), nil
	}
	if dtype == dtypes.BFloat16 {
		values := make([]bfloat16.BFloat16, len(data))
		for index, value := range data {
			values[index] = bfloat16.FromFloat32(value)
		}
		return tensors.FromFlatDataAndDimensions(values, shape...), nil
	}
	return nil, errors.New("gomlx: unsupported adapter dtype")
}

func tensorKey(module, suffix string) string { return "base_model.model." + module + "." + suffix }

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
