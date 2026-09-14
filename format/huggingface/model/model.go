// Package model streams Hugging Face base-model SafeTensors checkpoints.
package model

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/surya-mp/go-peft/format/safetensors"
)

// Standard Hugging Face SafeTensors model filenames.
const (
	ConfigFile = "config.json"
	ModelFile  = "model.safetensors"
	IndexFile  = "model.safetensors.index.json"
)

// Errors returned while inspecting or streaming base-model checkpoints.
// Callers can test them with errors.Is.
var (
	ErrMissingModel  = errors.New("huggingface model: no SafeTensors model found")
	ErrInvalidIndex  = errors.New("huggingface model: invalid SafeTensors index")
	ErrInvalidShard  = errors.New("huggingface model: invalid shard path")
	ErrMissingTensor = errors.New("huggingface model: missing indexed tensor")
)

// Config contains common Hugging Face model configuration fields.
// Unknown architecture-specific fields remain in config.json for its framework.
type Config struct {
	// ModelType is Hugging Face's model-family identifier.
	ModelType string `json:"model_type"`
	// Architectures lists model implementation names from config.json.
	Architectures []string `json:"architectures"`
	// HiddenSize is the transformer hidden width.
	HiddenSize int `json:"hidden_size"`
	// IntermediateSize is the feed-forward hidden width.
	IntermediateSize int `json:"intermediate_size"`
	// NumHiddenLayers is the transformer block count.
	NumHiddenLayers int `json:"num_hidden_layers"`
	// NumAttentionHeads is the query-attention head count.
	NumAttentionHeads int `json:"num_attention_heads"`
	// NumKeyValueHeads is the key/value attention head count.
	NumKeyValueHeads int `json:"num_key_value_heads"`
	// VocabSize is the tokenizer vocabulary size.
	VocabSize int `json:"vocab_size"`
	// TorchDType is the source checkpoint's declared dtype.
	TorchDType string `json:"torch_dtype"`
}

// Manifest describes indexed, sharded SafeTensors checkpoints.
type Manifest struct {
	// Metadata contains Hugging Face index metadata.
	Metadata map[string]json.RawMessage `json:"metadata"`
	// WeightMap maps tensor names to SafeTensors shard names.
	WeightMap map[string]string `json:"weight_map"`
}

// Tensor is one decoded base-model tensor.
type Tensor struct {
	// Name is the checkpoint tensor name.
	Name string
	// DType is the source SafeTensors dtype.
	DType string
	// Shape contains row-major dimensions.
	Shape []int
	// Data contains decoded F32 values.
	Data []float32
	// Shard is the source filename relative to the model directory.
	Shard string
}

// Options controls streaming model loading.
type Options struct {
	// Filter limits tensors delivered to OnTensor. All tensors are still
	// validated against an index when present.
	Filter func(name string) bool
	// OnTensor receives each selected tensor. It is required.
	OnTensor func(Tensor) error
	// Progress receives human-readable status messages while model shards load.
	Progress func(string)
	// ProgressEvery reports selected tensor decode progress every N tensors.
	// When zero, only model/shard milestones are reported.
	ProgressEvery int
}

// Info identifies a base model without loading its tensor data.
type Info struct {
	// Config is the parsed config.json subset.
	Config Config `json:"config"`
	// Shards lists SafeTensors files relative to the model directory.
	Shards []string `json:"shards"`
	// TensorCount is -1 until an unindexed model has been loaded.
	TensorCount int `json:"tensor_count"`
	// Indexed reports whether the model uses model.safetensors.index.json.
	Indexed bool `json:"indexed"`
}

// ReadConfig reads config.json from a Hugging Face model directory.
func ReadConfig(dir string) (Config, error) {
	data, err := os.ReadFile(filepath.Join(dir, ConfigFile))
	if err != nil {
		return Config{}, err
	}
	var config Config
	if err := json.Unmarshal(data, &config); err != nil {
		return Config{}, err
	}
	return config, nil
}

// ReadManifest reads the standard sharded-model index.
func ReadManifest(dir string) (Manifest, error) {
	data, err := os.ReadFile(filepath.Join(dir, IndexFile))
	if err != nil {
		return Manifest{}, err
	}
	var manifest Manifest
	if err := json.Unmarshal(data, &manifest); err != nil || len(manifest.WeightMap) == 0 {
		return Manifest{}, ErrInvalidIndex
	}
	return manifest, nil
}

// Inspect reads base-model metadata and its SafeTensors layout without weights.
func Inspect(dir string) (Info, error) {
	config, err := ReadConfig(dir)
	if err != nil {
		return Info{}, err
	}
	manifestPath := filepath.Join(dir, IndexFile)
	if _, err := os.Stat(manifestPath); err == nil {
		manifest, err := ReadManifest(dir)
		if err != nil {
			return Info{}, err
		}
		shards, err := manifestShards(manifest)
		if err != nil {
			return Info{}, err
		}
		return Info{Config: config, Shards: shards, TensorCount: len(manifest.WeightMap), Indexed: true}, nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return Info{}, err
	}
	if _, err := os.Stat(filepath.Join(dir, ModelFile)); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return Info{}, ErrMissingModel
		}
		return Info{}, err
	}
	return Info{Config: config, Shards: []string{ModelFile}, TensorCount: -1}, nil
}

// Load streams a Hugging Face base model. OnTensor is required so callers
// choose their framework's tensor allocation and never retain all weights here.
func Load(dir string, options Options) (Info, error) {
	if options.OnTensor == nil {
		return Info{}, errors.New("huggingface model: OnTensor is required")
	}
	progressf(options.Progress, "huggingface model: inspecting %s", dir)
	info, err := Inspect(dir)
	if err != nil {
		return Info{}, err
	}
	progressf(options.Progress, "huggingface model: found shards=%d indexed=%t tensor_count=%d", len(info.Shards), info.Indexed, info.TensorCount)
	var expected map[string]string
	if info.Indexed {
		progressf(options.Progress, "huggingface model: reading index")
		manifest, err := ReadManifest(dir)
		if err != nil {
			return Info{}, err
		}
		expected = manifest.WeightMap
	}
	seen := make(map[string]struct{})
	for shardIndex, shard := range info.Shards {
		path := filepath.Join(dir, shard)
		progressf(options.Progress, "huggingface model: loading shard %d/%d %s", shardIndex+1, len(info.Shards), shard)
		_, err := safetensors.VisitFileWithOptions(path, nil, func(decoded safetensors.DecodedTensor) error {
			if info.Indexed {
				wanted, ok := expected[decoded.Name]
				if !ok || wanted != shard {
					return fmt.Errorf("%w: %s", ErrInvalidIndex, decoded.Name)
				}
				if _, duplicate := seen[decoded.Name]; duplicate {
					return fmt.Errorf("%w: duplicate %s", ErrInvalidIndex, decoded.Name)
				}
			}
			seen[decoded.Name] = struct{}{}
			if options.Filter != nil && !options.Filter(decoded.Name) {
				return nil
			}
			return options.OnTensor(Tensor{
				Name: decoded.Name, DType: decoded.DType, Shape: decoded.Shape, Data: decoded.Data, Shard: shard,
			})
		}, safetensors.VisitOptions{Progress: options.Progress, ProgressEvery: options.ProgressEvery})
		if err != nil {
			return Info{}, err
		}
		progressf(options.Progress, "huggingface model: finished shard %d/%d %s", shardIndex+1, len(info.Shards), shard)
	}
	if info.Indexed && len(seen) != len(expected) {
		for name := range expected {
			if _, ok := seen[name]; !ok {
				return Info{}, fmt.Errorf("%w: %s", ErrMissingTensor, name)
			}
		}
	}
	if !info.Indexed {
		info.TensorCount = len(seen)
	}
	progressf(options.Progress, "huggingface model: loaded tensors=%d", len(seen))
	return info, nil
}

func progressf(progress func(string), format string, args ...any) {
	if progress != nil {
		progress(fmt.Sprintf(format, args...))
	}
}

func manifestShards(manifest Manifest) ([]string, error) {
	set := make(map[string]struct{}, len(manifest.WeightMap))
	for _, shard := range manifest.WeightMap {
		if !validShard(shard) {
			return nil, fmt.Errorf("%w: %q", ErrInvalidShard, shard)
		}
		set[shard] = struct{}{}
	}
	shards := make([]string, 0, len(set))
	for shard := range set {
		shards = append(shards, shard)
	}
	sort.Strings(shards)
	return shards, nil
}

func validShard(shard string) bool {
	clean := filepath.Clean(shard)
	return shard != "" && filepath.Ext(clean) == ".safetensors" && !filepath.IsAbs(shard) && clean != "." && clean != ".." && !strings.HasPrefix(clean, ".."+string(filepath.Separator))
}
