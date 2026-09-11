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

const (
	ConfigFile = "config.json"
	ModelFile  = "model.safetensors"
	IndexFile  = "model.safetensors.index.json"
)

var (
	ErrMissingModel  = errors.New("huggingface model: no SafeTensors model found")
	ErrInvalidIndex  = errors.New("huggingface model: invalid SafeTensors index")
	ErrInvalidShard  = errors.New("huggingface model: invalid shard path")
	ErrMissingTensor = errors.New("huggingface model: missing indexed tensor")
)

// Config contains common Hugging Face model configuration fields.
// Unknown architecture-specific fields remain in config.json for its framework.
type Config struct {
	ModelType         string   `json:"model_type"`
	Architectures     []string `json:"architectures"`
	HiddenSize        int      `json:"hidden_size"`
	IntermediateSize  int      `json:"intermediate_size"`
	NumHiddenLayers   int      `json:"num_hidden_layers"`
	NumAttentionHeads int      `json:"num_attention_heads"`
	NumKeyValueHeads  int      `json:"num_key_value_heads"`
	VocabSize         int      `json:"vocab_size"`
	TorchDType        string   `json:"torch_dtype"`
}

// Manifest describes indexed, sharded SafeTensors checkpoints.
type Manifest struct {
	Metadata  map[string]json.RawMessage `json:"metadata"`
	WeightMap map[string]string          `json:"weight_map"`
}

// Tensor is one decoded base-model tensor.
type Tensor struct {
	Name  string
	DType string
	Shape []int
	Data  []float32
	Shard string
}

// Options controls streaming model loading.
type Options struct {
	// Filter limits tensors delivered to OnTensor. All tensors are still
	// validated against an index when present.
	Filter   func(name string) bool
	OnTensor func(Tensor) error
}

// Info identifies a base model without loading its tensor data.
type Info struct {
	Config      Config   `json:"config"`
	Shards      []string `json:"shards"`
	TensorCount int      `json:"tensor_count"` // -1 when an unindexed model has not been loaded
	Indexed     bool     `json:"indexed"`
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
	info, err := Inspect(dir)
	if err != nil {
		return Info{}, err
	}
	var expected map[string]string
	if info.Indexed {
		manifest, err := ReadManifest(dir)
		if err != nil {
			return Info{}, err
		}
		expected = manifest.WeightMap
	}
	seen := make(map[string]struct{})
	for _, shard := range info.Shards {
		path := filepath.Join(dir, shard)
		_, err := safetensors.VisitFile(path, nil, func(decoded safetensors.DecodedTensor) error {
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
		})
		if err != nil {
			return Info{}, err
		}
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
	return info, nil
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
