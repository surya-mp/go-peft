// Package profiles provides target-module presets for common transformer families.
package profiles

import (
	"errors"
	"sort"
	"strings"
)

var (
	ErrUnknownFamily = errors.New("profiles: unknown model family")
	ErrUnknownMode   = errors.New("profiles: unknown target mode")
	ErrNoMatch       = errors.New("profiles: no target modules matched")
)

// Family identifies a transformer projection naming convention.
type Family string

const (
	Llama   Family = "llama"
	Mistral Family = "mistral"
	Qwen2   Family = "qwen2"
	Gemma   Family = "gemma"
	Phi3    Family = "phi3"
)

// Mode selects attention-only or QLoRA-style all-linear targeting.
type Mode string

const (
	Attention Mode = "attention"
	AllLinear Mode = "all-linear"
)

// Profile maps a model family to stable linear-module suffixes.
type Profile struct {
	Family    Family
	Attention []string
	Linear    []string
}

// Resolve returns a copy of the profile for one supported model family.
func Resolve(family Family) (Profile, error) {
	profile, ok := profiles[Family(strings.ToLower(string(family)))]
	if !ok {
		return Profile{}, ErrUnknownFamily
	}
	profile.Attention = append([]string(nil), profile.Attention...)
	profile.Linear = append([]string(nil), profile.Linear...)
	return profile, nil
}

// Families returns all supported profile names in deterministic order.
func Families() []Family {
	result := make([]Family, 0, len(profiles))
	for family := range profiles {
		result = append(result, family)
	}
	sort.Slice(result, func(i, j int) bool { return result[i] < result[j] })
	return result
}

// Targets returns configuration-ready target module suffixes.
func (p Profile) Targets(mode Mode) ([]string, error) {
	var targets []string
	switch mode {
	case Attention:
		targets = p.Attention
	case AllLinear:
		targets = p.Linear
	default:
		return nil, ErrUnknownMode
	}
	return append([]string(nil), targets...), nil
}

// Plan reports exactly which discovered module names a profile would inject.
func Plan(family Family, mode Mode, modules []string) (InjectionPlan, error) {
	profile, err := Resolve(family)
	if err != nil {
		return InjectionPlan{}, err
	}
	targets, err := profile.Targets(mode)
	if err != nil {
		return InjectionPlan{}, err
	}
	plan := InjectionPlan{Family: profile.Family, Mode: mode, Targets: targets}
	matchedTargets := make(map[string]bool, len(targets))
	seen := make(map[string]struct{}, len(modules))
	for _, module := range modules {
		if module == "" {
			continue
		}
		for _, target := range targets {
			if module == target || strings.HasSuffix(module, "."+target) {
				if _, exists := seen[module]; !exists {
					plan.Matched = append(plan.Matched, module)
					seen[module] = struct{}{}
				}
				matchedTargets[target] = true
				break
			}
		}
	}
	for _, target := range targets {
		if !matchedTargets[target] {
			plan.Missing = append(plan.Missing, target)
		}
	}
	sort.Strings(plan.Matched)
	return plan, nil
}

// InjectionPlan is a side-effect-free target matching result.
type InjectionPlan struct {
	Family  Family   `json:"family"`
	Mode    Mode     `json:"mode"`
	Targets []string `json:"targets"`
	Matched []string `json:"matched"`
	Missing []string `json:"missing"`
}

// Validate rejects a plan that would mutate no model modules.
func (p InjectionPlan) Validate() error {
	if len(p.Matched) == 0 {
		return ErrNoMatch
	}
	return nil
}

var profiles = map[Family]Profile{
	Llama:   {Family: Llama, Attention: []string{"q_proj", "v_proj"}, Linear: []string{"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}},
	Mistral: {Family: Mistral, Attention: []string{"q_proj", "v_proj"}, Linear: []string{"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}},
	Qwen2:   {Family: Qwen2, Attention: []string{"q_proj", "v_proj"}, Linear: []string{"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}},
	Gemma:   {Family: Gemma, Attention: []string{"q_proj", "v_proj"}, Linear: []string{"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}},
	Phi3:    {Family: Phi3, Attention: []string{"qkv_proj", "o_proj"}, Linear: []string{"qkv_proj", "o_proj", "gate_up_proj", "down_proj"}},
}
