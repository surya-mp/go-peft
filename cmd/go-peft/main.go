// Command go-peft inspects adapters and plans model-family injection.
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/surya-mp/go-peft/format/gguf"
	"github.com/surya-mp/go-peft/format/huggingface"
	hfmodel "github.com/surya-mp/go-peft/format/huggingface/model"
	"github.com/surya-mp/go-peft/profiles"
)

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "go-peft:", err)
		os.Exit(1)
	}
}

func run(args []string, output io.Writer) error {
	if len(args) == 0 {
		usage(output)
		return nil
	}
	switch args[0] {
	case "targets":
		return targets(args[1:], output)
	case "plan":
		return plan(args[1:], output)
	case "inspect", "validate":
		return inspect(args[1:], output)
	case "inspect-model":
		return inspectModel(args[1:], output)
	case "inspect-gguf":
		return inspectGGUF(args[1:], output)
	case "help", "-h", "--help":
		usage(output)
		return nil
	default:
		return fmt.Errorf("unknown command %q", args[0])
	}
}

func inspectModel(args []string, output io.Writer) error {
	flags := flag.NewFlagSet("inspect-model", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	dir := flags.String("model", "", "Hugging Face base model directory")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *dir == "" {
		return errors.New("--model is required")
	}
	info, err := hfmodel.Inspect(*dir)
	if err != nil {
		return err
	}
	return writeJSON(output, info)
}

func inspectGGUF(args []string, output io.Writer) error {
	flags := flag.NewFlagSet("inspect-gguf", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	path := flags.String("model", "", "GGUF model file")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *path == "" {
		return errors.New("--model is required")
	}
	info, err := gguf.Read(*path)
	if err != nil {
		return err
	}
	return writeJSON(output, info)
}

func targets(args []string, output io.Writer) error {
	flags := flag.NewFlagSet("targets", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	family := flags.String("family", "llama", "model family")
	mode := flags.String("mode", string(profiles.Attention), "attention or all-linear")
	if err := flags.Parse(args); err != nil {
		return err
	}
	profile, err := profiles.Resolve(profiles.Family(*family))
	if err != nil {
		return err
	}
	values, err := profile.Targets(profiles.Mode(*mode))
	if err != nil {
		return err
	}
	return writeJSON(output, map[string]any{"family": profile.Family, "mode": *mode, "targets": values})
}

func plan(args []string, output io.Writer) error {
	flags := flag.NewFlagSet("plan", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	family := flags.String("family", "llama", "model family")
	mode := flags.String("mode", string(profiles.AllLinear), "attention or all-linear")
	modules := flags.String("modules", "", "newline-delimited module names")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *modules == "" {
		return errors.New("--modules is required")
	}
	data, err := os.ReadFile(*modules)
	if err != nil {
		return err
	}
	values := strings.FieldsFunc(string(data), func(value rune) bool { return value == '\n' || value == '\r' })
	result, err := profiles.Plan(profiles.Family(*family), profiles.Mode(*mode), values)
	if err != nil {
		return err
	}
	if err := result.Validate(); err != nil {
		return err
	}
	return writeJSON(output, result)
}

func inspect(args []string, output io.Writer) error {
	flags := flag.NewFlagSet("inspect", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	dir := flags.String("adapter", "", "adapter directory")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *dir == "" {
		return errors.New("--adapter is required")
	}
	config, tensors, metadata, err := huggingface.LoadTensors(*dir)
	if err != nil {
		return err
	}
	type tensorInfo struct {
		Name  string `json:"name"`
		Shape []int  `json:"shape"`
	}
	items := make([]tensorInfo, 0, len(tensors))
	for name, tensor := range tensors {
		items = append(items, tensorInfo{Name: name, Shape: tensor.Shape})
	}
	return writeJSON(output, map[string]any{"valid": true, "config": config, "metadata": metadata, "tensors": items})
}

func writeJSON(output io.Writer, value any) error {
	encoder := json.NewEncoder(output)
	encoder.SetIndent("", "  ")
	return encoder.Encode(value)
}

func usage(output io.Writer) {
	fmt.Fprint(output, `go-peft commands:
  targets --family llama --mode attention
  plan --family llama --mode all-linear --modules modules.txt
  inspect --adapter adapter-dir
  validate --adapter adapter-dir
  inspect-model --model model-dir
  inspect-gguf --model model.gguf
`)
}
