package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestTargetsAndPlan(t *testing.T) {
	var output bytes.Buffer
	if err := run([]string{"targets", "--family", "llama", "--mode", "attention"}, &output); err != nil || !strings.Contains(output.String(), "q_proj") {
		t.Fatalf("targets = %q, %v", output.String(), err)
	}
	path := filepath.Join(t.TempDir(), "modules.txt")
	if err := os.WriteFile(path, []byte("layers.0.self_attn.q_proj\nlayers.0.mlp.down_proj\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	output.Reset()
	if err := run([]string{"plan", "--family", "llama", "--modules", path}, &output); err != nil || !strings.Contains(output.String(), "down_proj") {
		t.Fatalf("plan = %q, %v", output.String(), err)
	}
}

func TestInspectCommandsRequirePaths(t *testing.T) {
	for _, command := range []string{"inspect", "inspect-model", "inspect-gguf"} {
		if err := run([]string{command}, &bytes.Buffer{}); err == nil {
			t.Fatalf("%s accepted missing path", command)
		}
	}
}
