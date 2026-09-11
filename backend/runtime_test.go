package backend

import (
	"errors"
	"testing"
)

func TestRuntimeRequiresCUDA(t *testing.T) {
	var warning error
	runtime, err := NewRuntime(RuntimeOptions{Warn: func(err error) { warning = err }})
	if runtime.Device != DeviceUnavailable || runtime.Engine != nil || !errors.Is(err, ErrCUDAUnavailable) || !errors.Is(runtime.Warning, ErrCUDAUnavailable) || !errors.Is(warning, ErrCUDAUnavailable) {
		t.Fatalf("runtime = %#v, warning = %v", runtime, warning)
	}
}

func TestRuntimeUsesProvidedCUDAEngine(t *testing.T) {
	cuda := NewCPU()
	runtime, err := NewRuntime(RuntimeOptions{CUDA: cuda})
	if err != nil || runtime.Device != DeviceCUDA || runtime.Engine != cuda || runtime.Warning != nil {
		t.Fatalf("runtime = %#v", runtime)
	}
}

func TestRuntimeCanExplicitlyUseCPU(t *testing.T) {
	runtime, err := NewRuntime(RuntimeOptions{DisableCUDA: true})
	if err != nil || runtime.Device != DeviceCPU || runtime.Warning != nil {
		t.Fatalf("runtime = %#v", runtime)
	}
}
