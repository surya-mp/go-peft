package backend

import (
	"errors"
	"log"
)

// ErrCUDAUnavailable reports that GPU execution could not be selected.
var ErrCUDAUnavailable = errors.New("backend: CUDA is unavailable")

// Device identifies the selected execution device.
type Device uint8

// Device values identify runtime selection outcomes.
const (
	DeviceUnavailable Device = iota
	DeviceCPU
	DeviceCUDA
)

// RuntimeOptions controls accelerator selection.
type RuntimeOptions struct {
	// DisableCUDA bypasses accelerator selection and uses CPU without a warning.
	DisableCUDA bool
	// CUDA is an optional linked CUDA engine. A nil value rejects GPU execution.
	CUDA EagerEngine
	// Warn receives fallback diagnostics. The default writes a warning to the logger.
	Warn func(error)
}

// Runtime describes the explicitly selected execution engine.
type Runtime struct {
	// Engine is the selected eager engine. It is nil when DeviceUnavailable.
	Engine EagerEngine
	// Device identifies the selected execution device.
	Device Device
	// Warning is ErrCUDAUnavailable when CUDA selection failed.
	Warning error
}

// NewRuntime requires CUDA by default. Set DisableCUDA to opt into CPU execution.
func NewRuntime(options RuntimeOptions) (Runtime, error) {
	if options.DisableCUDA {
		return Runtime{Engine: NewCPU(), Device: DeviceCPU}, nil
	}
	if options.CUDA != nil {
		return Runtime{Engine: options.CUDA, Device: DeviceCUDA}, nil
	}
	runtime := Runtime{Device: DeviceUnavailable, Warning: ErrCUDAUnavailable}
	if options.Warn != nil {
		options.Warn(runtime.Warning)
	} else {
		log.Printf("go-peft WARNING: %v; training was not started", runtime.Warning)
	}
	return runtime, runtime.Warning
}
