// Package trainer coordinates backend-owned PEFT training steps.
package trainer

import (
	"context"
	"errors"
	"fmt"
	"time"
)

var ErrInvalidConfig = errors.New("trainer: invalid configuration")

// Loop owns batches, forward/backward execution, and optimizer state.
// TrainMicroBatch must accumulate gradients without applying an optimizer step.
type Loop interface {
	ZeroGrad(context.Context) error
	TrainMicroBatch(context.Context, float32) (float64, error)
	ApplyGradients(context.Context) error
}

// GradientClipper is implemented by loops with native gradient clipping.
type GradientClipper interface {
	ClipGradients(context.Context, float32) error
}

// Checkpointer persists adapter and host-owned training state.
type Checkpointer interface {
	SaveCheckpoint(context.Context, State) error
}

// Config controls generic accumulation, reporting, and checkpoint cadence.
type Config struct {
	Steps             int
	AccumulationSteps int
	GradientClip      float32
	CheckpointEvery   int
}

// Validate rejects invalid loop settings before work starts.
func (c Config) Validate() error {
	if c.Steps <= 0 || c.AccumulationSteps <= 0 || c.GradientClip < 0 || c.CheckpointEvery < 0 {
		return ErrInvalidConfig
	}
	return nil
}

// State describes a completed optimizer step.
type State struct {
	Step      int
	Loss      float64
	Duration  time.Duration
	Timestamp time.Time
}

// Runner executes a backend-neutral training loop.
type Runner struct {
	Config       Config
	Loop         Loop
	Checkpointer Checkpointer
	Report       func(State)
}

// Run executes configured optimizer steps or stops on context cancellation.
func (r Runner) Run(ctx context.Context) error {
	if err := r.Config.Validate(); err != nil {
		return err
	}
	if r.Loop == nil {
		return fmt.Errorf("%w: loop is required", ErrInvalidConfig)
	}
	scale := 1 / float32(r.Config.AccumulationSteps)
	for step := 1; step <= r.Config.Steps; step++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		started := time.Now()
		if err := r.Loop.ZeroGrad(ctx); err != nil {
			return err
		}
		var loss float64
		for micro := 0; micro < r.Config.AccumulationSteps; micro++ {
			value, err := r.Loop.TrainMicroBatch(ctx, scale)
			if err != nil {
				return err
			}
			loss += value
		}
		if r.Config.GradientClip != 0 {
			if clipper, ok := r.Loop.(GradientClipper); ok {
				if err := clipper.ClipGradients(ctx, r.Config.GradientClip); err != nil {
					return err
				}
			}
		}
		if err := r.Loop.ApplyGradients(ctx); err != nil {
			return err
		}
		state := State{Step: step, Loss: loss / float64(r.Config.AccumulationSteps), Duration: time.Since(started), Timestamp: time.Now().UTC()}
		if r.Report != nil {
			r.Report(state)
		}
		if r.Checkpointer != nil && r.Config.CheckpointEvery != 0 && step%r.Config.CheckpointEvery == 0 {
			if err := r.Checkpointer.SaveCheckpoint(ctx, state); err != nil {
				return err
			}
		}
	}
	return nil
}
