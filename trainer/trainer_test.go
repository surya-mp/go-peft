package trainer

import (
	"context"
	"testing"
)

type testLoop struct {
	zero, micro, apply, clip int
	scales                   []float32
}

func (l *testLoop) ZeroGrad(context.Context) error { l.zero++; return nil }
func (l *testLoop) TrainMicroBatch(_ context.Context, scale float32) (float64, error) {
	l.micro++
	l.scales = append(l.scales, scale)
	return float64(l.micro), nil
}
func (l *testLoop) ApplyGradients(context.Context) error         { l.apply++; return nil }
func (l *testLoop) ClipGradients(context.Context, float32) error { l.clip++; return nil }

type testCheckpoint struct{ states []State }

func (c *testCheckpoint) SaveCheckpoint(_ context.Context, state State) error {
	c.states = append(c.states, state)
	return nil
}

func TestRunnerAccumulatesClipsAndCheckpoints(t *testing.T) {
	loop := &testLoop{}
	checkpoint := &testCheckpoint{}
	runner := Runner{Config: Config{Steps: 3, AccumulationSteps: 2, GradientClip: 1, CheckpointEvery: 2}, Loop: loop, Checkpointer: checkpoint}
	if err := runner.Run(context.Background()); err != nil {
		t.Fatal(err)
	}
	if loop.zero != 3 || loop.micro != 6 || loop.apply != 3 || loop.clip != 3 {
		t.Fatalf("counts = %#v", loop)
	}
	for _, scale := range loop.scales {
		if scale != .5 {
			t.Fatalf("scale = %v", scale)
		}
	}
	if len(checkpoint.states) != 1 || checkpoint.states[0].Step != 2 {
		t.Fatalf("checkpoints = %#v", checkpoint.states)
	}
}
