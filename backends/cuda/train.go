//go:build cuda && linux && cgo

package cuda

import (
	"errors"
	"math"
	"math/rand"

	"github.com/surya-mp/go-peft/backend"
	"github.com/surya-mp/go-peft/lora"
	"github.com/surya-mp/go-peft/qlora"
)

var ErrLearningRate = errors.New("cuda: learning rate must be finite and positive")

// TrainLoRA applies one squared-error LoRA adapter update on CUDA.
func TrainLoRA(engine *Engine, layer *lora.Linear, input, target backend.Tensor, learningRate float32) (float32, error) {
	if layer == nil {
		return 0, backend.ErrInvalidTensor
	}
	trainingRNG, dropoutSeed, err := trainingRandom(engine, layer.Dropout())
	if err != nil {
		return 0, err
	}
	output, workspace, err := layer.NewWorkspace(input)
	if err != nil {
		return 0, err
	}
	defer closeTensor(output)
	defer closeTensor(workspace)
	if err := layer.ForwardInto(output, workspace, input, layer.Dropout() != 0, trainingRNG); err != nil {
		return 0, err
	}
	return train(engine, output, workspace, input, target, layer.A(), layer.B(), layer.Scaling(), layer.Dropout(), dropoutSeed, learningRate)
}

// TrainQLoRA applies one squared-error QLoRA adapter update on CUDA.
func TrainQLoRA(engine *Engine, layer *qlora.Linear, input, target backend.Tensor, learningRate float32) (float32, error) {
	if layer == nil {
		return 0, backend.ErrInvalidTensor
	}
	trainingRNG, dropoutSeed, err := trainingRandom(engine, layer.Dropout())
	if err != nil {
		return 0, err
	}
	output, workspace, err := layer.NewWorkspace(input)
	if err != nil {
		return 0, err
	}
	defer closeTensor(output)
	defer closeTensor(workspace)
	if err := layer.ForwardInto(output, workspace, input, layer.Dropout() != 0, trainingRNG); err != nil {
		return 0, err
	}
	return train(engine, output, workspace, input, target, layer.A(), layer.B(), layer.Scaling(), layer.Dropout(), dropoutSeed, learningRate)
}

func trainingRandom(engine *Engine, dropout float32) (*rand.Rand, uint64, error) {
	if dropout == 0 {
		return nil, 0, nil
	}
	if engine == nil {
		return nil, 0, ErrLearningRate
	}
	seed := engine.nextDropoutSeed()
	dropoutSeed := rand.New(rand.NewSource(seed)).Uint64()
	return rand.New(rand.NewSource(seed)), dropoutSeed, nil
}

func train(engine *Engine, output, workspace, input, target, a, b backend.Tensor, scaling, dropout float32, dropoutSeed uint64, learningRate float32) (float32, error) {
	if engine == nil || learningRate <= 0 || math.IsNaN(float64(learningRate)) || math.IsInf(float64(learningRate), 0) {
		return 0, ErrLearningRate
	}
	rows, cols, err := engine.Shape(output)
	if err != nil {
		return 0, err
	}
	targetRows, targetCols, err := engine.Shape(target)
	if err != nil {
		return 0, err
	}
	if rows != targetRows || cols != targetCols {
		return 0, backend.ErrShapeMismatch
	}
	_, in, err := engine.Shape(input)
	if err != nil {
		return 0, err
	}
	residual, err := engine.New(rows, cols)
	if err != nil {
		return 0, err
	}
	defer closeTensor(residual)
	if err := engine.Copy(residual, output); err != nil {
		return 0, err
	}
	if err := engine.axpy(residual, target, -1); err != nil {
		return 0, err
	}
	values, _, _, err := engine.EncodeFloat32(residual)
	if err != nil {
		return 0, err
	}
	var loss float32
	for _, value := range values {
		loss += value * value
	}
	_, rank, err := engine.Shape(b)
	if err != nil {
		return 0, err
	}
	gradB, err := engine.New(cols, rank)
	if err != nil {
		return 0, err
	}
	defer closeTensor(gradB)
	gradZ, err := engine.New(rows, rank)
	if err != nil {
		return 0, err
	}
	defer closeTensor(gradZ)
	gradA, err := engine.New(rank, in)
	if err != nil {
		return 0, err
	}
	defer closeTensor(gradA)
	var mask *Tensor
	if dropout != 0 {
		mask, err = engine.dropoutMask(rows, rank, dropout, dropoutSeed)
		if err != nil {
			return 0, err
		}
		defer mask.Close()
	}
	if err := engine.Gemm(gradB, residual, true, workspace, false, 2*scaling, 0); err != nil {
		return 0, err
	}
	if err := engine.Gemm(gradZ, residual, false, b, false, 2*scaling, 0); err != nil {
		return 0, err
	}
	if mask != nil {
		if err := engine.multiply(gradZ, mask); err != nil {
			return 0, err
		}
	}
	if err := engine.Gemm(gradA, gradZ, true, input, false, 1, 0); err != nil {
		return 0, err
	}
	if err := engine.axpy(a, gradA, -learningRate); err != nil {
		return 0, err
	}
	if err := engine.axpy(b, gradB, -learningRate); err != nil {
		return 0, err
	}
	return loss, nil
}

func closeTensor(value backend.Tensor) {
	if tensor, ok := value.(*Tensor); ok {
		_ = tensor.Close()
	}
}
