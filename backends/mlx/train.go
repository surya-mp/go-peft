//go:build darwin && arm64 && mlx

package mlx

/*
#include "mlx/c/mlx.h"

static int mlx_go_adapter_loss(
	mlx_vector_array* result,
	const mlx_vector_array input,
	void* payload) {
	if (mlx_vector_array_size(input) != 7 || payload == NULL) return 1;
	mlx_array base = mlx_array_new(), x = mlx_array_new(), target = mlx_array_new(), mask = mlx_array_new();
	mlx_array a = mlx_array_new(), b = mlx_array_new(), scale = mlx_array_new();
	mlx_array at = mlx_array_new(), bt = mlx_array_new(), projected = mlx_array_new();
	mlx_array dropped = mlx_array_new(), delta = mlx_array_new(), scaled = mlx_array_new(), output = mlx_array_new();
	mlx_array residual = mlx_array_new(), squared = mlx_array_new(), loss = mlx_array_new();
	mlx_stream stream = *(mlx_stream*)payload;
	int code = 1;
	if (mlx_vector_array_get(&base, input, 0) || mlx_vector_array_get(&x, input, 1) ||
		mlx_vector_array_get(&target, input, 2) || mlx_vector_array_get(&mask, input, 3) ||
		mlx_vector_array_get(&a, input, 4) || mlx_vector_array_get(&b, input, 5) ||
		mlx_vector_array_get(&scale, input, 6)) goto done;
	if (mlx_transpose(&at, a, stream) || mlx_matmul(&projected, x, at, stream) ||
		mlx_multiply(&dropped, projected, mask, stream) || mlx_transpose(&bt, b, stream) || mlx_matmul(&delta, dropped, bt, stream) ||
		mlx_multiply(&scaled, delta, scale, stream) || mlx_add(&output, base, scaled, stream) ||
		mlx_subtract(&residual, output, target, stream) || mlx_square(&squared, residual, stream) ||
		mlx_sum(&loss, squared, false, stream) || mlx_vector_array_set_value(result, loss)) goto done;
	code = 0;
done:
	mlx_array_free(loss); mlx_array_free(squared); mlx_array_free(residual);
	mlx_array_free(output); mlx_array_free(scaled); mlx_array_free(delta); mlx_array_free(dropped);
	mlx_array_free(projected); mlx_array_free(bt); mlx_array_free(at);
	mlx_array_free(scale); mlx_array_free(b); mlx_array_free(a); mlx_array_free(mask);
	mlx_array_free(target); mlx_array_free(x); mlx_array_free(base);
	return code;
}

static mlx_vector_array mlx_go_training_inputs(
	mlx_array base, mlx_array x, mlx_array target, mlx_array mask, mlx_array a, mlx_array b, mlx_array scale) {
	mlx_array values[7] = {base, x, target, mask, a, b, scale};
	return mlx_vector_array_new_data(values, 7);
}

static int mlx_go_adapter_value_and_grad(
	mlx_vector_array* values,
	mlx_vector_array* gradients,
	const mlx_vector_array input,
	mlx_stream stream) {
	mlx_closure fun = mlx_closure_new_func_payload(mlx_go_adapter_loss, &stream, NULL);
	mlx_closure_value_and_grad value_and_grad = mlx_closure_value_and_grad_new();
	int argnums[2] = {4, 5};
	int code = mlx_value_and_grad(&value_and_grad, fun, argnums, 2);
	if (!code) code = mlx_closure_value_and_grad_apply(values, gradients, value_and_grad, input);
	mlx_closure_value_and_grad_free(value_and_grad);
	mlx_closure_free(fun);
	return code;
}
*/
import "C"

import (
	"errors"
	"math"
)

var ErrLearningRate = errors.New("mlx: learning rate must be finite and positive")

// TrainStep applies one squared-error adapter update and returns its pre-update loss.
func (l *LoRALinear) TrainStep(input, target *Array, learningRate float32) (float32, error) {
	if l == nil || input == nil || target == nil || input.context != l.weight.context || target.context != l.weight.context {
		return 0, ErrContext
	}
	mask, err := l.trainingMask(input)
	if err != nil {
		return 0, err
	}
	defer mask.Close()
	base, err := l.baseOutput(input)
	if err != nil {
		return 0, err
	}
	defer base.Close()
	return l.train(base, input, target, mask, learningRate)
}

// TrainStep applies one squared-error QLoRA adapter update and returns its pre-update loss.
func (l *QLoRALinear) TrainStep(input, target *Array, learningRate float32) (float32, error) {
	if l == nil || input == nil || target == nil || input.context != l.weight.context || target.context != l.weight.context {
		return 0, ErrContext
	}
	mask, err := l.trainingMask(input)
	if err != nil {
		return 0, err
	}
	defer mask.Close()
	base, err := l.baseOutput(input)
	if err != nil {
		return 0, err
	}
	defer base.Close()
	return trainAdapter(base, input, target, mask, l.a, l.b, l.scaling, learningRate, func(a, b *Array) {
		l.a, l.b = a, b
	})
}

func (l *LoRALinear) baseOutput(input *Array) (*Array, error) {
	weightT, err := l.weight.Transpose()
	if err != nil {
		return nil, err
	}
	defer weightT.Close()
	output, err := input.MatMul(weightT)
	if err != nil || l.bias == nil {
		return output, err
	}
	withBias, err := output.Add(l.bias)
	_ = output.Close()
	return withBias, err
}

func (l *QLoRALinear) baseOutput(input *Array) (*Array, error) {
	output, err := input.QuantizedMatMul(l.weight, l.scales, l.biases, true, l.groupSize, 4)
	if err != nil || l.bias == nil {
		return output, err
	}
	withBias, err := output.Add(l.bias)
	_ = output.Close()
	return withBias, err
}

func (l *LoRALinear) train(base, input, target, mask *Array, learningRate float32) (float32, error) {
	return trainAdapter(base, input, target, mask, l.a, l.b, l.scaling, learningRate, func(a, b *Array) {
		l.a, l.b = a, b
	})
}

func trainAdapter(base, input, target, mask, a, b *Array, scaling, learningRate float32, replace func(*Array, *Array)) (float32, error) {
	if learningRate <= 0 || math.IsNaN(float64(learningRate)) || math.IsInf(float64(learningRate), 0) {
		return 0, ErrLearningRate
	}
	if err := sameArrayShape(base, target); err != nil {
		return 0, err
	}
	scale, err := base.context.Float32(nil, []float32{scaling})
	if err != nil {
		return 0, err
	}
	defer scale.Close()
	inputs := C.mlx_go_training_inputs(base.value, input.value, target.value, mask.value, a.value, b.value, scale.value)
	defer C.mlx_vector_array_free(inputs)
	values := C.mlx_vector_array_new()
	gradients := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(values)
	defer C.mlx_vector_array_free(gradients)
	if err := nativeError("adapter value and gradient", C.mlx_go_adapter_value_and_grad(&values, &gradients, inputs, base.context.stream)); err != nil {
		return 0, err
	}
	if C.mlx_vector_array_size(values) != 1 || C.mlx_vector_array_size(gradients) != 2 {
		return 0, ErrUnsupported
	}
	loss, err := vectorArrayAt(base.context, values, 0)
	if err != nil {
		return 0, err
	}
	defer loss.Close()
	gradA, err := vectorArrayAt(base.context, gradients, 0)
	if err != nil {
		return 0, err
	}
	defer gradA.Close()
	gradB, err := vectorArrayAt(base.context, gradients, 1)
	if err != nil {
		return 0, err
	}
	defer gradB.Close()
	lossValues, err := loss.Float32Values()
	if err != nil || len(lossValues) != 1 {
		return 0, err
	}
	nextA, err := gradientStep(a, gradA, learningRate)
	if err != nil {
		return 0, err
	}
	defer func() {
		if nextA != nil {
			_ = nextA.Close()
		}
	}()
	nextB, err := gradientStep(b, gradB, learningRate)
	if err != nil {
		return 0, err
	}
	defer func() {
		if nextB != nil {
			_ = nextB.Close()
		}
	}()
	if err := nextA.Eval(); err != nil {
		return 0, err
	}
	if err := nextB.Eval(); err != nil {
		return 0, err
	}
	oldA, oldB := a, b
	replace(nextA, nextB)
	nextA, nextB = nil, nil
	_ = oldA.Close()
	_ = oldB.Close()
	return lossValues[0], nil
}

func gradientStep(value, gradient *Array, learningRate float32) (*Array, error) {
	update, err := gradient.Scale(-learningRate)
	if err != nil {
		return nil, err
	}
	defer update.Close()
	return value.Add(update)
}

func sameArrayShape(left, right *Array) error {
	leftShape, err := left.Shape()
	if err != nil {
		return err
	}
	rightShape, err := right.Shape()
	if err != nil {
		return err
	}
	if len(leftShape) != len(rightShape) {
		return ErrShape
	}
	for index := range leftShape {
		if leftShape[index] != rightShape[index] {
			return ErrShape
		}
	}
	return nil
}

func vectorArrayAt(context *Context, values C.mlx_vector_array, index int) (*Array, error) {
	value := C.mlx_array_new()
	if err := nativeError("vector array result", C.mlx_vector_array_get(&value, values, C.size_t(index))); err != nil {
		_ = C.mlx_array_free(value)
		return nil, err
	}
	return newArray(context, value), nil
}
