package main

import (
	"math"

	"github.com/gonum/matrix/mat64"
)

// The rnn represents the neural network
// This RNNs parameters are the three matrices whh, wxh, why.
// h is the hidden state, which is actually the memory of the RNN
type rnn struct {
	whh *mat64.Dense  // size is hiddenDimension * hiddenDimension
	wxh *mat64.Dense  //
	why *mat64.Dense  //
	h   *mat64.Vector // This is the hidden vector, which actually represents the memory of the RNN
	bh  *mat64.Vector // This is the hidden vector, which actually represents the memory of the RNN
}

// newRNN creates a new RNN with input size of x, outputsize of y and hidden dimension of h
// The hidden state h is initialized with the zero vector.
func newRNN(x, y, h int) *rnn {
	return nil
}

// step updates the hidden state of the RNN
// The above specifies the forward pass of a vanilla RNN.
// The np.tanh function implements a non-linearity that squashes the activations to the range [-1, 1].
// There are two terms inside of the tanh:
// * one is based on the previous hidden state and one is based on the current input.
// The two intermediates interact with addition, and then get squashed by the tanh into the new state vector.
func (r *rnn) step(x *mat64.Vector) *mat64.Vector {
	r.h = tanh(add(dot(r.whh, add(r.h, r.bh)), dot(r.wxh, x)))
	return dot(r.why, r.h)
}

// tanh applies an element-wise tanh to the parameter and returns a new vector
func tanh(x *mat64.Vector) *mat64.Vector {
	y := mat64.NewVector(x.Len(), nil)
	for i := 0; i < x.Len(); i++ {
		y.SetVec(i, math.Tanh(x.At(i, 0)))
	}
	return y
}

// dot is a matrix multiplication the returns a Vector
func dot(x mat64.Matrix, y *mat64.Vector) *mat64.Vector {
	v := mat64.NewVector(y.Len(), nil)
	v.MulVec(x, y)
	return v
}

// TODO: check the size of the vectors...
func add(x, y *mat64.Vector) *mat64.Vector {
	v := mat64.NewVector(x.Len(), nil)
	v.AddVec(x, y)
	return v
}
