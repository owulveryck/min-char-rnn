package main

import (
	"math/rand"
	"time"

	"github.com/gonum/matrix/mat64"
)

// The rnn represents the neural network
// This RNNs parameters are the three matrices whh, wxh, why.
// h is the hidden state, which is actually the memory of the RNN
type rnn struct {
	whh   *mat64.Dense  // size is hiddenDimension * hiddenDimension
	wxh   *mat64.Dense  //
	why   *mat64.Dense  //
	h     *mat64.Vector // This is the hidden vector, which actually represents the memory of the RNN
	hprev *mat64.Vector // This is the hidden vector, which actually represents the memory of the RNN
	bh    *mat64.Vector // This is the biais
}

// newRNN creates a new RNN with input size of x, outputsize of y and hidden dimension of h
// The hidden state h is initialized with the zero vector.
func newRNN(x, y, h int) *rnn {
	var rnn rnn
	// Initialize the matrices with random parameters o
	// Initialize the random seed
	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	uData := make([]float64, x*h)
	for i := range uData {
		uData[i] = r.NormFloat64()
	}
	rnn.wxh = mat64.NewDense(x, h, uData)

	vData := make([]float64, y*h)
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	for i := range vData {
		vData[i] = r.NormFloat64()
	}
	rnn.why = mat64.NewDense(h, y, vData)

	wData := make([]float64, h*h)
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	for i := range wData {
		wData[i] = r.NormFloat64()
	}
	rnn.whh = mat64.NewDense(h, h, wData)

	return &rnn
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

// Estimate the loss function
func (r *rnn) loss(inputs, targets *mat64.Vector) {
}
