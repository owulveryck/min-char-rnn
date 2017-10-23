package main

import (
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

// The rnn represents the neural network
// This RNNs parameters are the three matrices whh, wxh, why.
// h is the hidden state, which is actually the memory of the RNN
type rnn struct {
	whh    *mat.Dense // size is hiddenDimension * hiddenDimension
	wxh    *mat.Dense //
	why    *mat.Dense //
	h      *mat.Dense // This is the hidden vector, which actually represents the memory of the RNN
	hprev  *mat.Dense // This is the hidden vector, which actually represents the memory of the RNN
	bh     *mat.Dense // This is the biais
	by     *mat.Dense // This is the biais
	config neuralNetConfig
}

// neuralNetConfig defines our neural network
// architecture and learning parameters.
type neuralNetConfig struct {
	inputNeurons  int
	outputNeurons int
	hiddenNeurons int
	memorySize    int
	numEpochs     int
	learningRate  float64
}

// newRNN creates a new RNN with input size of x, outputsize of y and hidden dimension of h
// The hidden state h is initialized with the zero vector.
//func newRNN(x, y, h int) *rnn {
func newRNN(config neuralNetConfig) *rnn {
	var rnn rnn
	rnn.config = config
	// Initialize biases/weights.
	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)

	rnn.wxh = mat.NewDense(config.inputNeurons, config.hiddenNeurons, nil)
	rnn.whh = mat.NewDense(config.hiddenNeurons, config.hiddenNeurons, nil)
	rnn.bh = mat.NewDense(1, config.hiddenNeurons, nil)
	rnn.why = mat.NewDense(config.hiddenNeurons, config.outputNeurons, nil)
	rnn.by = mat.NewDense(1, config.outputNeurons, nil)

	wHiddenRaw := rnn.wxh.RawMatrix().Data
	wHiddenHiddenRaw := rnn.whh.RawMatrix().Data
	bHiddenRaw := rnn.bh.RawMatrix().Data
	wOutRaw := rnn.why.RawMatrix().Data
	bOutRaw := rnn.by.RawMatrix().Data

	for _, param := range [][]float64{
		wHiddenRaw,
		wHiddenHiddenRaw,
		bHiddenRaw,
		wOutRaw,
		bOutRaw,
	} {
		for i := range param {
			param[i] = randGen.Float64()
		}
	}

	return &rnn
}

// step updates the hidden state of the RNN
// The above specifies the forward pass of a vanilla RNN.
// The np.tanh function implements a non-linearity that squashes the activations to the range [-1, 1].
// There are two terms inside of the tanh:
// one is based on the previous hidden state and one is based on the current input.
// The two intermediates interact with addition, and then get squashed by the tanh into the new state vector.
func (r *rnn) step(x *mat.Dense) *mat.Dense {
	return nil
}

// Estimate the loss function between inputs and targets
// returns the loss and gradients on model parameters
// The last hidden state is modified
func (r *rnn) loss(inputs, targets *mat.Vector) (loss float64, dwxh, dwhh, dwhy, dbh, dby *mat.Dense) {
	return
}

// Update the rnn with Adagrad method
func (r *rnn) adagrad(dwxh, dwhh, dwhy *mat.Dense, dbhh, dby *mat.Dense) {
	return
}
