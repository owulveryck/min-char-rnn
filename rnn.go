package main

import (
	"log"
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
	hprev  []float64  // This is the last element of the hidden vector, which actually represents the memory of the RNN
	bh     []float64  // This is the biais
	by     []float64  // This is the biais
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

	rnn.wxh = mat.NewDense(config.hiddenNeurons, config.inputNeurons, nil)
	rnn.whh = mat.NewDense(config.hiddenNeurons, config.hiddenNeurons, nil)
	rnn.why = mat.NewDense(config.hiddenNeurons, config.outputNeurons, nil)
	rnn.bh = make([]float64, config.hiddenNeurons)
	rnn.by = make([]float64, config.outputNeurons)

	wHiddenRaw := rnn.wxh.RawMatrix().Data
	wHiddenHiddenRaw := rnn.whh.RawMatrix().Data
	wOutRaw := rnn.why.RawMatrix().Data

	for _, param := range [][]float64{
		wHiddenRaw,
		wHiddenHiddenRaw,
		wOutRaw,
		rnn.bh,
		rnn.by,
	} {
		for i := range param {
			param[i] = randGen.Float64()
		}
	}

	// initialise the hidden vector to zero
	rnn.hprev = make([]float64, config.hiddenNeurons)
	return &rnn
}

// Estimate the loss function between inputs and targets
// returns the loss and gradients on model parameters
// The last hidden state is modified
func (r *rnn) loss(inputs, targets []int) (loss float64, dwxh, dwhh, dwhy, dbh, dby *mat.Dense) {
	// Do the forward pass
	// do the 1-of-k encoding of the input
	// xs is len(inputs)*len(vocabulary)
	xs := make([][]float64, len(inputs))
	// hidden state
	hs := make([][]float64, len(inputs))
	// un-normalized log probabilities for next chars
	ys := make([][]float64, len(inputs))
	// probabilities for next chars
	ps := make([][]float64, len(inputs))
	loss = 0
	for t := 0; t < len(inputs); t++ {
		xs[t] = make([]float64, r.config.inputNeurons)
		hs[t] = make([]float64, r.config.outputNeurons)
		ys[t] = make([]float64, r.config.outputNeurons)
		ps[t] = make([]float64, r.config.outputNeurons)
		xs[t][inputs[t]] = 1
		hs[t] = tanh(
			add(
				dot(r.wxh, xs[t]),
				dot(r.whh, r.hprev),
				r.bh,
			))
		r.hprev = hs[t]
		ys[t] = add(
			dot(r.why, hs[t]),
			r.bh)
		expYS := exp(ys[t])
		ps[t] = div(expYS, sum(expYS))
	}
	log.Println(loss)
	return
}

// Update the rnn with Adagrad method
func (r *rnn) adagrad(dwxh, dwhh, dwhy *mat.Dense, dbhh, dby *mat.Dense) {
	return
}
