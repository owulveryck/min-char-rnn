package main

import (
	"math/rand"
	"time"

	"github.com/gonum/matrix/mat64"
	"gonum.org/v1/gonum/mat"
)

// The rnn represents the neural network
// This RNNs parameters are the three matrices whh, wxh, why.
// h is the hidden state, which is actually the memory of the RNN
type rnn struct {
	whh    *mat.Dense    // size is hiddenDimension * hiddenDimension
	wxh    *mat.Dense    //
	why    *mat.Dense    //
	hprev  *mat.VecDense // This is the last element of the hidden vector, which actually represents the memory of the RNN
	bh     *mat.Dense    // This is the biais
	by     *mat.Dense    // This is the biais
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

	// initialise the hidden vector to zero
	rnn.hprev = mat.NewVecDense(config.hiddenNeurons, nil)
	return &rnn
}

// Estimate the loss function between inputs and targets
// returns the loss and gradients on model parameters
// The last hidden state is modified
func (r *rnn) loss(inputs, targets *mat64.Vector) (loss float64, dwxh, dwhh, dwhy, dbh, dby *mat.Dense) {
	// Do the forward pass
	// do the 1-of-k encoding of the input
	xs := mat.NewDense((*inputs).Len(), r.config.inputNeurons, nil)
	hs := mat.NewDense((*inputs).Len(), r.config.hiddenNeurons, nil)
	for t := 0; t < (*inputs).Len(); t++ {
		xs.Set(t, int((*inputs).At(t, 0)), 1)
		hs.SetRow(t, tanh(
			add(
				dot(r.wxh.T(), xs.RowView(t)),
				dot(r.whh, r.hprev),
				r.bh.T(),
			).T()).RawRowView(0))
		// TODO: update hprev
	}
	return
}

// Update the rnn with Adagrad method
func (r *rnn) adagrad(dwxh, dwhh, dwhy *mat.Dense, dbhh, dby *mat.Dense) {
	return
}
