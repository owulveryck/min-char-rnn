package main

import (
	"math"
	"math/rand"
	"sync"
	"time"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

// The rnn represents the neural network
// This RNNs parameters are the three matrices whh, wxh, why.
// h is the hidden state, which is actually the memory of the RNN
// bh, and by are the biais vectors respectivly for the hidden layer and the output layer
type rnn struct {
	whh    *mat.Dense // size is hiddenDimension * hiddenDimension
	wxh    *mat.Dense //
	why    *mat.Dense //
	h      []float64  // This is the hidden vector that represents the memory of the RNN
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

	rnn.wxh = mat.NewDense(config.hiddenNeurons, config.inputNeurons, nil)
	rnn.whh = mat.NewDense(config.hiddenNeurons, config.hiddenNeurons, nil)
	rnn.why = mat.NewDense(config.outputNeurons, config.hiddenNeurons, nil)
	rnn.bh = make([]float64, config.hiddenNeurons)
	rnn.by = make([]float64, config.outputNeurons)
	wHiddenRaw := rnn.wxh.RawMatrix().Data
	wHiddenHiddenRaw := rnn.whh.RawMatrix().Data
	wOutRaw := rnn.why.RawMatrix().Data

	for _, param := range [][]float64{
		wHiddenRaw,
		wHiddenHiddenRaw,
		wOutRaw,
	} {
		for i := range param {
			randSource := rand.NewSource(time.Now().UnixNano())
			randGen := rand.New(randSource)
			param[i] = randGen.NormFloat64() * 0.01
		}
	}

	// initialise the hidden vector to zero
	rnn.h = make([]float64, config.hiddenNeurons)

	return &rnn
}

// RNNs have a deceptively simple API:
// They accept an input vector x and give you an output vector y.
// However, crucially this output vector’s contents are influenced
// not only by the input you just fed in,
// but also on the entire history of inputs you’ve fed in in the past.
// Written as a class, the RNN’s API consists of a single step function:
func (r *rnn) step(x []float64) (y []float64) {
	r.h = tanh(
		add(
			dot(r.wxh, x),
			dot(r.whh, r.h),
			r.bh,
		))
	return add(
		dot(r.why, r.h),
		r.by)
}

// forwardPass takes a matrix of inputs and returns
// the corresponding outputs matrix
// and a matrix of the hidden states that will be used
// for the backpropagation
func (r *rnn) forwardPass(xs [][]float64) (ys, hs [][]float64) {
	inputSize := len(xs)
	// un-normalized log probabilities for next chars
	ys = make([][]float64, inputSize)
	hs = make([][]float64, inputSize)
	for t := 0; t < inputSize; t++ {
		// Initialization of the arrays
		ys[t] = make([]float64, r.config.outputNeurons)
		hs[t] = make([]float64, r.config.hiddenNeurons)
		ys[t] = r.step(xs[t])
		hs[t] = r.h
	}
	return
}

// Do a backpropagation of the RNNs and returns the derivates
// xs is the input matrix
// ts is the target matrices
// ps is the normalized log probability
// hs is a matrix of hidden vector
func (r *rnn) backPropagation(xs, ps, hs, ts [][]float64) (dwxh, dwhh, dwhy *mat.Dense, dbh, dby []float64) {
	inputSize := len(xs)
	dwxh = mat.NewDense(r.config.hiddenNeurons, r.config.inputNeurons, nil)
	dwhh = mat.NewDense(r.config.hiddenNeurons, r.config.hiddenNeurons, nil)
	dwhy = mat.NewDense(r.config.outputNeurons, r.config.hiddenNeurons, nil)
	dhnext := make([]float64, r.config.outputNeurons)
	dbh = make([]float64, r.config.hiddenNeurons)
	dby = make([]float64, r.config.outputNeurons)
	dhraw := make([]float64, len(hs[0]))

	for t := inputSize - 1; t >= 0; t-- {
		dy := make([]float64, r.config.outputNeurons)
		for i := range ps[t] {
			dy[i] = ps[t][i] - ts[t][i]
		}
		dwhy.Add(dwhy,
			dotVec(dy, hs[t]),
		)
		dby = add(dby, dy)
		dh := add(
			dot(r.why.T(), dy),
			dhnext,
		)

		for i := range hs[t] {
			dhraw[i] = (1 - hs[t][i]*hs[t][i]) * dh[i]
		}

		dbh = add(dbh, dhraw)
		dwxh.Add(dwxh, dotVec(dhraw, xs[t]))
		if t == 0 {
			dwhh.Add(dwhh, dotVec(dhraw, hs[inputSize-1]))
		} else {
			dwhh.Add(dwhh, dotVec(dhraw, hs[t-1]))
		}
		dhnext = dot(r.whh.T(), dhraw)
	}

	return
}

// Estimate the loss function between inputs and targets
// returns the loss and gradients on model parameters
// The last hidden state is modified
func (r *rnn) loss(inputs, targets []int) (loss float64, dwxh, dwhh, dwhy *mat.Dense, dbh, dby []float64) {
	wg := sync.WaitGroup{}
	inputSize := len(inputs)
	outputSize := len(targets)
	// Do the forward pass
	// do the 1-of-k encoding of the input and the target
	// xs is len(inputs)*len(vocabulary)
	xs := make([][]float64, inputSize)
	ts := make([][]float64, outputSize)
	for t := 0; t < len(inputs); t++ {
		xs[t] = make([]float64, r.config.inputNeurons)
		xs[t][inputs[t]] = 1
		ts[t] = make([]float64, r.config.outputNeurons)
		ts[t][targets[t]] = 1
	}
	ys, hs := r.forwardPass(xs)
	ps := normalizeByRow(ys)
	loss = 0
	// evaluate the loss softmax (cross-entropy loss)
	for t := 0; t < inputSize; t++ {
		loss -= math.Log(ps[t][targets[t]])
	}

	dwxh, dwhh, dwhy, dbh, dby = r.backPropagation(xs, ps, hs, ts)
	// Clip to mitigate exploding gradients
	for _, param := range [][]float64{
		dwxh.RawMatrix().Data,
		dwhh.RawMatrix().Data,
		dwhy.RawMatrix().Data,
		dby,
		dbh,
	} {
		wg.Add(1)
		go func(param []float64) {
			for i := range param {
				if param[i] > 5 {
					param[i] = 5
				}
				if param[i] < -5 {
					param[i] = -5
				}
			}
			wg.Done()
		}(param)
	}
	wg.Wait()
	return
}

func (r *rnn) sample(seed, n int) []int {
	res := make([]int, n)
	h := make([]float64, len(r.h))
	copy(h, r.h)
	for i := 0; i < n; i++ {
		x := make([]float64, r.config.inputNeurons)
		if i == 0 {
			x[seed] = 1
		} else {
			x[res[i-1]] = 1
		}

		h = tanh(
			add(
				dot(r.wxh, x),
				dot(r.whh, h),
				r.bh,
			),
		)
		y := add(
			dot(r.why, h),
			r.by,
		)
		expY := exp(y)
		p := div(expY, sum(expY))
		// find the best match
		sample := distuv.NewCategorical(p, rand.New(rand.NewSource(time.Now().UnixNano())))
		bestIdx := int(sample.Rand())
		res[i] = bestIdx
	}
	return res
}
