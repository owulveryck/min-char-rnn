package rnn

import (
	"math"
	"math/rand"
	"sync"
	"time"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

// RNN represents the neural network
// This RNNs parameters are the three matrices whh, wxh, why.
// h is the hidden state, which is actually the memory of the RNN
// bh, and by are the biais vectors respectivly for the hidden layer and the output layer
type RNN struct {
	sync.Mutex
	whh        *mat.Dense // size is hiddenDimension * hiddenDimension
	wxh        *mat.Dense //
	why        *mat.Dense //
	h          []float64  // This is the hidden vector that represents the memory of the RNN
	bh         []float64  // This is the biais
	by         []float64  // This is the biais
	config     NeuralNetConfig
	adagrad    *adagrad
	smoothLoss float64
}

// NeuralNetConfig defines our neural network
// architecture and learning parameters.
type NeuralNetConfig struct {
	InputNeurons  int
	OutputNeurons int
	HiddenNeurons int
	MemorySize    int
	NumEpochs     int
	LearningRate  float64
}

// NewRNN creates a new RNN with input size of x, outputsize of y and hidden dimension of h
// The hidden state h is initialized with the zero vector.
//func newRNN(x, y, h int) *RNN {
func NewRNN(config NeuralNetConfig) *RNN {
	var rnn RNN
	rnn.config = config
	// Initialize biases/weights.

	rnn.wxh = mat.NewDense(config.HiddenNeurons, config.InputNeurons, nil)
	rnn.whh = mat.NewDense(config.HiddenNeurons, config.HiddenNeurons, nil)
	rnn.why = mat.NewDense(config.OutputNeurons, config.HiddenNeurons, nil)
	rnn.bh = make([]float64, config.HiddenNeurons)
	rnn.by = make([]float64, config.OutputNeurons)
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
			param[i] = randGen.NormFloat64() * 1e-4
		}
	}

	// initialise the hidden vector to zero
	rnn.h = make([]float64, config.HiddenNeurons)
	// Initialise the adaptative gradient object
	rnn.adagrad = newAdagrad(config)
	rnn.smoothLoss = -math.Log10(float64(1)/float64(config.InputNeurons)) * float64(config.MemorySize)

	return &rnn
}

// RNNs have a deceptively simple API:
// They accept an input vector x and give you an output vector y.
// However, crucially this output vector’s contents are influenced
// not only by the input you just fed in,
// but also on the entire history of inputs you’ve fed in in the past.
// Written as a class, the RNN’s API consists of a single step function:
func (r *RNN) step(x []float64) (y []float64) {
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
func (r *RNN) forwardPass(xs [][]float64) (ys, hs [][]float64) {
	inputSize := len(xs)
	// un-normalized log probabilities for next chars
	ys = make([][]float64, inputSize)
	hs = make([][]float64, inputSize)
	for t := 0; t < inputSize; t++ {
		// Initialization of the arrays
		ys[t] = make([]float64, r.config.OutputNeurons)
		hs[t] = make([]float64, r.config.HiddenNeurons)
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
func (r *RNN) backPropagation(xs, ps, hs, ts [][]float64) (dwxh, dwhh, dwhy *mat.Dense, dbh, dby []float64) {
	inputSize := len(xs)
	dwxh = mat.NewDense(r.config.HiddenNeurons, r.config.InputNeurons, nil)
	dwhh = mat.NewDense(r.config.HiddenNeurons, r.config.HiddenNeurons, nil)
	dwhy = mat.NewDense(r.config.OutputNeurons, r.config.HiddenNeurons, nil)
	dhnext := make([]float64, r.config.OutputNeurons)
	dbh = make([]float64, r.config.HiddenNeurons)
	dby = make([]float64, r.config.OutputNeurons)
	dhraw := make([]float64, len(hs[0]))

	for t := inputSize - 1; t >= 0; t-- {
		dy := make([]float64, r.config.OutputNeurons)
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

// TrainingSet represents an input matrix and the expected
// result when passed through a rnn
type TrainingSet struct {
	Inputs  [][]float64
	Targets [][]float64
}

// GetSmoothLoss returns the current loss of the RNN
func (r *RNN) GetSmoothLoss() float64 {
	r.Lock()
	sl := r.smoothLoss
	r.Unlock()
	return sl
}

// Train the network.
// The train mechanisme is launched in a seperate go-routine
// it is waiting for an input to be sent in the feeding channel
func (r *RNN) Train() (feed chan TrainingSet) {
	feed = make(chan TrainingSet, 1)
	go func(feed chan TrainingSet) {
		// When we have new data
		for tset := range feed {
			// Forward pass
			xs := tset.Inputs
			ts := tset.Targets
			r.Lock()
			ys, hs := r.forwardPass(xs)
			r.Unlock()
			ps := normalizeByRow(ys)
			// Loss evaluation
			loss := float64(0)
			for t := 0; t < len(ps); t++ {
				l := float64(0)
				for i := 0; i < len(ps[t]); i++ {
					l += ps[t][i] * ts[t][i]
				}
				loss -= math.Log(l)
			}
			r.Lock()
			r.smoothLoss = r.smoothLoss*0.999 + loss*0.001
			r.Unlock()

			// Backpass
			r.Lock()
			dwxh, dwhh, dwhy, dbh, dby := r.backPropagation(xs, ps, hs, ts)
			r.Unlock()
			// Clip to mitigate exploding gradients
			for _, param := range [][]float64{
				dwxh.RawMatrix().Data,
				dwhh.RawMatrix().Data,
				dwhy.RawMatrix().Data,
				dby,
				dbh,
			} {
				func(param []float64) {
					for i := range param {
						if param[i] > 5 {
							param[i] = 5
						}
						if param[i] < -5 {
							param[i] = -5
						}
					}
				}(param)
			}
			// Adaptation
			r.Lock()
			r.adagrad.apply(r, dwxh, dwhh, dwhy, dbh, dby)
			r.Unlock()
		}
	}(feed)
	return feed
}

// Sample the rnn
// TODO: this function needs to be rewritten
func (r *RNN) Sample(seed, n int) []int {
	r.Lock()
	res := make([]int, n)
	h := make([]float64, len(r.h))
	copy(h, r.h)
	for i := 0; i < n; i++ {
		x := make([]float64, r.config.InputNeurons)
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

	r.Unlock()
	return res
}
