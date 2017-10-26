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
type rnn struct {
	whh    *mat.Dense // size is hiddenDimension * hiddenDimension
	wxh    *mat.Dense //
	mwhy   *mat.Dense //
	mwhh   *mat.Dense // memory for the adaptative gradient updagte
	mwxh   *mat.Dense //
	why    *mat.Dense //
	hprev  []float64  // This is the last element of the hidden vector, which actually represents the memory of the RNN
	bh     []float64  // This is the biais
	by     []float64  // This is the biais
	mbh    []float64  // This is the biais
	mby    []float64  // This is the biais
	config neuralNetConfig
	hs     [][]float64
	ys     [][]float64
	ps     [][]float64
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
	rnn.mwxh = mat.NewDense(config.hiddenNeurons, config.inputNeurons, nil)
	rnn.mwhh = mat.NewDense(config.hiddenNeurons, config.hiddenNeurons, nil)
	rnn.mwhy = mat.NewDense(config.outputNeurons, config.hiddenNeurons, nil)
	rnn.mbh = make([]float64, config.hiddenNeurons)
	rnn.mby = make([]float64, config.outputNeurons)

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
	rnn.hprev = make([]float64, config.hiddenNeurons)

	// hidden state
	rnn.hs = make([][]float64, config.inputNeurons)
	// un-normalized log probabilities for next chars
	rnn.ys = make([][]float64, config.inputNeurons)
	// probabilities for next chars
	rnn.ps = make([][]float64, config.inputNeurons)

	return &rnn
}

func (r *rnn) step(x []float64) (y []float64) {
	return nil
}

// Estimate the loss function between inputs and targets
// returns the loss and gradients on model parameters
// The last hidden state is modified
func (r *rnn) loss(inputs, targets []int) (loss float64, dwxh, dwhh, dwhy *mat.Dense, dbh, dby []float64) {
	wg := sync.WaitGroup{}
	// Do the forward pass
	// do the 1-of-k encoding of the input
	// xs is len(inputs)*len(vocabulary)
	xs := make([][]float64, r.config.inputNeurons)
	for t := 0; t < len(inputs); t++ {
		xs[t] = make([]float64, r.config.inputNeurons)
		xs[t][inputs[t]] = 1
	}
	loss = 0
	for t := 0; t < len(inputs); t++ {
		r.hs[t] = make([]float64, r.config.outputNeurons)
		r.ys[t] = make([]float64, r.config.outputNeurons)
		r.ps[t] = make([]float64, r.config.outputNeurons)
		if t > 0 {
			r.hs[t] = tanh(
				add(
					dot(r.wxh, xs[t]),
					dot(r.whh, r.hs[t-1]),
					r.bh,
				))

		} else {
			r.hs[t] = tanh(
				add(
					dot(r.wxh, xs[t]),
					dot(r.whh, r.hprev),
					r.bh,
				))
		}
		r.ys[t] = add(
			dot(r.why, r.hs[t]),
			r.by)
		expYS := exp(r.ys[t])
		r.ps[t] = div(expYS, sum(expYS))
		// softmax (cross-entropy loss)
		loss -= math.Log(r.ps[t][targets[t]])
	}
	r.hprev = r.hs[len(inputs)-1]
	// backward pass: compute gradients going backwards

	dwxh = mat.NewDense(r.config.hiddenNeurons, r.config.inputNeurons, nil)
	dwhh = mat.NewDense(r.config.hiddenNeurons, r.config.hiddenNeurons, nil)
	dwhy = mat.NewDense(r.config.outputNeurons, r.config.hiddenNeurons, nil)
	dhnext := make([]float64, r.config.outputNeurons)
	dbh = make([]float64, r.config.hiddenNeurons)
	dby = make([]float64, r.config.outputNeurons)
	dhraw := make([]float64, len(r.hs[0]))

	for t := len(inputs) - 1; t >= 0; t-- {
		dy := make([]float64, r.config.outputNeurons)
		copy(dy, r.ps[t])
		dy[targets[t]]--
		dwhy.Add(dwhy,
			dotVec(dy, r.hs[t]),
		)
		dby = add(dby, dy)
		dh := add(
			dot(r.why.T(), dy),
			dhnext,
		)

		for i := range r.hs[t] {
			dhraw[i] = (1 - r.hs[t][i]*r.hs[t][i]) * dh[i]
		}

		dbh = add(dbh, dhraw)
		dwxh.Add(dwxh, dotVec(dhraw, xs[t]))
		if t == 0 {
			dwhh.Add(dwhh, dotVec(dhraw, r.hs[len(inputs)-1]))
		} else {
			dwhh.Add(dwhh, dotVec(dhraw, r.hs[t-1]))
		}
		dhnext = dot(r.whh.T(), dhraw)
	}
	// Clip to mitigate exploding gradients
	for _, param := range [][]float64{
		dwxh.RawMatrix().Data,
		dwhh.RawMatrix().Data,
		dwhy.RawMatrix().Data,
		dby,
		dbh,
	} {
		wg.Add(1)
		go func() {
			for i := range param {
				if param[i] > 5 {
					param[i] = 5
				}
				if param[i] < -5 {
					param[i] = -5
				}
			}
			wg.Done()
		}()
	}
	wg.Wait()
	return
}

// Update the rnn with Adagrad method
func (r *rnn) adagrad(dwxh, dwhh, dwhy *mat.Dense, dbh, dby []float64) {
	memFunc := func(_, _ int, v float64) float64 {
		return math.Sqrt(v + 1e-8)
	}
	learningRateFunc := func(_, _ int, v float64) float64 {
		return -r.config.learningRate * v
	}

	wg := sync.WaitGroup{}
	for _, params := range [][3]*mat.Dense{
		[3]*mat.Dense{
			r.wxh, dwxh, r.mwxh,
		},
		[3]*mat.Dense{
			r.whh, dwhh, r.mwhh,
		},
		[3]*mat.Dense{
			r.why, dwhy, r.mwhy,
		},
		[3]*mat.Dense{
			mat.NewDense(len(r.bh), 1, r.bh), mat.NewDense(len(dbh), 1, dbh), mat.NewDense(len(r.mbh), 1, r.mbh),
		},
		[3]*mat.Dense{
			mat.NewDense(len(r.by), 1, r.by), mat.NewDense(len(dby), 1, dby), mat.NewDense(len(r.mby), 1, r.mby),
		},
	} {
		wg.Add(1)
		go func() {
			param := params[0]
			dparam := params[1]
			mem := params[2]
			tmp := new(mat.Dense)
			tmp.MulElem(dparam, dparam)
			mem.Add(mem, tmp)
			tmp.Reset()
			tmp.Apply(memFunc, mem)
			tmp2 := new(mat.Dense)
			tmp2.Apply(learningRateFunc, dparam)
			tmp3 := new(mat.Dense)
			tmp3.DivElem(tmp2, tmp)
			param.Add(param, tmp3)
			wg.Done()
		}()
	}
	wg.Wait()
}

func (r *rnn) sample(seed, n int) []int {
	res := make([]int, n)
	h := make([]float64, len(r.hprev))
	copy(h, r.hprev)
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
