package rnn

import (
	"bytes"
	"encoding/gob"
	"log"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/gonum/matrix/mat64"
	"github.com/kelseyhightower/envconfig"
)

// RNN represents the neural network
// This RNNs parameters are the three mat64rices whh, wxh, why.
// hprev is the last known hidden vector, which is actually the memory of the RNN
// bh, and by are the biais vectors respectivly for the hidden layer and the output layer
type RNN struct {
	sync.Mutex
	whh *mat64.Dense // size is hiddenDimension * hiddenDimension
	wxh *mat64.Dense //
	why *mat64.Dense //
	// This is the last known hidden vector that represents the memory of the RNN
	// This is used only for training
	hprev  []float64
	bh     []float64 // This is the biais
	by     []float64 // This is the biais
	config neuralNetConfig
}

// GobDecode the rnn for restoring
func (rnn *RNN) GobDecode(b []byte) error {
	input := bytes.NewBuffer(b)
	dec := gob.NewDecoder(input) // Will read from network.

	type bkp struct {
		Whh *mat64.Dense // size is hiddenDimension * hiddenDimension
		Wxh *mat64.Dense //
		Why *mat64.Dense //
		// This is the last known hidden vector that represents the memory of the RNN
		// This is used only for training
		Hprev  []float64
		Bh     []float64 // This is the biais
		By     []float64 // This is the biais
		Config neuralNetConfig
	}
	var backup bkp
	err := dec.Decode(&backup)
	rnn.bh = make([]float64, len(backup.Bh))
	rnn.by = make([]float64, len(backup.By))
	rnn.hprev = make([]float64, len(backup.Hprev))
	if err == nil {
		*rnn.whh = *backup.Whh
		*rnn.why = *backup.Why
		*rnn.wxh = *backup.Wxh
		rnn.config = backup.Config
		copy(rnn.bh, backup.Bh)
		copy(rnn.by, backup.By)
		copy(rnn.hprev, backup.Hprev)
	}
	return err
}

// GobEncode the RNN for backup
func (rnn *RNN) GobEncode() ([]byte, error) {
	var output bytes.Buffer // Stand-in for a network connection

	enc := gob.NewEncoder(&output) // Will write to network.

	type bkp struct {
		Whh *mat64.Dense // size is hiddenDimension * hiddenDimension
		Wxh *mat64.Dense //
		Why *mat64.Dense //
		// This is the last known hidden vector that represents the memory of the RNN
		// This is used only for training
		Hprev  []float64
		Bh     []float64 // This is the biais
		By     []float64 // This is the biais
		Config neuralNetConfig
	}
	rnn.Lock()
	err := enc.Encode(bkp{
		rnn.whh,
		rnn.wxh,
		rnn.why,
		rnn.hprev,
		rnn.bh,
		rnn.by,
		rnn.config,
	})
	rnn.Unlock()
	return output.Bytes(), err
}

// NewRNN creates a new RNN with input size of x, outputsize of y and hidden dimension of h
// The hidden state h is initialized with the zero vectornn.
//func newRNN(x, y, h int) *RNN {
func NewRNN(inputNeurons, outputNeurons int) *RNN {
	var conf neuralNetConfig
	if inputNeurons == 0 || outputNeurons == 0 {
		err := envconfig.Usage("RNN", &conf)
		if err != nil {
			log.Fatal(err)
		}
		return nil
	}
	err := envconfig.Process("RNN", &conf)
	if err != nil {
		log.Fatal(err)
	}

	//func NewRNN(config NeuralNetConfig) *RNN {
	var rnn RNN
	conf.InputNeurons = inputNeurons
	conf.OutputNeurons = outputNeurons
	rnn.config = conf
	// Initialize biases/weights.

	rnn.wxh = mat64.NewDense(conf.HiddenNeurons, conf.InputNeurons, nil)
	rnn.whh = mat64.NewDense(conf.HiddenNeurons, conf.HiddenNeurons, nil)
	rnn.why = mat64.NewDense(conf.OutputNeurons, conf.HiddenNeurons, nil)
	rnn.bh = make([]float64, conf.HiddenNeurons)
	rnn.by = make([]float64, conf.OutputNeurons)
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
			param[i] = randGen.NormFloat64() * conf.RandomFactor
		}
	}

	// initialise the hidden vector to zero
	rnn.hprev = make([]float64, conf.HiddenNeurons)

	return &rnn
}

// RNNs have a deceptively simple API:
// They accept an input vector x and give you an output vector y.
// However, crucially this output vector’s contents are influenced
// not only by the input you just fed in,
// but also on the entire history of inputs you’ve fed in in the past.
// Written as a class, the RNN’s API consists of a single step function:
func (rnn *RNN) step(x, hprev []float64) (y, h []float64) {
	h = tanh(
		add(
			dot(rnn.wxh, x),
			dot(rnn.whh, hprev),
			rnn.bh,
		))
	y = add(
		dot(rnn.why, h),
		rnn.by)
	return
}

// forwardPass takes a mat64rix of inputs and returns
// the corresponding outputs mat64rix
// and a mat64rix of the hidden states that will be used
// for the backpropagation
func (rnn *RNN) forwardPass(xs [][]float64, hprev []float64) (ys, hs [][]float64) {
	inputSize := len(xs)
	hp := make([]float64, len(hprev))
	copy(hp, hprev)
	// un-normalized log probabilities for next chars
	ys = make([][]float64, inputSize)
	hs = make([][]float64, inputSize)
	for t := 0; t < inputSize; t++ {
		// Initialization of the arrays
		ys[t] = make([]float64, rnn.config.OutputNeurons)
		hs[t] = make([]float64, rnn.config.HiddenNeurons)
		ys[t], hs[t] = rnn.step(xs[t], hp)
		hp = hs[t]
	}
	return
}

// Do a backpropagation of the RNNs and returns the derivates
// xs is the input mat64rix
// ts is the target mat64rices
// ps is the normalized log probability
// hs is a mat64rix of hidden vector
func (rnn *RNN) backPropagation(xs, ps, hs, ts [][]float64) (dwxh, dwhh, dwhy *mat64.Dense, dbh, dby []float64) {
	inputSize := len(xs)
	dwxh = mat64.NewDense(rnn.config.HiddenNeurons, rnn.config.InputNeurons, nil)
	dwhh = mat64.NewDense(rnn.config.HiddenNeurons, rnn.config.HiddenNeurons, nil)
	dwhy = mat64.NewDense(rnn.config.OutputNeurons, rnn.config.HiddenNeurons, nil)
	dhnext := make([]float64, rnn.config.OutputNeurons)
	dbh = make([]float64, rnn.config.HiddenNeurons)
	dby = make([]float64, rnn.config.OutputNeurons)
	dhraw := make([]float64, len(hs[0]))

	for t := inputSize - 1; t >= 0; t-- {
		dy := make([]float64, rnn.config.OutputNeurons)
		for i := range ps[t] {
			dy[i] = ps[t][i] - ts[t][i]
		}
		dwhy.Add(dwhy,
			dotVec(dy, hs[t]),
		)
		dby = add(dby, dy)

		dh := add(
			dot(rnn.why.T(), dy),
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
		dhnext = dot(rnn.whh.T(), dhraw)
	}

	return
}

// TrainingSet represents an input mat64rix and the expected
// result when passed through a rnn
type TrainingSet struct {
	Inputs  [][]float64
	Targets [][]float64
}

// Train the network.
// The train mechanisme is launched in a seperate go-routine
// it is waiting for an input to be sent in the feeding channel
func (rnn *RNN) Train() (feed chan TrainingSet, info chan float64) {
	feed = make(chan TrainingSet, 1)
	info = make(chan float64, 1)

	adagrad := newAdagrad(rnn.config)
	go func(feed chan TrainingSet, info chan float64) {
		// When we have new data
		for tset := range feed {
			// Forward pass
			xs := make([][]float64, len(tset.Inputs))
			copy(xs, tset.Inputs)
			ts := make([][]float64, len(tset.Targets))
			copy(ts, tset.Targets)
			rnn.Lock()
			ys, hs := rnn.forwardPass(xs, rnn.hprev)
			// Save the last state for future training
			copy(rnn.hprev, hs[len(hs)-1])
			//rnn.hprev = hs[len(hs)-1]
			rnn.Unlock()
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
			// Send info on a non blocking channel
			select {
			case info <- loss:
			default:
			}

			// Backpass
			rnn.Lock()
			dwxh, dwhh, dwhy, dbh, dby := rnn.backPropagation(xs, ps, hs, ts)
			rnn.Unlock()
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
			rnn.Lock()
			adagrad.apply(rnn, dwxh, dwhh, dwhy, dbh, dby)
			rnn.Unlock()
		}
	}(feed, info)
	return feed, info
}

// Predict n element of  output that corresponds to the input xs
// At every iteration, the output is processed by the adapt function
func (rnn *RNN) Predict(xs [][]float64, n int, adapt func([]float64) []float64) [][]float64 {
	ys := make([][]float64, n+len(xs))
	h := make([]float64, len(rnn.hprev))
	y := make([]float64, rnn.config.OutputNeurons)
	for i := 0; i < n+len(xs); i++ {
		x := make([]float64, rnn.config.InputNeurons)
		if i < len(xs) {
			copy(x, xs[i])
		} else {
			copy(x, ys[i-1])
		}

		yr, hr := rnn.step(x, h)
		copy(y, yr)
		copy(h, hr)
		expY := exp(y)
		p := div(expY, sum(expY))
		ys[i] = p
		if i < len(xs) {
			for j := 0; j < len(xs[i]); j++ {
				if xs[i][j] == float64(1) {
					ys[i][j] = float64(1)
				}
			}
		} else {
			ys[i] = adapt(p)
		}
	}
	res := make([][]float64, n)
	copy(res, ys[len(xs):])
	return res
}

// Sample the rnn
func (rnn *RNN) Sample(xs [][]float64, n int, choose func([]float64) int) []int {
	res := make([]int, n+len(xs))
	h := make([]float64, len(rnn.hprev))
	//copy(h, rnn.hprev)
	y := make([]float64, rnn.config.OutputNeurons)
	for i := 0; i < n; i++ {
		x := make([]float64, rnn.config.InputNeurons)
		if i < len(xs) {
			copy(x, xs[i])
		} else {
			x[res[i-1]] = 1
		}

		yr, hr := rnn.step(x, h)
		copy(y, yr)
		copy(h, hr)
		expY := exp(y)
		p := div(expY, sum(expY))
		if i < len(xs) {
			for j := 0; j < len(xs[i]); j++ {
				if xs[i][j] == float64(1) {
					res[i] = j
				}
			}
		} else {
			res[i] = choose(p)
		}
	}

	return res
}
