package rnn

import (
	"math"

	"github.com/gonum/matrix/mat64"
)

// adagrad is a structure that holds the memory of the adaptative gradient
type adagrad struct {
	mwhy    *mat64.Dense //
	mwhh    *mat64.Dense // memory for the adaptative gradient updagte
	mwxh    *mat64.Dense //
	mbh     []float64    // This is the biais
	mby     []float64    // This is the biais
	epsilon float64
}

// Create a new adaptative gradient structure suitable to the rnn shape
func newAdagrad(c neuralNetConfig) *adagrad {
	a := &adagrad{}
	a.mwxh = mat64.NewDense(c.HiddenNeurons, c.InputNeurons, nil)
	a.mwhh = mat64.NewDense(c.HiddenNeurons, c.HiddenNeurons, nil)
	a.mwhy = mat64.NewDense(c.OutputNeurons, c.HiddenNeurons, nil)
	a.mbh = make([]float64, c.HiddenNeurons)
	a.mby = make([]float64, c.OutputNeurons)
	a.epsilon = c.AdagradEpsilon
	return a
}

// apply the Adaptative gradient to the rnn
func (a *adagrad) apply(r *RNN, dwxh, dwhh, dwhy *mat64.Dense, dbh, dby []float64) {
	memFunc := func(_, _ int, v float64) float64 {
		return math.Sqrt(v + a.epsilon)
	}
	learningRateFunc := func(_, _ int, v float64) float64 {
		return -r.config.LearningRate * v
	}

	for _, params := range [][3]*mat64.Dense{
		[3]*mat64.Dense{
			r.wxh, dwxh, a.mwxh,
		},
		[3]*mat64.Dense{
			r.whh, dwhh, a.mwhh,
		},
		[3]*mat64.Dense{
			r.why, dwhy, a.mwhy,
		},
		[3]*mat64.Dense{
			mat64.NewDense(len(r.bh), 1, r.bh), mat64.NewDense(len(dbh), 1, dbh), mat64.NewDense(len(a.mbh), 1, a.mbh),
		},
		[3]*mat64.Dense{
			mat64.NewDense(len(r.by), 1, r.by), mat64.NewDense(len(dby), 1, dby), mat64.NewDense(len(a.mby), 1, a.mby),
		},
	} {
		param := params[0]
		dparam := params[1]
		mem := params[2]
		tmp := new(mat64.Dense)
		tmp.MulElem(dparam, dparam)
		mem.Add(mem, tmp)
		tmp.Reset()
		tmp.Apply(memFunc, mem)
		tmp2 := new(mat64.Dense)
		tmp2.Apply(learningRateFunc, dparam)
		tmp3 := new(mat64.Dense)
		tmp3.DivElem(tmp2, tmp)
		param.Add(param, tmp3)
	}
}
