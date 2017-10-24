package main

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func tanh(a mat.Matrix) *mat.Dense {
	t := new(mat.Dense)
	applyTanh := func(_, _ int, v float64) float64 { return math.Tanh(v) }
	t.Apply(applyTanh, a)
	return t
}
func dot(a, b mat.Matrix) *mat.Dense {
	t := new(mat.Dense)
	t.Mul(a, b)
	return t
}

func exp(a mat.Matrix) *mat.Dense {
	t := new(mat.Dense)
	applyExp := func(_, _ int, v float64) float64 { return math.Exp(v) }
	t.Apply(applyExp, a)
	return t
}

func add(a ...mat.Matrix) *mat.Dense {
	t := new(mat.Dense)
	for _, m := range a {
		if t.IsZero() {
			t = mat.DenseCopyOf(m)
		} else {
			t.Add(t, m)
		}
	}
	return t
}
