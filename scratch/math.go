package main

import (
	"math"

	"github.com/gonum/matrix/mat64"
)

// tanh applies an element-wise tanh to the parameter and returns a new vector
func tanh(x *mat64.Vector) *mat64.Vector {
	y := mat64.NewVector(x.Len(), nil)
	for i := 0; i < x.Len(); i++ {
		y.SetVec(i, math.Tanh(x.At(i, 0)))
	}
	return y
}

// dot is a matrix multiplication the returns a Vector
func dot(x mat64.Matrix, y *mat64.Vector) *mat64.Vector {
	v := mat64.NewVector(y.Len(), nil)
	v.MulVec(x, y)
	return v
}

// TODO: check the size of the vectors...
func add(x, y *mat64.Vector) *mat64.Vector {
	v := mat64.NewVector(x.Len(), nil)
	v.AddVec(x, y)
	return v
}
