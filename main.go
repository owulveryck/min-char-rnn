package main

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"os"

	"github.com/gonum/matrix/mat64"
)

func main() {
	// Open the sample text file
	file, err := os.Open("input.txt")
	if err != nil {
		log.Fatal(err)
	}
	data, runesToIx, ixToRune := getDataAndVocab(file)
	file.Close()
	dataSize := len(data)
	vocabSize := len(runesToIx)
	fmt.Printf("data has %d runes, %d unique.\n", dataSize, vocabSize)

	// hyperparameters
	hiddenSize := 100
	seqLength := 25
	//learningRate := 1e-1

	// Model parameters
	rnd := make([]float64, hiddenSize*vocabSize)
	for i := range rnd {
		rnd[i] = rand.NormFloat64()
	}
	// Input to hidden
	////wxh := mat64.NewDense(hiddenSize, vocabSize, rnd)
	// Hidden to output
	////why := mat64.NewDense(vocabSize, hiddenSize, rnd)
	rnd = make([]float64, hiddenSize*hiddenSize)
	for i := range rnd {
		rnd[i] = rand.NormFloat64() * 0.01
	}
	// Hidden to hidden
	////whh := mat64.NewDense(hiddenSize, hiddenSize, rnd)
	// hidden bias
	////bh := mat64.NewVector(hiddenSize, make([]float64, hiddenSize))
	////by := mat64.NewVector(vocabSize, make([]float64, vocabSize))

	// p is the position in the data array
	p := 0
	// n is the iteration number
	n := 0
	// hprev is a hiddenSizex1 array of initial hidden state
	hprev := mat64.NewVector(hiddenSize, make([]float64, hiddenSize))

	// Those array will contain the indexes of the current sequence
	inputs := make([]int, seqLength+1)
	targets := make([]int, seqLength+1)
	// loss at ieration 0
	smoothLoss := -math.Log(1.0/float64(vocabSize)) * float64(seqLength)
	for {
		// If we are starting or at the end of the data...
		if p+seqLength+1 >= len(data) || n == 0 {
			// reset RNN memory
			hprev = mat64.NewVector(hiddenSize, make([]float64, hiddenSize))
			p = 0 // go from start of data
		}
		for i := 0; i <= seqLength; i++ {
			inputs[i] = runesToIx[data[i+p]]
			targets[i] = runesToIx[data[i+p+1]]
		}
		// Every x iteration, sample a text
		if n%100 == 0 {
			fmt.Println("sampling...")
			sampleIx, err := sample(hprev, inputs[0], 200)
			if err != nil {
				log.Println("Sampling error", err)
			}

			var s string
			for _, ix := range sampleIx {
				s = fmt.Sprintf("%v%c", s, ixToRune[ix])
			}
			fmt.Printf("----\n %s \n----\n", s)
		}
		var loss float64
		loss, _, _, _, _, _, hprev = lossFun(inputs, targets, hprev)
		smoothLoss = smoothLoss*0.999 + loss*0.001
		// Display the progress
		if n%100 == 0 {
			fmt.Printf("iter %d, loss: %f\n", n, smoothLoss)
		}
		// update the parameters stochastic gradient descent (AdaGrad)
		//# perform parameter update with Adagrad
		//for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
		//                              [dWxh, dWhh, dWhy, dbh, dby],
		//                              [mWxh, mWhh, mWhy, mbh, mby]):
		//  mem += dparam * dparam
		//  param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

		p += seqLength
		n++
	}
}

//  inputs,targets are both list of integers.
//  hprev is Hx1 array of initial hidden state
//  returns the loss, gradients on model parameters, and last hidden state
func lossFun(inputs, targets []int, hprev *mat64.Vector) (float64, *mat64.Dense, *mat64.Dense, *mat64.Dense, *mat64.Dense, *mat64.Dense, *mat64.Vector) {
	return 0.0, nil, nil, nil, nil, nil, nil
}

// sample a sequence of integers from the model
// h is memory state, seedIx is seed letter for first time step
// length is the sample size
func sample(h *mat64.Vector, seedIx, length int) ([]int, error) {
	return []int{0}, nil
}

func getDataAndVocab(input io.Reader) ([]rune, map[rune]int, map[int]rune) {
	d, err := ioutil.ReadAll(input)
	if err != nil {
		log.Fatal(err)
	}
	// Extract the rune list
	runeToIx := make(map[rune]int)
	data := bytes.Runes(d)
	for _, v := range data {
		runeToIx[v] = 0
	}
	ixToRune := make(map[int]rune, len(runeToIx))
	i := 0
	for k := range runeToIx {
		runeToIx[k] = i
		ixToRune[i] = k
		i++
	}
	return data, runeToIx, ixToRune
}
