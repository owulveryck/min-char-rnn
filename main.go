package main

import (
	"log"
	"os"

	"github.com/gonum/matrix/mat64"
)

// Implementation of http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
func main() {
	// Open the sample text file
	file, err := os.Open("data/input.txt")
	if err != nil {
		log.Fatal(err)
	}
	//data, runesToIx, ixToRune := getDataAndVocab(file)
	_, runesToIx, _ := getDataAndVocab(file)
	file.Close()
	//dataSize := len(data)
	//fmt.Printf("data has %d runes, %d unique.\n", dataSize, vocabDimension)

	// Create a new RNNs
	// the first argument is the size of the input (the size of the vocabulary)
	// the second input is the size of the output vector (which is also the size of the vocabulary)
	// the lase argument is the size of the hidden layer
	_ = newRNN(len(runesToIx), len(runesToIx), 100)

}

// inputs,targets are both list of integers.
// hprev is Hx1 array of initial hidden state
// returns the loss, gradients on model parameters, and last hidden state
func lossFun(inputs, targets []int, hprev *mat64.Vector) (float64, *mat64.Dense, *mat64.Dense, *mat64.Dense, *mat64.Dense, *mat64.Dense, *mat64.Vector) {
	return 0.0, nil, nil, nil, nil, nil, nil
}

// sample a sequence of integers from the model
// h is memory state, seedIx is seed letter for first time step
// length is the sample size
func sample(h *mat64.Vector, seedIx, length int) ([]int, error) {
	return []int{0}, nil
}
