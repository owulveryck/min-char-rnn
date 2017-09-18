package main

import (
	"bytes"
	"io"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/gonum/matrix/mat64"
)

type rnnType struct {
	vocabSize    int // the size of the vocabulary
	hiddenSize   int // the size of the hidden layer
	bpttTruncate int
	u            *mat64.Dense // size is hiddenSize * vocabSize
	v            *mat64.Dense // size is vocabSize * hiddenSize
	w            *mat64.Dense // size is hiddenSize * hiddenSize
}

// Implementation of http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
func main() {
	// Open the sample text file
	file, err := os.Open("input.txt")
	if err != nil {
		log.Fatal(err)
	}
	//data, runesToIx, ixToRune := getDataAndVocab(file)
	_, runesToIx, _ := getDataAndVocab(file)
	file.Close()
	//dataSize := len(data)
	//fmt.Printf("data has %d runes, %d unique.\n", dataSize, vocabSize)

	rnn := &rnnType{
		vocabSize:  len(runesToIx),
		hiddenSize: 100,
	}
	// Initialize the matrices with random parameters o
	// Initialize the random seed
	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	uData := make([]float64, rnn.vocabSize*rnn.hiddenSize)
	for i := range uData {
		uData[i] = r.NormFloat64()
	}
	rnn.u = mat64.NewDense(rnn.hiddenSize, rnn.vocabSize, uData)

	vData := make([]float64, rnn.vocabSize*rnn.hiddenSize)
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	for i := range vData {
		vData[i] = r.NormFloat64()
	}
	rnn.v = mat64.NewDense(rnn.vocabSize, rnn.hiddenSize, vData)

	wData := make([]float64, rnn.hiddenSize*rnn.hiddenSize)
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	for i := range wData {
		wData[i] = r.NormFloat64()
	}
	rnn.w = mat64.NewDense(rnn.hiddenSize, rnn.hiddenSize, wData)
	log.Println(rnn)

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
