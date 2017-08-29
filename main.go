package main

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"log"
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
	//data, runesToIx, ixToRune := getDataAndVocab(file)
	data, runesToIx, _ := getDataAndVocab(file)
	file.Close()
	dataSize := len(data)
	vocabSize := len(runesToIx)
	fmt.Printf("data has %d runes, %d unique.\n", dataSize, vocabSize)

	// hyperparameters
	hiddenSize := 100
	//seqLength := 25
	//learningRate := 1e-1

	// Model parameters
	rnd := make([]float64, hiddenSize*vocabSize)
	for i := range rnd {
		rnd[i] = rand.NormFloat64()
	}
	wxh := mat64.NewDense(hiddenSize, vocabSize, rnd)
	why := mat64.NewDense(vocabSize, hiddenSize, rnd)
	rnd = make([]float64, hiddenSize*hiddenSize)
	for i := range rnd {
		rnd[i] = rand.NormFloat64() * 0.01
	}
	whh := mat64.NewDense(hiddenSize, hiddenSize, rnd)
	log.Println(wxh)
	log.Println(whh)
	log.Println(why)

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
