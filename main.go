package main

import (
	"bufio"
	"io"
	"log"
	"os"

	"github.com/gonum/matrix/mat64"
)

func main() {
	seqLength := 25
	runesToIx, _, err := getVocabIndexesFromFile("data/vocab.txt")
	if err != nil {
		log.Fatal(err)
	}
	// Define our network architecture and learning parameters.
	config := neuralNetConfig{
		inputNeurons:  len(runesToIx), // the input is the size of the vocabulary
		outputNeurons: len(runesToIx), // the output size is also the size of the vocablulary
		hiddenNeurons: 100,
		numEpochs:     100,
		memorySize:    25, // This corresponds to seq_length in the initial implementation
		learningRate:  0.3,
	}

	// Create a new RNNs
	// the first argument is the size of the input (the size of the vocabulary)
	// the second input is the size of the output vector (which is also the size of the vocabulary)
	// the lase argument is the size of the hidden layer
	rnn := newRNN(config)
	// Open the sample text file
	data, err := os.Open("data/input.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer data.Close()

	r := bufio.NewReader(data)
	i := 0
	x := mat64.NewVector(seqLength, nil)
	for epoch := 0; epoch < config.numEpochs; epoch++ {
		log.Println("Epoch: ", epoch)
		if _, err := data.Seek(10, io.SeekStart); err != nil {
			log.Fatal(err)
		}

		// Do the batch processing
		for {
			// Reading the file one rune at a time
			if c, _, err := r.ReadRune(); err != nil {
				if err == io.EOF {
					break
				} else {
					log.Fatal(err)
				}
			} else {
				// Now filling the input vector with the index of the rune
				x.SetVec(i, float64(runesToIx[c]))
				i++
				if i%seqLength == 0 {
					// The vector is complete, evaluate the lossFunction and perform the parameters adaptation
					_, dwxh, dwhh, dwhy, dbh, dby := rnn.loss(x, x)
					rnn.adagrad(dwxh, dwhh, dwhy, dbh, dby)
					i = 0
				}
			}
		}
	}
}
