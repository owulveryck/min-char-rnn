package main

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"time"
)

func main() {
	runesToIx, ixToRunes, err := getVocabIndexesFromFile("data/vocab.txt")
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
		learningRate:  1e-1,
	}

	// Create a new RNNs
	rnn := newRNN(config)
	// Open the sample text file
	data, err := os.Open("data/input.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer data.Close()

	r := bufio.NewReader(data)
	i := 0
	inputs := make([]int, config.memorySize+1)
	smoothLoss := -math.Log10(float64(1)/float64(config.inputNeurons)) * float64(config.memorySize)
	for epoch := 0; epoch < config.numEpochs; epoch++ {
		if _, err := data.Seek(0, io.SeekStart); err != nil {
			log.Fatal(err)
		}

		// Do the batch processing
		n := 0
		rnn.hprev = make([]float64, config.hiddenNeurons)
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
				inputs[i] = runesToIx[c]
				i++
				if i%(config.memorySize+1) == 0 {
					// The vector is complete, evaluate the lossFunction and perform the parameters adaptation
					loss, dwxh, dwhh, dwhy, dbh, dby := rnn.loss(inputs[0:config.memorySize], inputs[1:config.memorySize+1])
					smoothLoss = smoothLoss*0.999 + loss*0.001
					if n%100 == 0 {
						log.Printf("Epoch %v, iter %d, loss: %f", epoch, n, smoothLoss)
					}
					rnn.adagrad(dwxh, dwhh, dwhy, dbh, dby)
					// Do not loose the last element of the vector
					inputs[0] = inputs[config.memorySize]
					// the first index is already set, let's restart in the second position
					i = 1
					n++
					if n%1000 == 0 {
						rand.Seed(time.Now().UnixNano())
						seed := rand.Intn(config.inputNeurons)
						fmt.Printf("%c", ixToRunes[seed])

						index := rnn.sample(seed, 1000)
						for _, idx := range index {
							fmt.Printf("%c", ixToRunes[idx])
						}
						fmt.Println("")

					}
				}
			}
		}
	}
}
