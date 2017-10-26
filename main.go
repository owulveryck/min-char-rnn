package main

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"
	"time"
)

func main() {
	runesToIx, ixToRunes, err := getVocabIndexesFromFile("data/vocab.txt")
	if err != nil {
		log.Fatal(err)
	}
	maxEpoch := 100
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
	// Triggering the Training
	feed := rnn.Train()
	// Open the sample text file
	data, err := os.Open("data/input.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer data.Close()

	r := bufio.NewReader(data)
	tset := TrainingSet{
		inputs:  make([][]float64, config.memorySize),
		targets: make([][]float64, config.memorySize),
	}

	n := 0
	epoch := 1
	for {
		// Filling a training set
		for i := 0; i < config.memorySize+1; i++ {
			// Create the 1-of-k encoder vector
			oneOfK := make([]float64, config.inputNeurons)
			// Read a character
			if c, _, err := r.ReadRune(); err != nil {
				if err == io.EOF {
					// Restart the training if it's not the last epoch
					if epoch < maxEpoch {
						epoch++
						break
					} else {
						return
					}
				} else {
					log.Fatal(err)
				}
			} else {
				oneOfK[runesToIx[c]] = 1
			}

			switch i {
			case 0:
				tset.inputs[i] = oneOfK
			case config.memorySize:
				tset.targets[i-1] = oneOfK
			default:
				var copyOfOneOfK []float64
				copy(copyOfOneOfK, oneOfK)
				tset.inputs[i] = oneOfK
				tset.targets[i-1] = oneOfK
			}
		}
		// Feeding the network
		feed <- tset
		if n%100 == 0 {
			log.Printf("Epoch %v, iteration: %v, loss: %v\n", epoch, n, rnn.GetSmoothLoss())
		}
		if n%1000 == 0 {
			sampling(rnn, config.inputNeurons, ixToRunes)
		}
		n++
	}
}

func sampling(rnn *rnn, vocabSize int, ixToRunes map[int]rune) {
	rand.Seed(time.Now().UnixNano())
	seed := rand.Intn(vocabSize)
	fmt.Printf("%c", ixToRunes[seed])

	index := rnn.sample(seed, 1000)
	for _, idx := range index {
		fmt.Printf("%c", ixToRunes[idx])
	}
	fmt.Println("")

}
