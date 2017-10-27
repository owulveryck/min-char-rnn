package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/kelseyhightower/envconfig"
)

type configuration struct {
	HiddenNeurons int     `default:"100" required:"true"`
	Epochs        int     `default:"100" required:"true"`
	MemorySize    int     `default:"20" required:"true"`
	LearningRate  float64 `default:"1e-1" required:"true"`
}

func usage(err error) error {
	var conf configuration
	flag.Usage()
	err = envconfig.Usage("RNN", &conf)
	if err != nil {
		log.Fatal(err)
	}
	return err
}

func main() {
	vocab := flag.String("vocab", "data/vocab.txt", "the file holds the vocabulary")
	input := flag.String("input", "data/input.txt", "the input text to train the network")
	help := flag.Bool("h", false, "display help")
	flag.Parse()
	if *help {
		log.Fatal(usage(nil))
	}
	var conf configuration
	err := envconfig.Process("RNN", &conf)
	if err != nil {
		log.Fatal(usage(err))
	}

	runesToIx, ixToRunes, err := getVocabIndexesFromFile(*vocab)
	if err != nil {
		log.Fatal(usage(err))
	}
	// Open the sample text file
	data, err := os.Open(*input)
	if err != nil {
		log.Fatal(usage(err))
	}
	defer data.Close()

	maxEpoch := 100
	// Define our network architecture and learning parameters.
	config := neuralNetConfig{
		inputNeurons:  len(runesToIx), // the input is the size of the vocabulary
		outputNeurons: len(runesToIx), // the output size is also the size of the vocablulary
		hiddenNeurons: 65,
		numEpochs:     100,
		memorySize:    15, // This corresponds to seq_length in the initial implementation
		learningRate:  1e-1,
	}

	// Create a new RNNs
	rnn := newRNN(config)
	// Triggering the Training
	feed := rnn.Train()
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
						if _, err := data.Seek(0, io.SeekStart); err != nil {
							log.Fatal(err)
						}
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
			fmt.Printf("Epoch %v, iteration: %v, loss: %v\r", epoch, n, rnn.GetSmoothLoss())
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
	//fmt.Printf("\n%c", ixToRunes[seed])

	index := rnn.sample(seed, 1000)
	for _, idx := range index {
		fmt.Printf("%c", ixToRunes[idx])
	}
	fmt.Println("")

}
