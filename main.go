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
	"github.com/owulveryck/min-char-rnn/rnn"
)

type configuration struct {
	HiddenNeurons int     `default:"100" required:"true"`
	Epochs        int     `default:"100" required:"true"`
	MemorySize    int     `default:"20" required:"true"`
	LearningRate  float64 `default:"1e-3" required:"true"`
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
	config := rnn.NeuralNetConfig{
		InputNeurons:  len(runesToIx), // the input is the size of the vocabulary
		OutputNeurons: len(runesToIx), // the output size is also the size of the vocablulary
		HiddenNeurons: conf.HiddenNeurons,
		NumEpochs:     conf.Epochs,
		MemorySize:    conf.MemorySize, // This corresponds to seq_length in the initial implementation
		LearningRate:  conf.LearningRate,
	}
	if config.OutputNeurons > config.HiddenNeurons {
		log.Fatal("Bad parameter, too few hidden neurons")
	}

	// Create a new RNNs
	neuralNet := rnn.NewRNN(config)
	// Triggering the Training
	feed := neuralNet.Train()
	r := bufio.NewReader(data)
	tset := rnn.TrainingSet{
		Inputs:  make([][]float64, config.MemorySize),
		Targets: make([][]float64, config.MemorySize),
	}

	n := 0
	epoch := 1
	for {
		// Filling a training set
		for i := 0; i < config.MemorySize+1; i++ {
			// Create the 1-of-k encoder vector
			oneOfK := make([]float64, config.InputNeurons)
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
				tset.Inputs[i] = oneOfK
			case config.MemorySize:
				tset.Targets[i-1] = oneOfK
			default:
				var copyOfOneOfK []float64
				copy(copyOfOneOfK, oneOfK)
				tset.Inputs[i] = oneOfK
				tset.Targets[i-1] = oneOfK
			}
		}
		// Feeding the network
		feed <- tset
		if n%100 == 0 {
			fmt.Printf("Epoch %v, iteration: %v, loss: %v\r", epoch, n, neuralNet.GetSmoothLoss())
		}
		if n%1000 == 0 {
			sampling(neuralNet, config.InputNeurons, ixToRunes)
		}
		n++
	}
}

func sampling(neuralNet *rnn.RNN, vocabSize int, ixToRunes map[int]rune) {
	rand.Seed(time.Now().UnixNano())
	seed := rand.Intn(vocabSize)
	//fmt.Printf("\n%c", ixToRunes[seed])

	index := neuralNet.Sample(seed, 1000)
	for _, idx := range index {
		fmt.Printf("%c", ixToRunes[idx])
	}
	fmt.Println("")

}
