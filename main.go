package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/kelseyhightower/envconfig"
	"github.com/owulveryck/min-char-rnn/rnn"
	"gonum.org/v1/gonum/stat/distuv"
)

type configuration struct {
	Choice     string `default:"soft" required:"true"`
	Epochs     int    `default:"100" required:"true"`
	BatchSize  int    `default:"20" required:"true"`
	SampleSize int    `default:"100" required:"true"`
}

var conf configuration

func usage(err error) error {
	flag.Usage()
	err = envconfig.Usage("MIN_CHAR", &conf)
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
	err := envconfig.Usage("MIN_CHAR", &conf)
	if err != nil {
		log.Fatal(err)
	}
	err = envconfig.Process("MIN_CHAR", &conf)
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
	inputNeurons := len(runesToIx)
	outputNeurons := len(ixToRunes)

	// Create a new RNNs
	neuralNet := rnn.NewRNN(inputNeurons, outputNeurons)
	// Triggering the Training
	feed, info := neuralNet.Train()
	r := bufio.NewReader(data)
	tset := rnn.TrainingSet{
		Inputs:  make([][]float64, conf.BatchSize),
		Targets: make([][]float64, conf.BatchSize),
	}

	n := 0
	epoch := 1
	// loss at iteration 0
	smoothLoss := -math.Log(float64(1)/float64(len(runesToIx))) * float64(conf.BatchSize)
	log.Println(smoothLoss)
	for {
		// Filling a training set
		for i := 0; i < conf.BatchSize+1; i++ {
			// Create the 1-of-k encoder vector
			oneOfK := make([]float64, inputNeurons)
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
			case conf.BatchSize:
				tset.Targets[i-1] = oneOfK
			default:
				//var copyOfOneOfK []float64
				// copy(copyOfOneOfK, oneOfK)
				tset.Inputs[i] = oneOfK
				tset.Targets[i-1] = oneOfK
			}
		}
		// Feeding the network
		feed <- tset
		loss := <-info

		smoothLoss = smoothLoss*0.999 + loss*0.001
		if n%100 == 0 {
			fmt.Printf("Epoch %v, iteration: %v, loss: %v\r", epoch, n, smoothLoss)
			//fmt.Printf("Epoch %v, iteration: %v, loss: %v\r", epoch, n, neuralNet.GetLoss())
		}
		if n%1000 == 0 {
			sampling(neuralNet, outputNeurons, ixToRunes)
		}
		n++
	}
}

func sampling(neuralNet *rnn.RNN, vocabSize int, ixToRunes map[int]rune) {
	rand.Seed(time.Now().UnixNano())
	seed := rand.Intn(vocabSize)
	//fmt.Printf("\n%c", ixToRunes[seed])
	bestProb := func(p []float64) int {
		best := float64(0)
		bestIdx := 0
		for i, v := range p {
			if v > best {
				best = v
				bestIdx = i
			}
		}
		return bestIdx
	}

	randNormalDist := func(p []float64) int {
		sample := distuv.NewCategorical(p, rand.New(rand.NewSource(time.Now().UnixNano())))
		return int(sample.Rand())
	}

	var index []int
	switch conf.Choice {
	case "hard":
		index = neuralNet.Sample(seed, conf.SampleSize, bestProb)
	case "soft":
		index = neuralNet.Sample(seed, conf.SampleSize, randNormalDist)
	default:
		log.Println("Unknown choice")
	}

	for _, idx := range index {
		fmt.Printf("%c", ixToRunes[idx])
	}
	fmt.Println("")

}
