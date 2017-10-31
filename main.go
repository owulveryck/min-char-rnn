package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"regexp"
	"time"

	"github.com/kelseyhightower/envconfig"
	"github.com/owulveryck/min-char-rnn/rnn"
	"gonum.org/v1/gonum/stat/distuv"
)

type configuration struct {
	Choice          string `default:"soft" required:"true"`
	Epochs          int    `default:"100" required:"true"`
	BatchSize       int    `default:"20" required:"true"`
	SampleSize      int    `default:"100" required:"true"`
	SampleFrequency int    `default:"1000" required:"true"`
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
	start := flag.String("sampleStart", "Hello,", "the input text to train the network")
	num := flag.Int("sampleSize", 500, "size of the sample to generate")
	endRegexp := flag.String("sampleEndRegexp", "", "If ca generated char match the regexp, it stops")
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
	// Create the sampling object
	sample := newSample(ixToRunes, runesToIx, *start, *endRegexp, conf.Choice, *num)

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
					if epoch < conf.Epochs {
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
				tset.Inputs[i] = oneOfK
				tset.Targets[i-1] = oneOfK
			}
		}
		// Feeding the network
		feed <- tset
		loss := <-info

		smoothLoss = smoothLoss*0.999 + loss*0.001
		if n%100 == 0 {
			log.Printf("Epoch %v, iteration: %v, loss: %v", epoch, n, smoothLoss)
		}
		if n%conf.SampleFrequency == 0 {
			sample.sampling(neuralNet)
		}
		n++
	}
}

type sample struct {
	ixToRune     map[int]rune
	runeToIx     map[rune]int
	sampleStart  [][]float64 // The sampling in form of an array of 1-of-k encoded chars
	numChars     int
	endRegexp    *regexp.Regexp
	distribution func(p []float64) int // This is for the character generation
}

func newSample(ixToRune map[int]rune, runesToIx map[rune]int, start, end, choice string, num int) *sample {
	s := &sample{}
	// check if end is a number of characters
	s.numChars = num
	if end != "" {
		s.endRegexp = regexp.MustCompile(end)
	}
	xs := make([][]float64, 0)
	r := bufio.NewReader(bytes.NewBufferString(start))
	for {
		oneOfK := make([]float64, len(runesToIx))
		if c, _, err := r.ReadRune(); err != nil {
			if err == io.EOF {
				break
			} else {
				return nil
			}
		} else {
			oneOfK[runesToIx[c]] = 1
			xs = append(xs, oneOfK)
		}
	}

	s.sampleStart = xs
	s.ixToRune = ixToRune
	s.runeToIx = runesToIx
	switch choice {
	case "hard":
		s.distribution = func(p []float64) int {
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
	case "soft":
		s.distribution = func(p []float64) int {
			sample := distuv.NewCategorical(p, rand.New(rand.NewSource(time.Now().UnixNano())))
			return int(sample.Rand())
		}

	default:
		log.Println("Unknown choice")
		return nil
	}

	return s
}

func (s *sample) sampling(neuralNet *rnn.RNN) {

	var index []int
	index = neuralNet.Sample(s.sampleStart, s.numChars, s.distribution)

	fmt.Printf("\n------------\n")
	for _, idx := range index {
		str := fmt.Sprintf("%c", s.ixToRune[idx])
		fmt.Printf("%v", str)
		if s.endRegexp != nil {
			if s.endRegexp.MatchString(str) {
				break
			}
		}
	}
	fmt.Printf("\n------------\n")

}
