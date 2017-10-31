package main

import (
	"bufio"
	"bytes"
	"encoding/gob"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"math"
	"os"
	"time"

	"github.com/owulveryck/min-char-rnn/rnn"
)

func training(vocab, input, start, endRegexp, restore, backup *string, num *int) {
	// Create a new RNNs
	var neuralNet = rnn.NewRNN(1, 1)
	var ixToRunes map[int]rune
	var runesToIx map[rune]int
	if *restore != "" {
		b, err := ioutil.ReadFile(*restore)
		if err != nil {
			log.Fatal("Cannot read backup file", err)
		}
		bkp := bkpStruct{}
		backupBytes := bytes.NewBuffer(b)
		dec := gob.NewDecoder(backupBytes) // Will read from network.getVocabIndexesFromFile(
		err = dec.Decode(&bkp)
		if err != nil {
			log.Fatal("Cannot decode backup", err)
		}
		err = neuralNet.GobDecode(bkp.RNN)
		if err != nil {
			log.Fatal("Cannot decode backup", err)
		}
		runesToIx = bkp.RunesToIx
		ixToRunes = bkp.IxToRunes
	} else {
		var err error
		runesToIx, ixToRunes, err = getVocabIndexesFromFile(*vocab)
		if err != nil {
			log.Fatal(usage(err))
		}
		inputNeurons := len(runesToIx)
		outputNeurons := len(ixToRunes)
		neuralNet = rnn.NewRNN(inputNeurons, outputNeurons)
	}
	// Open the sample text file
	data, err := os.Open(*input)
	if err != nil {
		log.Fatal(usage(err))
	}
	defer data.Close()
	// Create the sampling object
	sample := newSample(ixToRunes, runesToIx, *start, *endRegexp, conf.Choice, *num)

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
			oneOfK := make([]float64, len(runesToIx))
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
			fmt.Printf("\n------%v------\n", time.Now())
			sample.sampling(neuralNet)
			fmt.Printf("\n------------\n")
			if *backup != "" {
				b, err := neuralNet.GobEncode()
				if err != nil {
					log.Println("Cannot backup", err)
				}
				var output bytes.Buffer        // Stand-in for a network connection
				enc := gob.NewEncoder(&output) // Will write to network.
				enc.Encode(bkpStruct{
					RNN:       b,
					IxToRunes: ixToRunes,
					RunesToIx: runesToIx,
				})

				err = ioutil.WriteFile(*backup, output.Bytes(), 0644)
				if err != nil {
					log.Println("Cannot backup", err)
				}
			}
		}
		n++
	}
	if *backup != "" {
		b, err := neuralNet.GobEncode()
		if err != nil {
			log.Println("Cannot backup", err)
		}
		var output bytes.Buffer        // Stand-in for a network connection
		enc := gob.NewEncoder(&output) // Will write to network.
		enc.Encode(bkpStruct{
			RNN:       b,
			IxToRunes: ixToRunes,
			RunesToIx: runesToIx,
		})

		err = ioutil.WriteFile(*backup, output.Bytes(), 0644)
		if err != nil {
			log.Println("Cannot backup", err)
		}
	}

}
