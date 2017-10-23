package main

import (
	"bufio"
	"io"
	"log"
	"os"

	"github.com/gonum/matrix/mat64"
)

// Implementation of http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
func main() {
	seqLength := 25
	file, err := os.Open("data/vocab.txt")
	_, runesToIx, _ := getDataAndVocab(file)
	file.Close()
	// Open the sample text file
	data, err := os.Open("data/input.txt")
	if err != nil {
		log.Fatal(err)
	}
	//data, runesToIx, ixToRune := getDataAndVocab(file)
	defer file.Close()
	//dataSize := len(data)
	//fmt.Printf("data has %d runes, %d unique.\n", dataSize, vocabDimension)

	// Create a new RNNs
	// the first argument is the size of the input (the size of the vocabulary)
	// the second input is the size of the output vector (which is also the size of the vocabulary)
	// the lase argument is the size of the hidden layer
	_ = newRNN(len(runesToIx), len(runesToIx), 100)
	// Read the io.Reader
	r := bufio.NewReader(data)
	i := 0
	x := mat64.NewVector(seqLength, nil)
	for {
		if c, _, err := r.ReadRune(); err != nil {
			if err == io.EOF {
				break
			} else {
				log.Fatal(err)
			}
		} else {
			x.SetVec(i, float64(runesToIx[c]))
			i++
			if i%seqLength == 0 {
				//loss, dwxh, dwhh, dwhy, dbh, dby := rnn.loss(nil, nil)
				//log.Println(loss)
				//rnn.adagrad(dwxh, dwhh, dwhy, dbh, dby)
				i = 0
			}
		}
	}
}
