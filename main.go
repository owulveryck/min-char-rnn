package main

import (
	"log"
	"os"
)

// Implementation of http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
func main() {
	// Open the sample text file
	file, err := os.Open("data/input.txt")
	if err != nil {
		log.Fatal(err)
	}
	//data, runesToIx, ixToRune := getDataAndVocab(file)
	_, runesToIx, _ := getDataAndVocab(file)
	defer file.Close()
	//dataSize := len(data)
	//fmt.Printf("data has %d runes, %d unique.\n", dataSize, vocabDimension)

	// Create a new RNNs
	// the first argument is the size of the input (the size of the vocabulary)
	// the second input is the size of the output vector (which is also the size of the vocabulary)
	// the lase argument is the size of the hidden layer
	rnn := newRNN(len(runesToIx), len(runesToIx), 100)
	// Read the io.Reader
	sample := make([]byte, 25)
	for err != nil {
		_, err = file.Read(sample)
		loss, dwxh, dwhh, dwhy, dbh, dby := rnn.loss(nil, nil)
		log.Println(loss)
		rnn.adagrad(dwxh, dwhh, dwhy, dbh, dby)
	}
}
