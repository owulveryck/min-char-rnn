package main

import (
	"bytes"
	"encoding/gob"
	"flag"
	"io/ioutil"
	"log"

	"github.com/kelseyhightower/envconfig"
	"github.com/owulveryck/min-char-rnn/rnn"
)

// Learner ...
type Learner interface {
	SetInputVectorSize(int)
	SetOutputVectorSize(int)
	Decode([]float64)
}

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
	// To display help
	rnn.NewRNN(0, 0)
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
	train := flag.Bool("train", false, "Training a rnn")
	backup := flag.String("backup", "", "backup file")
	restore := flag.String("restore", "", "backup file to restore")
	num := flag.Int("sampleSize", 500, "size of the sample to generate")
	endRegexp := flag.String("sampleEndRegexp", "", "If ca generated char match the regexp, it stops")
	help := flag.Bool("h", false, "display help")
	flag.Parse()
	if *help {
		log.Fatal(usage(nil))
	}
	err := envconfig.Process("MIN_CHAR", &conf)
	if err != nil {
		log.Fatal(usage(err))
	}
	switch *train {
	case true:
		training(vocab, input, start, endRegexp, restore, backup, num)
	case false:
		neuralNet := rnn.NewRNN(1, 1)
		if *restore == "" {
			log.Fatal("please specify the rnn backup to use")
		}
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

		sample := newSample(bkp.IxToRunes, bkp.RunesToIx, *start, *endRegexp, conf.Choice, *num)
		sample.sampling(neuralNet)
	}
}

type bkpStruct struct {
	RNN       []byte
	IxToRunes map[int]rune
	RunesToIx map[rune]int
}
