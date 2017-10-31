package main

import (
	"flag"
	"io/ioutil"
	"log"

	"github.com/kelseyhightower/envconfig"
	"github.com/owulveryck/min-char-rnn/rnn"
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
	train := flag.Bool("train", false, "Training a rnn")
	backup := flag.String("backup", "backup.bin", "backup file")
	restore := flag.String("restore", "", "backup file to restore")
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
	switch *train {
	case true:
		training(vocab, input, start, endRegexp, restore, backup, num)
	case false:
		var neuralNet *rnn.RNN
		if *restore == "" {
			log.Fatal("please specify the rnn backup to use")
		}
		b, err := ioutil.ReadFile(*restore)
		if err != nil {
			log.Fatal("Cannot read backup file", err)
		}
		err = neuralNet.GobDecode(b)
		if err != nil {
			log.Fatal("Cannot decode backup", err)
		}

		runesToIx, ixToRunes, err := getVocabIndexesFromFile(*vocab)
		if err != nil {
			log.Fatal(usage(err))
		}
		sample := newSample(ixToRunes, runesToIx, *start, *endRegexp, conf.Choice, *num)
		sample.sampling(neuralNet)
	}
}
