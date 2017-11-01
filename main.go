package main

import (
	"bytes"
	"errors"
	"flag"
	"io"
	"io/ioutil"
	"log"
	"os"

	"github.com/kelseyhightower/envconfig"
	"github.com/owulveryck/min-char-rnn/codec"
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

var (
	restoreFile *string
	backupFile  *string
)

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
	//vocab := flag.String("vocab", "data/vocab.txt", "the file holds the vocabulary")
	//input := flag.String("input", "data/input.txt", "the input text to train the network")
	start := flag.String("sampleStart", "Hello,", "the input text to train the network")
	train := flag.Bool("train", false, "Training a rnn")
	backupFile = flag.String("backup", "", "backup file")
	restoreFile = flag.String("restore", "", "backup file to restoreFile")
	num := flag.Int("sampleSize", 500, "size of the sample to generate")
	//endRegexp := flag.String("sampleEndRegexp", "", "If ca generated char match the regexp, it stops")
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
		//training(vocab, input, start, endRegexp, restoreFile, backup, num)
		var cdc codec.Codec
		var rnn *rnn.RNN
		cdc, rnn, err = restore()
		if err != nil {
			log.Println("Cannot restore from backup, creating new entries", err)
			rnn = cdc.NewRNN()
		}
		feed, info := rnn.Train()
		feeder := cdc.Feed()
		for tset := range feeder {
			feed <- tset
			loss := <-info
			cdc.SetLoss(loss)
			cdc.GetInfos()
		}
		err = backup(cdc, rnn)
		if err != nil {
			log.Println("Cannot backup ", err)
		}
	case false:
		cdc, rnn, err := restore()
		if err != nil {
			log.Fatal("Unable to restore", err)
		}
		xs := cdc.Encode(bytes.NewBufferString(*start))

		ys := rnn.Predict(xs, *num, cdc.Choose)
		io.Copy(os.Stdout, cdc.Decode(ys))
	}
}
func restore() (codec.Codec, *rnn.RNN, error) {
	if restoreFile == nil {
		return nil, nil, errors.New("No restore file specified")
	}
	b, err := ioutil.ReadFile(*restoreFile)
	if err != nil {
		return nil, nil, err
	}
	return codec.Restore(b)
}
func backup(cdc codec.Codec, rnn *rnn.RNN) error {
	if *backupFile != "" {
		b, err := codec.Save(cdc, rnn)
		if err != nil {
			return err
		}
		err = ioutil.WriteFile(*backupFile, b, 0644)
		if err != nil {
			return err
		}
	}
	return nil
}
