package main

import (
	"bytes"
	"errors"
	"flag"
	"io"
	"io/ioutil"
	"log"
	"os"
	"time"

	"github.com/kelseyhightower/envconfig"
	"github.com/owulveryck/min-char-rnn/codec"
	"github.com/owulveryck/min-char-rnn/codec/char"
	"github.com/owulveryck/min-char-rnn/rnn"
)

type configuration struct {
	SampleSize      int `default:"100" required:"true"`
	SampleFrequency int `default:"1000" required:"true"`
	InfoFrequency   int `default:"100" required:"true"`
	BackupFrequency int `default:"1000" required:"true"`
	// Backup prefix: default no backup
	BackupPrefix string `default:""`
	// Backup Suffix, should be compatible with time.Format()
	BackupSuffix string `default:""`
}

var (
	conf        configuration
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
	train := flag.Bool("train", false, "Training a rnn")
	backupFile = flag.String("backup", "", "backup file")
	restoreFile = flag.String("restore", "", "backup file to restoreFile")
	//endRegexp := flag.String("sampleEndRegexp", "", "If ca generated char match the regexp, it stops")
	help := flag.Bool("h", false, "display help")
	flag.Parse()
	if *help {
		log.Fatal(usage(nil))
	}
	err := envconfig.Process("MIN_CHAR", &conf)
	if err != nil {
		log.Println(usage(err))
		log.Fatal(err)
	}
	switch *train {
	case true:
		//training(vocab, input, start, endRegexp, restoreFile, backup, num)
		var cdc codec.Codec
		var nn *rnn.RNN
		cdc, nn, err = restore()
		if err != nil {
			log.Println("Cannot restore from backup, creating new entries", err)
			cdc, err = char.NewChar()
			if err != nil {
				log.Fatal(err)
			}
			nn = cdc.NewRNN()
		}
		feed, info := nn.Train()
		feeder := cdc.Feed()
		// Create the sampling
		var sample [][]float64
		if conf.SampleFrequency != 0 {
			b, err := ioutil.ReadAll(os.Stdin)
			if err != nil {
				log.Println("No start provided, sampling will be done ", err)
			} else {
				sample = cdc.Encode(bytes.NewBuffer(b))
			}
		}
		n := 0
		for tset := range feeder {
			feed <- rnn.CopyOf(tset)
			select {
			case inf := <-info:
				cdc.SetLoss(inf)
				if n%conf.InfoFrequency == 0 && conf.InfoFrequency != 0 {
					log.Printf("[%v] %v", n, cdc.GetInfos())
				}
			default:
			}
			if n%conf.BackupFrequency == 0 {
				err = backup(cdc, nn)
				if err != nil {
					log.Println("Cannot backup ", err)
				}
			}
			if n%conf.SampleFrequency == 0 && n != 0 && conf.SampleFrequency != 0 && len(sample) > 0 {
				ys := nn.Predict(sample, conf.SampleSize, cdc.ApplyDist)
				io.Copy(os.Stdout, cdc.Decode(ys))
			}
			n++
		}
		log.Println("end")
		close(feed)
		err = backup(cdc, nn)
		if err != nil {
			log.Println("Cannot backup ", err)
		}
	case false:
		cdc, nn, err := restore()
		if err != nil {
			log.Fatal("Unable to restore ", err)
		}
		xs := cdc.Encode(os.Stdin)
		log.Println(len(xs))

		ys := nn.Predict(xs, conf.SampleSize, cdc.ApplyDist)
		io.Copy(os.Stdout, cdc.Decode(ys))
	}
}
func restore() (codec.Codec, *rnn.RNN, error) {
	if *restoreFile == "" {
		return nil, nil, errors.New("No restore file specified")
	}
	b, err := ioutil.ReadFile(*restoreFile)
	if err != nil {
		return nil, nil, err
	}
	cdcb, nn, err := codec.Restore(b)
	cdc := &char.Char{}
	cdc.UnmarshalBinary(cdcb)
	return cdc, nn, err
}
func backup(cdc codec.Codec, rnn *rnn.RNN) error {
	if conf.BackupPrefix != "" {
		b, err := codec.Save(cdc, rnn)
		if err != nil {
			return err
		}
		err = ioutil.WriteFile(conf.BackupPrefix+time.Now().Format(conf.BackupSuffix)+".bin", b, 0644)
		if err != nil {
			return err
		}
	}
	return nil
}
