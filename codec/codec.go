package codec

import (
	"bytes"
	"encoding/gob"
	"encoding/json"
	"io"
	"log"

	"github.com/owulveryck/min-char-rnn/rnn"
)

// Codec ...
type Codec interface {
	// Decode an array of inputs and returns an io.Reader
	Decode([][]float64) io.Reader
	Encode(io.Reader) [][]float64
	Feed() <-chan rnn.TrainingSet
	// NewRNN returns a neural network suitable for the codec
	NewRNN() *rnn.RNN
	// ApplyDist applies  a distribution according to the configuration of the neural network
	ApplyDist([]float64) []float64
	SetLoss(float64)
	GetInfos() json.Marshaler
}

// Backup ...
type backup struct {
	Codec Codec
	rnn   rnn.RNN
}

// Save the Codec and the RNN for future use
func Save(c Codec, r *rnn.RNN) ([]byte, error) {
	bkp := backup{
		c,
		*r,
	}
	var output bytes.Buffer
	enc := gob.NewEncoder(&output)
	err := enc.Encode(&bkp)
	return output.Bytes(), err
}

// Restore the learner and the RNN
func Restore(b []byte) (Codec, *rnn.RNN, error) {
	var bkp backup
	input := bytes.NewBuffer(b)
	dec := gob.NewDecoder(input)
	err := dec.Decode(&bkp)
	log.Println(bkp.rnn)
	return bkp.Codec, &bkp.rnn, err
}
