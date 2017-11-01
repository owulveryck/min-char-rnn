package codec

import (
	"bytes"
	"encoding/gob"
	"encoding/json"
	"io"

	"github.com/owulveryck/min-char-rnn/rnn"
)

// Codec ...
type Codec interface {
	Decode([][]float64) io.Reader
	Encode(io.Reader) [][]float64
	Feed() chan rnn.TrainingSet
	NewRNN() *rnn.RNN
	// Choose function is applied in the prediction mechanism
	// it is used to choose the correct category.
	Choose([]float64) []float64
	SetLoss(float64)
	GetInfos() json.Marshaler
	gob.GobDecoder
	gob.GobEncoder
}

type backup struct {
	CDC Codec
	RNN *rnn.RNN
}

// Save the Codec and the RNN for future use
func Save(c Codec, r *rnn.RNN) ([]byte, error) {
	bkp := backup{
		c,
		r,
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
	return bkp.CDC, bkp.RNN, err
}
