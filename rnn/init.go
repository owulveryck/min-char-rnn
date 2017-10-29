package rnn

import (
	"log"

	"github.com/kelseyhightower/envconfig"
)

// NeuralNetConfig defines our neural network
// architecture and learning parameters.
type neuralNetConfig struct {
	inputNeurons   int
	outputNeurons  int
	HiddenNeurons  int     `default:"100" required:"true"`
	LearningRate   float64 `default:"1e-1" required:"true"`
	AdagradEpsilon float64 `default:"1e-8" required:"true"`
	RandomFactor   float64 `default:"0.01" required":"true"`
}

var conf neuralNetConfig

func init() {
	err := envconfig.Usage("RNN", &conf)
	if err != nil {
		log.Fatal(err)
	}
	err = envconfig.Process("RNN", &conf)
	if err != nil {
		log.Fatal(err)
	}
}
