package main

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"log"
	"math/rand"
	"regexp"
	"time"

	"github.com/owulveryck/min-char-rnn/rnn"
	"gonum.org/v1/gonum/stat/distuv"
)

type sample struct {
	ixToRune     map[int]rune
	runeToIx     map[rune]int
	sampleStart  [][]float64 // The sampling in form of an array of 1-of-k encoded chars
	numChars     int
	endRegexp    *regexp.Regexp
	distribution func(p []float64) int // This is for the character generation
}

func newSample(ixToRune map[int]rune, runesToIx map[rune]int, start, end, choice string, num int) *sample {
	s := &sample{}
	// check if end is a number of characters
	s.numChars = num
	if end != "" {
		s.endRegexp = regexp.MustCompile(end)
	}
	xs := make([][]float64, 0)
	r := bufio.NewReader(bytes.NewBufferString(start))
	for {
		oneOfK := make([]float64, len(runesToIx))
		if c, _, err := r.ReadRune(); err != nil {
			if err == io.EOF {
				break
			} else {
				return nil
			}
		} else {
			oneOfK[runesToIx[c]] = 1
			xs = append(xs, oneOfK)
		}
	}

	s.sampleStart = xs
	s.ixToRune = ixToRune
	s.runeToIx = runesToIx
	switch choice {
	case "hard":
		s.distribution = func(p []float64) int {
			best := float64(0)
			bestIdx := 0
			for i, v := range p {
				if v > best {
					best = v
					bestIdx = i
				}
			}
			return bestIdx
		}
	case "soft":
		s.distribution = func(p []float64) int {
			sample := distuv.NewCategorical(p, rand.New(rand.NewSource(time.Now().UnixNano())))
			return int(sample.Rand())
		}

	default:
		log.Println("Unknown choice")
		return nil
	}

	return s
}

func (s *sample) sampling(neuralNet *rnn.RNN) {

	var index []int
	index = neuralNet.Sample(s.sampleStart, s.numChars, s.distribution)

	fmt.Printf("\n------%v------\n", time.Now())
	for _, idx := range index {
		str := fmt.Sprintf("%c", s.ixToRune[idx])
		fmt.Printf("%v", str)
		if s.endRegexp != nil {
			if s.endRegexp.MatchString(str) {
				break
			}
		}
	}
	fmt.Printf("\n------------\n")
}
