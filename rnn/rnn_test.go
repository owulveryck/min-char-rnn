package rnn

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestGob(t *testing.T) {
	// Create a new RNN
	rnn := NewRNN(5, 5)
	rnn.by[1] = 1.0
	rnn.by[2] = 2.0
	rnn.by[3] = 3.0
	rnn.by[4] = 4.0
	rnn.bh[1] = 1.0
	rnn.bh[2] = 2.0
	rnn.bh[3] = 3.0
	rnn.bh[4] = 4.0
	rnn.hprev[1] = 1.0
	rnn.hprev[2] = 2.0
	rnn.hprev[3] = 3.0
	rnn.hprev[4] = 4.0
	b, err := rnn.GobEncode()
	if err != nil {
		t.Fatal(err)
	}
	rnnBkp := NewRNN(1, 1)
	err = rnnBkp.GobDecode(b)
	if err != nil {
		t.Fatal(err)
	}
	if !mat.Equal(rnn.whh, rnnBkp.whh) {
		t.Fatal("whh differs")
	}
	if !mat.Equal(rnn.wxh, rnnBkp.wxh) {
		t.Fatal("wxh differs")
	}
	if !mat.Equal(rnn.why, rnnBkp.why) {
		t.Fatal("why differs")
	}
	if !testEq(rnn.bh, rnnBkp.bh) {
		t.Fatal("bh differs")
	}
	if !testEq(rnn.by, rnnBkp.by) {
		t.Fatal("by differs")
	}
	if !testEq(rnn.hprev, rnnBkp.hprev) {
		t.Fatal("hprev differs")
	}
}

func testEq(a, b []float64) bool {

	if a == nil && b == nil {
		return true
	}

	if a == nil || b == nil {
		return false
	}

	if len(a) != len(b) {
		return false
	}

	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}

	return true
}
