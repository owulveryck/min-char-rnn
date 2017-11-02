package char

import (
	"bytes"
	"io/ioutil"
)

func getVocabIndexesFromFile(filename string) (map[rune]int, map[int]rune, error) {
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, nil, err
	}
	runexToIx, ixToRune := getVocabIndexes(data)
	return runexToIx, ixToRune, err
}

// getVocabIndexes reads all the input, fill in an array of runes,
// and returns a map that maps a rune to its index, and another
// one that maps the index to the rune
func getVocabIndexes(input []byte) (map[rune]int, map[int]rune) {
	// Extract the rune list
	runeToIx := make(map[rune]int)
	data := bytes.Runes(input)
	for _, v := range data {
		runeToIx[v] = 0
	}
	ixToRune := make(map[int]rune, len(runeToIx))
	i := 0
	for k := range runeToIx {
		runeToIx[k] = i
		ixToRune[i] = k
		i++
	}
	return runeToIx, ixToRune

}
