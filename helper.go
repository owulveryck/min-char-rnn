package main

import (
	"bytes"
	"io"
	"io/ioutil"
	"log"
)

func getDataAndVocab(input io.Reader) ([]rune, map[rune]int, map[int]rune) {
	d, err := ioutil.ReadAll(input)
	if err != nil {
		log.Fatal(err)
	}
	// Extract the rune list
	runeToIx := make(map[rune]int)
	data := bytes.Runes(d)
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
	return data, runeToIx, ixToRune
}
