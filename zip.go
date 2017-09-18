package main

import (
	"fmt"
	"reflect"
)

func zip(a, b, c interface{}) error {

	ta, tb, tc := reflect.TypeOf(a), reflect.TypeOf(b), reflect.TypeOf(c)

	if ta.Kind() != reflect.Slice || tb.Kind() != reflect.Slice || ta != tb {
		return fmt.Errorf("zip: first two arguments must be slices of the same type")
	}

	if tc.Kind() != reflect.Ptr {
		return fmt.Errorf("zip: third argument must be pointer to slice")
	}

	for tc.Kind() == reflect.Ptr {
		tc = tc.Elem()
	}

	if tc.Kind() != reflect.Slice {
		return fmt.Errorf("zip: third argument must be pointer to slice")
	}

	eta, _, etc := ta.Elem(), tb.Elem(), tc.Elem()

	if etc.Kind() != reflect.Array || etc.Len() != 2 {
		return fmt.Errorf("zip: third argument's elements must be an array of length 2")
	}

	if etc.Elem() != eta {
		return fmt.Errorf("zip: third argument's elements must be an array of elements of the same type that the first two arguments are slices of")
	}

	va, vb, vc := reflect.ValueOf(a), reflect.ValueOf(b), reflect.ValueOf(c)

	for vc.Kind() == reflect.Ptr {
		vc = vc.Elem()
	}

	if va.Len() != vb.Len() {
		return fmt.Errorf("zip: first two arguments must have same length")
	}

	for i := 0; i < va.Len(); i++ {
		ea, eb := va.Index(i), vb.Index(i)
		tt := reflect.New(etc).Elem()
		tt.Index(0).Set(ea)
		tt.Index(1).Set(eb)
		vc.Set(reflect.Append(vc, tt))
	}

	return nil
}
