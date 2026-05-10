// Package quant holds int16 quantization helpers used by the IVF index.
//
// All upstream computation runs in float64 to match the data-generator's
// reference vectorization. We only collapse to int16 at the very last step.
//
// Sentinel value (used in dims 5 and 6 when last_transaction is null) maps to
// -Scale.
package quant

import (
	"math"

	"rinha26/vector-search/vec"
)

const Scale int = 10000

// EncodeFloat encodes a float64 in [-1, 1] (where -1 is the sentinel) to
// int16.
//
// Uses round-half-away-from-zero (math.Round) to match libc's round() — the
// data-generator applies round4(v) = round(v * 10000) / 10000 to BOTH
// references and query vectors before running k-NN, so we mimic the same
// rounding here for query quantization.
func EncodeFloat(v float64) int16 {
	if v <= -0.999 {
		return int16(-Scale)
	}
	if v < 0 {
		v = 0
	}
	if v > 1 {
		v = 1
	}
	q := int(math.Round(v * float64(Scale)))
	if q > Scale {
		q = Scale
	}
	if q < 0 {
		q = 0
	}
	return int16(q)
}

// EncodeVec quantizes a 14-D float64 vector into a 14-int16 array.
func EncodeVec(in *[vec.Dim]float64, out *[vec.Dim]int16) {
	for i := 0; i < vec.Dim; i++ {
		out[i] = EncodeFloat(in[i])
	}
}

// DistSqRaw computes squared L2 distance between a query int16 vec and a raw
// int16 slice from mmap.
func DistSqRaw(query *[vec.Dim]int16, ref []int16) int64 {
	_ = ref[vec.Dim-1] // bounds check hint
	d0 := int64(query[0]) - int64(ref[0])
	d1 := int64(query[1]) - int64(ref[1])
	d2 := int64(query[2]) - int64(ref[2])
	d3 := int64(query[3]) - int64(ref[3])
	d4 := int64(query[4]) - int64(ref[4])
	d5 := int64(query[5]) - int64(ref[5])
	d6 := int64(query[6]) - int64(ref[6])
	d7 := int64(query[7]) - int64(ref[7])
	d8 := int64(query[8]) - int64(ref[8])
	d9 := int64(query[9]) - int64(ref[9])
	d10 := int64(query[10]) - int64(ref[10])
	d11 := int64(query[11]) - int64(ref[11])
	d12 := int64(query[12]) - int64(ref[12])
	d13 := int64(query[13]) - int64(ref[13])
	return d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 +
		d7*d7 + d8*d8 + d9*d9 + d10*d10 + d11*d11 + d12*d12 + d13*d13
}
