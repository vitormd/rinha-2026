package quant

import (
	"math"
	"testing"

	"rinha26/vector-search/vec"
)

func TestEncodeFloatBoundaries(t *testing.T) {
	cases := []struct {
		in   float64
		want int16
		desc string
	}{
		{0, 0, "zero"},
		{1, int16(Scale), "one"},
		{0.5, int16(Scale / 2), "half"},
		{0.0001, 1, "smallest"},
		{1.5, int16(Scale), "above 1 clamps to Scale"},
		{-0.5, 0, "negative non-sentinel clamps to 0"},
		{-1.0, int16(-Scale), "sentinel -1 maps to -Scale"},
	}
	for _, tc := range cases {
		got := EncodeFloat(tc.in)
		if got != tc.want {
			t.Errorf("EncodeFloat(%v) [%s]: got %v want %v", tc.in, tc.desc, got, tc.want)
		}
	}
}

func TestEncodeFloatRoundHalfAwayFromZero(t *testing.T) {
	// 0.00005 is exactly representable in float64 and 0.00005*10000 = 0.5,
	// a true midpoint. math.Round (half-away-from-zero) sends 0.5 → 1,
	// matching C's round() used by the data-generator's round4 step.
	if got := EncodeFloat(0.00005); got != 1 {
		t.Errorf("EncodeFloat(0.00005): got %v want 1 (half-away-from-zero)", got)
	}
	// 0.000049 is just below the midpoint → rounds down to 0.
	if got := EncodeFloat(0.000049); got != 0 {
		t.Errorf("EncodeFloat(0.000049): got %v want 0", got)
	}
	// 0.000051 is just above → rounds up to 1.
	if got := EncodeFloat(0.000051); got != 1 {
		t.Errorf("EncodeFloat(0.000051): got %v want 1", got)
	}
}

func TestEncodeVec(t *testing.T) {
	in := [vec.Dim]float64{
		0.5, 0, 1, 0.0001, -1.0,
		0.25, 0.75, 0.123, 0.999, 0.001,
		1.5, -0.5, 0.42, 0.0,
	}
	var got [vec.Dim]int16
	EncodeVec(&in, &got)

	wantD0 := int16(Scale / 2)
	if got[0] != wantD0 {
		t.Errorf("dim 0: got %v want %v", got[0], wantD0)
	}
	if got[4] != int16(-Scale) {
		t.Errorf("dim 4 (sentinel): got %v want %v", got[4], -Scale)
	}
	if got[10] != int16(Scale) { // 1.5 clamps
		t.Errorf("dim 10 (clamp high): got %v want %v", got[10], Scale)
	}
	if got[11] != 0 { // -0.5 clamps to 0 (not sentinel)
		t.Errorf("dim 11 (clamp low): got %v want 0", got[11])
	}
}

// silence unused
var _ = math.Round
