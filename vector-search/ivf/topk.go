package ivf

import "math"

// pickTopFromDists returns the IDs of the n_probe centroids with the
// smallest precomputed distances. Uses an unsorted bounded-set + linear
// "find-current-worst" — fine for small n_probe (≤64) and avoids heap
// allocations.
func pickTopFromDists(distances []float64, K, nProbe int) []uint32 {
	chosen := make([]uint32, 0, nProbe)
	chosenDistances := make([]float64, 0, nProbe)
	worst := math.MaxFloat64
	worstIdx := 0

	for c := 0; c < K; c++ {
		d := distances[c]
		if len(chosen) < nProbe {
			chosen = append(chosen, uint32(c))
			chosenDistances = append(chosenDistances, d)
			if len(chosen) == nProbe {
				worstIdx = indexOfMax(chosenDistances)
				worst = chosenDistances[worstIdx]
			}
			continue
		}
		if d < worst {
			chosen[worstIdx] = uint32(c)
			chosenDistances[worstIdx] = d
			worstIdx = indexOfMax(chosenDistances)
			worst = chosenDistances[worstIdx]
		}
	}
	return chosen
}

// indexOfMax returns the position of the largest value in xs (xs is assumed
// non-empty).
func indexOfMax(xs []float64) int {
	idx := 0
	for i := 1; i < len(xs); i++ {
		if xs[i] > xs[idx] {
			idx = i
		}
	}
	return idx
}
