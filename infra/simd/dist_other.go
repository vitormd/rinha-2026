//go:build !amd64

package simd

// DistBlock is the scalar Go fallback used on non-x86 architectures. It
// matches the AVX2 implementation semantically (i64 accumulator, 8 dims
// partial + threshold prune, then 6 more if any lane survives).
func DistBlock(query *Query, block *Block, out *Distances, threshold int64) {
	for s := 0; s < BlockSize; s++ {
		out[s] = 0
	}

	// First 8 dims.
	for d := 0; d < 8; d++ {
		qd := int64(query[d])
		base := d * BlockSize
		for s := 0; s < BlockSize; s++ {
			diff := qd - int64(block[base+s])
			out[s] += diff * diff
		}
	}

	// Prune: if every slot already exceeds threshold, skip the rest.
	pruned := true
	for s := 0; s < BlockSize; s++ {
		if out[s] <= threshold {
			pruned = false
			break
		}
	}
	if pruned {
		return
	}

	// Remaining 6 dims.
	for d := 8; d < Dim; d++ {
		qd := int64(query[d])
		base := d * BlockSize
		for s := 0; s < BlockSize; s++ {
			diff := qd - int64(block[base+s])
			out[s] += diff * diff
		}
	}
}
