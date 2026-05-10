//go:build amd64

package simd

// DistBlock computes 8 squared L2 int64 distances between the int32 query
// (sign-extended from the int16-quantized query) and the 8 int16 reference
// vectors stored in `block` (dim-major, scale 10000).
//
//	out[s] = sum_{d in 0..Dim} (query[d] - block[d*BlockSize + s])^2
//
// Threshold pruning: after accumulating the first 8 dims, if every slot's
// partial sum already exceeds `threshold` (the current worst-of-top-5
// distance), the remaining 6 dims are skipped — those slots can't enter
// top-5 since adding non-negative squared diffs only grows the distance.
// The caller's update loop sees the partial sums (still ≥ threshold) and
// naturally rejects them.
//
// Implemented in dist_amd64.s using AVX2 + VPMOVZXDQ widening to int64.
//
//go:noescape
func DistBlock(query *Query, block *Block, out *Distances, threshold int64)
