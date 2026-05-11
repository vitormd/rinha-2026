package ivf

import (
	"math"
	"sync"
	"unsafe"

	"rinha26/infra/simd"
	"rinha26/vector-search/quant"
	"rinha26/vector-search/vec"
)

// topK is the k-NN k from the spec — the number of nearest reference
// vectors counted to compute fraud_score. Distinct from `K` (the number of
// IVF centroids).
const topK = 5

// distsBufPool reuses the centroid-distance slice across requests. Each
// allocation is K × 8 bytes (~33 KB at K=4096); avoiding it on the hot path
// removes GC pressure and one large make() per request.
var distsBufPool = sync.Pool{
	New: func() any {
		buf := make([]float64, 0, 4096)
		return &buf
	},
}

// FraudScore returns how many of the top-K nearest references carry the
// fraud label.
//
// Two-stage probing: the fast stage scans the `fast` clusters whose
// centroids are closest to the query. If the fraud count among the resulting
// top-K is exactly 2 or 3 (right at the approve/deny boundary at score 0.6),
// the search retries with `full` clusters; otherwise the fast result is
// returned. This keeps ~95% of queries on the cheap path.
func (i *Index) FraudScore(query *[vec.Dim]float64, fast, full int) int {
	K := int(i.K)
	if fast <= 0 {
		fast = 1
	}
	if fast > K {
		fast = K
	}
	if full > K {
		full = K
	}

	queryInt32 := prepareQueryForSIMD(query)

	// Borrow a centroid-distance buffer for this request.
	bufPtr := distsBufPool.Get().(*[]float64)
	dists := (*bufPtr)[:0]
	if cap(dists) < K {
		dists = make([]float64, K)
	} else {
		dists = dists[:K]
	}
	defer func() {
		*bufPtr = dists[:0]
		distsBufPool.Put(bufPtr)
	}()

	computeCentroidDistances(query, i.CentroidsF64, i.CentroidNormsSq, K, dists)

	fastChosen := pickTopFromDists(dists, K, fast)
	fastCount := i.scanClusters(&queryInt32, fastChosen)

	if full <= fast || (fastCount != 2 && fastCount != 3) {
		return fastCount
	}

	fullChosen := pickTopFromDists(dists, K, full)
	return i.scanClusters(&queryInt32, fullChosen)
}

// prepareQueryForSIMD quantizes the float64 query to int16 (scale 10000),
// then sign-extends to int32 with padding to 16 lanes. The padded buffer is
// what the SIMD inner loop expects: each dim is broadcast as an int32 and
// matched against 8 sign-extended refs from a block.
func prepareQueryForSIMD(query *[vec.Dim]float64) [16]int32 {
	var quantized [vec.Dim]int16
	quant.EncodeVec(query, &quantized)
	var padded [16]int32
	for j := 0; j < vec.Dim; j++ {
		padded[j] = int32(quantized[j])
	}
	return padded
}

// computeCentroidDistances fills `out` with a rank-equivalent score of the
// squared L2 distance from the query to each centroid:
//
//	||q - c||² = ||q||² + ||c||² - 2·<q, c>
//
// ||q||² is constant per query so it drops out of the ranking; the loop
// computes only `||c||² - 2·<q, c>` per centroid (one FMA per dim instead
// of sub+mul+add). Pre-computed normsSq comes from Index.CentroidNormsSq.
//
// `out[c]` is therefore NOT the true squared distance — it's an additive
// constant away from it. Top-N ordering and tie-breaking are preserved
// exactly because ||q||² is shared by all candidates.
func computeCentroidDistances(query *[vec.Dim]float64, centroids, normsSq []float64, K int, out []float64) {
	for c := 0; c < K; c++ {
		base := c * vec.Dim
		var dot float64
		for j := 0; j < vec.Dim; j++ {
			dot += query[j] * centroids[base+j]
		}
		out[c] = normsSq[c] - 2.0*dot
	}
}

// scanClusters scans the given clusters one block at a time, computing 8
// squared int64 distances per block (via simd.DistBlock — AVX2 on amd64,
// scalar fallback elsewhere) and tracking the global top-K.
//
// Distances are int64 because the int16 scale-10000 quantization yields
// per-block distance sums up to ~5.6e9, which exceeds int32. int64 also
// preserves the rank order exactly for 5th-vs-6th-nearest tie cases that
// f32 would round together.
func (i *Index) scanClusters(query *[16]int32, clusters []uint32) int {
	var topDistances [topK]int64
	var topLabels [topK]uint8
	for j := range topDistances {
		topDistances[j] = math.MaxInt64
	}
	worstIdx := 0

	blocks := i.Blocks
	labels := i.Labels
	offsets := i.Offsets

	var blockDistances [BlockSize]int64

	for _, clusterID := range clusters {
		blockStart := offsets[clusterID]
		blockEnd := offsets[clusterID+1]
		for blockIdx := int(blockStart); blockIdx < int(blockEnd); blockIdx++ {
			blockOffset := blockIdx * vec.Dim * BlockSize
			labelOffset := blockIdx * BlockSize

			simd.DistBlock(query,
				(*simd.Block)(unsafe.Pointer(&blocks[blockOffset])),
				(*simd.Distances)(&blockDistances),
				topDistances[worstIdx])

			for slot := 0; slot < BlockSize; slot++ {
				worstIdx = updateTopK(&topDistances, &topLabels, worstIdx,
					blockDistances[slot], labels[labelOffset+slot])
			}
		}
	}

	frauds := 0
	for j := 0; j < topK; j++ {
		if topLabels[j] == 1 {
			frauds++
		}
	}
	return frauds
}

// updateTopK conditionally replaces the worst entry of the top-K with a new
// candidate, returning the (possibly updated) index of the new worst.
func updateTopK(topDistances *[topK]int64, topLabels *[topK]uint8, worstIdx int, candidateDist int64, candidateLabel uint8) int {
	if candidateDist >= topDistances[worstIdx] {
		return worstIdx
	}
	topDistances[worstIdx] = candidateDist
	topLabels[worstIdx] = candidateLabel
	newWorst := 0
	for k := 1; k < topK; k++ {
		if topDistances[k] > topDistances[newWorst] {
			newWorst = k
		}
	}
	return newWorst
}
