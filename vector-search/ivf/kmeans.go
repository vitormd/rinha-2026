package ivf

import (
	"log"
	"math"
	"runtime"
	"sync"
	"sync/atomic"

	"rinha26/vector-search/vec"
)

// trainKMeans returns flat centroids ([k][dim]float64) as a flat []float64
// of length k*dim.
//
// Sampling and k-means++ initialization use the LCG with the seed picked by
// the caller. Lloyd iterations stop early when fewer than 0.1% of sample
// assignments change between iterations.
func trainKMeans(vectors []float64, k, sampleSize, maxIter int, rng *lcg) []float64 {
	sampleIdx := makeTrainingSample(vectors, sampleSize, rng)
	centroids := kmeansPlusPlusInit(vectors, sampleIdx, k, rng)

	assignments := make([]uint32, len(sampleIdx))
	for iter := 0; iter < maxIter; iter++ {
		changed := assignParallelCount(vectors, sampleIdx, centroids, k, assignments)
		updateCentroidsFromSample(vectors, sampleIdx, assignments, centroids, k, rng)

		if iter == 0 || (iter+1)%5 == 0 || iter == maxIter-1 {
			log.Printf("ivf: kmeans iter %d/%d (%.2f%% changed)",
				iter+1, maxIter, float64(changed)/float64(len(sampleIdx))*100)
		}
		if changed*1000 < len(sampleIdx) {
			log.Printf("ivf: kmeans converged at iter %d", iter+1)
			break
		}
	}
	return centroids
}

// makeTrainingSample picks `sampleSize` indices from the vector set. When
// sampleSize equals the full size, it returns the trivial 0..n permutation.
// Otherwise samples with replacement (matches the reference's
// `(0..sample_size).map(|_| rng.next_usize(n)).collect()`); duplicates are
// rare for sample << n and don't materially affect k-means quality.
func makeTrainingSample(vectors []float64, sampleSize int, rng *lcg) []int {
	n := len(vectors) / vec.Dim
	sampleIdx := make([]int, sampleSize)
	if sampleSize == n {
		for i := range sampleIdx {
			sampleIdx[i] = i
		}
		return sampleIdx
	}
	for i := 0; i < sampleSize; i++ {
		sampleIdx[i] = rng.intN(n)
	}
	return sampleIdx
}

// kmeansPlusPlusInit picks k centroids by D²-weighted sampling over the
// training subset.
func kmeansPlusPlusInit(vectors []float64, sampleIdx []int, k int, rng *lcg) []float64 {
	centroids := make([]float64, 0, k*vec.Dim)

	first := sampleIdx[rng.intN(len(sampleIdx))]
	centroids = append(centroids, vectors[first*vec.Dim:first*vec.Dim+vec.Dim]...)

	// minDist[i] is the squared distance from sampleIdx[i] to its closest
	// already-chosen centroid.
	minDist := make([]float64, len(sampleIdx))
	for i := range minDist {
		minDist[i] = math.MaxFloat64
	}

	for c := 1; c < k; c++ {
		lastCentroidOffset := (c - 1) * vec.Dim
		for i, idx := range sampleIdx {
			d := squaredL2(vectors, idx*vec.Dim, centroids, lastCentroidOffset)
			if d < minDist[i] {
				minDist[i] = d
			}
		}

		// Pick weighted by minDist (D²-weighted).
		total := sum(minDist)
		threshold := rng.float64() * total
		var cum float64
		chosen := len(sampleIdx) - 1
		for i, d := range minDist {
			cum += d
			if cum >= threshold {
				chosen = i
				break
			}
		}
		picked := sampleIdx[chosen] * vec.Dim
		centroids = append(centroids, vectors[picked:picked+vec.Dim]...)
	}
	return centroids
}

// updateCentroidsFromSample replaces each centroid with the mean of the
// sample points currently assigned to it. Empty clusters are reseeded from a
// random sample point.
func updateCentroidsFromSample(vectors []float64, sampleIdx []int, assignments []uint32, centroids []float64, k int, rng *lcg) {
	sums := make([]float64, k*vec.Dim)
	counts := make([]uint32, k)
	for s, idx := range sampleIdx {
		c := assignments[s]
		counts[c]++
		vectorOffset := idx * vec.Dim
		centroidOffset := int(c) * vec.Dim
		for j := 0; j < vec.Dim; j++ {
			sums[centroidOffset+j] += vectors[vectorOffset+j]
		}
	}
	for c := 0; c < k; c++ {
		centroidOffset := c * vec.Dim
		if counts[c] == 0 {
			idx := sampleIdx[rng.intN(len(sampleIdx))]
			copy(centroids[centroidOffset:centroidOffset+vec.Dim],
				vectors[idx*vec.Dim:idx*vec.Dim+vec.Dim])
			continue
		}
		invCount := 1.0 / float64(counts[c])
		for j := 0; j < vec.Dim; j++ {
			centroids[centroidOffset+j] = sums[centroidOffset+j] * invCount
		}
	}
}

// assignParallelCount assigns each sample point its nearest centroid id,
// returning the number of assignments that changed (used to detect
// convergence).
func assignParallelCount(vectors []float64, sampleIdx []int, centroids []float64, k int, out []uint32) int {
	workers := workerCount(len(sampleIdx))
	chunk := (len(sampleIdx) + workers - 1) / workers

	var wg sync.WaitGroup
	var totalChanged int64
	for w := 0; w < workers; w++ {
		start := w * chunk
		end := start + chunk
		if end > len(sampleIdx) {
			end = len(sampleIdx)
		}
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			localChanged := 0
			for s := start; s < end; s++ {
				idx := sampleIdx[s]
				best, _ := nearestCentroid(vectors, idx*vec.Dim, centroids, k)
				if out[s] != best {
					localChanged++
				}
				out[s] = best
			}
			atomic.AddInt64(&totalChanged, int64(localChanged))
		}(start, end)
	}
	wg.Wait()
	return int(totalChanged)
}

// assignAll assigns every reference vector to its nearest centroid. Used
// once after k-means converges, to lay vectors out per cluster.
func assignAll(vectors, centroids []float64) []uint32 {
	n := len(vectors) / vec.Dim
	k := len(centroids) / vec.Dim
	out := make([]uint32, n)

	workers := workerCount(n)
	chunk := (n + workers - 1) / workers

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		start := w * chunk
		end := start + chunk
		if end > n {
			end = n
		}
		if start >= end {
			continue
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				best, _ := nearestCentroid(vectors, i*vec.Dim, centroids, k)
				out[i] = best
			}
		}(start, end)
	}
	wg.Wait()
	return out
}

// nearestCentroid finds the centroid id with the smallest squared L2
// distance to the vector at vectors[vectorOffset : vectorOffset+Dim]. Returns
// the id and the distance for callers that need it.
func nearestCentroid(vectors []float64, vectorOffset int, centroids []float64, k int) (uint32, float64) {
	bestCluster := uint32(0)
	bestDistance := math.MaxFloat64
	for c := 0; c < k; c++ {
		d := squaredL2(vectors, vectorOffset, centroids, c*vec.Dim)
		if d < bestDistance {
			bestDistance = d
			bestCluster = uint32(c)
		}
	}
	return bestCluster, bestDistance
}

// squaredL2 returns the squared Euclidean distance between two Dim-length
// slices of float64, each given by a slice + offset.
func squaredL2(a []float64, aOffset int, b []float64, bOffset int) float64 {
	var d float64
	for j := 0; j < vec.Dim; j++ {
		x := a[aOffset+j] - b[bOffset+j]
		d += x * x
	}
	return d
}

func sum(xs []float64) float64 {
	var s float64
	for _, v := range xs {
		s += v
	}
	return s
}

func workerCount(items int) int {
	w := runtime.GOMAXPROCS(0)
	if w < 1 {
		w = 1
	}
	if w > items {
		w = 1
	}
	return w
}
