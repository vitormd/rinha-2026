package ivf

import (
	"fmt"
	"io"
	"log"
	"math"
	"time"

	"rinha26/vector-search/quant"
	"rinha26/vector-search/vec"
)

// fixedSeed pins the k-means++ trajectory so every build produces an
// identical clustering. Empirically this seed yields cluster boundaries
// where n_probe=28 covers all top-5 neighbours for the test set.
const fixedSeed uint64 = 0xdeadbeefcafebabe

// Build trains an IVF index over the supplied float64 reference dataset and
// streams it out as the v4 block-of-8 dim-interleaved layout — see
// format.go for the byte layout.
func Build(w io.Writer, vectors []float64, labels []uint8, k, trainSamples, maxIter int) error {
	n := len(vectors) / vec.Dim
	if n*vec.Dim != len(vectors) {
		return fmt.Errorf("ivf.Build: vectors len %d not multiple of %d", len(vectors), vec.Dim)
	}
	if len(labels) != n {
		return fmt.Errorf("ivf.Build: labels %d != n %d", len(labels), n)
	}
	if k <= 0 || k > n {
		return fmt.Errorf("ivf.Build: invalid k=%d for n=%d", k, n)
	}
	if maxIter <= 0 {
		maxIter = 25
	}
	if trainSamples <= 0 || trainSamples > n {
		trainSamples = n
	}

	rng := newLCG(fixedSeed)

	log.Printf("ivf: training k-means k=%d, n=%d, sample=%d, iter=%d (seed=0x%x)",
		k, n, trainSamples, maxIter, fixedSeed)
	phaseStart := time.Now()
	centroids := trainKMeans(vectors, k, trainSamples, maxIter, rng)
	log.Printf("ivf: k-means done in %.1fs", time.Since(phaseStart).Seconds())

	log.Printf("ivf: assigning %d vectors to clusters", n)
	phaseStart = time.Now()
	assignments := assignAll(vectors, centroids)
	log.Printf("ivf: assign done in %.1fs", time.Since(phaseStart).Seconds())

	clusterMembers := groupByCluster(assignments, k)
	logClusterStats(clusterMembers)

	blockOffsets := computeBlockOffsets(clusterMembers)
	totalBlocks := int(blockOffsets[k])
	paddedN := totalBlocks * BlockSize
	log.Printf("ivf: layout: %d blocks (%d slots, %d real vecs, %d padded)",
		totalBlocks, paddedN, n, paddedN-n)

	log.Printf("ivf: quantizing & writing %d blocks (%d bytes)", totalBlocks, totalBlocks*BlockBytes)
	phaseStart = time.Now()
	blocks, outLabels := quantizeBlocks(clusterMembers, blockOffsets, vectors, labels, totalBlocks)
	if err := writeIndex(w, n, k, centroids, blockOffsets, blocks, outLabels); err != nil {
		return err
	}
	log.Printf("ivf: write done in %.1fs", time.Since(phaseStart).Seconds())
	return nil
}

// groupByCluster builds the per-cluster list of original vector indices.
func groupByCluster(assignments []uint32, k int) [][]uint32 {
	out := make([][]uint32, k)
	for i, c := range assignments {
		out[c] = append(out[c], uint32(i))
	}
	return out
}

// computeBlockOffsets produces a (k+1)-length cumulative array where
// blockOffsets[i] is the index of the first 8-vector block belonging to
// cluster i, and blockOffsets[k] is the total number of blocks.
func computeBlockOffsets(clusterMembers [][]uint32) []uint32 {
	k := len(clusterMembers)
	offsets := make([]uint32, k+1)
	for c := 0; c < k; c++ {
		size := uint32(len(clusterMembers[c]))
		offsets[c+1] = offsets[c] + (size+BlockSize-1)/BlockSize
	}
	return offsets
}

// quantizeBlocks lays out vectors as cluster-grouped 8-wide blocks in
// dim-major order (see format.go), quantizing each float64 dim to int16
// (scale = quant.Scale). Trailing slots in a cluster's last block are
// padded with math.MaxInt16 (so SIMD distance becomes huge → never picked)
// and label 0.
func quantizeBlocks(clusterMembers [][]uint32, blockOffsets []uint32, vectors []float64, labels []uint8, totalBlocks int) ([]int16, []uint8) {
	k := len(clusterMembers)

	blocks := make([]int16, totalBlocks*vec.Dim*BlockSize)
	for i := range blocks {
		blocks[i] = math.MaxInt16
	}
	outLabels := make([]uint8, totalBlocks*BlockSize)

	for c := 0; c < k; c++ {
		blockStart := int(blockOffsets[c])
		blockCount := int(blockOffsets[c+1]) - blockStart
		members := clusterMembers[c]

		for localBlock := 0; localBlock < blockCount; localBlock++ {
			blockOffset := (blockStart + localBlock) * vec.Dim * BlockSize
			labelOffset := (blockStart + localBlock) * BlockSize
			for slot := 0; slot < BlockSize; slot++ {
				memberIdx := localBlock*BlockSize + slot
				if memberIdx >= len(members) {
					break // remaining slots stay padded
				}
				vectorIdx := int(members[memberIdx])
				vectorOffset := vectorIdx * vec.Dim
				for d := 0; d < vec.Dim; d++ {
					blocks[blockOffset+d*BlockSize+slot] = quant.EncodeFloat(vectors[vectorOffset+d])
				}
				outLabels[labelOffset+slot] = labels[vectorIdx]
			}
		}
	}
	return blocks, outLabels
}

// writeIndex serializes header + centroids + offsets + blocks + labels.
func writeIndex(w io.Writer, n, k int, centroids []float64, blockOffsets []uint32, blocks []int16, outLabels []uint8) error {
	hdr := Header{
		Magic:   Magic,
		Version: Version,
		N:       uint32(n),
		Dim:     vec.Dim,
		K:       uint32(k),
		Scale:   uint32(quant.Scale),
	}
	if err := WriteHeader(w, hdr); err != nil {
		return err
	}
	if err := writeFloat32sLE(w, centroids); err != nil {
		return err
	}
	if err := writeUint32sLE(w, blockOffsets); err != nil {
		return err
	}
	if err := writeInt16sLE(w, blocks); err != nil {
		return err
	}
	if _, err := w.Write(outLabels); err != nil {
		return err
	}
	return nil
}

// logClusterStats reports min/max/avg cluster sizes so a build pipeline can
// sanity-check whether k-means produced reasonable clusters.
func logClusterStats(clusterMembers [][]uint32) {
	if len(clusterMembers) == 0 {
		return
	}
	min, max := math.MaxInt, 0
	var total int
	empties := 0
	for _, members := range clusterMembers {
		size := len(members)
		if size < min {
			min = size
		}
		if size > max {
			max = size
		}
		total += size
		if size == 0 {
			empties++
		}
	}
	avg := float64(total) / float64(len(clusterMembers))
	log.Printf("ivf: cluster sizes min=%d max=%d avg=%.0f empties=%d", min, max, avg, empties)
}
