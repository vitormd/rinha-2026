package ivf

import (
	"fmt"
	"log"
	"os"
	"syscall"
)

// Index is a memory-mapped IVF index ready for search. The slices held are
// unsafe views over the mmap region — never mutate them.
type Index struct {
	Header

	Centroids   []float32 // K × Dim
	Offsets     []uint32  // K+1, values are BLOCK indices
	Blocks      []int16   // total_blocks × Dim × BlockSize (dim-major within block)
	Labels      []uint8   // total_blocks × BlockSize
	TotalBlocks int

	// CentroidsF64 is the centroid table pre-cast to float64 (the search
	// path needs float64 distances to keep top-N ordering stable). Computed
	// once at Open so the centroid scan doesn't redo K×Dim casts per request.
	CentroidsF64 []float64

	// CentroidNormsSq[c] = sum_j centroids[c][j]² (the squared L2 norm of
	// each centroid). Precomputed at Open so the per-request centroid
	// distance can use the identity
	//   ||q - c||² = ||q||² + ||c||² - 2·<q, c>
	// dropping the constant ||q||² (irrelevant for ranking), the hot path
	// just does ||c||² - 2·<q, c> per centroid — one fused multiply-add
	// per dim instead of sub+mul+add. ~3× fewer FP ops in the centroid
	// scan with no precision loss for the top-N ordering.
	CentroidNormsSq []float64

	mmap []byte
}

// Open mmaps an ivf.bin file produced by Build.
func Open(path string) (*Index, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	fileInfo, err := f.Stat()
	if err != nil {
		return nil, err
	}
	size := int(fileInfo.Size())
	if size < HeaderSize {
		return nil, fmt.Errorf("ivf: file too small: %d", size)
	}

	mapped, err := syscall.Mmap(int(f.Fd()), 0, size, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, err
	}

	hdr, err := ReadHeader(mapped)
	if err != nil {
		_ = syscall.Munmap(mapped)
		return nil, err
	}

	sectionOffsets := SectionOffsets(hdr.K)
	offsets := bytesToUint32(mapped[sectionOffsets.OffsetsT:sectionOffsets.BlocksT])
	if len(offsets) != int(hdr.K)+1 {
		_ = syscall.Munmap(mapped)
		return nil, fmt.Errorf("ivf: offsets len %d != K+1 %d", len(offsets), hdr.K+1)
	}
	totalBlocks := int(offsets[hdr.K])

	blocksBytes := totalBlocks * BlockBytes
	labelsLen := totalBlocks * BlockSize
	expectedEnd := sectionOffsets.BlocksT + blocksBytes + labelsLen
	if expectedEnd != size {
		_ = syscall.Munmap(mapped)
		return nil, fmt.Errorf("ivf: size mismatch: header expects %d got %d (totalBlocks=%d)",
			expectedEnd, size, totalBlocks)
	}

	centroidsF32 := bytesToFloat32(mapped[sectionOffsets.Centroids:sectionOffsets.OffsetsT])
	centroidsF64 := make([]float64, len(centroidsF32))
	for i, v := range centroidsF32 {
		centroidsF64[i] = float64(v)
	}

	// Pre-compute per-centroid squared norms — see CentroidNormsSq doc.
	K := int(hdr.K)
	dim := int(hdr.Dim)
	normsSq := make([]float64, K)
	for c := 0; c < K; c++ {
		base := c * dim
		var s float64
		for j := 0; j < dim; j++ {
			v := centroidsF64[base+j]
			s += v * v
		}
		normsSq[c] = s
	}

	return &Index{
		Header:          hdr,
		Centroids:       centroidsF32,
		CentroidsF64:    centroidsF64,
		CentroidNormsSq: normsSq,
		Offsets:         offsets,
		Blocks:          bytesToInt16(mapped[sectionOffsets.BlocksT : sectionOffsets.BlocksT+blocksBytes]),
		Labels:          mapped[sectionOffsets.BlocksT+blocksBytes : expectedEnd],
		TotalBlocks:     totalBlocks,
		mmap:            mapped,
	}, nil
}

// Close unmaps the underlying file.
func (i *Index) Close() error {
	if i.mmap == nil {
		return nil
	}
	err := syscall.Munmap(i.mmap)
	i.mmap = nil
	return err
}

// PreTouch reads through the mmap'd pages once at startup so that the page
// cache is populated before the API starts accepting traffic, and tries to
// pin them with mlock(2) so the kernel cannot evict them under memory
// pressure (this eliminates rare page-fault outliers in p99).
//
// mlock can fail without CAP_IPC_LOCK or sufficient `ulimit -l`; the error
// is logged but not fatal.
func (i *Index) PreTouch() {
	const stride = 4096
	var sink byte
	for k := 0; k < len(i.mmap); k += stride {
		sink ^= i.mmap[k]
	}
	preTouchSink = sink

	if err := syscall.Mlock(i.mmap); err != nil {
		log.Printf("warning: mlock failed (%v) — pages may be evicted under pressure", err)
	}
}

var preTouchSink byte
