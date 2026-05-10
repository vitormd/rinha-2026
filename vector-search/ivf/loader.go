package ivf

import (
	"fmt"
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

	return &Index{
		Header:      hdr,
		Centroids:   bytesToFloat32(mapped[sectionOffsets.Centroids:sectionOffsets.OffsetsT]),
		Offsets:     offsets,
		Blocks:      bytesToInt16(mapped[sectionOffsets.BlocksT : sectionOffsets.BlocksT+blocksBytes]),
		Labels:      mapped[sectionOffsets.BlocksT+blocksBytes : expectedEnd],
		TotalBlocks: totalBlocks,
		mmap:        mapped,
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
// cache is populated before the API starts accepting traffic. Without this
// the first few thousand requests pay a page-fault tax that shows up in p99.
func (i *Index) PreTouch() {
	const stride = 4096
	var sink byte
	for k := 0; k < len(i.mmap); k += stride {
		sink ^= i.mmap[k]
	}
	preTouchSink = sink
}

var preTouchSink byte
