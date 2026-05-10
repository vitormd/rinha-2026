// Package ivf implements an Inverted File index for the fraud-score k-NN.
//
// On-disk layout of ivf.bin (single mmap'able file, all little-endian):
//
//	[0..64)            Header (Header struct, 64 bytes)
//	[64..C)            Centroids: K × Dim × float32 (in [0,1] space, sentinels possible)
//	[C..O)             Offsets:   (K+1) × uint32 — offsets[i] is the start *block* index of cluster i; offsets[K]=total_blocks
//	[O..V)             Blocks:    total_blocks × (Dim × 8) × int16, dim-major within block
//	[V..V+padded_n)    Labels:    total_blocks × 8 × uint8 (1 per block-slot)
//
// Vectors are grouped into 8-wide blocks per cluster. Within a block, layout
// is [d0_v0..d0_v7, d1_v0..d1_v7, ..., d13_v0..d13_v7] — 14 contiguous
// dim-arrays of 8 int16 each (224 bytes per block). This lets a SIMD
// implementation compute 8 squared L2 distances in parallel using broadcast
// loads of one query dim at a time.
//
// Trailing slots in a cluster's last block are padded: vector dims set to
// math.MaxInt16 (so distance becomes huge and the slot is never picked) and
// labels set to 0 (irrelevant since the slot can't enter top-5).
//
// padded_n = total_blocks * 8; the original N (header.N) is the count of
// real reference vectors, which may be < padded_n.
package ivf

import (
	"encoding/binary"
	"fmt"
	"io"

	"rinha26/vector-search/vec"
)

const (
	HeaderSize = 64
	Magic      = uint64(0x52494630_30303034) // "RIVF0004"
	Version    = uint32(4)
)

// Header is the fixed-size prefix of ivf.bin. Sizes in bytes are computed
// from these counts.
type Header struct {
	Magic   uint64
	Version uint32
	N       uint32
	Dim     uint32
	K       uint32
	Scale   uint32 // quantization scale (100)
	_pad    [HeaderSize - 32]byte
}

func WriteHeader(w io.Writer, h Header) error {
	var buf [HeaderSize]byte
	binary.LittleEndian.PutUint64(buf[0:], h.Magic)
	binary.LittleEndian.PutUint32(buf[8:], h.Version)
	binary.LittleEndian.PutUint32(buf[12:], h.N)
	binary.LittleEndian.PutUint32(buf[16:], h.Dim)
	binary.LittleEndian.PutUint32(buf[20:], h.K)
	binary.LittleEndian.PutUint32(buf[24:], h.Scale)
	_, err := w.Write(buf[:])
	return err
}

func ReadHeader(b []byte) (Header, error) {
	if len(b) < HeaderSize {
		return Header{}, fmt.Errorf("ivf: header too short: %d", len(b))
	}
	h := Header{
		Magic:   binary.LittleEndian.Uint64(b[0:]),
		Version: binary.LittleEndian.Uint32(b[8:]),
		N:       binary.LittleEndian.Uint32(b[12:]),
		Dim:     binary.LittleEndian.Uint32(b[16:]),
		K:       binary.LittleEndian.Uint32(b[20:]),
		Scale:   binary.LittleEndian.Uint32(b[24:]),
	}
	if h.Magic != Magic {
		return Header{}, fmt.Errorf("ivf: bad magic %x", h.Magic)
	}
	if h.Version != Version {
		return Header{}, fmt.Errorf("ivf: unsupported version %d", h.Version)
	}
	if h.Dim != vec.Dim {
		return Header{}, fmt.Errorf("ivf: dim mismatch %d != %d", h.Dim, vec.Dim)
	}
	return h, nil
}

// BlockSize is the number of vectors per block. SIMD code expects 8 i16 lanes
// — changing this requires changing the assembly.
const BlockSize = 8

// BlockBytes is the size in bytes of one block's vector data: 14 dims × 8
// vectors × 2 bytes = 224 bytes.
const BlockBytes = vec.Dim * BlockSize * 2

// Offsets returns the byte offsets of each section given the header counts.
// The number of blocks must come from offsets[K] (after that array is
// loaded), so this function stops before the blocks/labels section.
type Offsets struct {
	Centroids int
	OffsetsT  int
	BlocksT   int // start of blocks region
}

// SectionOffsets returns the offsets up to (and including) the start of the
// blocks region. Block count is data-dependent (offsets[K]) so the caller
// derives it after reading offsets.
func SectionOffsets(k uint32) Offsets {
	c := HeaderSize
	o := c + int(k)*int(vec.Dim)*4
	b := o + (int(k)+1)*4
	return Offsets{
		Centroids: c,
		OffsetsT:  o,
		BlocksT:   b,
	}
}
