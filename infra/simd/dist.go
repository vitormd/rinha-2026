// Package simd holds the architecture-specific 8-wide squared L2 distance
// kernel used by the IVF cluster scan.
//
// The kernel computes 8 distances in lock-step against an int16 reference
// block and supports threshold pruning to skip blocks whose partial sums
// already exceed the running top-5 worst.
//
//   - amd64: AVX2 implementation in dist_amd64.s
//   - other: scalar Go fallback in dist_other.go
//
// The constants Dim and BlockSize are layout invariants — the asm is
// hand-written for these specific values, so changing them requires
// rewriting dist_amd64.s.
package simd

// Layout constants. The asm in dist_amd64.s is unrolled for these values;
// changing them requires rewriting the assembly.
const (
	Dim       = 14
	BlockSize = 8
	// QueryLanes is the padded query buffer length; only the first Dim
	// entries are read. Padded to 16 for AVX2 alignment in case future
	// asm uses a wider broadcast load.
	QueryLanes = 16
)

// Block is the dim-major int16 layout of 8 vectors quantized at scale 10000.
//
//	block[d*BlockSize + s]  ->  dim d of vector at slot s.
//
// Bytes per block: Dim * BlockSize * 2 = 224.
type Block = [Dim * BlockSize]int16

// Query is the SIMD-prepared query buffer: the int16-quantized query
// sign-extended to int32 and padded to QueryLanes lanes.
type Query = [QueryLanes]int32

// Distances is the output array — one squared int64 distance per slot in a
// scanned block. int64 because the per-block sum (≤ 5.6e9 worst case)
// exceeds int32.
type Distances = [BlockSize]int64
