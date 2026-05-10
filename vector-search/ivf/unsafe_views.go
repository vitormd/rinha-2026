package ivf

import "unsafe"

// The functions in this file build typed slices that alias bytes from the
// memory-mapped index file. Returned slices share storage with the mmap, so:
//
//   - Lifetime: only valid while the owning Index is open. Munmap invalidates
//     them.
//   - Mutability: never write through these slices — the underlying region is
//     mapped PROT_READ and writes will SIGSEGV.
//   - Alignment: each helper panics if the input bytes aren't a whole multiple
//     of the target type's size. The byte offsets in format.go are designed
//     so this always holds.

func bytesToFloat32(b []byte) []float32 {
	if len(b) == 0 {
		return nil
	}
	if len(b)%4 != 0 {
		panic("ivf: bytesToFloat32: byte length not a multiple of 4")
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), len(b)/4)
}

func bytesToUint32(b []byte) []uint32 {
	if len(b) == 0 {
		return nil
	}
	if len(b)%4 != 0 {
		panic("ivf: bytesToUint32: byte length not a multiple of 4")
	}
	return unsafe.Slice((*uint32)(unsafe.Pointer(&b[0])), len(b)/4)
}

func bytesToInt16(b []byte) []int16 {
	if len(b) == 0 {
		return nil
	}
	if len(b)%2 != 0 {
		panic("ivf: bytesToInt16: byte length not a multiple of 2")
	}
	return unsafe.Slice((*int16)(unsafe.Pointer(&b[0])), len(b)/2)
}
