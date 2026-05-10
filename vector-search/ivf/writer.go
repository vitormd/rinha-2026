package ivf

import (
	"encoding/binary"
	"io"
	"math"
)

// writeFloat32sLE writes each float64 as a little-endian float32 to w.
func writeFloat32sLE(w io.Writer, values []float64) error {
	buf := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(float32(v)))
	}
	_, err := w.Write(buf)
	return err
}

// writeUint32sLE writes each uint32 as little-endian to w.
func writeUint32sLE(w io.Writer, values []uint32) error {
	buf := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(buf[i*4:], v)
	}
	_, err := w.Write(buf)
	return err
}

// writeInt16sLE writes each int16 as little-endian to w.
func writeInt16sLE(w io.Writer, values []int16) error {
	buf := make([]byte, len(values)*2)
	for i, v := range values {
		binary.LittleEndian.PutUint16(buf[i*2:], uint16(v))
	}
	_, err := w.Write(buf)
	return err
}
