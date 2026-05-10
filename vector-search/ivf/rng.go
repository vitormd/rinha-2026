package ivf

// lcg is a 64-bit linear-congruential generator (Numerical Recipes' MMIX
// constants). With a fixed seed it produces a deterministic sequence —
// necessary so that the k-means++ centroid trajectory is reproducible across
// builds and matches a known-good clustering.
type lcg struct {
	state uint64
}

func newLCG(seed uint64) *lcg { return &lcg{state: seed} }

// next advances the state and returns the new 64-bit value.
func (l *lcg) next() uint64 {
	l.state = l.state*6364136223846793005 + 1442695040888963407
	return l.state
}

// intN returns a non-negative int in [0, n). Takes the top 31 bits of the
// stream to avoid LCG low-bit periodicity, then reduces modulo n.
func (l *lcg) intN(n int) int {
	return int(l.next()>>33) % n
}

// float64 returns a uniform float64 in [0, 1). Takes the top 53 bits and
// divides by 2^53.
func (l *lcg) float64() float64 {
	return float64(l.next()>>11) / float64(uint64(1)<<53)
}
