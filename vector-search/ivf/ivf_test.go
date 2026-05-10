package ivf

import (
	"bytes"
	"math/rand"
	"testing"

	"rinha26/vector-search/vec"
)

// TestBuildOpenRoundTrip exercises the full pipeline on a tiny synthetic
// dataset: build → write → mmap → search. The dataset is small enough that
// k-NN results can be hand-verified.
func TestBuildOpenRoundTrip(t *testing.T) {
	const n = 128
	const k = 4

	rng := rand.New(rand.NewSource(42))
	vectors := make([]float64, n*vec.Dim)
	labels := make([]uint8, n)
	for i := 0; i < n; i++ {
		for d := 0; d < vec.Dim; d++ {
			vectors[i*vec.Dim+d] = rng.Float64()
		}
		if i%3 == 0 {
			labels[i] = 1
		}
	}

	var buf bytes.Buffer
	if err := Build(&buf, vectors, labels, k, n, 5); err != nil {
		t.Fatalf("Build: %v", err)
	}

	tmpFile := writeTempFile(t, buf.Bytes())
	idx, err := Open(tmpFile)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer idx.Close()

	if int(idx.N) != n {
		t.Errorf("N: got %d want %d", idx.N, n)
	}
	if int(idx.K) != k {
		t.Errorf("K: got %d want %d", idx.K, k)
	}
	if int(idx.Dim) != vec.Dim {
		t.Errorf("Dim: got %d want %d", idx.Dim, vec.Dim)
	}

	// FraudScore on a known query: pick reference vector 0 directly. The
	// nearest neighbor should be itself, so the search must terminate in
	// the cluster containing it. Result must be in [0, 5].
	var query [vec.Dim]float64
	copy(query[:], vectors[:vec.Dim])
	count := idx.FraudScore(&query, k, k) // brute force: scan all clusters
	if count < 0 || count > 5 {
		t.Errorf("fraud count out of range: %d", count)
	}
}

func TestPickTopFromDists(t *testing.T) {
	dists := []float64{5, 1, 4, 2, 3, 9, 0, 7}
	got := pickTopFromDists(dists, len(dists), 3)
	if len(got) != 3 {
		t.Fatalf("len: got %d want 3", len(got))
	}
	// The 3 smallest are at indices 6 (val 0), 1 (val 1), 3 (val 2).
	want := map[uint32]bool{6: true, 1: true, 3: true}
	for _, id := range got {
		if !want[id] {
			t.Errorf("unexpected id %d in top-3", id)
		}
	}
}

func TestUpdateTopK(t *testing.T) {
	var dists [topK]int64
	var labels [topK]uint8
	for i := range dists {
		dists[i] = 1_000_000
	}
	worstIdx := 0

	// Insert 5 distinct candidates; all should land.
	candidates := []struct {
		dist  int64
		label uint8
	}{{100, 0}, {50, 1}, {200, 0}, {25, 1}, {300, 0}}
	for _, c := range candidates {
		worstIdx = updateTopK(&dists, &labels, worstIdx, c.dist, c.label)
	}

	// Now top-K = {100, 50, 200, 25, 300}; worst = 300.
	if dists[worstIdx] != 300 {
		t.Errorf("worst after seeding: got %d want 300", dists[worstIdx])
	}

	// Insert a candidate worse than the worst — should NOT replace.
	prevWorst := dists[worstIdx]
	worstIdx = updateTopK(&dists, &labels, worstIdx, 400, 0)
	if dists[worstIdx] != prevWorst {
		t.Errorf("worse-than-worst replaced something: now %d", dists[worstIdx])
	}

	// Insert better than worst — should replace.
	worstIdx = updateTopK(&dists, &labels, worstIdx, 10, 1)
	for _, d := range dists {
		if d == 300 {
			t.Errorf("worst (300) should have been displaced")
		}
	}
	if dists[worstIdx] <= 10 {
		t.Errorf("worst should be > new value, got %d", dists[worstIdx])
	}
}

func TestLCGDeterministic(t *testing.T) {
	a := newLCG(42)
	b := newLCG(42)
	for i := 0; i < 100; i++ {
		if a.next() != b.next() {
			t.Fatalf("LCG diverged at iter %d", i)
		}
	}
}

// writeTempFile writes data to a temp file and returns its path.
func writeTempFile(t *testing.T, data []byte) string {
	t.Helper()
	f, err := tempFile(t)
	if err != nil {
		t.Fatalf("tempFile: %v", err)
	}
	if _, err := f.Write(data); err != nil {
		t.Fatalf("write: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	return f.Name()
}
