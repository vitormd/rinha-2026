// indexer reads references.json.gz, trains an IVF index, and writes ivf.bin.
//
// Run via Dockerfile multi-stage build; produces an mmap-friendly binary the
// API container loads at startup.
package main

import (
	"bufio"
	"compress/gzip"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"runtime"
	"time"

	"rinha26/vector-search/ivf"
	"rinha26/vector-search/vec"
)

func main() {
	in := flag.String("in", "dataset/references.json.gz", "path to references.json.gz")
	out := flag.String("out", "dataset/ivf.bin", "output IVF index path")
	limit := flag.Int("limit", 0, "max records to process (0 = all)")
	k := flag.Int("k", 4096, "number of IVF centroids")
	trainSamples := flag.Int("train-samples", 50000, "k-means training sample size (0 = full N)")
	iter := flag.Int("iter", 25, "max k-means iterations (early stop at 0.1% changed)")
	flag.Parse()

	runtime.GOMAXPROCS(runtime.NumCPU())

	start := time.Now()
	log.Printf("indexer: reading %s", *in)

	vectors, labels, err := readReferences(*in, *limit)
	if err != nil {
		log.Fatalf("read references: %v", err)
	}
	n := len(labels)
	log.Printf("indexer: loaded %d records in %.1fs", n, time.Since(start).Seconds())

	if *k > n {
		log.Printf("indexer: k=%d > n=%d, lowering k to n", *k, n)
		*k = n
	}
	if *trainSamples > n {
		*trainSamples = n
	}

	if err := writeIndex(*out, vectors, labels, *k, *trainSamples, *iter); err != nil {
		log.Fatalf("write index: %v", err)
	}

	fileInfo, _ := os.Stat(*out)
	log.Printf("indexer: wrote %s (%.1f MB) in total %.1fs",
		*out, float64(fileInfo.Size())/1e6, time.Since(start).Seconds())
	fmt.Printf("ivf bytes: %d records=%d\n", fileInfo.Size(), n)
}

// referenceEntry is the schema of each item in the references.json.gz array.
type referenceEntry struct {
	Vector [vec.Dim]float64 `json:"vector"`
	Label  string           `json:"label"`
}

// readReferences streams references.json.gz with encoding/json's token API,
// returning the flat float64 vectors and uint8 labels arrays. The streaming
// approach avoids holding the full ~280 MB JSON in memory while still using
// the standard library's parser (build-time perf is not critical).
func readReferences(path string, limit int) ([]float64, []uint8, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return nil, nil, err
	}
	defer gz.Close()

	br := bufio.NewReaderSize(gz, 4<<20)
	dec := json.NewDecoder(br)
	dec.UseNumber()

	tok, err := dec.Token()
	if err != nil {
		return nil, nil, fmt.Errorf("read opening token: %w", err)
	}
	if delim, ok := tok.(json.Delim); !ok || delim != '[' {
		return nil, nil, fmt.Errorf("expected '[' at top level, got %v", tok)
	}

	initialCapacity := 4_000_000
	if limit > 0 {
		initialCapacity = limit
	}
	vectors := make([]float64, 0, initialCapacity*vec.Dim)
	labels := make([]uint8, 0, initialCapacity)

	count := 0
	progressStart := time.Now()
	for dec.More() {
		var entry referenceEntry
		if err := dec.Decode(&entry); err != nil {
			return nil, nil, fmt.Errorf("decode entry %d: %w", count, err)
		}
		vectors = append(vectors, entry.Vector[:]...)
		var lbl uint8
		if entry.Label == "fraud" {
			lbl = 1
		}
		labels = append(labels, lbl)

		count++
		if count%500_000 == 0 {
			log.Printf("indexer: read %d (%.1fs)", count, time.Since(progressStart).Seconds())
		}
		if limit > 0 && count >= limit {
			break
		}
	}
	return vectors, labels, nil
}

// writeIndex creates the output file and streams the IVF index into it.
func writeIndex(path string, vectors []float64, labels []uint8, k, trainSamples, maxIter int) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	bufWriter := bufio.NewWriterSize(f, 4<<20)
	if err := ivf.Build(bufWriter, vectors, labels, k, trainSamples, maxIter); err != nil {
		return err
	}
	if err := bufWriter.Flush(); err != nil {
		return err
	}
	return nil
}

