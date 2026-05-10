package main

import (
	"io"
	"log"
	"net/http"
	"os"
	"runtime"
	"runtime/debug"

	"rinha26/vector-search/ivf"
	"rinha26/vector-search/vec"
)

// responses are pre-formatted JSON bodies indexed by fraud count [0..5].
//
// fraud_score = count/5; approved = score < 0.6 (i.e. count < 3).
var responses = [6][]byte{
	[]byte(`{"approved":true,"fraud_score":0}`),
	[]byte(`{"approved":true,"fraud_score":0.2}`),
	[]byte(`{"approved":true,"fraud_score":0.4}`),
	[]byte(`{"approved":false,"fraud_score":0.6}`),
	[]byte(`{"approved":false,"fraud_score":0.8}`),
	[]byte(`{"approved":false,"fraud_score":1}`),
}

const maxBodyBytes = 8 << 10

type server struct {
	index     *ivf.Index
	norm      *vec.Norm
	mcc       vec.MccRisk
	probeFast int
	probeFull int
}

func main() {
	runtime.GOMAXPROCS(envIntDefault("GOMAXPROCS", 1))
	debug.SetGCPercent(envIntDefault("GOGC", 100))

	dataDir := envDefault("DATA_DIR", "/data")
	listenAddr := envDefault("LISTEN_ADDR", ":8080")
	probeFast := envIntDefault("N_PROBE_FAST", 8)
	probeFull := envIntDefault("N_PROBE_FULL", 28)

	norm, err := vec.LoadNorm(dataDir + "/normalization.json")
	if err != nil {
		log.Fatalf("load normalization.json: %v", err)
	}
	mcc, err := vec.LoadMccRisk(dataDir + "/mcc_risk.json")
	if err != nil {
		log.Fatalf("load mcc_risk.json: %v", err)
	}
	index, err := ivf.Open(dataDir + "/ivf.bin")
	if err != nil {
		log.Fatalf("open ivf.bin: %v", err)
	}
	log.Printf("loaded IVF: N=%d K=%d Dim=%d probe(fast/full)=%d/%d",
		index.N, index.K, index.Dim, probeFast, probeFull)

	log.Printf("pre-touching pages...")
	index.PreTouch()

	s := &server{
		index:     index,
		norm:      norm,
		mcc:       mcc,
		probeFast: probeFast,
		probeFull: probeFull,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/ready", s.handleReady)
	mux.HandleFunc("/fraud-score", s.handleFraudScore)

	log.Printf("listening on %s", listenAddr)
	srv := &http.Server{Addr: listenAddr, Handler: mux}
	if err := srv.ListenAndServe(); err != nil {
		log.Fatalf("listen: %v", err)
	}
}

func (s *server) handleReady(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusOK)
}

func (s *server) handleFraudScore(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(io.LimitReader(r.Body, maxBodyBytes))
	if err != nil {
		http.Error(w, "", http.StatusBadRequest)
		return
	}

	var query [vec.Dim]float64
	if err := vec.FromPayload(body, s.norm, s.mcc, &query); err != nil {
		http.Error(w, "", http.StatusBadRequest)
		return
	}

	fraudCount := s.index.FraudScore(&query, s.probeFast, s.probeFull)
	if fraudCount < 0 || fraudCount > 5 {
		fraudCount = 5
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(responses[fraudCount])
}

func envDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func envIntDefault(key string, def int) int {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	n := 0
	for _, c := range v {
		if c < '0' || c > '9' {
			return def
		}
		n = n*10 + int(c-'0')
	}
	return n
}
