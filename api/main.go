package main

import (
	"log"
	"net"
	"os"
	"runtime"
	"runtime/debug"
	"strings"

	"github.com/valyala/fasthttp"

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

var (
	pathReady       = []byte("/ready")
	pathFraudScore  = []byte("/fraud-score")
	contentTypeJSON = []byte("application/json")
)

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
	listenAddr := envDefault("LISTEN_ADDR", "/run/sock/api.sock")
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

	listener, err := listen(listenAddr)
	if err != nil {
		log.Fatalf("listen %s: %v", listenAddr, err)
	}

	srv := &fasthttp.Server{
		Handler:                       s.handler,
		Name:                          "rinha26",
		ReadBufferSize:                4096,
		WriteBufferSize:               1024,
		MaxRequestBodySize:            8 << 10,
		NoDefaultServerHeader:         true,
		NoDefaultContentType:          true,
		NoDefaultDate:                 true,
		DisableHeaderNamesNormalizing: true,
	}

	log.Printf("listening on %s", listenAddr)
	if err := srv.Serve(listener); err != nil {
		log.Fatalf("serve: %v", err)
	}
}

// listen returns a net.Listener for either a Unix domain socket (path starts
// with '/' or '@') or a TCP address.
//
// For UDS, any stale file at the path is removed first, and the socket is
// chmod'd to 0666 so a different uid (e.g. nginx) can connect.
func listen(addr string) (net.Listener, error) {
	if strings.HasPrefix(addr, "/") || strings.HasPrefix(addr, "@") {
		if strings.HasPrefix(addr, "/") {
			_ = os.Remove(addr)
		}
		l, err := net.Listen("unix", addr)
		if err != nil {
			return nil, err
		}
		if strings.HasPrefix(addr, "/") {
			if err := os.Chmod(addr, 0o666); err != nil {
				log.Printf("warning: chmod %s: %v", addr, err)
			}
		}
		return l, nil
	}
	return net.Listen("tcp", addr)
}

func (s *server) handler(ctx *fasthttp.RequestCtx) {
	path := ctx.Path()
	switch {
	case bytesEqual(path, pathFraudScore):
		s.handleFraudScore(ctx)
	case bytesEqual(path, pathReady):
		ctx.SetStatusCode(fasthttp.StatusOK)
	default:
		ctx.SetStatusCode(fasthttp.StatusNotFound)
	}
}

func (s *server) handleFraudScore(ctx *fasthttp.RequestCtx) {
	if !ctx.IsPost() {
		ctx.SetStatusCode(fasthttp.StatusMethodNotAllowed)
		return
	}

	var query [vec.Dim]float64
	if err := vec.FromPayload(ctx.PostBody(), s.norm, s.mcc, &query); err != nil {
		ctx.SetStatusCode(fasthttp.StatusBadRequest)
		return
	}

	fraudCount := s.index.FraudScore(&query, s.probeFast, s.probeFull)
	if fraudCount < 0 || fraudCount > 5 {
		fraudCount = 5
	}
	ctx.SetContentTypeBytes(contentTypeJSON)
	ctx.SetBody(responses[fraudCount])
}

func bytesEqual(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
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
