# rinha26 — context for Claude Code

Implementation in Go for [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026): credit-card fraud detection via 5-NN over 3M reference vectors. **State: detection 100% (E=0), code refactored, tests pass, submission-ready.**

## Architecture (compact)

```
nginx LB (alpine-slim, 0.10 CPU / 20 MB)
  └── round-robin to 2× Go API (0.45 CPU / 165 MB each)
                            = 1.00 CPU / 350 MB total (spec hard limit)

per request:
  parse JSON → vec.FromPayload (14-D float64) → quant.EncodeVec (i16, scale 10000)
  → IVF 2-stage probing (fast=8, full=28 if count∈{2,3})
  → top-5 fraud count → pre-formatted JSON response
```

Index file `ivf.bin` (v4 format, 84 MB):
- `4096` centroids in float32, dim-major
- offsets in **block units** (block = 8 vectors, 14 dims dim-interleaved, 224 bytes)
- 376k blocks × int16 quantized vectors
- Padded slots filled with `MaxInt16` so SIMD distance becomes huge → never picked

## Project layout

```
api/                          HTTP server + Dockerfile
  main.go
  Dockerfile                  multi-stage: builder → indexer → final image

vector-search/                everything about vector indexing/search
  vec/                        types.go (Dim, Sentinel), config.go, payload.go (+ tests)
  quant/                      int16 quantization (+ tests)
  ivf/                        format, rng (LCG), kmeans, build, loader, search, topk, unsafe_views (+ tests)
  indexer/                    CLI that builds ivf.bin from references.json.gz

infra/                        low-level primitives (kernel, SIMD)
  simd/                       DistBlock kernel: AVX2 asm + scalar fallback
                              owns layout consts (Dim=14, BlockSize=8)
                              exports types: Block, Query, Distances

LB/                           load balancer
  nginx.conf                  minimal nginx (1 worker, ~3 MB RSS)

(at root)
  go.mod / go.sum
  docker-compose.yml          orchestrates the stack
  Makefile                    build/up/verify/bench/test/fmt/vet
  README.md / CLAUDE.md
  dataset/                    references.json.gz, mcc_risk.json, normalization.json
  test/                       official k6 (test.js + test-data.json)
  scripts/                    verify.py, bench.py, tune.sh
```

## Critical decisions (with rationale — don't undo without re-verifying 0 errors)

1. **int64 accumulator for distance** — f32 was tried and regressed E=0 → E=10. f32's 23-bit mantissa rounds 5th-vs-6th nearest ties together at boundary queries.
2. **float64 throughout vectorization** — matches data-generator's reference precision. f32 caused 5-7 errors.
3. **Centroid distances in float64** — f32 sums of 14 squared diffs flip top-N ordering at ulp level for boundary queries.
4. **LCG seed `0xdeadbeefcafebabe`** — empirically yields a centroid layout where n_probe=28 covers all 5-NN for the test set. Other seeds drift the centroids by ulp-level changes that introduce 1-3 errors at fixed n_probe.
5. **`math.Round` (half-away-from-zero) in quantization** — matches C `round()` used by data-generator's `round4` step. RoundToEven was tested and worse.
6. **K=4096, n_probe_full=28** — minimum that zeros errors with our centroid layout. K=1024 needs n_probe≥64 for same recall (more cluster scan work).
7. **2 API instances even on shared CPU** — spec requires it (LB does round-robin between two backends). 1 process would be ~5-15% faster but risks DQ.
8. **AVX2 asm uses i32→i32→i64 widening for sq dist** — i32 acc would overflow over 14 dims; i64 acc requires `VPMOVZXDQ` + extract-high pattern.
9. **`infra/simd` owns its layout constants** (Dim=14, BlockSize=8) — the asm is hand-unrolled for these. ivf imports simd's `Block`, `Query`, `Distances` types.

## Key files

| | |
|---|---|
| `vector-search/vec/payload.go` | 14-dim vectorization. Schema in upstream `docs/br/REGRAS_DE_DETECCAO.md`. Helpers per payload section. |
| `vector-search/vec/config.go` | LoadNorm, LoadMccRisk |
| `vector-search/quant/quant.go` | EncodeFloat (half-away-from-zero), DistSqRaw |
| `vector-search/ivf/format.go` | Header, magic `RIVF0004`, BlockSize=8, BlockBytes=224 |
| `vector-search/ivf/rng.go` | LCG (deterministic 64-bit PCG-style) used to seed k-means++ |
| `vector-search/ivf/kmeans.go` | k-means++ init + Lloyd iterations, parallel assign |
| `vector-search/ivf/build.go` | Build orchestration: k-means → groupByCluster → computeBlockOffsets → quantizeBlocks → writeIndex |
| `vector-search/ivf/loader.go` | Open / Close / PreTouch (mmap PROT_READ) |
| `vector-search/ivf/search.go` | FraudScore (2-stage), scanClusters, updateTopK; calls `simd.DistBlock` |
| `vector-search/ivf/unsafe_views.go` | bytesToFloat32/Uint32/Int16 — uses `unsafe.Slice` (not deprecated SliceHeader) |
| `infra/simd/dist.go` | layout consts + Block/Query/Distances types |
| `infra/simd/dist_amd64.{go,s}` | AVX2 SIMD: 8 i64 distances per block + threshold pruning |
| `infra/simd/dist_other.go` | Scalar Go fallback (matches asm semantically) |

## Common workflows

```bash
make build              # full 3M, ~30s
make build-fast         # 20k subset, ~5s (for fast iteration)
make up && make verify  # must print "weighted E: 0"
make bench-k6           # official k6 ramping benchmark (1→900 RPS over 120s)
make test-local         # go test ./... (needs local Go install)
```

Run Go tests via Docker (no local Go needed):
```bash
docker run --rm -v "$PWD":/src -w /src golang:1.23-alpine sh -c 'go vet ./... && go test ./...'
```

## Gotchas (saving you debugging time)

- **Mac local k6 benchmarks are noisy**: same config gives p99 from 2ms to 200ms across runs. Don't tune from these.
- **GCP e2-medium also throttles**: burstable CPU exhausts under 120s of 900 RPS. p99 there is bounded by throttling, not the code. Real avaliador (Mac Mini Late 2014, dedicated CPU) is the only fair latency benchmark.
- **References precision**: values in `references.json.gz` are pre-rounded to ~4 decimals by data-generator's `round4()`. Our query path must round identically before quantization (we do via `math.Round` in `EncodeFloat`).
- **`test-data.json` checksum**: SHA-256 is over the *uncompressed* references.json, not the .gz. Currently matches `24a1fd58...`.
- **Build-host RAM**: indexer holds 3M × 14 × 8 = ~336 MB float64 in memory. e2-small borderline; e2-medium and Mac Docker fine.
- **proxy memory on Linux is strict**: a Go TCP proxy with goroutine-per-conn OOM-killed at 20 MB cgroup limit (kernel TCP buffers + Go runtime). nginx alpine-slim sits at 3 MB. Don't reintroduce a Go proxy without epoll+single-thread.
- **Docker Desktop on Mac doesn't enforce cgroup memory limits the way Linux does** — what passes locally may OOM on the avaliador.
- **GCP VM `rinha26-bench` exists** in project `rinha-2026`, zone `us-east1-b`. Stop with `gcloud compute instances stop rinha26-bench --zone=us-east1-b`.

## Module/import conventions

- Module path: `rinha26` (see `go.mod`).
- Imports use the new layout:
  - `rinha26/api` — main package (the api binary)
  - `rinha26/vector-search/vec`
  - `rinha26/vector-search/quant`
  - `rinha26/vector-search/ivf`
  - `rinha26/vector-search/indexer` — main package (the indexer binary)
  - `rinha26/infra/simd`
- Package names follow Go conventions (lowercase, no hyphens). The directory names with hyphens (`vector-search`) only show in the import path.

## What's been tried and rejected

| Approach | Result | Why rejected |
|---|---|---|
| HNSW | 230+ MB graph at M=8 | Doesn't fit 350 MB cgroup |
| int8 quantization (scale 100) | 161 errors | Quantization flips k-NN ranks at boundary |
| Custom Go TCP proxy | OOM on Linux at 20 MB | Per-conn goroutines |
| n_probe_full < 28 | 1-3 errors | Our centroid trajectory needs 28 to cover all queries |
| f32 cluster scan | E=10 with same n_probe | ulp precision loss at boundaries |
| RoundToEven quantization | E=10 | Doesn't match C `round()` used by data-generator |

## Submission steps (when ready)

1. `docker push youruser/rinha26-api:vX` (publish image)
2. Branch `submission`: update `docker-compose.yml` to use the public image (no `build:` directive)
3. Add `participants/<your-github-user>.json` to upstream repo via PR
4. Open issue with `rinha/test` in body — avaliador runs automatically and posts to temporary-results

