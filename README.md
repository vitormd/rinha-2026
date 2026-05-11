# rinha26

Implementação Go para o desafio [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026) — detecção de fraude em transações de cartão via busca de vizinhos próximos (k=5) sobre 3 milhões de vetores 14-D.

## Arquitetura

```
client
  │
  ▼
nginx (porta 9999)            0.10 CPU /  20 MB
  ├─ round-robin via UDS
  │
  ├──▶ api 1 (Go)              0.45 CPU / 165 MB   /run/sock/api1.sock
  └──▶ api 2 (Go)              0.45 CPU / 165 MB   /run/sock/api2.sock
                              ───────────────────
                              1.00 CPU / 350 MB
```

nginx ↔ APIs comunicam por **Unix Domain Socket** num volume tmpfs compartilhado (`sock`) — sem TCP loopback no caminho dos dados.

Stack:
- Go 1.24
- [fasthttp](https://github.com/valyala/fasthttp) (substitui `net/http` — zero alloc por request)
- [jsonparser](https://github.com/buger/jsonparser) na vetorização

## Pipeline de inferência

```
HTTP POST /fraud-score
  │
  ▼
parse JSON (jsonparser)         vector-search/vec/payload.go
  │
  ▼
vetorização 14-D float64        vector-search/vec/payload.go
  │
  ▼
quantização int16 (×10000)      vector-search/quant/quant.go
  │
  ▼
IVF k-NN dois estágios:         vector-search/ivf/search.go
  • estágio rápido: 8 clusters
  • estágio completo: 28 clusters (só quando count = 2 ou 3)
  • SIMD AVX2 com threshold pruning  (infra/simd/dist_amd64.s)
  │
  ▼
contagem fraud no top-5
  │
  ▼
resposta pré-computada           api/main.go
```

## Layout

```
api/                          servidor HTTP (porta 8080; LB expõe 9999)
  main.go
  Dockerfile                  multi-stage: builder → indexer → final image

vector-search/                tudo relacionado à busca vetorial
  vec/                        vetorização do payload (14 dims, normalização)
  quant/                      quantização int16
  ivf/                        índice IVF: build, search, layout, k-means
  indexer/                    CLI que constrói ivf.bin a partir do dataset

infra/                        primitivas de baixo nível (kernel, SIMD)
  simd/                       AVX2 asm + scalar fallback (DistBlock kernel)

LB/                           load balancer
  nginx.conf                  config minimalista (1 worker, 3 MB RSS)

dataset/                      input data
  references.json.gz          3M referências (50 MB)
  mcc_risk.json               MCC → risco
  normalization.json          constantes de normalização

test/                         k6 oficial do desafio
  test.js                     ramping 1→900 RPS em 120s
  test-data.json              54.100 entries com expected_fraud_score

scripts/                      ferramentas de validação/benchmark
  verify.py                   confere 0 erros sobre 54.100 entries
  bench.py                    p99 com keep-alive single conn
  tune.sh                     varre N_PROBE_FAST/N_PROBE_FULL

docker-compose.yml            orquestração local
Makefile                      build / up / verify / bench / test / fmt / vet / clean
```

## Build & smoke test

```bash
make build          # docker compose build (full 3M, ~30s)
make up             # docker compose up -d
make verify         # python3 scripts/verify.py
make bench-k6       # k6 run test/test.js
make down
```

Para iterar rápido sem reindexar 3M registros:

```bash
make build-fast     # INDEX_LIMIT=20000, IVF_K=512, ~5s
```

## Tunables

Variáveis de ambiente (definidas em `docker-compose.yml`):

| | default | descrição |
|---|---|---|
| `N_PROBE_FAST` | 8 | clusters varridos no estágio 1 |
| `N_PROBE_FULL` | 28 | clusters varridos no estágio 2 (só quando contagem = 2 ou 3) |
| `GOMAXPROCS` | 1 | threads do Go scheduler (single-thread evita contenção) |
| `GOMEMLIMIT` | 150MiB | soft limit do heap do Go |

Build-args (passados ao `docker compose build`):

| | default | descrição |
|---|---|---|
| `INDEX_LIMIT` | 0 | número de refs (0 = todas as 3M) |
| `IVF_K` | 4096 | número de centroides |
| `IVF_TRAIN_SAMPLES` | 50000 | amostra para o k-means |
| `IVF_ITER` | 25 | iterações Lloyd (early stop a 0.1% de mudança) |

## Testes

```bash
make test-local                                      # precisa Go local
docker run --rm -v "$PWD":/src -w /src golang:1.23-alpine go test ./...
```
