# syntax=docker/dockerfile:1.7
#
# Multi-stage build:
#   builder  → compiles api and indexer Go binaries
#   dataset  → runs indexer to produce ivf.bin (build-time work)
#   api      → final small image: api binary + ivf.bin + json configs

# --- builder ---
FROM golang:1.24-alpine AS builder
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY api ./api
COPY vector-search ./vector-search
COPY infra ./infra
ENV CGO_ENABLED=0
RUN go build -trimpath -ldflags="-s -w" -o /out/api ./api
RUN go build -trimpath -ldflags="-s -w" -o /out/indexer ./vector-search/indexer

# --- dataset: run the indexer to produce ivf.bin ---
FROM builder AS dataset
WORKDIR /work
COPY dataset/references.json.gz dataset/mcc_risk.json dataset/normalization.json ./dataset/
ARG INDEX_LIMIT=0
ARG IVF_K=4096
ARG IVF_TRAIN_SAMPLES=50000
ARG IVF_ITER=25
RUN /out/indexer \
        -in=./dataset/references.json.gz \
        -out=./dataset/ivf.bin \
        -limit=${INDEX_LIMIT} \
        -k=${IVF_K} \
        -train-samples=${IVF_TRAIN_SAMPLES} \
        -iter=${IVF_ITER}
RUN ls -la ./dataset/

# --- final image: api binary + baked IVF index + json configs ---
FROM alpine:3.20 AS api
WORKDIR /data
COPY --from=builder /out/api /usr/local/bin/api
COPY --from=dataset /work/dataset/ivf.bin /data/ivf.bin
COPY dataset/mcc_risk.json /data/mcc_risk.json
COPY dataset/normalization.json /data/normalization.json
# Run as root so the api process can bind a Unix socket on the shared
# `/run/sock` volume that nginx (also root in its alpine-slim image) reads
# from. UID isolation isn't a concern in this single-tenant benchmark image.
ENV DATA_DIR=/data \
    LISTEN_ADDR=/run/sock/api.sock \
    GOMAXPROCS=1
ENTRYPOINT ["/usr/local/bin/api"]
