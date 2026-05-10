# syntax=docker/dockerfile:1.7
#
# Multi-stage build:
#   builder  → compiles api and indexer Go binaries
#   dataset  → runs indexer to produce ivf.bin (build-time work)
#   api      → final small image: api binary + ivf.bin + json configs

# --- builder ---
FROM golang:1.23-alpine AS builder
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
RUN adduser -D -u 10001 app
WORKDIR /data
COPY --from=builder /out/api /usr/local/bin/api
COPY --from=dataset /work/dataset/ivf.bin /data/ivf.bin
COPY dataset/mcc_risk.json /data/mcc_risk.json
COPY dataset/normalization.json /data/normalization.json
USER app
ENV DATA_DIR=/data \
    LISTEN_ADDR=:8080 \
    GOMAXPROCS=1
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/api"]
