# rinha26 — make targets for common workflows.
#
# Indexer build args can be overridden:
#   make build INDEX_LIMIT=20000 IVF_K=512
#
# Tunables at runtime (env, not args):
#   N_PROBE_FAST=8 N_PROBE_FULL=28 make up

INDEX_LIMIT ?= 0
IVF_K ?= 4096
IVF_TRAIN_SAMPLES ?= 50000
IVF_ITER ?= 25

BUILD_ARGS = \
	--build-arg INDEX_LIMIT=$(INDEX_LIMIT) \
	--build-arg IVF_K=$(IVF_K) \
	--build-arg IVF_TRAIN_SAMPLES=$(IVF_TRAIN_SAMPLES) \
	--build-arg IVF_ITER=$(IVF_ITER)

.PHONY: help build build-fast up down logs verify bench-py bench-k6 test fmt vet clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

build: ## Build api image with full 3M dataset (slow first time, ~30s with cache)
	docker compose build $(BUILD_ARGS) api1

build-fast: ## Build with INDEX_LIMIT=20000 for quick iteration (~5s)
	$(MAKE) build INDEX_LIMIT=20000 IVF_K=512 IVF_TRAIN_SAMPLES=5000 IVF_ITER=10

up: ## Start the stack (proxy + 2 APIs)
	docker compose up -d

down: ## Stop and remove containers
	docker compose down

logs: ## Tail compose logs
	docker compose logs -f

verify: ## Run scripts/verify.py against the running stack (full 54k entries)
	python3 scripts/verify.py

verify-quick: ## Verify against first 5k entries (10× faster, useful for iteration)
	python3 scripts/verify.py --n 5000

bench-py: ## Local Python benchmark with single keep-alive connection
	python3 scripts/bench.py --n 1000 --warmup 100

bench-k6: ## Run the official k6 ramping benchmark (1→900 RPS over 120s)
	K6_NO_USAGE_REPORT=true k6 run test/test.js

test: ## Run Go unit tests
	docker compose run --rm --no-deps --entrypoint='' api1 sh -c 'cd /src && go test ./...' || \
		echo "(api image doesn't ship the source — to run tests, install Go and run: go test ./...)"

test-local: ## Run Go unit tests using local Go toolchain (must have Go installed)
	go test ./...

fmt: ## go fmt the source
	gofmt -w api vector-search infra

vet: ## go vet the source
	go vet ./...

clean: ## Remove built artifacts
	docker compose down --rmi local --volumes 2>/dev/null || true
	rm -f dataset/ivf.bin dataset/dataset.bin dataset/labels.bin
