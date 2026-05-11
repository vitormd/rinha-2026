#!/usr/bin/env bash
# Sweep a config knob via env vars + cpu/mem overrides, capturing the final
# k6 score per variant.
#
# Usage: scripts/sweep.sh <label> <api_cpus> <proxy_cpus> [N_PROBE_FAST] [N_PROBE_FULL]
set -euo pipefail
cd "$(dirname "$0")/.."

LABEL=${1:?label required}
API_CPUS=${2:?api cpus required}
PROXY_CPUS=${3:?proxy cpus required}
NPF=${4:-8}
NPL=${5:-28}

# Write an override file that pins the requested resources & probes.
cat > docker-compose.override.yml <<EOF
services:
  api1:
    image: rinha26-api:local
    build:
      context: .
      dockerfile: api/Dockerfile
    environment:
      LISTEN_ADDR: "/run/sock/api1.sock"
      GOMAXPROCS: "1"
      GOMEMLIMIT: "150MiB"
      N_PROBE_FAST: "$NPF"
      N_PROBE_FULL: "$NPL"
    deploy:
      resources:
        limits:
          cpus: "$API_CPUS"

  api2:
    image: rinha26-api:local
    build:
      context: .
      dockerfile: api/Dockerfile
    environment:
      LISTEN_ADDR: "/run/sock/api2.sock"
      GOMAXPROCS: "1"
      GOMEMLIMIT: "150MiB"
      N_PROBE_FAST: "$NPF"
      N_PROBE_FULL: "$NPL"
    deploy:
      resources:
        limits:
          cpus: "$API_CPUS"

  proxy:
    deploy:
      resources:
        limits:
          cpus: "$PROXY_CPUS"
EOF

docker compose down >/dev/null 2>&1 || true
docker compose up -d >/dev/null 2>&1
sleep 4

K6_NO_USAGE_REPORT=true k6 run test/test-quick.js >/dev/null 2>&1

python3 - "$LABEL" "$API_CPUS" "$PROXY_CPUS" "$NPF" "$NPL" <<'PY'
import json, sys
label, ac, pc, npf, npl = sys.argv[1:6]
d = json.load(open("test/results.json"))
s = d["scoring"]; b = s["breakdown"]
print(f"{label:>16}  api={ac:>5} proxy={pc:>5} np={npf}/{npl:<2}  "
      f"p99={d['p99']:>9}  sp99={s['p99_score']['value']:>7.1f}  "
      f"E={s['weighted_errors_E']:>3}  FINAL={s['final_score']:>7.1f}")
PY
