#!/usr/bin/env bash
# Tune N_PROBE (and other env vars) by re-creating the api containers and
# re-running the official k6 test.
#
# Usage: scripts/tune.sh <label> <N_PROBE> [extra-env=val ...]
set -euo pipefail
cd "$(dirname "$0")/.."

LABEL=${1:?label required}
N_PROBE=${2:?N_PROBE required}
shift 2

# Stop everything first.
docker compose down >/dev/null 2>&1 || true

# Up with overridden env. Compose merges environment from yaml + N_PROBE here.
N_PROBE=$N_PROBE docker compose up -d >/dev/null

# Wait for /ready.
for i in $(seq 1 30); do
  code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:9999/ready || true)
  if [[ "$code" == "200" ]]; then break; fi
  sleep 1
done

echo "[$LABEL] N_PROBE=$N_PROBE — running k6..."
K6_NO_USAGE_REPORT=true k6 run test/test.js >/dev/null 2>&1

mkdir -p results
out="results/$LABEL.json"
cp test/results.json "$out"

python3 - "$out" "$LABEL" "$N_PROBE" <<'PY'
import json, sys
path, label, nprobe = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(path))
s = d['scoring']
b = s['breakdown']
print(f"{label:>16}  N_PROBE={nprobe:>3}  p99={d['p99']:>8}  "
      f"sp99={s['p99_score']['value']:>7.1f}  "
      f"fp={b['false_positive_detections']:>4}  fn={b['false_negative_detections']:>4}  "
      f"errs={b['http_errors']:>3}  fail%={s['failure_rate']:>6}  "
      f"sdet={s['detection_score']['value']:>7.1f}  final={s['final_score']:>7.1f}")
PY

docker compose down >/dev/null 2>&1 || true
