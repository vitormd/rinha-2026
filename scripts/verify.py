#!/usr/bin/env python3
"""Sequentially POST every test entry and tally detection errors against the
ground truth in test-data.json. Single persistent connection — pure
correctness check, not a load test.

Usage:
  python3 scripts/verify.py [--n N] [--show-errors K]
"""
import argparse
import http.client
import json
import socket
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--n", type=int, default=0, help="limit entries; 0 = all")
    ap.add_argument("--show-errors", type=int, default=0,
                    help="dump first K mismatching entries to stderr")
    ap.add_argument("--data", default="test/test-data.json")
    args = ap.parse_args()

    print(f"loading {args.data}...", file=sys.stderr)
    data = json.load(open(args.data))
    entries = data["entries"]
    if args.n > 0:
        entries = entries[:args.n]
    n = len(entries)
    print(f"verifying {n} entries against {args.host}:{args.port}", file=sys.stderr)

    conn = http.client.HTTPConnection(args.host, args.port)
    conn.connect()
    if conn.sock is not None:
        conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    headers = {"content-type": "application/json", "connection": "keep-alive"}

    tp = tn = fp = fn = errs = score_diff = 0
    mismatch_examples = []
    err_by_expected_score = {}
    edge_total = 0
    edge_errors = 0

    t0 = time.perf_counter()
    last_print = t0
    for i, entry in enumerate(entries):
        body = json.dumps(entry["request"]).encode()
        expected_approved = entry["expected_approved"]
        expected_score = entry.get("expected_fraud_score", None)

        conn.request("POST", "/fraud-score", body=body, headers=headers)
        r = conn.getresponse()
        raw = r.read()
        if r.status != 200:
            errs += 1
            continue
        try:
            j = json.loads(raw)
        except Exception:
            errs += 1
            continue
        approved = bool(j.get("approved"))
        got_score = j.get("fraud_score")

        if expected_score is not None and got_score is not None and got_score != expected_score:
            score_diff += 1

        is_edge = expected_score == 0.6
        if is_edge:
            edge_total += 1
        if approved == expected_approved:
            if approved:
                tn += 1
            else:
                tp += 1
        else:
            err_by_expected_score[expected_score] = err_by_expected_score.get(expected_score, 0) + 1
            if is_edge:
                edge_errors += 1
            if approved:
                fn += 1  # missed fraud
            else:
                fp += 1
            if len(mismatch_examples) < args.show_errors:
                mismatch_examples.append({
                    "idx": i,
                    "id": entry["request"].get("id"),
                    "expected_approved": expected_approved,
                    "got_approved": approved,
                    "expected_score": expected_score,
                    "got_score": got_score,
                })

        now = time.perf_counter()
        if now - last_print >= 2.0:
            elapsed = now - t0
            done = i + 1
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n - done) / rate if rate > 0 else float("inf")
            print(f"  {done}/{n}  {rate:.0f} req/s  eta {eta:.0f}s", file=sys.stderr)
            last_print = now

    elapsed = time.perf_counter() - t0
    conn.close()

    total = tp + tn + fp + fn + errs
    print()
    print(f"verified {total} entries in {elapsed:.1f}s ({total/elapsed:.0f} req/s)")
    print(f"  tp (correct fraud blocked):  {tp}")
    print(f"  tn (correct legit approved): {tn}")
    print(f"  fp (legit blocked):          {fp}")
    print(f"  fn (fraud approved):         {fn}")
    print(f"  http errors:                 {errs}")
    print(f"  score mismatches:            {score_diff} (decisions can match while scores differ)")
    weighted = fp * 1 + fn * 3 + errs * 5
    print(f"  weighted E:                  {weighted}")
    print(f"  failure rate:                {(fp+fn+errs)/total*100:.3f}%")
    print(f"  edge cases (expected=0.6):   {edge_total} total, {edge_errors} mismatches "
          f"({edge_errors/edge_total*100:.1f}% of edges)" if edge_total else f"  edge cases: 0")
    if err_by_expected_score:
        print("  errors by expected_fraud_score:")
        for k in sorted(err_by_expected_score.keys()):
            print(f"    {k}: {err_by_expected_score[k]}")

    if mismatch_examples:
        print("\nfirst mismatches:", file=sys.stderr)
        for m in mismatch_examples:
            print(f"  {m}", file=sys.stderr)


if __name__ == "__main__":
    main()
