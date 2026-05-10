#!/usr/bin/env python3
"""Tiny load test: hammer /fraud-score with persistent HTTP keep-alive
connection and report latency percentiles.

Stdlib only. By default, draws payloads round-robin from test/test-data.json
(the official k6 dataset). Pass --payloads to use a different file with the
same {entries: [{request: ...}]} shape, or a flat [payload, ...] array.
"""
import argparse
import http.client
import json
import socket
import time
from statistics import median


def percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--payloads", default="test/test-data.json")
    ap.add_argument("--connections", type=int, default=1, help="parallel persistent connections")
    args = ap.parse_args()

    with open(args.payloads) as f:
        raw = json.load(f)
    # Accept either the test-data.json shape ({entries: [{request: ...}]}) or
    # a flat list of payload objects.
    if isinstance(raw, dict) and "entries" in raw:
        payloads = [e["request"] for e in raw["entries"]]
    else:
        payloads = raw
    bodies = [json.dumps(p).encode() for p in payloads]

    if args.connections > 1:
        import threading

        results = []
        lock = threading.Lock()

        def worker(per_n):
            local = bench(args.host, args.port, bodies, per_n, args.warmup // args.connections)
            with lock:
                results.extend(local)

        threads = []
        per = args.n // args.connections
        t0 = time.perf_counter()
        for _ in range(args.connections):
            t = threading.Thread(target=worker, args=(per,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0
        report(results, wall, args.connections)
    else:
        t0 = time.perf_counter()
        results = bench(args.host, args.port, bodies, args.n, args.warmup)
        wall = time.perf_counter() - t0
        report(results, wall, 1)


def bench(host, port, bodies, n, warmup):
    conn = http.client.HTTPConnection(host, port)
    conn.connect()
    sock = conn.sock
    if sock is not None:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    headers = {"content-type": "application/json", "connection": "keep-alive"}

    # Warmup
    for i in range(warmup):
        body = bodies[i % len(bodies)]
        conn.request("POST", "/fraud-score", body=body, headers=headers)
        r = conn.getresponse()
        r.read()

    samples = []
    for i in range(n):
        body = bodies[i % len(bodies)]
        t0 = time.perf_counter_ns()
        conn.request("POST", "/fraud-score", body=body, headers=headers)
        r = conn.getresponse()
        r.read()
        t1 = time.perf_counter_ns()
        if r.status != 200:
            print(f"non-200: {r.status}")
        samples.append((t1 - t0) / 1e6)  # ms
    conn.close()
    return samples


def report(samples, wall, conns):
    if not samples:
        print("no samples")
        return
    samples.sort()
    n = len(samples)
    rps = n / wall if wall > 0 else 0
    print(f"n={n} wall={wall:.2f}s rps={rps:.0f} conns={conns}")
    print(f"  min  {samples[0]:.3f} ms")
    print(f"  p50  {percentile(samples, 0.50):.3f} ms")
    print(f"  p90  {percentile(samples, 0.90):.3f} ms")
    print(f"  p99  {percentile(samples, 0.99):.3f} ms")
    print(f"  p999 {percentile(samples, 0.999):.3f} ms")
    print(f"  max  {samples[-1]:.3f} ms")


if __name__ == "__main__":
    main()
