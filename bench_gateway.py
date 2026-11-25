import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby
from pathlib import Path
from statistics import fmean
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import duckdb
import numpy as np


def sample_requests(validation):
    with duckdb.connect(config={"threads": "2"}) as db:
        rows = db.execute("""
            WITH numbered AS (
                SELECT query_id, query, product_id, product_title,
                       row_number() OVER (PARTITION BY query_id ORDER BY product_id) AS position,
                       count(*) OVER (PARTITION BY query_id) AS available
                FROM read_parquet(?)
                WHERE length(trim(query)) BETWEEN 1 AND 512
                  AND length(trim(product_title)) BETWEEN 1 AND 2048
            )
            SELECT query_id, query, product_id, product_title FROM numbered
            WHERE position <= 8 AND available >= 8
            ORDER BY query_id, product_id LIMIT 256
        """, [str(validation)]).fetchall()
    payloads = []
    for _, group in groupby(rows, key=lambda row: row[0]):
        items = list(group)
        payloads.append(json.dumps({
            "query": items[0][1],
            "candidates": [{"id": row[2], "title": row[3]} for row in items],
        }).encode())
    if not payloads:
        raise ValueError("Validation data needs queries with at least eight usable candidates.")
    return payloads


def run(url, payloads, requests, concurrency):
    def send(index):
        request = Request(url, data=payloads[index % len(payloads)],
                          headers={"Content-Type": "application/json"})
        started = perf_counter()
        try:
            with urlopen(request, timeout=3) as response:
                parsed = json.load(response)
        except HTTPError as exc:
            exc.close()
            return str(exc.code), 0.0, None
        except (URLError, TimeoutError):
            return "transport_error", 0.0, None
        return "200", (perf_counter() - started) * 1000, parsed

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        warmup = list(pool.map(send, range(concurrency)))
        if any(status != "200" for status, _, _ in warmup):
            raise ValueError(f"Gateway warmup failed: {dict(Counter(row[0] for row in warmup))}")
        started = perf_counter()
        responses = list(pool.map(send, range(requests)))
        elapsed = perf_counter() - started

    successful = [(elapsed_ms, data) for status, elapsed_ms, data in responses if status == "200"]
    report = {
        "concurrency": concurrency, "requests": requests,
        "status_counts": dict(Counter(row[0] for row in responses)),
        "successful_requests_per_second": len(successful) / elapsed,
    }
    if successful:
        report.update({
            "request_p95_ms": float(np.percentile([ms for ms, _ in successful], 95)),
            "queue_p95_ms": float(np.percentile([row["queue_ms"] for _, row in successful], 95)),
            "worker_p95_ms": float(np.percentile([row["worker_ms"] for _, row in successful], 95)),
            "mean_batch_pairs_per_request": fmean(row["batch_pairs"] for _, row in successful),
            "model_versions": sorted({row["model_version"] for _, row in successful}),
        })
    return report


def main():
    parser = argparse.ArgumentParser(description="Measure actual gateway HTTP requests on ESCI samples.")
    parser.add_argument("--url", default="http://127.0.0.1:8082/rank")
    parser.add_argument("--validation", type=Path, default=Path("data/esci-us-small/validation.parquet"))
    parser.add_argument("--requests", type=int, default=160)
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= args.requests:
        parser.error("use at least one concurrent client and at least as many requests as clients")
    try:
        results = run(args.url, sample_requests(args.validation), args.requests, args.concurrency)
    except (ValueError, duckdb.Error) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(results, indent=2))
    if results["status_counts"].get("200", 0) != args.requests:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
