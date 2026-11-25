import argparse
import json
import subprocess
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory

import duckdb

from bench_gateway import sample_requests
from esci import SNAPSHOT, sha256_file
from export_onnx import ranking_results
from release_replay import (
    CLIENTS, MAX_LATENCY_RATIO, MAX_NDCG_DROP, REPEATS, REQUESTS, ROOT, THREADS,
    check_ports, decision, replay,
)
from serve_ranker import Ranker, bundle_manifest
from train_ranker import Candidate


def load_rows(snapshot):
    path = snapshot / "validation.parquet"
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    if sha256_file(path) != manifest["splits"]["validation"]["sha256"]:
        raise ValueError("Validation Parquet checksum does not match the snapshot.")

    with duckdb.connect(config={"threads": "2"}) as db:
        rows = db.execute("""
            SELECT query_id, query, product_id, product_title, esci_label
            FROM read_parquet(?) ORDER BY query_id, product_id
        """, [str(path)]).fetchall()
    return [Candidate(*row) for row in rows]


def quality(bundle, rows):
    ranker = Ranker(bundle, THREADS)
    scores = []
    for start in range(0, len(rows), 64):
        scores.extend(ranker.score(rows[start:start + 64]))
    return fmean(score for _, score in ranking_results(rows, scores))


def run(args):
    check_ports()
    digest = sha256_file(args.snapshot / "manifest.json")
    rows = load_rows(args.snapshot)
    payloads = sample_requests(args.snapshot / "validation.parquet")
    versions = {}

    for name in ("current", "candidate"):
        bundle = getattr(args, name)
        manifest, version = bundle_manifest(bundle)
        if manifest["snapshot_manifest_sha256"] != digest:
            raise ValueError(f"The {name} bundle belongs to a different dataset snapshot.")
        print(f"Scoring {name} on {len(rows):,} validation pairs", flush=True)
        versions[name] = {
            "version": version,
            "ndcg_at_10": quality(bundle, rows),
            "model_bytes": (bundle / "model.onnx").stat().st_size,
        }

    if versions["current"]["version"] == versions["candidate"]["version"]:
        raise ValueError("Choose two different model bundles.")

    with TemporaryDirectory(prefix="nlp-release-") as temp:
        binary = Path(temp) / "gateway"
        subprocess.run(
            ["go", "build", "-buildvcs=false", "-o", str(binary), "./cmd/gateway"],
            cwd=ROOT, check=True,
        )
        for name, result in versions.items():
            print(f"Replaying HTTP traffic against {name}", flush=True)
            result["http"] = replay(binary, getattr(args, name), result["version"], payloads)

    verdict = decision(versions["current"], versions["candidate"])
    report = {
        "snapshot_manifest_sha256": digest,
        "split": "validation",
        "candidate_pairs": len(rows),
        "limits": {"max_ndcg_drop": MAX_NDCG_DROP, "max_latency_ratio": MAX_LATENCY_RATIO},
        "workload": {"requests_per_run": REQUESTS, "concurrency": CLIENTS,
                     "repeats": REPEATS, "cpu_threads": THREADS},
        **versions,
        "decision": verdict,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temp:
        pending = Path(temp) / "report.json"
        pending.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        pending.replace(args.output)

    for name, result in versions.items():
        latency = result["http"]["median_http_p95_ms"]
        latency_text = f"{latency:.2f} ms" if latency is not None else "unavailable"
        print(f"{name}: nDCG@10 {result['ndcg_at_10']:.6f}, median HTTP p95 {latency_text}")
    print(f"Decision: {verdict['status']}")
    for reason in verdict["reasons"]:
        print(f"  {reason}")
    print(f"Saved {args.output}")
    return 3 if verdict["status"] == "blocked" else 0


def main():
    parser = argparse.ArgumentParser(description="Compare two ONNX releases.")
    parser.add_argument("--current", type=Path, default=Path("artifacts/esci-tinybert-v1-onnx"))
    parser.add_argument("--candidate", type=Path, default=Path("artifacts/esci-tinybert-v1-int8"))
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--output", type=Path, default=Path("artifacts/int8-release-check.json"))
    args = parser.parse_args()
    try:
        raise SystemExit(run(args))
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError, duckdb.Error) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
