import argparse
import json
from pathlib import Path

from bench_gateway import sample_requests
from esci import sha256_file
from release_replay import CLIENTS, REPEATS, REQUESTS, THREADS, replay
from serve_ranker import bundle_manifest


def main():
    parser = argparse.ArgumentParser(description="Benchmark the Go gateway and ONNX worker.")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--gateway", type=Path, required=True)
    args = parser.parse_args()

    bundle, version = bundle_manifest(args.bundle)
    manifest_path = args.snapshot / "manifest.json"
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != bundle["snapshot_manifest_sha256"]:
        raise SystemExit("The snapshot manifest does not match the model bundle.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = args.snapshot / "validation.parquet"
    validation_sha = sha256_file(validation)
    if validation_sha != manifest["splits"]["validation"]["sha256"]:
        raise SystemExit("The validation file does not match the snapshot manifest.")

    payloads = sample_requests(validation)
    http = replay(args.gateway.resolve(), args.bundle, version, payloads)
    report = {
        "status": "failed" if http["problems"] else "passed",
        "model_version": version,
        "snapshot_manifest_sha256": manifest_sha,
        "validation_sha256": validation_sha,
        "workload": {
            "queries": len(payloads),
            "requests_per_run": REQUESTS,
            "concurrency": CLIENTS,
            "runs": REPEATS,
            "worker_threads": THREADS,
        },
        "http": http,
    }
    print("BENCHMARK_REPORT " + json.dumps(report, separators=(",", ":")), flush=True)
    if http["problems"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
