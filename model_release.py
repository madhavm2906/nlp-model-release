import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median

import psycopg

from release_replay import CLIENTS, MAX_LATENCY_RATIO, MAX_NDCG_DROP, REPEATS, REQUESTS, THREADS, decision
from release_store import active_release, connect, initialize, lock_releases, record, rollback
from serve_ranker import bundle_manifest


def checked_metric(value, *, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Report contains an invalid metric.")
    if value < 0 or (maximum is not None and value > maximum):
        raise ValueError("Report metric is out of range.")
    return value


def verify_evidence(report, current_version, candidate_version, snapshot):
    if report["decision"]["status"] != "eligible":
        reasons = "; ".join(report["decision"]["reasons"])
        raise ValueError(f"Release report is blocked. {reasons}")
    if report["split"] != "validation" or report["snapshot_manifest_sha256"] != snapshot:
        raise ValueError("Release report does not match the validation snapshot.")
    limits = {"max_ndcg_drop": MAX_NDCG_DROP, "max_latency_ratio": MAX_LATENCY_RATIO}
    workload = {"requests_per_run": REQUESTS, "concurrency": CLIENTS,
                "repeats": REPEATS, "cpu_threads": THREADS}
    if report["limits"] != limits or report["workload"] != workload:
        raise ValueError("Release report does not match the configured limits and workload.")

    results = {}
    for name, version in (("current", current_version), ("candidate", candidate_version)):
        row = report[name]
        if row["version"] != version:
            raise ValueError(f"Report {name} version does not match the verified bundle.")
        http = row["http"]
        if http["problems"] or len(http["runs"]) != REPEATS:
            raise ValueError(f"Report lacks successful replays for {name}.")
        replay_latencies = []
        for replay in http["runs"]:
            if replay["status_counts"] != {"200": REQUESTS} or replay["model_versions"] != [version]:
                raise ValueError(f"Report lacks successful replays for {name}.")
            p95 = checked_metric(replay["request_p95_ms"])
            if p95 == 0:
                raise ValueError("Report contains a zero HTTP latency.")
            replay_latencies.append(p95)
        results[name] = {
            "ndcg_at_10": checked_metric(row["ndcg_at_10"], maximum=1),
            "http": {"problems": [], "median_http_p95_ms": median(replay_latencies)},
        }

    verdict = decision(results["current"], results["candidate"])
    if verdict["status"] != "eligible":
        raise ValueError("Release metrics failed the gate. " + " ".join(verdict["reasons"]))


def promote(db, bundle, report_path):
    bundle = bundle.resolve()
    candidate_manifest, candidate_version = bundle_manifest(bundle)
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    lock_releases(db)
    current = active_release(db)
    if current["version"] == candidate_version:
        raise ValueError("Candidate is already active.")
    current_manifest, version = bundle_manifest(Path(current["bundle"]))
    if version != current["version"]:
        raise ValueError("Active bundle no longer matches the recorded version.")
    snapshot = candidate_manifest["snapshot_manifest_sha256"]
    if current_manifest["snapshot_manifest_sha256"] != snapshot:
        raise ValueError("Current and candidate bundles use different validation snapshots.")
    verify_evidence(report, version, candidate_version, snapshot)
    evidence = {"sha256": hashlib.sha256(report_bytes).hexdigest(), "report": report}
    return record(db, "promote", candidate_version, bundle, current, evidence)


def main():
    parser = argparse.ArgumentParser(description="Manage verified model releases in PostgreSQL.")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Set the first active bundle")
    init.add_argument("--bundle", type=Path, default=Path("artifacts/esci-tinybert-v1-onnx"))
    promotion = commands.add_parser("promote", help="Activate a candidate with passing evidence")
    promotion.add_argument("--bundle", type=Path, required=True)
    promotion.add_argument("--report", type=Path, required=True)
    commands.add_parser("rollback", help="Restore the previous active bundle")
    commands.add_parser("status", help="Show the desired release")
    args = parser.parse_args()
    try:
        with connect() as db:
            if args.command == "init":
                release = initialize(db, args.bundle)
            elif args.command == "promote":
                release = promote(db, args.bundle, args.report)
            elif args.command == "rollback":
                release = rollback(db)
            else:
                release = active_release(db)
    except (OSError, TypeError, ValueError, KeyError, psycopg.Error) as exc:
        raise SystemExit(f"Release command failed: {exc}") from exc
    print(json.dumps(release, indent=2, default=str))


if __name__ == "__main__":
    main()
