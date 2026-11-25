import argparse
import json
import math
from itertools import groupby
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb
import numpy as np
import psycopg
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from baseline import fit_bm25, ndcg, tokens
from esci import SNAPSHOT, sha256_file
from release_store import active_release, connect
from serve_ranker import MAX_BATCH, Ranker, bundle_manifest
from train_ranker import MODEL, REVISION, Candidate, logits

BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 42


def load_snapshot(snapshot):
    path = snapshot / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for split in ("train", "validation", "test"):
        data = snapshot / f"{split}.parquet"
        if sha256_file(data) != manifest["splits"][split]["sha256"]:
            raise ValueError(f"Dataset checksum mismatch: {data}")
    return manifest, sha256_file(path)


def baseline_and_test_rows(snapshot):
    train = str(snapshot / "train.parquet")
    validation = str(snapshot / "validation.parquet")
    test = str(snapshot / "test.parquet")
    with duckdb.connect(config={"threads": "2"}) as db:
        overlap = db.execute("""
            SELECT query_key FROM read_parquet($test)
            INTERSECT
            SELECT query_key FROM read_parquet([$train, $validation])
            LIMIT 1
        """, {"test": test, "train": train, "validation": validation}).fetchone()
        if overlap is not None:
            raise ValueError("A held out query also appears in training or validation.")
        bm25 = fit_bm25(db, Path(train))
        rows = db.execute("""
            SELECT query_id, query, product_id, product_title, esci_label
            FROM read_parquet(?) ORDER BY query_id, product_id
        """, [test]).fetchall()
    return bm25, [Candidate(*row) for row in rows]


def query_metrics(rows, scores):
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("A model returned a nonfinite score.")
    ndcg_values, exact_top1 = [], []
    for _, items in groupby(zip(rows, scores, strict=True), key=lambda item: item[0].query_id):
        ranked = sorted(items, key=lambda item: (-item[1], item[0].product))
        labels = [row.label for row, _ in ranked]
        ndcg_values.append(ndcg(labels))
        if "E" in labels:
            exact_top1.append(labels[0] == "E")
    return np.asarray(ndcg_values), exact_top1


def interval(deltas, *, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 64):
        stop = min(start + 64, draws)
        ids = rng.integers(0, len(deltas), size=(stop - start, len(deltas)))
        means[start:stop] = deltas[ids].mean(axis=1)
    return np.percentile(means, [2.5, 97.5]).tolist()


@torch.inference_mode()
def pretrained_scores(rows, max_length):
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, revision=REVISION, use_safetensors=True,
    ).eval()
    scores = []
    for start in range(0, len(rows), MAX_BATCH):
        scores.extend(logits(model, tokenizer, rows[start:start + MAX_BATCH], max_length, "cpu").tolist())
    return scores


def selected_bundle(path):
    if path is not None:
        return path, None
    with connect() as db:
        release = active_release(db)
    return Path(release["bundle"]), release


def evaluate(args):
    path, release = selected_bundle(args.bundle)
    bundle, version = bundle_manifest(path)
    if release is not None and release["version"] != version:
        raise ValueError("The active bundle does not match its PostgreSQL release record.")
    manifest, snapshot_digest = load_snapshot(args.snapshot)
    if bundle["snapshot_manifest_sha256"] != snapshot_digest:
        raise ValueError("The selected model belongs to a different dataset snapshot.")

    bm25, rows = baseline_and_test_rows(args.snapshot)
    counts = manifest["splits"]["test"]
    if len(rows) != counts["pairs"] or not rows:
        raise ValueError("Held out evaluation rows do not match the dataset manifest.")
    torch.set_num_threads(args.threads)
    print(f"Scoring {len(rows):,} held out pairs with BM25, pretrained TinyBERT, and the ONNX release", flush=True)
    plain = []
    for _, group in groupby(rows, key=lambda row: row.query_id):
        candidates = list(group)
        query_terms = set(tokens(candidates[0].query))
        plain.extend(bm25.score(query_terms, row.title) for row in candidates)
    bm25_metrics = query_metrics(rows, plain)
    print("Scoring with pretrained TinyBERT", flush=True)
    pretrained = query_metrics(rows, pretrained_scores(rows, bundle["max_length"]))
    print("Scoring with the selected ONNX release", flush=True)
    ranker = Ranker(path, args.threads)
    if ranker.version != version:
        raise ValueError("The selected bundle changed during evaluation.")
    output_scores = []
    for start in range(0, len(rows), MAX_BATCH):
        output_scores.extend(ranker.score(rows[start:start + MAX_BATCH]))
    active = query_metrics(rows, output_scores)

    results = {}
    for name, (values, exact_top1) in (("bm25", bm25_metrics), ("pretrained", pretrained), ("release", active)):
        if len(values) != counts["queries"]:
            raise ValueError("Scored query count does not match the dataset manifest.")
        results[name] = {"ndcg_at_10": float(values.mean()),
                         "exact_top1_rate": float(np.mean(exact_top1)) if exact_top1 else None,
                         "queries_with_exact_candidate": len(exact_top1)}

    comparisons = {}
    for name, baseline in (("bm25", bm25_metrics[0]), ("pretrained", pretrained[0])):
        deltas = active[0] - baseline
        comparisons[name] = {"delta_ndcg_at_10": float(deltas.mean()),
                             "paired_query_bootstrap_95pct": interval(deltas)}
    return {
        "split": "test", "snapshot_manifest_sha256": snapshot_digest,
        "model_version": version, "release_id": release["id"] if release else None,
        "pretrained_model": MODEL, "pretrained_revision": REVISION,
        "queries": counts["queries"], "candidate_pairs": len(rows), "results": results,
        "release_vs": comparisons,
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "unit": "query"},
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the frozen release on the official ESCI test split.")
    parser.add_argument("--bundle", type=Path, help="An explicit bundle. Defaults to the active PostgreSQL release.")
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--output", type=Path, default=Path("artifacts/final-test-report.json"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    try:
        if args.output.exists():
            raise FileExistsError(f"Report already exists: {args.output}. Choose a new --output path.")
        report = evaluate(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=args.output.parent) as tmp:
            pending = Path(tmp) / "report.json"
            pending.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            args.output.hardlink_to(pending)
    except (OSError, ValueError, KeyError, RuntimeError, duckdb.Error, psycopg.Error) as exc:
        raise SystemExit(f"Test evaluation failed: {exc}") from exc

    for name, scores in report["results"].items():
        print(f"{name} test nDCG@10: {scores['ndcg_at_10']:.6f}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
