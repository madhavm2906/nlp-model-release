import argparse
import hashlib
import json
import math
import random
from itertools import combinations, groupby
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory
from typing import NamedTuple

import duckdb
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from baseline import GAINS, evaluate as evaluate_bm25, ndcg
from esci import SNAPSHOT, sha256_file

MODEL = "cross-encoder/ms-marco-TinyBERT-L2-v2"
REVISION = "81d1926f67cb8eee2c2be17ca9f793c7c3bd20cc"


class Candidate(NamedTuple):
    query_id: int
    query: str
    product: str
    title: str
    label: str


class Comparison(NamedTuple):
    better: Candidate
    worse: Candidate
    weight: float


def sample_comparisons(candidates: list[Candidate], limit: int, seed: int) -> list[Comparison]:
    rng = random.Random(seed)
    comparisons = []
    for _, group in groupby(candidates, key=lambda row: row.query_id):
        available = []
        for left, right in combinations(group, 2):
            difference = GAINS[left.label] - GAINS[right.label]
            if difference > 0:
                available.append(Comparison(left, right, difference))
            elif difference < 0:
                available.append(Comparison(right, left, -difference))
        comparisons.extend(rng.sample(available, min(limit, len(available))))
    return comparisons


def load_validation_and_training(snapshot: Path, train_queries: int, seed: int):
    with duckdb.connect(config={"threads": "2"}) as db:
        train = db.execute("""
            WITH eligible AS (
                SELECT query_id FROM read_parquet($train)
                GROUP BY query_id HAVING count(DISTINCT esci_label) > 1
            ), selected AS (
                SELECT query_id FROM eligible
                ORDER BY sha256($seed || ':' || query_id::VARCHAR), query_id LIMIT $budget
            )
            SELECT query_id, query, product_id, product_title, esci_label
            FROM read_parquet($train) JOIN selected USING (query_id)
            ORDER BY query_id, product_id
        """, {"train": str(snapshot / "train.parquet"), "budget": train_queries, "seed": str(seed)})
        training = [Candidate(*row) for row in train.fetchall()]
        validation = db.execute("""
            SELECT query_id, query, product_id, product_title, esci_label
            FROM read_parquet(?) ORDER BY query_id, product_id
        """, [str(snapshot / "validation.parquet")])
        return training, [Candidate(*row) for row in validation.fetchall()]


def pairwise_loss(better, worse, weights):
    return (weights * F.softplus(worse - better)).sum() / weights.sum()


def logits(model, tokenizer, rows, max_length, device):
    inputs = tokenizer(
        [row.query for row in rows], [row.title for row in rows],
        padding=True, truncation=True, max_length=max_length, return_tensors="pt",
    )
    return model(**inputs.to(device)).logits.squeeze(-1)


@torch.inference_mode()
def score_validation(model, tokenizer, rows, args, device):
    model.eval()
    scores = []
    for start in range(0, len(rows), args.eval_batch_size):
        batch = rows[start:start + args.eval_batch_size]
        scores.extend(logits(model, tokenizer, batch, args.max_length, device).tolist())
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("Model produced nonfinite validation scores.")

    by_query = {}
    for query_id, group in groupby(zip(rows, scores, strict=True), key=lambda item: item[0].query_id):
        ranked = sorted(group, key=lambda item: (-item[1], item[0].product))
        by_query[query_id] = ndcg([row.label for row, _ in ranked])
    return by_query


def train_epoch(model, tokenizer, comparisons, optimizer, args, device):
    model.train()
    losses = []
    for start in range(0, len(comparisons), args.batch_size):
        batch = comparisons[start:start + args.batch_size]
        rows = [pair.better for pair in batch] + [pair.worse for pair in batch]
        weights = torch.tensor([pair.weight for pair in batch], device=device)
        optimizer.zero_grad(set_to_none=True)
        better, worse = logits(model, tokenizer, rows, args.max_length, device).chunk(2)
        loss = pairwise_loss(better, worse, weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(loss.item())
        if len(losses) % 200 == 0:
            print(f"  Trained on {start + len(batch):,}/{len(comparisons):,} comparisons", flush=True)
    return fmean(losses)


def run(args):
    if args.output.exists():
        raise FileExistsError(f"Run already exists: {args.output}. Choose a new --output directory.")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    use_cuda = torch.cuda.is_available()
    if args.device == "cuda" and not use_cuda:
        raise ValueError("CUDA is unavailable in this PyTorch environment. Use --device cpu.")
    device = torch.device("cuda" if use_cuda else "cpu") if args.device == "auto" else torch.device(args.device)

    bm25 = evaluate_bm25(args.snapshot)
    training, validation = load_validation_and_training(args.snapshot, args.train_queries, args.seed)
    print(f"Training on {len({row.query_id for row in training}):,} sampled queries on {device}", flush=True)
    print(f"Scoring {len(validation):,} validation candidates with the pretrained model", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, revision=REVISION, use_safetensors=True,
    ).to(device)
    initial = score_validation(model, tokenizer, validation, args, device)
    print(f"Pretrained nDCG@10: {fmean(initial.values()):.6f}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history = []
    for epoch in range(args.epochs):
        comparisons = sample_comparisons(training, args.pairs_per_query, args.seed + epoch)
        if not comparisons:
            raise ValueError("Training needs eligible comparisons with different relevance labels.")
        random.Random(args.seed + epoch).shuffle(comparisons)
        loss = train_epoch(model, tokenizer, comparisons, optimizer, args, device)
        history.append({"epoch": epoch + 1, "comparisons": len(comparisons), "batch_mean_loss": loss})
        print(f"Epoch {epoch + 1}: pairwise loss {loss:.6f}", flush=True)

    print("Scoring the trained model on validation", flush=True)
    final = score_validation(model, tokenizer, validation, args, device)
    report = {
        "model_source": MODEL, "model_revision": REVISION,
        "snapshot_manifest_sha256": bm25["snapshot_manifest_sha256"],
        "split": "validation", "queries": len(final), "candidate_pairs": len(validation),
        "settings": {key: value for key, value in vars(args).items() if key not in {"snapshot", "output"}},
        "runtime": {"device": str(device), "torch": torch.__version__, "transformers": transformers.__version__},
        "training_query_ids": sorted({row.query_id for row in training}),
        "epochs": history,
        "ndcg_at_10": {"bm25": bm25["ndcg_at_10"], "pretrained": fmean(initial.values()),
                       "fine_tuned": fmean(final.values())},
        "per_query_ndcg_delta_from_pretrained": {
            "improved": sum(final[key] > initial[key] for key in final),
            "unchanged": sum(final[key] == initial[key] for key in final),
            "declined": sum(final[key] < initial[key] for key in final),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".ranker-", dir=args.output.parent) as temp:
        pending = Path(temp) / "run"
        model.save_pretrained(pending, safe_serialization=True)
        tokenizer.save_pretrained(pending)
        report["model_sha256"] = sha256_file(pending / "model.safetensors")
        (pending / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        pending.rename(args.output)

    for name, score in report["ndcg_at_10"].items():
        print(f"{name} validation nDCG@10: {score:.6f}")
    print(f"Saved model, tokenizer and metrics to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Fine tune and validate a compact ESCI cross encoder.")
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--output", type=Path, default=Path("artifacts/esci-tinybert-v1"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--train-queries", type=int, default=2500)
    parser.add_argument("--pairs-per-query", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    args = parser.parse_args()
    for name in ("train_queries", "pairs_per_query", "epochs", "batch_size", "eval_batch_size", "cpu_threads"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 8 <= args.max_length <= 512:
        parser.error("--max-length must be between 8 and 512")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, duckdb.Error) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
