import argparse
import json
import re
import tempfile
from collections import Counter
from itertools import groupby
from math import log1p, log2
from pathlib import Path
from statistics import fmean

import duckdb

from esci import SNAPSHOT, normalize_query, sha256_file

GAINS = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
K1 = 1.2
B = 0.75
K = 10


def tokens(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", normalize_query(text))


class BM25:
    def __init__(self, titles):
        frequencies = Counter()
        total_length = 0
        total_documents = 0
        for title in titles:
            terms = tokens(title)
            frequencies.update(set(terms))
            total_length += len(terms)
            total_documents += 1
        if not total_length:
            raise ValueError("BM25 needs training titles with at least one token.")
        self.document_count = total_documents
        self.average_length = total_length / total_documents
        self.idf = {
            term: log1p((total_documents - count + 0.5) / (count + 0.5))
            for term, count in frequencies.items()
        }

    def score(self, query_terms: set[str], title: str) -> float:
        title_tokens = tokens(title)
        counts = Counter(title_tokens)
        norm = K1 * (1 - B + B * len(title_tokens) / self.average_length)
        return sum(
            self.idf.get(term, 0.0) * counts[term] * (K1 + 1) / (counts[term] + norm)
            for term in sorted(query_terms)
            if term in counts
        )


def ndcg(labels: list[str], k: int = K) -> float:
    gains = [GAINS[label] for label in labels]
    ideal = sorted(gains, reverse=True)

    def discounted(values):
        return sum(gain / log2(rank + 2) for rank, gain in enumerate(values[:k]))

    best = discounted(ideal)
    return discounted(gains) / best if best else 0.0


def fit_bm25(db, train: Path) -> BM25:
    titles = db.execute("""
        SELECT DISTINCT product_id, product_title FROM read_parquet(?)
    """, [str(train)]).fetchall()
    return BM25(title for _, title in titles)


def evaluate(snapshot: Path) -> dict:
    manifest_path = snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train, validation = (snapshot / f"{split}.parquet" for split in ("train", "validation"))
    for path in (train, validation):
        if sha256_file(path) != manifest["splits"][path.stem]["sha256"]:
            raise ValueError(f"Snapshot checksum mismatch: {path}")

    with duckdb.connect(config={"threads": "2"}) as db:
        model = fit_bm25(db, train)
        examples = db.execute("""
            SELECT query_id, query, product_id, product_title, esci_label
            FROM read_parquet(?) ORDER BY query_id, product_id
        """, [str(validation)]).fetchall()

    scores = []
    zero_score_queries = 0
    for _, group in groupby(examples, key=lambda row: row[0]):
        candidates = list(group)
        query_terms = set(tokens(candidates[0][1]))
        ranked = sorted(
            ((model.score(query_terms, title), product, label)
             for _, _, product, title, label in candidates),
            key=lambda row: (-row[0], row[1]),
        )
        zero_score_queries += ranked[0][0] == 0
        scores.append(ndcg([label for _, _, label in ranked]))
    if not scores:
        raise ValueError("Validation has no queries to score.")
    return {
        "model": "BM25 over product titles",
        "parameters": {"k1": K1, "b": B},
        "tie_break": "product ID ascending",
        "split": "validation",
        "snapshot_manifest_sha256": sha256_file(manifest_path),
        "training_documents": model.document_count,
        "queries": len(scores),
        "candidate_pairs": len(examples),
        "zero_score_queries": zero_score_queries,
        "gains": GAINS,
        "ndcg_at_10": fmean(scores),
    }


def main():
    parser = argparse.ArgumentParser(description="Score the validation set with BM25.")
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--output", type=Path, default=Path("artifacts/bm25-validation.json"))
    args = parser.parse_args()
    try:
        result = evaluate(args.snapshot)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.output.parent) as directory:
            pending = Path(directory) / "report.json"
            pending.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            pending.replace(args.output)
    except (OSError, ValueError, duckdb.Error) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Validation nDCG@10: {result['ndcg_at_10']:.6f} across {result['queries']:,} queries")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
