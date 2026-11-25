import argparse
import hashlib
import json
import subprocess
import tempfile
import unicodedata
from pathlib import Path

import duckdb

SOURCE = json.loads(Path(__file__).with_name("esci_source.json").read_text(encoding="utf-8"))
SPLITS = ("train", "validation", "test")
RAW = Path("data/raw")
SNAPSHOT = Path("data/esci-us-small")
EXAMPLES = "shopping_queries_dataset_examples.parquet"
PRODUCTS = "shopping_queries_dataset_products.parquet"
VALIDATION_PERCENT = 15


def sha256_file(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def normalize_query(query: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", query).casefold().split())


def assign_splits(queries):
    items = [(query_id, normalize_query(query), split) for query_id, query, split in queries]
    if len(set(query_id for query_id, _, _ in items)) != len(items):
        raise ValueError("A query ID has multiple query texts or source splits.")
    if any(not key or split not in {"train", "test"} for _, key, split in items):
        raise ValueError("Every query must have text and a known source split.")

    test_keys = {key for _, key, split in items if split == "test"}
    assignments = []
    dropped = 0
    for query_id, key, split in items:
        if split == "test":
            assignments.append((query_id, key, "test"))
        elif key in test_keys:
            dropped += 1
        else:
            bucket = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100
            assigned = "validation" if bucket < VALIDATION_PERCENT else "train"
            assignments.append((query_id, key, assigned))
    return assignments, dropped


def download(raw: Path):
    raw.mkdir(parents=True, exist_ok=True)
    base_url = (
        "https://media.githubusercontent.com/media/amazon-science/esci-data/"
        f"{SOURCE['commit']}/shopping_queries_dataset/"
    )
    for name, expected in SOURCE["files"].items():
        path = raw / name
        if path.is_file() and sha256_file(path) == expected:
            print(f"Verified {path}", flush=True)
            continue

        with tempfile.TemporaryDirectory(prefix=".download-", dir=raw) as tmp:
            pending = Path(tmp) / name
            print(f"Downloading {name}", flush=True)
            subprocess.run(
                ["curl", "--fail", "--location", "--retry", "2", "--connect-timeout", "15",
                 "--output", str(pending), base_url + name],
                check=True,
            )
            if sha256_file(pending) != expected:
                raise ValueError(f"Checksum mismatch for {name}.")
            pending.replace(path)


def build_snapshot(db, raw: Path, staging: Path) -> dict:
    db.execute("""
        CREATE TABLE examples AS
        SELECT example_id, query_id, query, product_id, esci_label, split
        FROM read_parquet(?) WHERE product_locale = 'us' AND small_version = 1
    """, [str(raw / EXAMPLES)])
    queries = db.execute("SELECT DISTINCT query_id, query, split FROM examples").fetchall()
    assignments, dropped = assign_splits(queries)
    if not assignments:
        raise ValueError("The US small subset has no queries.")
    ids, keys, splits = zip(*assignments)
    db.execute("""
        CREATE TABLE assignments AS
        SELECT unnest(?::BIGINT[]) AS query_id,
               unnest(?::VARCHAR[]) AS query_key,
               unnest(?::VARCHAR[]) AS dataset_split
    """, [list(ids), list(keys), list(splits)])
    db.execute("""
        CREATE TABLE pairs AS
        SELECT e.example_id, e.query_id, a.query_key, e.query, e.product_id,
               coalesce(trim(p.product_title), '') AS product_title,
               e.esci_label, a.dataset_split, p.product_id IS NULL AS missing_product
        FROM examples e
        JOIN assignments a USING (query_id)
        LEFT JOIN (SELECT product_id, product_title FROM read_parquet(?)
                   WHERE product_locale = 'us') p USING (product_id)
    """, [str(raw / PRODUCTS)])
    invalid, rows, unique_ids = db.execute("""
        SELECT count(*) FILTER (WHERE missing_product OR esci_label IS NULL
                               OR esci_label NOT IN ('E', 'S', 'C', 'I')),
               count(*), count(DISTINCT example_id)
        FROM pairs
    """).fetchone()
    if invalid or rows != unique_ids:
        raise ValueError("The prepared pairs contain invalid labels, missing products, or duplicate IDs.")

    stats = {}
    for split in SPLITS:
        counts = db.execute("""
            SELECT count(*), count(DISTINCT query_id), count(DISTINCT query_key),
                   count(*) FILTER (WHERE product_title = '')
            FROM pairs WHERE dataset_split = ?
        """, [split]).fetchone()
        if counts[0] == 0:
            raise ValueError(f"The {split} split is empty.")
        output = staging / f"{split}.parquet"
        db.execute("""
            COPY (SELECT * EXCLUDE (dataset_split, missing_product) FROM pairs
                  WHERE dataset_split = $split ORDER BY query_id, product_id)
            TO $output (FORMAT PARQUET, COMPRESSION ZSTD)
        """, {"split": split, "output": str(output)})
        stats[split] = dict(zip(("pairs", "queries", "query_groups", "empty_titles"), counts))
        stats[split]["sha256"] = sha256_file(output)

    return {
        "source": SOURCE,
        "engine": f"duckdb {duckdb.__version__}",
        "locale": "us",
        "dataset": "small",
        "split_policy": "Keep official test; split official train by normalized query SHA256 bucket.",
        "validation_percent": VALIDATION_PERCENT,
        "train_queries_dropped_for_test_overlap": dropped,
        "splits": stats,
    }


def prepare(raw: Path, snapshot: Path):
    if snapshot.exists():
        raise FileExistsError(f"Snapshot already exists: {snapshot}")
    for name, expected in SOURCE["files"].items():
        path = raw / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Missing or invalid source file {path}. Run python esci.py download.")
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=snapshot.parent) as tmp:
        staging = Path(tmp) / "snapshot"
        staging.mkdir()
        config = {"threads": "2", "memory_limit": "512MB", "temp_directory": str(Path(tmp) / "spill")}
        with duckdb.connect(config=config) as db:
            manifest = build_snapshot(db, raw, staging)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staging.rename(snapshot)
    print(f"Saved {snapshot / 'manifest.json'}")
    for split, counts in manifest["splits"].items():
        print(f"{split}: {counts['queries']:,} queries, {counts['pairs']:,} pairs")


def main():
    parser = argparse.ArgumentParser(description="Download and freeze the US ESCI data.")
    parser.add_argument("action", choices=("download", "prepare"))
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    args = parser.parse_args()
    try:
        if args.action == "download":
            download(args.raw)
        else:
            prepare(args.raw, args.snapshot)
    except (OSError, ValueError, subprocess.CalledProcessError, duckdb.Error) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
