import argparse
import json
import platform
from itertools import groupby
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory
from time import perf_counter

import duckdb
import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from baseline import ndcg
from esci import SNAPSHOT, sha256_file
from train_ranker import Candidate

INPUTS = ("input_ids", "attention_mask", "token_type_ids")
OUTPUT = "scores"
OPSET = 17
SCORE_TOLERANCE = 0.0001
NDCG_TOLERANCE = 0.0001
VALIDATION_BATCH = 64
BENCHMARK_BATCHES = (1, 8, 32)
WARMUP = 15


class RankerGraph(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
        )
        return output.logits.squeeze(-1)


def encode(tokenizer, rows, max_length):
    batch = tokenizer(
        [row.query for row in rows], [row.title for row in rows],
        padding=True, truncation=True, max_length=max_length, return_tensors="np",
    )
    return {name: batch[name] for name in INPUTS}


def as_tensors(batch):
    return {name: torch.from_numpy(value) for name, value in batch.items()}


def export_graph(graph, sample, path):
    with torch.inference_mode():
        torch.onnx.export(
            graph, tuple(as_tensors(sample).values()), path,
            input_names=list(INPUTS), output_names=[OUTPUT], opset_version=OPSET,
            dynamic_axes={**{name: {0: "batch", 1: "sequence"} for name in INPUTS},
                          OUTPUT: {0: "batch"}},
            dynamo=False, external_data=False,
        )
    onnx.checker.check_model(path)


def cpu_session(path, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def scores_match(reference, exported):
    if reference.shape != exported.shape:
        raise ValueError(f"Export returned {exported.shape}, expected {reference.shape}.")
    if not np.isfinite(reference).all() or not np.isfinite(exported).all():
        raise ValueError("Inference returned a nonfinite score.")
    error = float(np.max(np.abs(reference - exported)))
    if error > SCORE_TOLERANCE:
        raise ValueError(f"ONNX score error {error:.8f} exceeds {SCORE_TOLERANCE:.8f}.")
    return error


def ranking_results(rows, scores):
    for _, group in groupby(zip(rows, scores, strict=True), key=lambda item: item[0].query_id):
        ranked = sorted(group, key=lambda item: (-item[1], item[0].product))
        yield ranked[0][0].product, ndcg([row.label for row, _ in ranked])


@torch.inference_mode()
def check_validation(graph, session, tokenizer, rows, max_length):
    pytorch_scores, onnx_scores = [], []
    max_error = 0.0
    for start in range(0, len(rows), VALIDATION_BATCH):
        batch = encode(tokenizer, rows[start:start + VALIDATION_BATCH], max_length)
        reference = graph(**as_tensors(batch)).numpy()
        result, = session.run([OUTPUT], batch)
        max_error = max(max_error, scores_match(reference, result))
        pytorch_scores.extend(reference.tolist())
        onnx_scores.extend(result.tolist())

    reference = list(ranking_results(rows, pytorch_scores))
    exported = list(ranking_results(rows, onnx_scores))
    pytorch_ndcg = fmean(score for _, score in reference)
    onnx_ndcg = fmean(score for _, score in exported)
    if abs(onnx_ndcg - pytorch_ndcg) > NDCG_TOLERANCE:
        raise ValueError("The exported model changed validation nDCG beyond tolerance.")
    return {
        "split": "validation", "queries": len(reference), "candidate_pairs": len(rows),
        "pytorch_ndcg_at_10": pytorch_ndcg, "onnx_ndcg_at_10": onnx_ndcg,
        "max_absolute_score_error": max_error, "score_tolerance": SCORE_TOLERANCE,
        "ndcg_tolerance": NDCG_TOLERANCE,
        "changed_first_products": sum(left[0] != right[0] for left, right in zip(reference, exported, strict=True)),
    }


def measure(infer, inputs, batch_size):
    for batch in inputs[:WARMUP]:
        infer(batch)
    elapsed = []
    for batch in inputs:
        start = perf_counter()
        infer(batch)
        elapsed.append(perf_counter() - start)
    return {
        "p50_ms": float(np.percentile(elapsed, 50) * 1000),
        "p95_ms": float(np.percentile(elapsed, 95) * 1000),
        "pairs_per_second": len(inputs) * batch_size / sum(elapsed),
        "timed_batches": len(inputs),
    }


@torch.inference_mode()
def benchmark(graph, session, tokenizer, rows, max_length, iterations):
    results = []
    for size in BENCHMARK_BATCHES:
        if iterations * size > len(rows):
            raise ValueError(f"Validation has too few pairs for {iterations} batches of {size}.")
        starts = np.linspace(0, len(rows) - size, num=iterations, dtype=int)
        inputs = [encode(tokenizer, rows[start:start + size], max_length) for start in starts]
        tensors = [as_tensors(batch) for batch in inputs]
        results.append({
            "batch_size": size,
            "pytorch": measure(lambda batch: graph(**batch), tensors, size),
            "onnx": measure(lambda batch: session.run([OUTPUT], batch), inputs, size),
        })
    return results


def validation_rows(snapshot, training):
    manifest_path = snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = snapshot / "validation.parquet"
    if (sha256_file(manifest_path) != training["snapshot_manifest_sha256"]
            or sha256_file(path) != manifest["splits"]["validation"]["sha256"]):
        raise ValueError("Validation data does not match the training run.")
    with duckdb.connect(config={"threads": "2"}) as db:
        rows = db.execute("""
            SELECT query_id, query, product_id, product_title, esci_label
            FROM read_parquet(?) ORDER BY query_id, product_id
        """, [str(path)]).fetchall()
    if len(rows) != training["candidate_pairs"] or not rows:
        raise ValueError("Validation candidate count does not match the training run.")
    return [Candidate(*row) for row in rows]


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def run(args):
    if args.output.exists():
        raise FileExistsError(f"Export already exists: {args.output}. Choose a new --output directory.")
    training = json.loads((args.model / "metrics.json").read_text(encoding="utf-8"))
    if sha256_file(args.model / "model.safetensors") != training["model_sha256"]:
        raise ValueError("The trained model does not match the recorded model checksum.")
    rows = validation_rows(args.snapshot, training)
    max_length = training["settings"]["max_length"]
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, local_files_only=True, use_safetensors=True, attn_implementation="sdpa",
    )
    graph = RankerGraph(model).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".onnx-", dir=args.output.parent) as temp:
        pending = Path(temp) / "bundle"
        pending.mkdir()
        path = pending / "model.onnx"
        export_model = AutoModelForSequenceClassification.from_pretrained(
            args.model, local_files_only=True, use_safetensors=True, attn_implementation="eager",
        )
        export_graph(RankerGraph(export_model).eval(), encode(tokenizer, rows[:2], max_length), path)
        del export_model
        tokenizer.save_pretrained(pending)
        session = cpu_session(path, args.threads)
        print(f"Checking both runtimes on {len(rows):,} validation pairs", flush=True)
        parity = check_validation(graph, session, tokenizer, rows, max_length)
        if (parity["queries"] != training["queries"]
                or abs(parity["pytorch_ndcg_at_10"] - training["ndcg_at_10"]["fine_tuned"]) > NDCG_TOLERANCE):
            raise ValueError("PyTorch validation does not match the training run.")
        print("Benchmarking warm CPU inference", flush=True)
        benchmark_results = benchmark(graph, session, tokenizer, rows, max_length, args.iterations)
        manifest = {
            "source_model_sha256": training["model_sha256"],
            "snapshot_manifest_sha256": training["snapshot_manifest_sha256"],
            "opset": OPSET, "max_length": max_length, "inputs": list(INPUTS), "output": OUTPUT,
            "files": {file.name: sha256_file(file) for file in sorted(pending.iterdir())},
        }
        write_json(pending / "manifest.json", manifest)
        write_json(pending / "benchmark.json", {
            "runtime": {"cpu_threads": args.threads, "machine": platform.machine(),
                        "pytorch": torch.__version__, "pytorch_attention": "sdpa",
                        "onnxruntime": ort.__version__},
            "scope": "Warm sequential batches; tokenization, model loading and network excluded.",
            "validation": parity, "latency": benchmark_results,
        })
        pending.rename(args.output)

    print(f"PyTorch validation nDCG@10: {parity['pytorch_ndcg_at_10']:.6f}")
    print(f"ONNX validation nDCG@10:    {parity['onnx_ndcg_at_10']:.6f}")
    print(f"Maximum absolute score difference: {parity['max_absolute_score_error']:.8f}")
    for row in benchmark_results:
        print(f"Batch {row['batch_size']:2d} p95: PyTorch {row['pytorch']['p95_ms']:.3f} ms, "
              f"ONNX {row['onnx']['p95_ms']:.3f} ms")
    print(f"Saved ONNX bundle and benchmark to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Export the trained ranker and verify CPU inference.")
    parser.add_argument("--model", type=Path, default=Path("artifacts/esci-tinybert-v1"))
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--output", type=Path, default=Path("artifacts/esci-tinybert-v1-onnx"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=150)
    args = parser.parse_args()
    if args.threads < 1 or args.iterations < 30:
        parser.error("Use at least one CPU thread and 30 benchmark iterations.")
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, duckdb.Error) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
