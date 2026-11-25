import argparse
import hashlib
import json
from pathlib import Path
from threading import Lock

import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from tokenizers import Tokenizer

INPUTS = ("input_ids", "attention_mask", "token_type_ids")
OUTPUT = "scores"
MAX_BATCH = 64


class Pair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=512)
    title: str = Field(min_length=1, max_length=2048)


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pairs: list[Pair] = Field(min_length=1, max_length=MAX_BATCH)


def bundle_manifest(directory):
    raw = (directory / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    if manifest["inputs"] != list(INPUTS) or manifest["output"] != OUTPUT:
        raise ValueError("Bundle has an incompatible input or output contract.")
    if not 8 <= manifest["max_length"] <= 512:
        raise ValueError("Bundle has an unsupported token limit.")
    if not {"model.onnx", "tokenizer.json"} <= manifest["files"].keys():
        raise ValueError("Bundle is missing model or tokenizer checksums.")
    for name, expected in manifest["files"].items():
        if name in {".", ".."} or Path(name).name != name:
            raise ValueError("Bundle contains an invalid file name.")
        with (directory / name).open("rb") as file:
            if hashlib.file_digest(file, "sha256").hexdigest() != expected:
                raise ValueError(f"Bundle checksum mismatch: {name}")
    return manifest, hashlib.sha256(raw).hexdigest()


class Ranker:
    def __init__(self, directory, threads):
        manifest, self.version = bundle_manifest(directory)
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        pad_id = self.tokenizer.token_to_id("[PAD]")
        if pad_id is None:
            raise ValueError("The BERT tokenizer is missing its padding token.")
        self.tokenizer.enable_truncation(max_length=manifest["max_length"])
        self.tokenizer.enable_padding(pad_id=pad_id, pad_token="[PAD]")

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(
            str(directory / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"],
        )
        self.score([Pair(query="usb cable", title="USB C cable")])

    def inputs(self, pairs):
        encoded = self.tokenizer.encode_batch([(pair.query, pair.title) for pair in pairs])
        return {
            "input_ids": np.asarray([row.ids for row in encoded], dtype=np.int64),
            "attention_mask": np.asarray([row.attention_mask for row in encoded], dtype=np.int64),
            "token_type_ids": np.asarray([row.type_ids for row in encoded], dtype=np.int64),
        }

    def score(self, pairs):
        scores, = self.session.run([OUTPUT], self.inputs(pairs))
        if scores.shape != (len(pairs),) or not np.isfinite(scores).all():
            raise ValueError("Model returned invalid scores.")
        return scores.tolist()


def create_app(ranker):
    app = FastAPI(title="ONNX ranking worker")
    scoring = Lock()

    @app.get("/healthz")
    def health():
        return {"status": "ok", "model_version": ranker.version}

    @app.post("/score")
    def score(request: ScoreRequest):
        if not scoring.acquire(blocking=False):
            raise HTTPException(status_code=503, detail="Scoring worker is busy")
        try:
            scores = ranker.score(request.pairs)
        except ValueError as exc:
            raise HTTPException(status_code=500, detail="Invalid inference result") from exc
        finally:
            scoring.release()
        return {"model_version": ranker.version, "scores": scores}

    return app


def main():
    parser = argparse.ArgumentParser(description="Serve a verified ONNX ranker bundle on CPU.")
    parser.add_argument("--bundle", type=Path, default=Path("artifacts/esci-tinybert-v1-onnx"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    try:
        ranker = Ranker(args.bundle, args.threads)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"Could not load ONNX bundle: {exc}") from exc
    uvicorn.run(create_app(ranker), host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
