import argparse
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process

from esci import sha256_file
from serve_ranker import Ranker, bundle_manifest


def build(source: Path, output: Path):
    if output.exists():
        raise FileExistsError(f"Candidate already exists: {output}")

    manifest, parent_version = bundle_manifest(source)
    source_graph = source / "model.onnx"
    output.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix=".int8-", dir=output.parent) as temp:
        pending = Path(temp) / "bundle"
        pending.mkdir()
        for name in manifest["files"]:
            if name != "model.onnx":
                shutil.copyfile(source / name, pending / name)

        prepared = Path(temp) / "prepared.onnx"
        # Keep standard ONNX shape inference for the graph's dynamic axes.
        quant_pre_process(
            str(source_graph), str(prepared),
            skip_optimization=True, skip_symbolic_shape=True,
        )
        graph = pending / "model.onnx"
        quantize_dynamic(
            str(prepared), str(graph), weight_type=QuantType.QInt8,
            per_channel=True, op_types_to_quantize=["MatMul"],
            extra_options={"MatMulConstBOnly": True},
        )
        onnx.checker.check_model(str(graph))
        model = onnx.load(str(graph))
        if not any(node.op_type == "MatMulInteger" for node in model.graph.node):
            raise ValueError("The candidate contains no quantized matrix multiplications.")

        manifest["parent_model_version"] = parent_version
        manifest["optimization"] = "Dynamic INT8 weights for constant matrix multiplications"
        manifest["files"] = {p.name: sha256_file(p) for p in sorted(pending.iterdir())}
        (pending / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
        )
        candidate = Ranker(pending, threads=1)
        pending.rename(output)

    print(f"FP32 model bytes: {source_graph.stat().st_size:,}")
    print(f"INT8 model bytes: {(output / 'model.onnx').stat().st_size:,}")
    print(f"Saved candidate {candidate.version} to {output}")


def main():
    parser = argparse.ArgumentParser(description="Build an INT8 ONNX candidate.")
    parser.add_argument("--source", type=Path, default=Path("artifacts/esci-tinybert-v1-onnx"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/esci-tinybert-v1-int8"))
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == "__main__":
    main()
