import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from serve_ranker import INPUTS, OUTPUT, bundle_manifest


class BundleTests(unittest.TestCase):
    def test_bundle_rejects_a_changed_tokenizer(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            files = {"model.onnx": b"model", "tokenizer.json": b"tokenizer"}
            for name, data in files.items():
                (root / name).write_bytes(data)
            manifest = {
                "inputs": list(INPUTS), "output": OUTPUT, "max_length": 96,
                "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            actual, version = bundle_manifest(root)
            self.assertEqual(actual, manifest)
            self.assertEqual(len(version), 64)

            (root / "tokenizer.json").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum mismatch: tokenizer.json"):
                bundle_manifest(root)
