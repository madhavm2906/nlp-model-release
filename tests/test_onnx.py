import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from transformers import BertConfig, BertForSequenceClassification

from export_onnx import OUTPUT, RankerGraph, as_tensors, cpu_session, export_graph, scores_match


class ExportTests(unittest.TestCase):
    def test_export_accepts_different_batch_sizes_and_sequence_lengths(self):
        config = BertConfig(
            vocab_size=20, hidden_size=8, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=16, num_labels=1,
        )
        config._attn_implementation = "eager"
        graph = RankerGraph(BertForSequenceClassification(config)).eval()

        def batch(size, length):
            ids = np.arange(size * length, dtype=np.int64).reshape(size, length) % 20
            mask = np.ones_like(ids)
            mask[:, -1] = 0
            return {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}

        with TemporaryDirectory() as temp, torch.inference_mode():
            path = Path(temp) / "model.onnx"
            export_graph(graph, batch(2, 5), path)
            session = cpu_session(path, 1)
            for size, length in ((1, 3), (4, 12)):
                with self.subTest(size=size, length=length):
                    inputs = batch(size, length)
                    expected = graph(**as_tensors(inputs)).numpy()
                    actual, = session.run([OUTPUT], inputs)
                    self.assertEqual(actual.shape, (size,))
                    scores_match(expected, actual)

    def test_parity_rejects_bad_score_and_shape(self):
        reference = np.array([0.1, 0.2])
        with self.assertRaisesRegex(ValueError, "score error"):
            scores_match(reference, np.array([0.1, 0.3]))
        with self.assertRaisesRegex(ValueError, "expected"):
            scores_match(reference, reference.reshape(2, 1))
