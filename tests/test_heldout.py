import unittest

import numpy as np

from evaluate_test import interval, query_metrics
from train_ranker import Candidate


class HeldoutMetricsTests(unittest.TestCase):
    def test_exact_first_has_a_defined_denominator_and_tie_break(self):
        rows = [
            Candidate(1, "usb cable", "b", "HDMI cable", "I"),
            Candidate(1, "usb cable", "a", "USB C cable", "E"),
            Candidate(2, "office chair", "c", "Desk chair", "S"),
        ]
        scores, exact_top1 = query_metrics(rows, [0.5, 0.5, 0.8])
        self.assertEqual(scores.tolist(), [1.0, 1.0])
        self.assertEqual(exact_top1, [True])

    def test_rejects_missing_and_nonfinite_scores(self):
        row = Candidate(1, "usb cable", "a", "USB C cable", "E")
        with self.assertRaises(ValueError):
            query_metrics([row], [])
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            query_metrics([row], [float("nan")])

    def test_paired_bootstrap_preserves_a_constant_delta(self):
        self.assertEqual(interval(np.array([0.125, 0.125, 0.125])), [0.125, 0.125])
