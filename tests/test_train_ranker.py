import unittest

import torch

from train_ranker import Candidate, pairwise_loss, sample_comparisons


class RankerTests(unittest.TestCase):
    def test_samples_only_different_labels_from_the_same_query(self):
        rows = [
            Candidate(1, "usb cable", "a", "HDMI cable", "I"),
            Candidate(1, "usb cable", "b", "USB C cable", "E"),
            Candidate(1, "usb cable", "c", "USB A cable", "S"),
            Candidate(2, "chair", "a", "Office chair", "E"),
        ]
        comparisons = sample_comparisons(rows, limit=6, seed=42)
        self.assertEqual(comparisons, sample_comparisons(rows, limit=6, seed=42))
        self.assertEqual(
            {(pair.better.product, pair.worse.product, pair.weight) for pair in comparisons},
            {("b", "a", 1.0), ("b", "c", 0.9), ("c", "a", 0.1)},
        )
        self.assertEqual(len(sample_comparisons(rows, limit=1, seed=42)), 1)
        self.assertTrue(all(pair.better.query_id == pair.worse.query_id == 1 for pair in comparisons))

    def test_loss_rewards_correct_ordering(self):
        high = torch.tensor([2.0, 1.5])
        low = torch.tensor([0.0, -1.0])
        weights = torch.tensor([1.0, 0.1])
        self.assertLess(pairwise_loss(high, low, weights).item(),
                        pairwise_loss(low, high, weights).item())
