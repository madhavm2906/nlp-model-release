import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb

from baseline import BM25, ndcg
from esci import EXAMPLES, PRODUCTS, assign_splits, build_snapshot


class DatasetTests(unittest.TestCase):
    def test_test_queries_are_held_out_even_when_spelled_differently(self):
        rows = [(1, "USB  Ｃ", "train"), (2, "usb c", "test"), (3, "headphones", "train")]
        assignments, dropped = assign_splits(rows)
        self.assertEqual(dropped, 1)
        self.assertEqual(assignments[0], (2, "usb c", "test"))
        self.assertEqual([row[0] for row in assignments], [2, 3])

    def test_query_aliases_stay_in_one_split(self):
        rows = [(1, "AA Batteries", "train"), (2, "aa   batteries", "train")]
        assignments, dropped = assign_splits(rows)
        self.assertEqual(dropped, 0)
        self.assertEqual(assignments[0][1:], assignments[1][1:])

    def test_snapshot_writes_the_assigned_queries_to_each_file(self):
        with TemporaryDirectory() as folder, duckdb.connect() as db:
            raw = Path(folder)
            snapshot = raw / "snapshot"
            snapshot.mkdir()
            db.execute("""
                COPY (SELECT * FROM (VALUES
                    (1, 1, 'usb cable', 'cable', 'us', 'E', 1, 'train'),
                    (2, 2, 'keyboard', 'keyboard', 'us', 'E', 1, 'train'),
                    (3, 3, 'speaker', 'speaker', 'us', 'E', 1, 'test')
                ) AS t(example_id, query_id, query, product_id, product_locale,
                       esci_label, small_version, split)) TO ? (FORMAT PARQUET)
            """, [str(raw / EXAMPLES)])
            db.execute("""
                COPY (SELECT * FROM (VALUES
                    ('cable', 'USB cable', 'us'), ('keyboard', 'Keyboard', 'us'),
                    ('speaker', 'Bluetooth speaker', 'us')
                ) AS t(product_id, product_title, product_locale)) TO ? (FORMAT PARQUET)
            """, [str(raw / PRODUCTS)])
            build_snapshot(db, raw, snapshot)
            expected = (("train", 1), ("validation", 2), ("test", 3))
            for split, query_id in expected:
                with self.subTest(split=split):
                    actual = db.execute(
                        "SELECT query_id FROM read_parquet(?)", [str(snapshot / f"{split}.parquet")]
                    ).fetchall()
                    self.assertEqual(actual, [(query_id,)])


class RankingTests(unittest.TestCase):
    def test_bm25_prefers_a_matching_title(self):
        model = BM25(["blue usb cable", "wireless headphones", "usb charger"])
        self.assertGreater(model.score({"usb", "cable"}, "blue usb cable"),
                           model.score({"usb", "cable"}, "usb charger"))
        self.assertEqual(model.score({"usb", "cable"}, "wireless headphones"), 0)

    def test_ndcg_rewards_better_ordering_and_handles_zero_gain(self):
        self.assertEqual(ndcg(["E", "S", "C", "I"]), 1.0)
        self.assertLess(ndcg(["I", "C", "S", "E"]), 1.0)
        self.assertEqual(ndcg(["I", "I"]), 0.0)
