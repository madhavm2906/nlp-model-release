import unittest

from release_replay import decision


def version(quality, latency, problems=None):
    return {
        "ndcg_at_10": quality,
        "http": {"median_http_p95_ms": latency, "problems": problems or []},
    }


class ReleaseCheckTests(unittest.TestCase):
    def test_allows_candidate_within_budgets(self):
        result = decision(version(0.720, 60), version(0.718, 55))
        self.assertEqual(result, {"status": "eligible", "reasons": []})

    def test_blocks_quality_and_latency_regressions(self):
        result = decision(version(0.720, 60), version(0.700, 70))
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(len(result["reasons"]), 2)

    def test_blocks_failed_replay(self):
        failed = version(0.720, None, ["Replay returned HTTP 502."])
        result = decision(version(0.720, 60), failed)
        self.assertEqual(result["status"], "blocked")
        self.assertIn("Candidate: Replay returned HTTP 502.", result["reasons"])
