import copy
import unittest

from model_release import verify_evidence
from release_replay import CLIENTS, MAX_LATENCY_RATIO, MAX_NDCG_DROP, REPEATS, REQUESTS, THREADS


def report_for_test():
    def version(name, latency):
        run = {"model_versions": [name], "status_counts": {"200": REQUESTS}, "request_p95_ms": latency}
        return {"version": name, "ndcg_at_10": 0.72,
                "http": {"runs": [copy.deepcopy(run) for _ in range(REPEATS)], "problems": []}}

    return {
        "decision": {"status": "eligible", "reasons": []}, "split": "validation",
        "snapshot_manifest_sha256": "validation-digest",
        "limits": {"max_ndcg_drop": MAX_NDCG_DROP, "max_latency_ratio": MAX_LATENCY_RATIO},
        "workload": {"cpu_threads": THREADS, "requests_per_run": REQUESTS,
                     "concurrency": CLIENTS, "repeats": REPEATS},
        "current": version("fp32", 61), "candidate": version("int8", 60),
    }


class EvidenceTests(unittest.TestCase):
    def test_verified_bundles_can_use_passing_evidence(self):
        verify_evidence(report_for_test(), "fp32", "int8", "validation-digest")

    def test_blocked_and_stale_reports_are_rejected(self):
        report = report_for_test()
        with self.assertRaisesRegex(ValueError, "Report current version"):
            verify_evidence(report, "another-model", "int8", "validation-digest")
        report["decision"] = {"status": "blocked", "reasons": ["Latency above budget."]}
        with self.assertRaisesRegex(ValueError, "Latency above budget"):
            verify_evidence(report, "fp32", "int8", "validation-digest")

    def test_decision_is_recomputed_from_replays(self):
        report = report_for_test()
        for run in report["candidate"]["http"]["runs"]:
            run["request_p95_ms"] = 80
        with self.assertRaisesRegex(ValueError, "p95"):
            verify_evidence(report, "fp32", "int8", "validation-digest")

    def test_failed_or_wrong_version_replay_is_rejected(self):
        report = report_for_test()
        report["candidate"]["http"]["runs"][0]["model_versions"] = ["fp32"]
        with self.assertRaisesRegex(ValueError, "lacks successful replays"):
            verify_evidence(report, "fp32", "int8", "validation-digest")
