import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from release_dashboard import read_report
from release_store import initialize


class DashboardTests(unittest.TestCase):
    def test_loads_optional_report(self):
        with TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            self.assertIsNone(read_report(report))
            report.write_text(json.dumps({"status": "blocked"}), encoding="utf-8")
            self.assertEqual(read_report(report), {"status": "blocked"})

    def test_bootstrap_keeps_an_existing_release(self):
        db = Mock()
        current = {"id": 2, "version": "new-version"}
        db.execute.return_value.fetchone.return_value = current
        with patch("release_store.bundle_manifest") as validate:
            result = initialize(db, Path("missing-default-bundle"), keep_existing=True)
        self.assertEqual(result, current)
        validate.assert_not_called()
        self.assertFalse(any("INSERT" in call.args[0] for call in db.execute.call_args_list))
