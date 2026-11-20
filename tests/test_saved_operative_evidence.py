"""Regression on published, truly executed pretrained-operator CI evidence.

This is archive-contract verification, NOT independent re-running of remote
tensor bytes or claims of full Qwen / DeepSeek decoder inference.
"""
from __future__ import annotations

import json
from pathlib import Path
import unittest

from vli.large_model_comparison import SOURCES
from vli.ground import _hex64

SOURCE=Path(__file__).resolve().parents[1]/"experiments/operative_model/datasets/2026-11-19-two-large-model-constructed-results.json"


class TestSavedOperativePretrainedObservations(unittest.TestCase):
    def test_original_models_two_actual_observer_inputs_each(self):
        report=json.loads(SOURCE.read_text(encoding="utf-8"))
        self.assertEqual(report["schema"],"vl-i.saved-operative-pretrained-observations.v1")
        self.assertEqual(report["run_id"],44646397252)
        self.assertEqual(set(report["models"]),set(SOURCES))
        self.assertFalse(report["interpreter_core_changed"])
        self.assertIn("not model token inference",report["scope"])
        for route,source in SOURCES.items():
            entry=report["models"][route]
            self.assertEqual(entry["model"],source["model"])
            self.assertEqual(entry["revision"],source["revision"])
            observations=entry["observations"]
            self.assertEqual(len(observations),2)
            self.assertEqual(len(set(r["coordinate"] for r in observations)),2)
            self.assertEqual(len(set(r["intent_digest"] for r in observations)),2)
            for row in observations:
                self.assertTrue(row["coordinate"].startswith("state://vli/"))
                for field in ("intent_digest","observation_digest","range_chain"):
                    self.assertTrue(_hex64(row[field]),field)
                self.assertGreater(row["weight_bytes"],0)
                self.assertGreater(row["range_reads"],0)
                self.assertEqual(row["rendered"],"RENDERED")
                self.assertEqual(row["reference_error"],0)

    def test_trained_mathematical_output_changed_on_observer_input(self):
        report=json.loads(SOURCE.read_text(encoding="utf-8"))
        a,b=report["models"]["qwen-27b"]["observations"]
        self.assertEqual(a["constructed"]["heads"],48)
        self.assertEqual(b["constructed"]["heads"],48)
        self.assertNotEqual(a["constructed"]["mean_beta"],b["constructed"]["mean_beta"])
        self.assertNotEqual(a["constructed"]["beta_above_half"],b["constructed"]["beta_above_half"])
        self.assertEqual(a["weight_bytes"],993472)
        self.assertEqual(b["weight_bytes"],993472)
        c,d=report["models"]["deepseek-v41-flash"]["observations"]
        self.assertEqual(len(c["constructed"]["selected"]),6)
        self.assertEqual(len(d["constructed"]["selected"]),6)
        self.assertEqual(len(set(c["constructed"]["selected"])),6)
        self.assertEqual(len(set(d["constructed"]["selected"])),6)
        self.assertNotEqual(c["constructed"]["selected"],d["constructed"]["selected"])
        self.assertNotEqual(c["constructed"]["weighted_partial"],d["constructed"]["weighted_partial"])
        self.assertEqual(c["weight_bytes"],3973680)
        self.assertEqual(d["weight_bytes"],3973680)


if __name__=="__main__":unittest.main()
