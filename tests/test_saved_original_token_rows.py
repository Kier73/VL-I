"""Original checkpoint reports, distinct token rows, no false full-decoder claim."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

from vli.ground import _hex64
from vli.large_model_comparison import SOURCES

SOURCE=Path(__file__).resolve().parents[1]/"experiments/token_embedding/datasets/2026-11-19-real-original-token-rows.json"


class TestArchivedOriginalTokenRows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipt=json.loads(SOURCE.read_text("utf-8"))

    def test_both_original_checkpoint_revisions_are_retained(self):
        report=self.receipt
        self.assertEqual(report["schema"],"vl-i.saved-original-token-embedding-evidence.v1")
        self.assertEqual(report["github_actions_run"],44648895450)
        self.assertEqual(set(report["models"]),set(SOURCES))
        self.assertTrue(report["no_pretrained_weight_bytes_stored_in_virtual_ground"])
        self.assertFalse(report["native_tokenizer_applied"])
        self.assertFalse(report["full_decoder_completed"])
        for route,model in SOURCES.items():
            rec=report["models"][route]
            self.assertEqual(rec["model"],model["model"])
            self.assertEqual(rec["revision"],model["revision"])
            self.assertEqual(rec["advertised_weight_bytes"],model["advertised_bytes"])
            self.assertTrue(rec["checkpoint_restoration_verified"])
            self.assertEqual([v["token_id"] for v in rec["states"]],[100,101])
            self.assertEqual(len({s["coordinate"] for s in rec["states"]}),2)
            self.assertEqual(len({s["embedding_sha256"] for s in rec["states"]}),2)
            for state in rec["states"]:
                self.assertEqual(state["embedding_bytes"],10240)
                self.assertGreater(state["trained_bytes"],state["embedding_bytes"])
                self.assertGreater(state["trained_ranges"],0)
                self.assertEqual(state["render_state"],"RENDERED")
                self.assertTrue(state["coordinate"].startswith("state://vli/"))
                for field in ("embedding_sha256","range_chain"):
                    self.assertTrue(_hex64(state[field]),field)

    def test_qwen_true_first_layer_trained_gate_values_change(self):
        q=self.receipt["models"]["qwen-27b"]
        self.assertIn("genuine_initial",q["scope"])
        a,b=[x["constructed"] for x in q["states"]]
        self.assertEqual(a["heads"],b["heads"])
        self.assertEqual(a["heads"],48)
        self.assertEqual(a["beta_above_half"],44)
        self.assertEqual(b["beta_above_half"],41)
        self.assertNotEqual(a["mean_beta"],b["mean_beta"])

    def test_deepseek_embedding_routing_is_not_promoted_to_full_prefill(self):
        d=self.receipt["models"]["deepseek-v41-flash"]
        self.assertIn("not_true_moe_causal_input",d["scope"])
        a,b=[x["constructed"] for x in d["states"]]
        self.assertEqual(len(a["selected"]),6)
        self.assertEqual(len(b["selected"]),6)
        self.assertEqual(len(set(a["selected"])),6)
        self.assertEqual(len(set(b["selected"])),6)
        self.assertNotEqual(a["selected"],b["selected"])
        self.assertNotEqual(a["weighted_partial"],b["weighted_partial"])


if __name__=="__main__":
    unittest.main()
