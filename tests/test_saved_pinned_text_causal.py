"""Regression on archived *actual* Qwen/DeepSeek paragraph-driven trained evidence.

This validates pinned receipt contracts and does not reproduce model weights
or independently rerun the original numerical ranges.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import unittest

from vli.ground import _hex64
from vli.large_model_comparison import SOURCES

ARCHIVE=Path(__file__).resolve().parents[1]/(
    "experiments/pinned_text/datasets/"
    "2026-11-19-real-paragraph-token-causal-results.json")


class TestArchivedPinnedTextToCausal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=json.loads(ARCHIVE.read_text("utf-8"))

    def test_exact_unscaled_weight_sources_and_native_text_positions(self):
        r=self.report
        self.assertEqual(r["schema"],"vl-i.saved-pinned-text-to-trained-causal-evidence.v1")
        self.assertEqual(r["run_id"],44652875978)
        self.assertEqual(set(r["original_sources"]),set(SOURCES))
        self.assertEqual(r["model_weights_persisted_inside_ground"],0)
        self.assertFalse(r["full_pretrained_decoder_completed"])
        self.assertFalse(r["complete_logits_observed"])
        self.assertTrue(_hex64(r["prompt_sha256"]))
        for route,src in SOURCES.items():
            model=r["original_sources"][route]
            self.assertEqual(model["model"],src["model"])
            self.assertEqual(model["revision"],src["revision"])
            self.assertEqual(model["model_index_sha256"],src["index_sha256"])
            self.assertEqual(model["full_checkpoint_advertised_bytes"],
                             src["advertised_bytes"])
            self.assertEqual(model["token_count"],13)
            self.assertEqual(len(model["native_token_ids"]),13)
            self.assertEqual(len(model["observed_steps"]),2)
            self.assertTrue(model["checkpoint_restored"])
            self.assertTrue(_hex64(model["checkpoint_head"]))
            self.assertTrue(_hex64(model["tokenizer_sha256"]))
            self.assertEqual(model["native_token_ids"][:2],
                             [z["token_id"] for z in model["observed_steps"]])
            self.assertNotEqual(model["observed_steps"][0]["embedding_sha256"],
                                model["observed_steps"][1]["embedding_sha256"])
            for item in model["observed_steps"]:
                self.assertEqual(item["embedding_bytes"],10240)
                self.assertTrue(_hex64(item["embedding_sha256"]))

    def test_true_qwen_stateful_recurrence_and_cross_token_outputs(self):
        q=self.report["original_sources"]["qwen-27b"]
        self.assertEqual(q["tokenizer_revision"],q["revision"])
        self.assertEqual(q["tokenizer_grade"],"SAME_PINNED_CHECKPOINT_TOKENIZER")
        self.assertIn("causal",q["trained_numeric_scope"])
        x,y=q["observed_steps"]
        self.assertLessEqual(x["independent_recurrence_max_abs_error"],2e-8)
        self.assertLessEqual(y["independent_recurrence_max_abs_error"],2e-8)
        self.assertNotEqual(x["convolution_digest"],y["convolution_digest"])
        self.assertNotEqual(x["recurrent_state_digest"],y["recurrent_state_digest"])
        self.assertNotEqual(x["selected_head_out_row0_contribution"],
                            y["selected_head_out_row0_contribution"])
        self.assertEqual(x["trained_head_ranges"],98)
        self.assertEqual(y["trained_head_ranges"],98)
        self.assertTrue(x["selected_head_out_row0_contribution"]<0)
        self.assertTrue(y["selected_head_out_row0_contribution"]>0)

    def test_deepseek_related_tokenizer_and_noncausal_router_not_promoted(self):
        d=self.report["original_sources"]["deepseek-v41-flash"]
        self.assertNotEqual(d["tokenizer_revision"],d["revision"])
        self.assertEqual(d["tokenizer_grade"],
             "RELATED_FLASH_RELEASE_NOT_PROVEN_IDENTICAL_TO_1017")
        self.assertIn("no model-causal post-attention",d["trained_numeric_scope"])
        x,y=d["observed_steps"]
        self.assertEqual(len(set(x["selected_experts"])),6)
        self.assertEqual(len(set(y["selected_experts"])),6)
        self.assertNotEqual(x["selected_experts"],y["selected_experts"])
        self.assertNotEqual(x["weighted_partial_contribution"],
                            y["weighted_partial_contribution"])
        self.assertTrue(all(math.isfinite(a["weighted_partial_contribution"])
                         for a in (x,y)))


if __name__=="__main__":
    unittest.main()
