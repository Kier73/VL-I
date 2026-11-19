"""B1/B2 comparative gates using the *actual* large-scale VL-64 archived receipts.

All tests use immutable source observations, not substitute/smaller models.
No fixture test makes claims about live pretrained text generation or equality.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from vli.ground import load_json
from vli.large_model_comparison import (
    ARCHIVE_DIR, BASELINE_COMMIT, SOURCES, TASKS, admit_archived_subgraphs,
    comparison, run,
)


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream" / "vl64"


def reports():
    assert UPSTREAM.is_dir(), "Pin and checkout the exact VL-64 source"
    base = UPSTREAM / ARCHIVE_DIR
    return (
        load_json(base / "2026-11-19-large-model-frontier-observations.json"),
        load_json(base / "2026-11-19-qwen-deepseek-trained-subgraphs.json"),
        load_json(base / "2026-11-19-real-trained-weight-glance-replays.json"),
    )


class TestUnscaledLargeModelEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.structural, cls.trained, cls.replays = reports()

    def test_pinned_upstream_checkout_only(self):
        import subprocess
        rev = subprocess.check_output(["git","-C",str(UPSTREAM),"rev-parse","HEAD"],
                                      text=True).strip()
        self.assertEqual(rev, BASELINE_COMMIT)

    def test_two_real_original_scale_models_without_downscaling(self):
        grounds = admit_archived_subgraphs(
            self.structural, self.trained, self.replays)
        self.assertEqual(set(grounds), set(SOURCES))
        q, d = grounds["qwen-27b"], grounds["deepseek-v41-flash"]
        self.assertEqual(q["layers"], 64)
        self.assertEqual(d["layers"], 40)
        self.assertEqual(q["full_model_advertised_weight_bytes"], 55562855904)
        self.assertEqual(d["full_model_advertised_weight_bytes"], 510286023000)
        self.assertEqual(q["archived_trained_weight_bytes_read"], 5277188)
        self.assertEqual(d["archived_trained_weight_bytes_read"], 3973680)
        self.assertTrue(all(not x["pretrained_forward_completed"]
                         for x in grounds.values()))
        self.assertTrue(all(not x["complete_full_vocab_logits_observed"]
                         for x in grounds.values()))
        self.assertTrue(all(x["replay_additional_remote_range_count"] == 0
                         for x in grounds.values()))

    def test_same_b1_intentions_are_compiled_for_both_models(self):
        report = comparison(self.structural, self.trained, self.replays)
        self.assertEqual(report["same_model_same_intention_full_output_parity"], "UNRESOLVED")
        self.assertEqual(report["b1_observation_grade"], "NO_PROVIDER_RUN")
        self.assertEqual(report["b2_execution_grade"], "REAL_PRETRAINED_PARTIAL_SUBGRAPHS_ONLY")
        for route in SOURCES:
            entry = report["models"][route]
            self.assertEqual([t["kind"] for t in entry["task_intentions"]], list(TASKS))
            self.assertTrue(all(len(t["exact_b1_intention_digest"]) == 64
                              for t in entry["task_intentions"]))
            self.assertTrue(all(t["parity"] == "UNRESOLVED_DIFFERENT_OBSERVATION_SCOPE"
                              for t in entry["task_intentions"]))
            self.assertFalse(entry["eligibility_for_B2_full_model"])
            self.assertFalse(entry["eligibility_for_B3_predictive_shortcut"])
            self.assertEqual(entry["nearest_unresolved_frontier"]["first"],
                             "pinned_tokenizer_and_chat_template")

    def test_checkpointed_real_b1_protocol_fixture_is_not_inference(self):
        from experiments.model_fabric.remote_large_recovery_benchmark import run_benchmark
        with tempfile.TemporaryDirectory() as tmp:
            b1 = run_benchmark(live=False, directory=Path(tmp)/"control", max_tokens=64)
        result = comparison(self.structural, self.trained, self.replays, b1=b1)
        self.assertEqual(result["b1_observation_grade"], "B1_CONTROL_FIXTURE_ONLY")
        self.assertEqual(set(result["models"]), set(SOURCES))
        for route in SOURCES:
            metrics=result["models"][route]["b1_metrics"]
            self.assertEqual(metrics["mode"], "B1_CONTROL_FIXTURE_ONLY")
            self.assertTrue(metrics["checkpoint_continuity_observed"])
            self.assertFalse(metrics["underlying_pinned_weight_snapshot_attested"])
            self.assertFalse(metrics["provider_KV_state_recovered"])
            self.assertFalse(any(t["b1_result_available"]
                             for t in result["models"][route]["task_intentions"]))

    def test_ground_digest_and_actual_observation_index_are_bound(self):
        changed=deepcopy(self.structural)
        changed["models"]["qwen"]["report"]["index_sha256"]="0"*64
        with self.assertRaisesRegex(ValueError,"inventory"):
            admit_archived_subgraphs(changed,self.trained,self.replays)

    def test_different_deepseek_snapshot_revision_is_not_comparable(self):
        changed=deepcopy(self.trained)
        changed["deepseek"]["revision"]="f"*40
        with self.assertRaisesRegex(ValueError,"revision"):
            admit_archived_subgraphs(self.structural,changed,self.replays)

    def test_qwen_fake_entire_layer_is_not_b2_parity(self):
        changed=deepcopy(self.trained)
        changed["qwen"]["status"]="FULL_PRETRAINED_MODEL_VERIFIED"
        with self.assertRaisesRegex(ValueError,"Qwen"):
            admit_archived_subgraphs(self.structural,changed,self.replays)

    def test_deepseek_incomplete_selected_expert_set_refused(self):
        changed=deepcopy(self.trained)
        changed["deepseek"]["selected_fp4_slices"]["slices"].pop()
        with self.assertRaisesRegex(ValueError,"DeepSeek"):
            admit_archived_subgraphs(self.structural,changed,self.replays)

    def test_deepseek_nonfinite_or_wrong_numeric_reference_refused(self):
        changed=deepcopy(self.trained)
        changed["deepseek"]["selected_fp4_slices"]["max_abs_independent_error"]=0.1
        with self.assertRaisesRegex(ValueError,"reference"):
            admit_archived_subgraphs(self.structural,changed,self.replays)

    def test_glance_replay_requires_zero_new_remote_ranges(self):
        changed=deepcopy(self.replays)
        changed["observations"]["qwen"]["replayed_remote_ranges"]=1
        with self.assertRaisesRegex(ValueError,"replay"):
            admit_archived_subgraphs(self.structural,self.trained,changed)

    def test_disguised_fixture_live_claim_rejected(self):
        from experiments.model_fabric.remote_large_recovery_benchmark import run_benchmark
        from experiments.model_fabric.operational_ground import digest
        with tempfile.TemporaryDirectory() as tmp:
            b1=run_benchmark(live=False,directory=Path(tmp)/"control",max_tokens=64)
        b1["mode"]="LIVE_PROVIDER";b1["status"]="REAL_PROVIDER_CHAT_BENCHMARK"
        b1["report_digest"]=digest(b1["schema"],{k:v for k,v in b1.items() if k!="report_digest"})
        with self.assertRaisesRegex(ValueError,"provider-mode"):
            comparison(self.structural,self.trained,self.replays,b1=b1)

    def test_wrong_immutable_checkout_is_refused_even_if_files_present(self):
        with self.assertRaisesRegex(ValueError,"commit differs"):
            run(ROOT,require_pinned_checkout=True)

    def test_full_cli_source_ground_has_same_model_intent(self):
        outcome=run(UPSTREAM)
        self.assertEqual(outcome["vl64_reference_commit"], BASELINE_COMMIT)
        self.assertEqual(outcome["status"],"QUALIFIED_EVIDENCE_GAPS_NOT_FULL_MODEL_PARITY")


if __name__=="__main__":
    unittest.main()
