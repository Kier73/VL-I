"""VL-I-001: independent structural checks, evidence refusals and adapter seam.

The fake numerical upstream in these unit tests only tests the binding ABI.
It does not assert that a pretrained model executed. The separate live test
must actually use VL-64's trained-data numerical runner.
"""
import copy
import os
from pathlib import Path
import unittest

from vli.ground import (
    _digest, lift_observation, validate_plan, verify_ground,
    intention_frontier, verify_frontier, load_json,
)
from vli.realization import bind_realization, run_pinned_decoder, verify_realization


def fixture():
    rev = "a" * 40
    model = "fixture/test"
    tensors = [
        {"name": "embedding", "dtype": "BF16", "shape": [2, 2],
         "data_offsets": [0, 8]},
        {"name": "out", "dtype": "BF16", "shape": [2, 2],
         "data_offsets": [8, 16]},
    ]
    obs = {
        "schema": "vl64.remote-model-topology-ground.v1",
        "model": model,
        "pinned_revision": rev,
        "ground_digest": "1" * 64,
        "config_sha256": "2" * 64,
        "tensor_header_status": "OBSERVED_COMPLETE_TENSOR_HEADER",
        "tensors": {
            "schema": "vl64.remote-tensor-structure-ground.v1",
            "ground_digest": "3" * 64,
            "header_sha256": "4" * 64,
            "source": "https://example.invalid/model.safetensors",
            "tensor_count": 2,
            "declared_payload_bytes": 16,
            "tensors": tensors,
        },
    }
    plan = {
        "schema": "vl64.typed-pretrained-execution-dag.v1",
        "model": model, "revision": rev, "model_ground_digest": "1"*64,
        "tensor_ground_digest": "3"*64,
        "dag_digest": "5"*64, "architecture": {"layers": 30, "vocab": 2},
        "tensors": {t["name"]: {k: t[k] for k in ("dtype", "shape", "data_offsets")}
                    for t in tensors},
        "nodes": [
            {"id": "tokens", "op": "input", "inputs": [], "weight": None},
            {"id": "emb", "op": "embedding", "inputs": ["tokens"], "weight": "embedding"},
            {"id": "hidden", "op": "copy", "inputs": ["emb"], "weight": None},
            {"id": "vocab_logits", "op": "linear",
             "inputs": ["hidden"], "weight": "out"},
        ],
    }
    return obs, plan


def fake_upstream(tokens=(0, 1)):
    return {
        "schema": "vl64.real-pretrained-30-layer-run.v1",
        "status": "REAL_PRETRAINED_30_LAYER_LOGITS_INDEPENDENTLY_VERIFIED",
        "model": "fixture/test", "revision": "a"*40, "dag_digest": "5"*64,
        "tokens": list(tokens), "layers_completed": 30,
        "total_declared_layers": 30, "full_vocab_size": 2,
        "selected_next_token": 1, "checkpoint_digest": "6"*64,
        "logits_f64_sha256": "7"*64, "result_receipt_digest": "8"*64,
        "independent_logit_max_abs_error": 0.0,
        "range_reads_including_logits": 3, "weight_bytes_including_logits": 16,
    }


class TestModelGround(unittest.TestCase):
    def setUp(self):
        self.obs, self.plan = fixture()

    def lift(self):
        return lift_observation(self.obs, compiler=lambda _: self.plan)

    def test_lift_does_not_claim_weights_or_inference(self):
        ground, plan = self.lift()
        self.assertTrue(verify_ground(ground))
        self.assertEqual(ground["source"]["payload_bytes_admitted"], 0)
        self.assertEqual(ground["readiness"]["numerical"],
                         "UNRESOLVED_WEIGHT_PAYLOAD_NOT_ACQUIRED")
        self.assertEqual(ground["interpretation"]["tensor_count"], 2)

    def test_structural_dependency_frontier(self):
        _, plan = self.lift()
        part = intention_frontier(plan, ["hidden"])
        full = intention_frontier(plan, ["vocab_logits"])
        self.assertTrue(verify_frontier(part, plan))
        self.assertEqual(part["required_nodes"], ["tokens", "emb", "hidden"])
        self.assertEqual(part["required_weight_bytes_declared"], 8)
        self.assertEqual(part["excluded_weight_bytes_declared"], 8)
        self.assertEqual(full["required_weight_bytes_declared"], 16)
        self.assertEqual(full["excluded_weight_bytes_declared"], 0)

    def test_ground_tamper_refused(self):
        ground, _ = self.lift()
        ground["source"]["payload_bytes_admitted"] = 16
        with self.assertRaises(ValueError):
            verify_ground(ground)

    def test_falsified_weight_binding_refused(self):
        self.plan["tensors"]["out"]["data_offsets"] = [0, 8]
        with self.assertRaises(ValueError):
            self.lift()

    def test_non_topological_dependencies_refused(self):
        self.plan["nodes"][1]["inputs"] = ["vocab_logits"]
        with self.assertRaises(ValueError):
            self.lift()

    def test_source_identity_refused(self):
        self.plan["revision"] = "b"*40
        with self.assertRaises(ValueError):
            self.lift()

    def test_missing_tensor_refused(self):
        del self.plan["tensors"]["out"]
        with self.assertRaises(ValueError):
            self.lift()

    def test_missing_target_refused(self):
        with self.assertRaises(ValueError):
            intention_frontier(self.plan, ["nonexistent"])

    def test_unit_binding_receipt_not_numerical_execution(self):
        ground, plan = self.lift()
        frontier = intention_frontier(plan, ["vocab_logits"])
        record = bind_realization(ground, frontier, fake_upstream(), (0, 1))
        self.assertTrue(verify_realization(ground, frontier, record))
        self.assertEqual(record["independent_vli_status"],
                         "STRUCTURAL_CONTRACT_CHECKED_NUMERICAL_ORACLE_NOT_REEXECUTED")

    def test_wrong_upstream_model_refused(self):
        ground, plan = self.lift()
        frontier = intention_frontier(plan, ["vocab_logits"])
        bad = fake_upstream()
        bad["model"] = "other/model"
        with self.assertRaises(ValueError):
            bind_realization(ground, frontier, bad, (0, 1))

    def test_no_actual_weight_ranges_refused(self):
        ground, plan = self.lift()
        frontier = intention_frontier(plan, ["vocab_logits"])
        bad = fake_upstream()
        bad["weight_bytes_including_logits"] = 0
        with self.assertRaises(ValueError):
            bind_realization(ground, frontier, bad, (0, 1))

    def test_partial_frontier_not_full_model(self):
        ground, plan = self.lift()
        frontier = intention_frontier(plan, ["hidden"])
        with self.assertRaises(ValueError):
            bind_realization(ground, frontier, fake_upstream(), (0, 1))

    def test_run_injectable_native_seam(self):
        # This is ONLY an interface/receipt fixture; no neural inference occurs.
        ground, plan = self.lift()
        import unittest.mock as mock
        with mock.patch("vli.realization.lift_observation", return_value=(ground, plan)):
            result = run_pinned_decoder(self.obs, (0, 1), state_dir="unused",
                                        result_path="unused.json",
                                        native_runner=lambda **_: fake_upstream())
        self.assertTrue(verify_realization(result["ground"], result["frontier"],
                                           result["realization"]))

    def test_strict_json_rejects_duplicate_keys(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"bad.json"
            p.write_text('{"a":1,"a":2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_json(p)


class TestActualPinnedModelHeader(unittest.TestCase):
    """Real 272-tensor source observation, zero weight data transfer.

    Runs on CI with the pinned VL-64 checkout in PYTHONPATH, not synthesized
    from the tiny two-tensor test fixture.
    """
    @unittest.skipUnless(os.environ.get("VL64_SOURCE"), "Pinned VL-64 checkout not supplied")
    def test_real_30_layer_lift_and_frontier(self):
        base = Path(os.environ["VL64_SOURCE"])
        file = base/"experiments/model_fabric/datasets/2026-11-18-pinned-smollm2-tensor-ground.json"
        source = load_json(file)
        ground, plan = lift_observation(source)
        self.assertTrue(verify_ground(ground))
        self.assertEqual(plan["architecture"]["layers"], 30)
        self.assertEqual(ground["interpretation"]["tensor_count"], 272)
        self.assertEqual(ground["source"]["payload_bytes_admitted"], 0)
        full = intention_frontier(plan, ["vocab_logits"])
        part = intention_frontier(plan, ["layer.0.ffn_residual"])
        self.assertEqual(full["excluded_weight_bytes_declared"], 0)
        self.assertGreater(part["excluded_weight_bytes_declared"], 0)
        self.assertEqual(full["required_nodes"], [n["id"] for n in plan["nodes"]])
        self.assertTrue(verify_frontier(full, plan))
        self.assertTrue(verify_frontier(part, plan))


if __name__ == "__main__":
    unittest.main()
