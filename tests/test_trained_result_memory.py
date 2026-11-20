"""SHAPE-006: exact pretrained result state, not merely weight-range caching.

This fixture has a Qwen-shaped *64-element synthetic* BF16 tensor; it never
claims the entire Qwen trained architecture. The separate live GitHub job
performs the same protocol on the original 5,120-dimensional pretrained
Qwen norm without selecting a smaller model.
"""
from __future__ import annotations
from copy import deepcopy
import json
import unittest

from test_pretrained_range_memory import (
    fixture,FakeQualifiedRangeHost,make_memory,MODEL,REVISION)
from vli.trained_result_memory import (
    VerifiedTrainedResultGround,intention)
from vli.ground import _digest


class TestTrainedResultGround(unittest.TestCase):
    def setUp(self):
        self.index,self.shard,_=fixture()
        self.host=FakeQualifiedRangeHost(self.shard)
        self.source=make_memory(self.host,self.index)
        self.ground=VerifiedTrainedResultGround(
            self.source,allow_synthetic_fixture=True).bind_source_index(self.index)

    def test_equal_intention_reuses_entire_original_verified_result_not_just_bytes(self):
        spec=intention(prompt="observe the same trained relation")
        fresh=self.ground.observe(spec,expected_head=self.ground.head)
        requests=self.host.calls
        transitions=self.ground.metrics()["verified_trained_evaluations"]
        again=self.ground.observe(spec,expected_head=self.ground.head)
        self.assertEqual(requests,3)
        self.assertEqual(self.host.calls,requests)
        self.assertEqual(fresh["result"]["output_sha256"],
                         again["result"]["output_sha256"])
        self.assertEqual(self.ground.metrics()["verified_trained_evaluations"],
                         transitions)
        self.assertEqual(self.ground.metrics()[
                         "reused_trained_evaluations_without_arithmetic"],1)
        self.assertEqual(again["event"]["new_trained_arithmetic_evaluations"],0)
        self.assertEqual(again["event"]["new_external_source_requests"],0)
        self.assertEqual(again["trained_compute_ns"],0)
        self.assertNotEqual(fresh["position"]["coordinate"],
                            again["position"]["coordinate"])
        self.assertEqual(len(fresh["position"]["coordinate"]),76)
        self.assertNotEqual(fresh["event"]["event_digest"],
                            again["event"]["event_digest"])
        self.assertFalse(fresh["original_observation_receipt"]["original_source"])

    def test_intention_changes_force_fresh_trained_arithmetic_even_with_cached_weights(self):
        a=intention(prompt="A",phase=0.1)
        b=intention(prompt="A",phase=0.9)
        self.ground.observe(a,expected_head=self.ground.head)
        seen=self.host.calls
        r=self.ground.observe(b,expected_head=self.ground.head)
        self.assertEqual(self.host.calls,seen)
        self.assertEqual(r["event"]["new_trained_arithmetic_evaluations"],1)
        self.assertEqual(self.ground.metrics()["verified_trained_evaluations"],2)
        self.assertNotEqual(self.ground.records[a["intent_digest"]]["receipt"][
                            "output_sha256"],r["result"]["output_sha256"])
        after=self.ground.observe(a,expected_head=self.ground.head)
        self.assertEqual(after["event"]["new_trained_arithmetic_evaluations"],0)
        self.assertEqual(after["result"]["output_sha256"],
                         self.ground.records[a["intent_digest"]]["receipt"][
                         "output_sha256"])

    def test_word_change_is_distinct_intention_even_when_hidden_vector_same(self):
        a=intention(prompt="Show norm output",views=("text","tokens"))
        b=intention(prompt="Show the norm output",views=("text","tokens"))
        self.ground.observe(a,expected_head=self.ground.head)
        other=self.ground.observe(b,expected_head=self.ground.head)
        self.assertEqual(other["event"]["new_trained_arithmetic_evaluations"],1)
        self.assertEqual(self.ground.metrics()["retained_trained_result_positions"],2)

    def test_train_output_is_generatively_rendered_in_requested_views(self):
        spec=intention(prompt="Compute grounded result",views=("text","code","tokens","json"))
        out=self.ground.observe(spec,expected_head=self.ground.head)
        self.assertEqual(set(out["views"]),{"text","code","tokens","json"})
        self.assertEqual(out["views"]["tokens"],list(out["views"]["text"].encode()))
        self.assertIn("def interpret_trained_result",out["views"]["code"])
        self.assertEqual(out["result"]["width"],64)
        self.assertEqual(out["result"]["independent_numeric_error"],0.0)
        self.assertIn(out["result"]["output_sha256"],out["views"]["text"])

    def test_checkpoint_recovery_new_process_identity_with_zero_original_source(self):
        spec=intention(prompt="recover this source-bound result")
        original=self.ground.observe(spec,expected_head=self.ground.head)
        checkpoint=self.ground.checkpoint()
        cold=self.host.calls
        restored=VerifiedTrainedResultGround.restore(
            index_raw=self.index,snapshot=checkpoint,
            expected_head=checkpoint["checkpoint_digest"])
        self.assertEqual(restored.checkpoint(),checkpoint)
        later=restored.observe(spec,expected_head=restored.head)
        self.assertEqual(self.host.calls,cold)
        self.assertEqual(later["event"]["new_original_source_bytes"],0)
        self.assertEqual(later["event"]["new_trained_arithmetic_evaluations"],0)
        self.assertEqual(later["result"]["output_sha256"],
                         original["result"]["output_sha256"])
        self.assertEqual(restored.metrics()["verified_trained_evaluations"],0)

    def test_changed_index_and_stale_head_refuse(self):
        spec=intention()
        with self.assertRaises(ValueError):
            self.ground.observe(spec,expected_head="0"*64)
        self.ground.observe(spec,expected_head=self.ground.head)
        saved=self.ground.checkpoint()
        with self.assertRaises(ValueError):
            VerifiedTrainedResultGround.restore(
                index_raw=self.index,snapshot=saved,expected_head="0"*64)
        changed=json.dumps({"weight_map":{"a":"b.safetensors"}}).encode()
        with self.assertRaises(ValueError):
            VerifiedTrainedResultGround.restore(
                index_raw=changed,snapshot=saved,
                expected_head=saved["checkpoint_digest"])
        self.assertEqual(self.ground.metrics()["verified_trained_evaluations"],1)

    def test_corrupt_retained_trained_vector_page_refused_after_rehashed_snapshot(self):
        spec=intention()
        self.ground.observe(spec,expected_head=self.ground.head)
        checkpoint=self.ground.checkpoint()
        forged=deepcopy(checkpoint)
        obj=forged["result_pages"]["local_pages"]
        k=next(iter(obj))
        obj[k]="00"
        forged["checkpoint_digest"]=_digest(
            forged["schema"],{k:v for k,v in forged.items()
                              if k!="checkpoint_digest"})
        with self.assertRaises(ValueError):
            VerifiedTrainedResultGround.restore(
                index_raw=self.index,snapshot=forged,
                expected_head=forged["checkpoint_digest"])

    def test_rehash_forged_operator_provenance_is_not_trained_proof(self):
        spec=intention()
        self.ground.observe(spec,expected_head=self.ground.head)
        saved=self.ground.checkpoint()
        forged=deepcopy(saved)
        forged["records"][0]["receipt"]["implementation_fingerprint"]="0"*64
        forged["checkpoint_digest"]=_digest(
            forged["schema"],{k:v for k,v in forged.items()
                              if k!="checkpoint_digest"})
        with self.assertRaises(ValueError):
            VerifiedTrainedResultGround.restore(
                index_raw=self.index,snapshot=forged,
                expected_head=forged["checkpoint_digest"])

    def test_changed_operator_fingerprint_refuses_before_any_result(self):
        self.ground.implementation_fingerprint="0"*64
        with self.assertRaises(ValueError):
            self.ground.observe(intention(),expected_head=self.ground.head)
        self.assertEqual(self.host.calls,0)

    def test_corrupt_intention_refuses_without_mutating_event_history(self):
        spec=intention()
        modified=deepcopy(spec);modified["frequency"]=0.4
        with self.assertRaises(ValueError):
            self.ground.observe(modified,expected_head=self.ground.head)
        self.assertEqual(self.ground.events,[])
        self.assertEqual(self.host.calls,0)

    def test_unknown_checkpoint_index_not_promoted_as_actual_pretrained(self):
        with self.assertRaises(ValueError):
            VerifiedTrainedResultGround(self.source)
        self.assertFalse(self.ground.original_pretrained)

    def test_original_source_change_after_result_certification_refuses_replay(self):
        spec=intention()
        self.ground.observe(spec,expected_head=self.ground.head)
        # A *different* range observed after the certified result changes its
        # source observation scope; a result may not be silently rebound.
        req={"model":MODEL,"revision":REVISION,
             "index_ground_digest":self.source.index_ground_digest,
             "shard":"model-00001-of-00020.safetensors",
             "byte_begin":2,"byte_end_exclusive":4,"purpose":"header"}
        self.source(req)
        with self.assertRaises(ValueError):
            self.ground.observe(spec,expected_head=self.ground.head)

    def test_no_model_parity_or_language_generation_claim_from_norm(self):
        spec=intention()
        self.ground.observe(spec,expected_head=self.ground.head)
        m=self.ground.metrics()
        self.assertEqual(m["scope"],
                         "one_pure_original_centered_rmsnorm_not_full_pretrained_decoder")
        self.assertFalse(m["original_pretrained_checkpoint"])
        self.assertFalse(hasattr(self.ground,"predict_next_token"))


if __name__=="__main__":
    unittest.main()
