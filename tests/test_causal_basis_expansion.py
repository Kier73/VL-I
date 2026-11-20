"""SHAPE-010: witnessed causal counterexample grows a source-bound simplex.

Fixture-only CI: actual two-layer native trained computation on a fabricated
small model. It is not proof of original Qwen 27B numerical parity.
"""
from __future__ import annotations

from copy import deepcopy
import math
import unittest

from test_qwen_layer_chain import two_layer_fixture, executor
from vli.qwen_all_heads import QwenVectorizedCompleteLayer
from vli.qwen_layer_chain import QwenCausalGDNChain
from vli.causal_hidden_manifold import CausalHiddenManifoldGround, TOL
from vli.causal_basis_expansion import (
    ExpandedCausalGround, _fit, _energy, _algebra, TRAIN, CHALLENGES,
    CANDIDATES, CHECKPOINT,
)
from vli.ground import _digest


INPUTS = ((63204, [.5,-.2,.3]), (628, [-.1,.2,.8]),
          (1761, [.7,.6,-.3]), (33303, [.3,-.4,.65]))


def sources():
    nav = two_layer_fixture()
    chain = QwenCausalGDNChain(executor(nav, QwenVectorizedCompleteLayer))
    receipts = [chain.step(h, token_id=token,
                           capture_intermediates=True)[1]
                for token, h in INPUTS]
    raw, desc = nav.get_raw("model.language_model.layers.1.input_layernorm.weight")
    parent = CausalHiddenManifoldGround(
        reports=receipts[:3], native_norm_raw=raw,
        native_norm_dtype=desc["dtype"], model=nav.index["model"],
        revision=nav.index["revision"],
        index_sha256=nav.index["index_sha256"],
        index_ground_digest=nav.index["ground_digest"])
    return parent, receipts[3]


def child(parent, report):
    return ExpandedCausalGround(
        parent, report,
        expected_parent_checkpoint_digest=parent.checkpoint()["checkpoint_digest"])


class TestCausalGroundExpansion(unittest.TestCase):
    def setUp(self):
        self.parent, self.fourth = sources()
        self.parent_checkpoint = self.parent.checkpoint()
        self.grown = child(self.parent, self.fourth)

    def test_actual_fourth_causal_transition_extends_rank_and_lineage(self):
        grown = self.grown
        self.assertEqual(grown.width, 3)
        self.assertFalse(grown.original_pretrained)
        self.assertEqual(grown.proof["source_revision"], 4)
        self.assertEqual(grown.proof["native_norm_observations"], 14)
        self.assertEqual(len(grown.proof["native_training_probes"]), 10)
        self.assertEqual(len(grown.proof["native_challenges"]), 4)
        self.assertEqual(grown.fourth_witness["position"], 3)
        self.assertEqual(grown.fourth_witness["previous_head"],
                         self.parent.final_chain_head)
        self.assertEqual(grown.fourth_witness["layer0_output_f64_digest"],
                         grown.fourth_witness["layer1_input_f64_digest"])
        self.assertGreater(grown.rank_probe["relative_residual"], 1e-7)
        self.assertFalse(grown.rank_probe["inside_declared_convex_hull"])
        self.assertEqual(self.parent.checkpoint(), self.parent_checkpoint)
        self.assertNotEqual(self.parent.ground_digest, grown.ground_digest)
        self.assertEqual(grown.parent_checkpoint, self.parent_checkpoint)

    def test_learned_three_variable_quadratic_falsifies_weaker_candidates(self):
        scores = self.grown.proof["candidate_max_output_error"]
        self.assertEqual(set(scores), set(CANDIDATES))
        for kind in CANDIDATES[:-1]:
            self.assertGreater(scores[kind], TOL)
        self.assertLess(scores["full_quadratic"], TOL)
        self.assertLess(max(abs(x-y) for x,y in
                            zip(self.grown.coefficients,
                                self.grown.proof["independent_gram_coefficients"])),
                        3e-8)

    def test_unseen_sparse_intentions_generated_before_native_oracle(self):
        grown = self.grown
        requests = ((.13,.27,.11),(.23,.17,.41),(.08,.11,.73))
        staged = [grown.generate(*p, (0,2), expected_head=grown.head)
                  for p in requests]
        self.assertEqual(len(set(p["position"] for p in staged)), 3)
        for p, formed in zip(requests, staged):
            self.assertEqual(formed["event"]["native_calls_during_generation"], 0)
            self.assertLess(max(abs(a-b) for a,b in
                                zip(formed["values"],
                                    grown.native_reference(*p, (0,2)))), TOL)
        self.assertEqual(self.parent.checkpoint(), self.parent_checkpoint)

    def test_parent_scope_is_not_broadened_by_child(self):
        grown = self.grown
        # Old 2D Ground retains its own domain and independent original receipts.
        original = self.parent.generate(.12,.23,(0,1),
                                        expected_head=self.parent.head)
        native = self.parent.native_reference(.12,.23,(0,1))
        self.assertLess(max(abs(a-b) for a,b in zip(original["values"],native)),
                        TOL)
        with self.assertRaises(ValueError):
            grown.generate(.5,.4,.2,(0,),expected_head=grown.head)
        with self.assertRaises(ValueError):
            grown.generate(.1,.2,.3,(0,0),expected_head=grown.head)
        with self.assertRaises(ValueError):
            grown.generate(.1,.2,.3,(0,),expected_head="0"*64)
        self.assertFalse(hasattr(grown,"predict_next_token"))
        self.assertFalse(hasattr(grown,"publish_complete_layer"))

    def test_pinned_source_free_restore_reconstructs_new_intention(self):
        grown = self.grown
        grown.generate(.12,.27,.16,(1,),expected_head=grown.head)
        saved = grown.checkpoint()
        restored = ExpandedCausalGround.restore(
            saved, expected_head=saved["checkpoint_digest"])
        self.assertEqual(restored.checkpoint(),saved)
        unseen = (.19,.31,.22)
        generated = restored.generate(*unseen,(0,1,2),
                                     expected_head=restored.head)
        oracle = grown.native_reference(*unseen,(0,1,2))
        self.assertLess(max(abs(a-b) for a,b in
                            zip(generated["values"],oracle)),TOL)

    def test_forged_or_stale_original_fourth_receipts_refused(self):
        parent, real = sources()
        altered = deepcopy(real)
        altered["layer_witnesses"][1]["input_f64_digest"] = "0"*64
        with self.assertRaises(ValueError):
            child(parent, altered)
        altered = deepcopy(real)
        altered["previous"] = "0"*64
        with self.assertRaises(ValueError):
            child(parent, altered)
        altered = deepcopy(real)
        altered["receiver_local_intermediates"][0] = tuple([altered["receiver_local_intermediates"][0][0]+.1, *altered["receiver_local_intermediates"][0][1:]])
        with self.assertRaises(ValueError):
            child(parent, altered)
        with self.assertRaises(ValueError):
            ExpandedCausalGround(
                parent, real, expected_parent_checkpoint_digest="0"*64)

    def test_adversarial_saved_ECPU_pages_and_new_state_receipts_rejected(self):
        saved = self.grown.checkpoint()
        forged = deepcopy(saved)
        key = next(iter(forged["fourth_ECPU_backing"]["local_pages"]))
        forged["fourth_ECPU_backing"]["local_pages"][key] = "00"
        forged["checkpoint_digest"] = _digest(CHECKPOINT,{
            k:v for k,v in forged.items() if k!="checkpoint_digest"})
        with self.assertRaises(ValueError):
            ExpandedCausalGround.restore(
                forged,expected_head=forged["checkpoint_digest"])
        forged = deepcopy(saved)
        forged["fourth_witness"]["layer0_output_f64_digest"] = "0"*64
        forged["checkpoint_digest"] = _digest(CHECKPOINT,{
            k:v for k,v in forged.items() if k!="checkpoint_digest"})
        with self.assertRaises(ValueError):
            ExpandedCausalGround.restore(
                forged,expected_head=forged["checkpoint_digest"])

    def test_saved_generated_event_cannot_be_rehashed_into_false_result(self):
        grown = self.grown
        grown.generate(.2,.3,.1,(0,1),expected_head=grown.head)
        saved = grown.checkpoint()
        forged = deepcopy(saved)
        event = forged["events"][0]
        event["output_sha256"] = "0"*64
        event["event_digest"] = _digest(event["schema"],{
            k:v for k,v in event.items() if k!="event_digest"})
        forged["head"] = event["event_digest"]
        forged["checkpoint_digest"] = _digest(CHECKPOINT,{
            k:v for k,v in forged.items() if k!="checkpoint_digest"})
        with self.assertRaises(ValueError):
            ExpandedCausalGround.restore(
                forged,expected_head=forged["checkpoint_digest"])

    def test_rehashed_wrong_coefficients_cannot_change_original_observation(self):
        forged = deepcopy(self.grown.checkpoint())
        forged["coefficients"][3] += .12
        forged["proof"]["inferred_coefficients"][3] += .12
        forged["proof"]["proof_digest"] = _digest(
            forged["proof"]["schema"], {
                k:v for k,v in forged["proof"].items() if k!="proof_digest"})
        forged["ground_digest"] = _digest(
            "vl-i.shape010-causal-basis-growth.v1",{
                "source_root":forged["source_root"],
                "proof":forged["proof"]["proof_digest"]})
        forged["checkpoint_digest"] = _digest(CHECKPOINT,{
            k:v for k,v in forged.items() if k!="checkpoint_digest"})
        with self.assertRaises(ValueError):
            ExpandedCausalGround.restore(
                forged,expected_head=forged["checkpoint_digest"])

    def test_closed_form_fit_recovers_generic_rank_three_quadratic(self):
        target = (2.1,.7,1.3,.4,-.31,.18,.5,-.2,.7,1.6)
        observations = {p:_energy(target,p) for p in TRAIN}
        inferred = _fit(observations)
        self.assertLess(max(abs(x-y) for x,y in zip(inferred,target)),1e-13)
        for p in CHALLENGES:
            self.assertAlmostEqual(_energy(inferred,p),_energy(target,p),
                                   places=12)


if __name__ == "__main__":
    unittest.main()
