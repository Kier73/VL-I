"""SHAPE-009: actual complete native VL64 causal layer0-to-layer1 hidden states.

The fixture is synthetic small geometry, but **its basis is not invented**:
each of the three input tokens is advanced through TWO original VL64
qualified *native GDN+SwiGLU layer implementations*. The derived basis
therefore consists of real causal layer0 outputs and native layer1 inputs
from the fixture's true trained recurrent-state trajectory.

Original 5120-wide pretrained real proof uses a separate opt-in live run.
"""
from __future__ import annotations

from copy import deepcopy
import math
import unittest

from test_qwen_layer_chain import two_layer_fixture,executor
from vli.qwen_all_heads import QwenVectorizedCompleteLayer
from vli.qwen_layer_chain import QwenCausalGDNChain
from vli.causal_hidden_manifold import (
    CausalHiddenManifoldGround,_off_line,TOL,CANDIDATES)
from vli.ground import _digest


def observed():
    nav=two_layer_fixture()
    chain=QwenCausalGDNChain(executor(nav,QwenVectorizedCompleteLayer))
    reports=[]
    for token,hidden in (
        (63204,[.5,-.2,.3]),
        (628,[-.1,.2,.8]),
        (1761,[.7,.6,-.3]),
    ):
        _,receipt=chain.step(hidden,token_id=token,capture_intermediates=True)
        reports.append(receipt)
    raw,desc=nav.get_raw(
        "model.language_model.layers.1.input_layernorm.weight")
    return nav,chain,reports,raw,desc


def ground(nav,reports,raw,desc):
    return CausalHiddenManifoldGround(
        reports=reports,native_norm_raw=raw,native_norm_dtype=desc["dtype"],
        model=nav.index["model"],revision=nav.index["revision"],
        index_sha256=nav.index["index_sha256"],
        index_ground_digest=nav.index["ground_digest"])


class TestActualCausalContextManifold(unittest.TestCase):
    def setUp(self):
        self.nav,self.chain,self.reports,self.norm_raw,self.norm_desc=observed()
        self.m=ground(self.nav,self.reports,self.norm_raw,self.norm_desc)

    def test_three_original_causal_positions_not_static_token_embeddings(self):
        m=self.m
        self.assertEqual(m.width,3)
        self.assertEqual(m.native_training_calls,9)
        self.assertFalse(m.original_pretrained)
        self.assertEqual([r["original_token_id"]
                          for r in m.recorded_causal_state_witnesses],
                         [63204,628,1761])
        self.assertEqual(m.final_chain_head,self.chain.head)
        for rec in m.recorded_causal_state_witnesses:
            self.assertEqual(rec["layer0_output_f64_digest"],
                             rec["layer1_input_f64_digest"])
            self.assertTrue(rec["native_complete_layers"])
        self.assertGreater(m.off_line_residual,1e-7)
        self.assertEqual(m.proof["admitted_family"],
                         "quadratic_with_cross_term")

    def test_observed_native_rejects_three_inadequate_candidate_relations(self):
        scores=self.m.proof["candidate_max_error"]
        self.assertEqual(set(scores),set(CANDIDATES))
        for hypothesis in CANDIDATES[:-1]:
            self.assertGreater(scores[hypothesis],TOL)
        self.assertLess(scores["full_quadratic"],TOL)
        self.assertEqual(len(self.m.proof["training_probes"]),6)
        self.assertEqual(len(self.m.proof["counterexamples"]),3)

    def test_prediction_on_new_original_causal_basis_mixtures_before_native_oracle(self):
        m=self.m
        idx=(0,1,2)
        mixtures=((.13,.27),(.45,.13),(.18,.7))
        formed=[]
        for s,t in mixtures:
            prediction=m.generate(s,t,idx,expected_head=m.head)
            self.assertEqual(
                prediction["event"]["original_native_trained_calls_during_generation"],0)
            formed.append(prediction)
        self.assertEqual(m.native_heldout_calls,0)
        self.assertEqual(m.native_training_calls,9)
        for (s,t),prediction in zip(mixtures,formed):
            original=m.native_reference(s,t,idx)
            self.assertLess(max(abs(x-y) for x,y in
                                zip(prediction["values"],original)),TOL)
        self.assertEqual(m.native_heldout_calls,3)
        self.assertEqual(len(set(r["position"] for r in formed)),3)

    def test_small_observer_projection_is_position_distinct_from_complete_output(self):
        m=self.m
        short=m.generate(.2,.3,(0,),expected_head=m.head)
        complete=m.generate(.2,.3,(0,1,2),expected_head=m.head)
        self.assertEqual(short["values"][0],complete["values"][0])
        self.assertEqual(short["event"]["indices"],[0])
        self.assertNotEqual(short["position"],complete["position"])
        self.assertTrue(short["event"]["ECPU_coefficient_pages_verified"]>=4)

    def test_checkpoint_carries_original_source_basis_and_new_unseen_output_law(self):
        m=self.m
        m.generate(.2,.2,(0,2),expected_head=m.head)
        checkpoint=m.checkpoint()
        restored=CausalHiddenManifoldGround.restore(
            checkpoint,expected_head=checkpoint["checkpoint_digest"])
        self.assertEqual(restored.checkpoint(),checkpoint)
        self.assertEqual(restored.native_training_calls,0)
        new=restored.generate(.41,.32,(0,1,2),expected_head=restored.head)
        self.assertEqual(new["event"]["original_native_trained_calls_during_generation"],0)
        expected=m.native_reference(.41,.32,(0,1,2))
        self.assertLess(max(abs(a-b) for a,b in
                            zip(new["values"],expected)),TOL)

    def test_wrong_native_source_digest_rejected(self):
        altered=deepcopy(self.reports)
        altered[1]["receiver_local_intermediates"][0]=(2.,3.,4.)
        with self.assertRaises(ValueError):
            ground(self.nav,altered,self.norm_raw,self.norm_desc)

    def test_forged_intermediate_chain_receipt_rejected(self):
        altered=deepcopy(self.reports)
        altered[1]["layer_witnesses"][1]["input_f64_digest"]="0"*64
        with self.assertRaises(ValueError):
            ground(self.nav,altered,self.norm_raw,self.norm_desc)

    def test_incomplete_causal_layer_and_wrong_order_are_refused(self):
        altered=deepcopy(self.reports)
        altered[2]["all_layers_completed"]=False
        with self.assertRaises(ValueError):
            ground(self.nav,altered,self.norm_raw,self.norm_desc)
        altered=deepcopy(self.reports)
        altered[2]["position"]=1
        with self.assertRaises(ValueError):
            ground(self.nav,altered,self.norm_raw,self.norm_desc)

    def test_changed_checkpoint_head_and_corrupt_ECPU_page_refused(self):
        m=self.m
        m.generate(.15,.2,(0,),expected_head=m.head)
        saved=m.checkpoint()
        with self.assertRaises(ValueError):
            CausalHiddenManifoldGround.restore(saved,expected_head="0"*64)
        forged=deepcopy(saved)
        page=next(iter(forged["ECPU_backing"]["local_pages"]))
        forged["ECPU_backing"]["local_pages"][page]="00"
        forged["checkpoint_digest"]=_digest(
            forged["schema"],{k:v for k,v in forged.items()
                              if k!="checkpoint_digest"})
        with self.assertRaises(ValueError):
            CausalHiddenManifoldGround.restore(
                forged,expected_head=forged["checkpoint_digest"])

    def test_rehashed_fake_generated_output_and_position_refused(self):
        m=self.m
        m.generate(.3,.2,(0,1),expected_head=m.head)
        saved=m.checkpoint()
        for field,bad in (
            ("output_sha256","0"*64),
            ("position","state://vli/"+"0"*64),
        ):
            forged=deepcopy(saved)
            event=forged["events"][0];event[field]=bad
            event["event_digest"]=_digest(
                event["schema"],{k:v for k,v in event.items()
                                 if k!="event_digest"})
            forged["head"]=event["event_digest"]
            forged["checkpoint_digest"]=_digest(
                forged["schema"],{k:v for k,v in forged.items()
                                  if k!="checkpoint_digest"})
            with self.subTest(field=field),self.assertRaises(ValueError):
                CausalHiddenManifoldGround.restore(
                    forged,expected_head=forged["checkpoint_digest"])

    def test_altered_algebraic_law_cannot_be_rehashed_into_qualified_ground(self):
        saved=self.m.checkpoint()
        forged=deepcopy(saved)
        forged["coefficients"][2]+=.1
        forged["proof"]["inferred_coefficients"][2]+=.1
        forged["proof"]["proof_digest"]=_digest(
            forged["proof"]["schema"],{k:v for k,v in forged["proof"].items()
                                       if k!="proof_digest"})
        forged["ground_digest"]=_digest(
            "vl-i.shape009-contextual-hidden-ground.v1",
            {"source":forged["source_root"],
             "proof":forged["proof"]["proof_digest"],
             "objects":{k:v["manifest_digest"]
                        for k,v in forged["ECPU_objects"].items()}})
        forged["checkpoint_digest"]=_digest(
            forged["schema"],{k:v for k,v in forged.items()
                              if k!="checkpoint_digest"})
        with self.assertRaises(ValueError):
            CausalHiddenManifoldGround.restore(
                forged,expected_head=forged["checkpoint_digest"])

    def test_future_causal_state_outside_span_requires_original_observation(self):
        m=self.m
        future=m.off_manifold([.41,-.2,.88])
        self.assertGreater(future["relative_residual"],1e-8)
        self.assertFalse(future["inside_declared_convex_hull"])
        self.assertEqual(future["next_action"],
                         "refuse_and_observe_new_original_causal_position")

    def test_no_claim_of_whole_decoder_or_independent_original_pytorch_oracle(self):
        m=self.m
        self.assertFalse(m.original_pretrained)
        self.assertFalse(hasattr(m,"predict_next_token"))
        self.assertFalse(hasattr(m,"independent_full_qwen_pytorch_oracle"))
        with self.assertRaises(ValueError):
            m.generate(.9,.9,(0,),expected_head=m.head)
        with self.assertRaises(ValueError):
            m.generate(.2,.2,(0,0),expected_head=m.head)


if __name__=="__main__":
    unittest.main()
