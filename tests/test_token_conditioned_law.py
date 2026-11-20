"""SHAPE-008: source-observed token embedding mixture -> discovered degree-2 Ground.

All fixtures are fake but match the exact VL-64 safetensors range protocol.
Original Qwen pretrained verification runs in a separate CI job with the
original full-size immutable checkpoint and token IDs used in INTERACT-004.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
import struct
import unittest

from test_pretrained_range_memory import FakeQualifiedRangeHost,MODEL,REVISION
from vli.pretrained_range_memory import QualifiedPretrainedRangeMemory
from vli.token_conditioned_law import TokenConditionedLawGround
from vli.ground import _digest

SHARD="model-00001-of-00020.safetensors"
EMBED="model.language_model.embed_tokens.weight"
NORM="model.language_model.layers.0.input_layernorm.weight"
TOKENS=(100,101)


def _bf16(number):
    return struct.pack("<H",struct.unpack("<I",struct.pack("<f",number))[0]>>16)


def original_shape_fixture():
    width=64;vocab=128
    left=[((j%11)-5)/32 for j in range(width)]
    right=[((j%7)-3)/16 for j in range(width)]
    embeddings=[]
    for k in range(vocab):
        values=(left if k==100 else right if k==101 else
                [((j+k)%5-2)/128 for j in range(width)])
        embeddings.extend(values)
    emb=b"".join(_bf16(v) for v in embeddings)
    norm=b"".join(_bf16((j%9)/256) for j in range(width))
    h={
        EMBED:{"dtype":"BF16","shape":[vocab,width],
               "data_offsets":[0,len(emb)]},
        NORM:{"dtype":"BF16","shape":[width],
              "data_offsets":[len(emb),len(emb)+len(norm)]},
    }
    hb=json.dumps(h,sort_keys=True,separators=(",",":")).encode()
    shard=struct.pack("<Q",len(hb))+hb+emb+norm
    index=json.dumps({
        "metadata":{"total_size":len(emb)+len(norm)},
        "weight_map":{EMBED:SHARD,NORM:SHARD},
    },sort_keys=True,separators=(",",":")).encode()
    return index,shard


class TestOriginalTokenConditionedLearning(unittest.TestCase):
    def setUp(self):
        self.index,self.shard=original_shape_fixture()
        self.host=FakeQualifiedRangeHost(self.shard)
        self.source=QualifiedPretrainedRangeMemory(
            model=MODEL,revision=REVISION,index_raw=self.index,
            upstream_host=self.host,
            approved_upstream_hosts={"synthetic-qualified-range-host"})
        self.model=TokenConditionedLawGround(
            self.source,self.index,token_ids=TOKENS,synthetic_fixture=True)

    def test_discovers_quadratic_from_original_operator_not_preprovided_coefficients(self):
        m=self.model
        self.assertEqual(m.training_calls,5)
        self.assertEqual(m.width,64)
        self.assertEqual(m.token_ids,TOKENS)
        self.assertEqual(m.chosen_law,"quadratic")
        scores=m.proof["candidate_challenge_errors"]
        self.assertLess(scores["quadratic"],2e-9)
        self.assertGreater(scores["constant"],2e-9)
        self.assertGreater(scores["affine"],2e-9)
        self.assertFalse(m.original_pretrained)
        self.assertEqual(len(m.proof["embedding_witnesses"]),2)

    def test_new_token_conditioned_states_generated_before_original_reference(self):
        m=self.model
        ids=(0,1,7,17,33,63)
        heldout=(.13,.37,.67,.91)
        orig=m.training_calls;baseline_calls=self.host.calls
        results=[]
        for t in heldout:
            formed=m.generate(t,ids,expected_head=m.head)
            results.append(formed)
            self.assertEqual(formed["event"]["original_native_operator_calls"],0)
            self.assertEqual(formed["event"]["original_host_range_calls"],0)
        self.assertEqual(m.native_heldout_calls,0)
        self.assertEqual(m.training_calls,orig)
        self.assertEqual(self.host.calls,baseline_calls)
        for t,formed in zip(heldout,results):
            ref=m.native_reference(t,ids)
            self.assertLess(max(abs(a-b)
                                for a,b in zip(formed["values"],ref)),2e-9)
        self.assertEqual(self.host.calls,baseline_calls)
        self.assertEqual(len({r["event"]["output_sha256"] for r in results}),4)

    def test_sparse_observer_reads_less_than_full_and_forms_different_positions(self):
        m=self.model
        x=m.generate(.21,(0,4),expected_head=m.head)
        y=m.generate(.21,tuple(range(64)),expected_head=m.head)
        self.assertEqual(x["values"][0],y["values"][0])
        self.assertEqual(x["values"][1],y["values"][4])
        self.assertEqual(x["event"]["ECPU_source_pages_verified"],3)
        self.assertEqual(y["event"]["ECPU_source_pages_verified"],3)
        self.assertNotEqual(x["coordinate"],y["coordinate"])
        self.assertEqual(len(x["coordinate"]),76)

    def test_restore_generates_fresh_token_mixture_without_source_and_without_native_calls(self):
        m=self.model
        m.generate(.27,(0,8,32),expected_head=m.head)
        saved=m.checkpoint()
        old_calls=self.host.calls
        restored=TokenConditionedLawGround.restore(
            self.index,saved,expected_head=saved["checkpoint_digest"])
        self.assertEqual(restored.checkpoint(),saved)
        fresh=restored.generate(.83,(1,7,12,62),expected_head=restored.head)
        self.assertEqual(restored.training_calls,0)
        self.assertEqual(restored.native_heldout_calls,0)
        self.assertEqual(fresh["event"]["original_native_operator_calls"],0)
        self.assertEqual(self.host.calls,old_calls)
        self.assertLess(max(abs(x-y) for x,y in
                            zip(fresh["values"],m.native_reference(.83,(1,7,12,62)))),
                        2e-9)

    def test_wrong_source_bytes_refuse(self):
        m=self.model;saved=m.checkpoint()
        with self.assertRaises(ValueError):
            TokenConditionedLawGround.restore(
                b"{}",saved,expected_head=saved["checkpoint_digest"])
        changed=deepcopy(saved)
        changed["ECPU_hashes"]["token_left"]="0"*64
        changed["checkpoint_digest"]=_digest(
            changed["schema"],{k:v for k,v in changed.items()
                               if k!="checkpoint_digest"})
        with self.assertRaises(ValueError):
            TokenConditionedLawGround.restore(
                self.index,changed,expected_head=changed["checkpoint_digest"])

    def test_forged_ground_law_with_rehashed_checkpoint_is_rejected_by_source_algebra(self):
        m=self.model;saved=m.checkpoint()
        changed=deepcopy(saved)
        changed["coefficients"][0]+=0.25
        changed["proof"]["polynomial_coefficients"][0]+=0.25
        changed["proof"]["proof_digest"]=_digest(
            changed["proof"]["schema"],{
                k:v for k,v in changed["proof"].items() if k!="proof_digest"})
        changed["ground_digest"]=_digest(
            "vl-i.shape008-token-conditioned-law.v1",
            {"source_root":changed["source_root"],
             "proof":changed["proof"]["proof_digest"]})
        changed["checkpoint_digest"]=_digest(
            changed["schema"],{k:v for k,v in changed.items()
                               if k!="checkpoint_digest"})
        with self.assertRaises(ValueError):
            TokenConditionedLawGround.restore(
                self.index,changed,expected_head=changed["checkpoint_digest"])

    def test_forged_output_and_causal_position_are_rejected_even_when_rehashed(self):
        m=self.model
        m.generate(.43,(0,3,7),expected_head=m.head)
        saved=m.checkpoint()
        for key,value in (("output_sha256","0"*64),
                          ("coordinate","state://vli/"+"0"*64)):
            altered=deepcopy(saved)
            event=altered["events"][0];event[key]=value
            event["event_digest"]=_digest(
                event["schema"],{k:v for k,v in event.items()
                                 if k!="event_digest"})
            altered["head"]=event["event_digest"]
            altered["checkpoint_digest"]=_digest(
                altered["schema"],{k:v for k,v in altered.items()
                                   if k!="checkpoint_digest"})
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    TokenConditionedLawGround.restore(
                        self.index,altered,
                        expected_head=altered["checkpoint_digest"])

    def test_source_epoch_change_refuses_existing_inferred_law(self):
        m=self.model
        other={"model":MODEL,"revision":REVISION,
               "index_ground_digest":self.source.index_ground_digest,
               "shard":SHARD,"byte_begin":11,"byte_end_exclusive":17,
               "purpose":"header"}
        self.source(other)
        with self.assertRaises(ValueError):
            m.generate(.31,(0,),expected_head=m.head)

    def test_offspan_and_unknown_token_input_refuse(self):
        m=self.model;old=m.head
        for t in (-.1,1.1,float("nan")):
            with self.assertRaises(ValueError):
                m.generate(t,(0,),expected_head=old)
        with self.assertRaises(ValueError):
            m.generate(.3,(-1,),expected_head=old)
        with self.assertRaises(ValueError):
            m.generate(.3,(0,0),expected_head=old)
        with self.assertRaises(ValueError):
            m.generate(.3,(0,),expected_head="0"*64)
        self.assertEqual(m.head,old)
        self.assertEqual(m.generation_calls,0)
        self.assertFalse(hasattr(m,"predict_next_token"))

    def test_observed_token_ids_are_distinct_not_a_text_tokenizer(self):
        m=self.model
        self.assertEqual([x["token_id"] for x in m.embedding_witnesses],
                         list(TOKENS))
        self.assertEqual(len(set(x["embedding_sha256"]
                                 for x in m.embedding_witnesses)),2)
        self.assertTrue(all(
            x["scope"]=="synthetic_control_not_original_token_evidence"
            for x in m.embedding_witnesses))
        self.assertEqual(len(m.byte_hashes),3)


if __name__=="__main__":
    unittest.main()
