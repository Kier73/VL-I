"""INTERACT-004: pinned literal language Ground and trained causal Qwen head.

Fixtures verify causal and provenance contracts. Live original checkpoint work
runs separately; these tests are not unscaled-model inference measurements.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
import unittest
from unittest.mock import patch

from vli.operative_model import ground
from vli.pinned_text_ground import (
    TOKENIZER_SOURCES,admit,text_intention,tokenize,validate_observation)
from vli.qwen_causal_head import QwenCausalHead

from test_operative_model import fake_source


def tiny_tokenizer_bytes():
    from tokenizers import Tokenizer,models,pre_tokenizers,decoders,trainers
    t=Tokenizer(models.BPE(unk_token="[UNK]"))
    t.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder=decoders.ByteLevel()
    t.train_from_iterator([
      "Communication can support computation through a virtual layer.",
      "This is a different paragraph with meaningful token boundaries.",
      "Text and code become input to trained mathematical operations."],
      trainer=trainers.BpeTrainer(vocab_size=384,
           initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
           special_tokens=["[UNK]"]))
    return t.to_str().encode("utf-8")


class TestPinnedTokenizerGround(unittest.TestCase):
    def setUp(self):
        self.g=ground("qwen-27b",fake_source("qwen-27b").index)
        self.raw=tiny_tokenizer_bytes()
        self.tg,self.runtime=admit(self.g,self.raw)

    def test_real_native_tokenization_and_inverse(self):
        text="Communication can support computation through a virtual layer."
        intent=text_intention(self.tg,text)
        receipt=tokenize(self.tg,self.runtime,intent)
        self.assertIs(validate_observation(self.g,self.tg,intent,receipt),receipt)
        self.assertGreater(receipt["token_count"],2)
        self.assertEqual(len(receipt["token_offsets"]),receipt["token_count"])
        self.assertTrue(receipt["decoded_roundtrip_exact"])
        self.assertEqual(receipt["text_sha256"],sha256(text.encode()).hexdigest())
        self.assertEqual(self.tg["source_grade"],"SAME_PINNED_CHECKPOINT_TOKENIZER")
        self.assertEqual(self.tg["trained_weight_bytes_stored"],0)
        self.assertEqual(self.tg["asset_sha256"],sha256(self.raw).hexdigest())

    def test_source_ground_is_a_recipe_and_not_native_token_bytes(self):
        self.assertNotIn("model",self.tg.get("tokenizer_bytes",{}))
        self.assertNotIn(self.raw.decode(),json.dumps(self.tg))
        self.assertEqual(self.tg["model_revision"],self.g["revision"])
        self.assertEqual(self.tg["asset_revision"],self.g["revision"])

    def test_tokenizer_source_grade_not_promoted_for_related_deepseek(self):
        d=ground("deepseek-v41-flash",fake_source("deepseek-v41-flash").index)
        tg,runtime=admit(d,self.raw)
        self.assertEqual(tg["source_grade"],
            "RELATED_FLASH_RELEASE_NOT_PROVEN_IDENTICAL_TO_1017")
        self.assertNotEqual(tg["asset_revision"],d["revision"])
        receipt=tokenize(tg,runtime,text_intention(tg,
            "Communication through a virtual layer."))
        self.assertGreater(receipt["token_count"],1)

    def test_tampered_intention_and_receipt_rejected(self):
        intent=text_intention(self.tg,"Text and code become input.")
        intent["text"]="Altered text"
        with self.assertRaisesRegex(ValueError,"changed"):
            tokenize(self.tg,self.runtime,intent)
        original=text_intention(self.tg,"Text and code become input.")
        observed=tokenize(self.tg,self.runtime,original)
        wrong=deepcopy(observed)
        wrong["token_ids"][0]=999
        with self.assertRaisesRegex(ValueError,"Unqualified"):
            validate_observation(self.g,self.tg,original,wrong)

    def test_wrong_tokenizer_source_not_relabelled_as_qwen(self):
        other=ground("deepseek-v41-flash",fake_source("deepseek-v41-flash").index)
        with self.assertRaisesRegex(ValueError,"mismatch"):
            validate_observation(other,self.tg,
                text_intention(self.tg,"Hello, model."),
                tokenize(self.tg,self.runtime,text_intention(
                    self.tg,"Hello, model.")))

    def test_non_roundtripping_or_unqualified_asset_fails(self):
        with self.assertRaises(ValueError):
            admit(self.g,b"{}")
        with self.assertRaises(ValueError):
            text_intention(self.tg,"\0payload")
        with self.assertRaises(ValueError):
            text_intention(self.tg,"a"*3000)


def mock_values(nav,name,*,start=0,count=None):
    from hashlib import sha256
    nav.range_count+=1
    nav.weight_bytes+=max(2,count or 256)
    nav.chain=sha256((nav.chain+name+str(start)).encode()).hexdigest()
    if name.endswith("input_layernorm.weight"):return [0.0]*5120
    if name.endswith("conv1d.weight"):return [z for _ in range(128) for z in (0.,0.,.2,.8)]
    if name.endswith("A_log"):return [-4.0]
    if name.endswith("dt_bias"):return [0.0]
    if name.endswith("norm.weight"):return [1.0]*128
    if name.endswith("out_proj.weight"):return [0.01]*128
    raise ValueError("Unqualified fake trained value "+name)


def mock_rows(nav,name,x,*,first=0,count=None):
    from hashlib import sha256
    nav.range_count+=1;nav.weight_bytes+=1024
    nav.chain=sha256((nav.chain+name+str(first)).encode()).hexdigest()
    z=math.fsum(x[:80])/80
    n=count if count is not None else 128
    if name.endswith("in_proj_qkv.weight"):
        return [0.12+0.12*z+(i%11)*.003 for i in range(n)]
    if name.endswith("in_proj_z.weight"):
        return [0.3+.05*z for _ in range(n)]
    if name.endswith("in_proj_a.weight"):
        return [0.2+z]*n
    if name.endswith("in_proj_b.weight"):
        return [0.3+.1*z]*n
    raise ValueError("Unqualified fake trained projection "+name)


def causal_source():
    nav=fake_source("qwen-27b")
    p="model.language_model.layers.0."
    names=("input_layernorm.weight","linear_attn.in_proj_qkv.weight",
           "linear_attn.in_proj_z.weight","linear_attn.in_proj_a.weight",
           "linear_attn.in_proj_b.weight","linear_attn.conv1d.weight",
           "linear_attn.A_log","linear_attn.dt_bias",
           "linear_attn.norm.weight","linear_attn.out_proj.weight")
    nav.weight_map.update({p+n:"model-00001-of-00018.safetensors" for n in names})
    def locate(name):
        suffix=name[len(p):]
        if suffix=="linear_attn.in_proj_qkv.weight":return {"shape":[10240,5120],"dtype":"BF16"}
        if suffix=="linear_attn.conv1d.weight":return {"shape":[10240,1,4],"dtype":"BF16"}
        if suffix=="linear_attn.out_proj.weight":return {"shape":[5120,6144],"dtype":"BF16"}
        return {"shape":[48,5120],"dtype":"BF16"}
    nav.locate=locate
    return nav,ground("qwen-27b",nav.index)


def row_witness(token=100):
    return {"tensor":"model.language_model.embed_tokens.weight",
            "token_id":token,"embedding_sha256":sha256(str(token).encode()).hexdigest()}


class TestCausalHead(unittest.TestCase):
    def setUp(self):
        a=patch("experiments.model_fabric.qualified_subgraphs.observed_values",
                side_effect=mock_values)
        b=patch("experiments.model_fabric.qualified_subgraphs.observed_rows",
                side_effect=mock_rows)
        a.start();b.start();self.addCleanup(a.stop);self.addCleanup(b.stop)
        self.nav,self.g=causal_source()
        self.s=QwenCausalHead(self.g,head=0)

    def test_two_actual_stateful_steps_with_independent_recurrence(self):
        x=[0.15+(i%7)*0.001 for i in range(5120)]
        first=self.s.next(self.nav,x,row_witness(100),expected_head=self.s.chain)
        second=self.s.next(self.nav,x,row_witness(101),expected_head=self.s.chain)
        self.assertEqual([first["position"],second["position"]],[0,1])
        self.assertNotEqual(first["causal_recurrent_state_digest"],
                            second["causal_recurrent_state_digest"])
        self.assertTrue(all(r["independent_recurrence_max_abs_error"]<=2e-8
                            for r in (first,second)))
        baseline=QwenCausalHead(self.g,head=0).next(
            self.nav,x,row_witness(101),expected_head=self.s.genesis)
        self.assertNotEqual(second["output_row0_selected_head_contribution"],
                            baseline["output_row0_selected_head_contribution"])
        self.assertGreater(first["new_weight_ranges"],0)

    def test_restore_next_step_uses_identical_numeric_state(self):
        x=[0.12+0.001*(j%10) for j in range(5120)]
        self.s.next(self.nav,x,row_witness(100),expected_head=self.s.chain)
        cp=self.s.checkpoint()
        restored=QwenCausalHead.restore(
            self.g,cp,expected_checkpoint_digest=cp["checkpoint_digest"])
        self.assertEqual(restored.chain,self.s.chain)
        a=self.s.next(self.nav,x,row_witness(101),expected_head=self.s.chain)
        b=restored.next(self.nav,x,row_witness(101),expected_head=restored.chain)
        self.assertEqual(a["output_row0_selected_head_contribution"],
                         b["output_row0_selected_head_contribution"])
        self.assertEqual(a["causal_recurrent_state_digest"],
                         b["causal_recurrent_state_digest"])

    def test_bad_checkpoint_cannot_be_restored(self):
        x=[0.1]*5120
        self.s.next(self.nav,x,row_witness(100),expected_head=self.s.chain)
        cp=self.s.checkpoint()
        changed=deepcopy(cp)
        changed["state"][0][0]+=1
        with self.assertRaisesRegex(ValueError,"checkpoint"):
            QwenCausalHead.restore(self.g,changed,
                      expected_checkpoint_digest=cp["checkpoint_digest"])

    def test_failed_projection_does_not_advance_causal_head(self):
        x=[.1]*5120
        before=self.s.chain
        with patch("experiments.model_fabric.qualified_subgraphs.observed_rows",
                   side_effect=ValueError("Qualified host unavailable")):
            with self.assertRaisesRegex(ValueError,"host unavailable"):
                self.s.next(self.nav,x,row_witness(100),expected_head=self.s.chain)
        self.assertEqual(self.s.position,0)
        self.assertEqual(self.s.chain,before)
        self.assertIsNone(self.s.state)

    def test_refuse_stale_or_nonfinite_input(self):
        with self.assertRaisesRegex(ValueError,"Stale"):
            self.s.next(self.nav,[0.0]*5120,row_witness(),expected_head="0"*64)
        with self.assertRaisesRegex(ValueError,"Incomplete"):
            self.s.next(self.nav,[float("nan")]*5120,row_witness(),
                        expected_head=self.s.chain)


if __name__=="__main__":
    unittest.main()
