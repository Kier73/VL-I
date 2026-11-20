"""INTERACT-003: explicit token row from original-size pretrained source into math.

The qualified range and dense operator functions are fixtures; they are not
pretrained weights. Real public-checkpoint execution is checked in a separate
CI job with the same token intention, and labeled accordingly.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
import struct
import subprocess
import sys
import unittest
from unittest.mock import patch

from vli.operative_model import ground
from vli.token_embedding_ground import (
    EMBEDDING_TENSORS, _decode_and_independently_check,
    token_intention, token_position, observe_token_row,
    TokenEmbeddingSession, SCOPE,
)

from test_operative_model import (
    fake_source, fake_observed_rows, fake_observed_values,
    fake_fp4_row,
)


def encode_bf16(values):
    raw=bytearray()
    for v in values:
        word=struct.unpack("<I",struct.pack("<f",float(v)))[0]>>16
        raw.extend(struct.pack("<H",word))
    return bytes(raw)


def attach_embedding(nav,route,*,dtype="BF16",width=5120,vocab=4096,
                     truncate=False):
    from hashlib import sha256
    tensor=EMBEDDING_TENSORS[route]
    nav.weight_map[tensor]="pinned-embedding.safetensors"
    original_locate=nav.locate
    def locate(name):
        if name==tensor:
            return {"dtype":dtype,"shape":[vocab,width]}
        return original_locate(name)
    nav.locate=locate
    def get_raw(name,*,flat_start,count):
        if name!=tensor or count!=width or flat_start%width:
            raise ValueError("Non-row or wrong pretrained tensor fragment")
        token=flat_start//width
        if not 0<=token<vocab:
            raise ValueError("Invalid token row")
        # BF16 values vary with original token ID. This models a source, not a
        # lookup of expected output values and not actual pretrained numbers.
        vals=[(token%17)*.03125 + ((j%7)-3)*.0078125 for j in range(width)]
        raw=encode_bf16(vals)
        if truncate:raw=raw[:-2]
        nav.range_count+=1;nav.weight_bytes+=len(raw)
        nav.chain=sha256((nav.chain+sha256(raw).hexdigest()).encode()).hexdigest()
        return raw,locate(name)
    nav.get_raw=get_raw
    return nav


class TestRealTokenEmbeddingControl(unittest.TestCase):
    def setUp(self):
        patches=[
            patch("experiments.model_fabric.qualified_subgraphs.observed_rows",
                  side_effect=fake_observed_rows),
            patch("experiments.model_fabric.qualified_subgraphs.observed_values",
                  side_effect=fake_observed_values),
            patch("experiments.model_fabric.qualified_subgraphs._mx_fp4_row",
                  side_effect=fake_fp4_row),
        ]
        for p in patches:p.start()
        self.addCleanup(lambda:[p.stop() for p in reversed(patches)])

    def new(self,route):
        nav=attach_embedding(fake_source(route),route)
        g=ground(route,nav.index)
        return nav,g,TokenEmbeddingSession(g)

    def test_qwen_real_embedding_sourced_into_all_48_gate_relations(self):
        nav,g,s=self.new("qwen-27b")
        intent=token_intention(g,100)
        pos=token_position(g,intent,previous_head=s.head)
        self.assertEqual(nav.range_count,0)
        result=s.operate(nav,intent,expected_head=s.head)
        self.assertEqual(result["coordinate"],pos["coordinate"])
        row=result["trained"]["embedding_witness"]
        self.assertEqual(row["token_id"],100)
        self.assertEqual(row["embedding_bytes"],10240)
        self.assertEqual(row["dtype"],"BF16")
        self.assertEqual(row["tensor"],"model.language_model.embed_tokens.weight")
        self.assertEqual(result["constructed"]["heads"],48)
        self.assertEqual(result["trained"]["causal_scope"],SCOPE["qwen-27b"])
        self.assertFalse(result["trained"]["pretrained_full_decoder_completed"])
        self.assertFalse(result["trained"]["native_tokenizer_applied"])
        self.assertNotIn("embedding_values",str(s.checkpoint()))

    def test_two_actual_token_rows_change_numeric_gate_state(self):
        nav,g,s=self.new("qwen-27b")
        first=s.operate(nav,token_intention(g,101),expected_head=s.head)
        other=s.operate(nav,token_intention(g,102),expected_head=s.head)
        self.assertNotEqual(first["coordinate"],other["coordinate"])
        self.assertNotEqual(first["trained"]["embedding_witness"]["embedding_sha256"],
                            other["trained"]["embedding_witness"]["embedding_sha256"])
        self.assertNotEqual(first["constructed"]["mean_beta"],
                            other["constructed"]["mean_beta"])
        self.assertEqual(len(s.events),2)

    def test_deepseek_embedding_fed_moe_is_explicitly_not_causal_prefill(self):
        nav,g,s=self.new("deepseek-v41-flash")
        res=s.operate(nav,token_intention(g,100),expected_head=s.head)
        self.assertEqual(res["trained"]["embedding_witness"]["tensor"],"embed.weight")
        self.assertIn("not_true_moe_causal_input",res["trained"]["causal_scope"])
        self.assertEqual(len(res["trained"]["numerical_result"]["expert_slices"]),6)
        self.assertEqual(len(set(res["constructed"]["selected"])),6)
        other=s.operate(nav,token_intention(g,101),expected_head=s.head)
        self.assertNotEqual(res["trained"]["embedding_witness"]["embedding_sha256"],
                            other["trained"]["embedding_witness"]["embedding_sha256"])

    def test_generated_code_recomposes_actual_computed_numeric_results(self):
        for route in ("qwen-27b","deepseek-v41-flash"):
            nav,g,s=self.new(route)
            res=s.operate(nav,token_intention(g,100),expected_head=s.head)
            prog=res["views"]["code"]+(
              "\nimport json,sys\nprint(json.dumps(construct(json.load(sys.stdin))))")
            p=subprocess.run([sys.executable,"-c",prog],
                input=json.dumps(res["trained"]["numerical_result"]),
                text=True,capture_output=True,timeout=15)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertEqual(json.loads(p.stdout),res["constructed"])

    def test_causal_resume_replays_no_pretrained_weight_bytes(self):
        nav,g,s=self.new("qwen-27b")
        s.operate(nav,token_intention(g,100),expected_head=s.head)
        cp=s.checkpoint()
        count=nav.range_count
        s2=TokenEmbeddingSession.restore(g,cp,expected_head=cp["checkpoint_digest"])
        self.assertEqual(s2.head,s.head)
        self.assertEqual(nav.range_count,count)
        result=s2.operate(nav,token_intention(g,101),expected_head=s2.head)
        self.assertEqual(result["event"]["sequence"],1)

    def test_exact_original_source_identity_must_remain_pinned(self):
        nav,g,s=self.new("qwen-27b")
        nav.index["revision"]="a"*40
        with self.assertRaisesRegex(ValueError,"Changed actual model"):
            s.operate(nav,token_intention(g,100),expected_head=s.head)
        self.assertFalse(s.events)

    def test_token_id_is_bounded_by_real_table(self):
        nav,g,s=self.new("qwen-27b")
        with self.assertRaisesRegex(ValueError,"Unqualified explicit"):
            token_intention(g,True)
        with self.assertRaisesRegex(ValueError,"Unqualified explicit"):
            token_intention(g,-1)
        with self.assertRaisesRegex(ValueError,"Token ID"):
            s.operate(nav,token_intention(g,4096),expected_head=s.head)
        self.assertFalse(s.events)

    def test_altered_token_intention_prevents_any_weight_read(self):
        nav,g,s=self.new("qwen-27b")
        v=token_intention(g,100);v["token_id"]=101
        with self.assertRaisesRegex(ValueError,"altered"):
            s.operate(nav,v,expected_head=s.head)
        self.assertEqual(nav.range_count,0)
        self.assertFalse(s.events)

    def test_missing_embedding_rejected_before_numeric_operation(self):
        nav,g,s=self.new("qwen-27b")
        nav.weight_map.pop(EMBEDDING_TENSORS["qwen-27b"])
        with self.assertRaisesRegex(ValueError,"absent"):
            s.operate(nav,token_intention(g,100),expected_head=s.head)
        self.assertFalse(s.events)

    def test_unrecognized_dtype_and_mismatched_width_rejected(self):
        for dtype,width in (("I8",5120),("BF16",4096)):
            nav,g,s=self.new("qwen-27b")
            attach_embedding(nav,"qwen-27b",dtype=dtype,width=width)
            with self.assertRaises(ValueError):
                s.operate(nav,token_intention(g,100),expected_head=s.head)
            self.assertFalse(s.events)

    def test_truncated_observation_cannot_claim_complete_pretrained_row(self):
        nav,g,s=self.new("qwen-27b")
        attach_embedding(nav,"qwen-27b",truncate=True)
        with self.assertRaisesRegex(ValueError,"complete qualified"):
            s.operate(nav,token_intention(g,100),expected_head=s.head)
        self.assertEqual(s.events,[])

    def test_wrong_hidden_vector_is_rejected_before_trained_layer(self):
        from vli.operative_model import _admit_hidden
        for values in ([float("nan")]*5120,[float("inf")]*5120,[0.0]*5119):
            with self.assertRaises(ValueError):
                _admit_hidden(values,5120)

    def test_refuse_tampered_or_stale_causal_record(self):
        nav,g,s=self.new("qwen-27b")
        with self.assertRaisesRegex(ValueError,"Stale"):
            s.operate(nav,token_intention(g,100),expected_head="0"*64)
        s.operate(nav,token_intention(g,100),expected_head=s.head)
        cp=s.checkpoint()
        wrong=deepcopy(cp)
        wrong["events"][0]["embedding_sha256"]="1"*64
        with self.assertRaisesRegex(ValueError,"checkpoint"):
            TokenEmbeddingSession.restore(g,wrong,expected_head=cp["checkpoint_digest"])

    def test_render_through_actual_e_cpu_vld_when_available(self):
        try:
            import vld.navigation.dag
            import mailcpu.data_ground
        except ImportError:
            self.skipTest("Pinned E-CPU/VLD not installed in this local process")
        nav,g,s=self.new("qwen-27b")
        out=s.operate(nav,token_intention(g,100,views=("text","page","code","tokens")),
                      expected_head=s.head,render=True)
        self.assertEqual(out["rendered"]["state"],"RENDERED")
        self.assertEqual({v["kind"] for v in out["rendered"]["projections"]},
                         {"text","page","code","tokens"})

    def test_independent_bf16_decoder_for_real_row(self):
        values=[0,1,-1,0.125,1.5,-3.5]
        raw=encode_bf16(values)
        observed=_decode_and_independently_check(raw,"BF16")
        self.assertEqual(observed,values)


if __name__=="__main__":
    unittest.main()
