"""INTERACT-002: executable pretrained tensor relations, constructed views and refusals.

The mocked *range source* verifies programme and operator contracts. It must
not be reported as trained Qwen/DeepSeek inference. A separate public-weight
CI probe exercises the actual original-scale source and precise ranges.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from vli.large_model_comparison import SOURCES
from vli.operative_model import (
    _compose, _program, _views, ground, intention, position,
    TrainedOperationSession,
)


def fake_source(route):
    from experiments.model_fabric.scaled_model_fabric import QualifiedShardNavigator
    cfg=SOURCES[route]
    nav=QualifiedShardNavigator.__new__(QualifiedShardNavigator)
    nav.index={
        "model":cfg["model"],"revision":cfg["revision"],
        "index_sha256":cfg["index_sha256"],
        "ground_digest":cfg["index_ground_digest"],
        "tensor_count":cfg["tensors"],"shard_count":cfg["shards"],
        "advertised_payload_bytes":cfg["advertised_bytes"],
    }
    nav.range_count=0; nav.weight_bytes=0; nav.chain="0"*64
    nav.weight_map={}
    if route=="qwen-27b":
        p="model.language_model.layers.0."
        names=[p+s for s in (
            "input_layernorm.weight","linear_attn.in_proj_a.weight",
            "linear_attn.in_proj_b.weight","linear_attn.A_log","linear_attn.dt_bias")]
        nav.weight_map={name:"model-00001-of-00018.safetensors" for name in names}
        nav.locate=lambda name: {
            "shape":([48,5120] if name.endswith(("in_proj_a.weight","in_proj_b.weight"))
                     else [5120] if name.endswith("input_layernorm.weight") else [48]),
            "dtype":"BF16",
        }
    else:
        nav.weight_map={"layers.0.ffn.gate.weight":"shard","layers.0.ffn.gate.bias":"shard"}
        nav.locate=lambda name: {
            "shape":([384,5120] if name.endswith("gate.weight") else
                     [384] if name.endswith("gate.bias") else
                     [1,2560] if name.endswith(("w1.weight","w3.weight")) else
                     [5120,32]),
            "dtype":"BF16" if "gate." in name else "I8",
        }
    return nav


def _bump(nav, value):
    from hashlib import sha256
    nav.range_count+=1
    nav.weight_bytes+=1024
    nav.chain=sha256((nav.chain+str(value)).encode()).hexdigest()


def fake_observed_values(nav,name,**kw):
    _bump(nav,name)
    if name.endswith("input_layernorm.weight"):
        return [0.0]*5120
    if name.endswith("A_log"):return [-4.0]*48
    if name.endswith("dt_bias"):return [0.0]*48
    if name.endswith("gate.bias"):return [0.0]*384
    raise ValueError("Not an admitted fixture tensor")


def fake_observed_rows(nav,name,x,*,first=0,count=None):
    _bump(nav,name)
    if name.endswith("gate.weight"):
        return [0.005*(i%17)+0.000001*sum(x[:4]) for i in range(384)]
    n=48 if count is None else count
    scale=(0.018 if "in_proj_a" in name else 0.12)
    return [scale*math.fsum(x[:20])/(j+1) for j in range(n)]


def fake_fp4_row(nav,name,x,**kw):
    _bump(nav,name)
    amp=math.fsum(x[:4])
    if name.endswith("w1.weight"):return 0.2+amp*0.02,0.0
    if name.endswith("w3.weight"):return 0.3+amp*0.01,0.0
    if name.endswith("w2.weight"):return 0.05+amp*0.03,0.0
    raise ValueError("Unqualified expert row")


class TestConstructiveTrainedOperations(unittest.TestCase):
    def setUp(self):
        self.patches=[
            patch("experiments.model_fabric.qualified_subgraphs.observed_rows",
                  side_effect=fake_observed_rows),
            patch("experiments.model_fabric.qualified_subgraphs.observed_values",
                  side_effect=fake_observed_values),
            patch("experiments.model_fabric.qualified_subgraphs._mx_fp4_row",
                  side_effect=fake_fp4_row),
        ]
        for p in self.patches:p.start()
        self.addCleanup(lambda:[p.stop() for p in reversed(self.patches)])

    def new(self,route):
        nav=fake_source(route)
        g=ground(route,nav.index)
        return nav,g,TrainedOperationSession(g)

    def test_position_is_located_before_numeric_operation(self):
        nav,g,s=self.new("qwen-27b")
        spec=intention(g,phase=.4)
        pos=position(g,spec,previous_head=s.head)
        self.assertEqual(nav.range_count,0)
        self.assertEqual(nav.weight_bytes,0)
        r=s.operate(nav,spec,expected_head=s.head)
        self.assertEqual(r["position"],pos)
        self.assertEqual(r["trained"]["coordinate"],pos["coordinate"])
        self.assertGreater(r["trained"]["new_weight_bytes"],0)
        self.assertEqual(r["trained"]["numerical_result"]["gated_delta_heads"],48)
        self.assertEqual(len(r["trained"]["numerical_result"]["beta"]),48)

    def test_intent_changes_real_operative_math_not_merely_output_format(self):
        nav,g,s=self.new("qwen-27b")
        a=s.operate(nav,intention(g,phase=0),expected_head=s.head)
        b=s.operate(nav,intention(g,phase=1.0),expected_head=s.head)
        self.assertNotEqual(a["position"]["coordinate"],b["position"]["coordinate"])
        self.assertNotEqual(a["trained"]["numerical_result"]["beta"],
                            b["trained"]["numerical_result"]["beta"])
        self.assertNotEqual(a["constructed"]["mean_beta"],b["constructed"]["mean_beta"])
        self.assertEqual(s.checkpoint()["events"][-1]["sequence"],1)

    def test_selected_only_deepseek_fp4_execution(self):
        nav,g,s=self.new("deepseek-v41-flash")
        r=s.operate(nav,intention(g,waveform="cos"),expected_head=s.head)
        calc=r["trained"]["numerical_result"]
        self.assertEqual(calc["router_expert_count"],384)
        self.assertEqual(len(calc["selected_experts"]),6)
        self.assertEqual(len(set(calc["selected_experts"])),6)
        self.assertEqual(len(calc["expert_slices"]),6)
        self.assertTrue(math.isfinite(calc["weighted_partial_output_channel0"]))
        self.assertTrue(calc["reference_max_abs_error"] <= 2e-8)
        self.assertEqual(set(r["constructed"]),{"selected","weighted_partial"})

    def test_generated_code_constructs_independent_result_in_new_process(self):
        for route in SOURCES:
            nav,g,s=self.new(route)
            r=s.operate(nav,intention(g),expected_head=s.head)
            code=r["views"]["code"]
            ast.parse(code)
            script=code+"\nimport json,sys\nprint(json.dumps(construct(json.load(sys.stdin))))"
            p=subprocess.run([sys.executable,"-c",script],
                input=json.dumps(r["trained"]["numerical_result"]),
                capture_output=True,text=True,timeout=10)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertEqual(json.loads(p.stdout),r["constructed"])

    def test_render_multiple_formats_from_trained_result(self):
        nav,g,s=self.new("qwen-27b")
        r=s.operate(nav,intention(g),expected_head=s.head)
        self.assertEqual(set(r["views"]),
             {"text","page","code","json","tokens"})
        self.assertIn("48",r["views"]["json"])
        self.assertIn("state://vli/",r["views"]["page"])
        self.assertEqual(bytes(r["views"]["tokens"]).decode("utf8"),
                         r["views"]["text"])
        self.assertEqual(json.loads(r["views"]["json"])["constructed"],r["constructed"])

    def test_causal_checkpoint_restore_without_new_weight_reads(self):
        nav,g,s=self.new("qwen-27b")
        a=s.operate(nav,intention(g),expected_head=s.head)
        cp=s.checkpoint()
        old_reads=nav.range_count
        restored=TrainedOperationSession.restore(g,cp,expected_head=cp["checkpoint_digest"])
        self.assertEqual(restored.head,s.head)
        self.assertEqual(nav.range_count,old_reads)
        b=restored.operate(nav,intention(g,phase=.7),expected_head=restored.head)
        self.assertNotEqual(a["position"]["coordinate"],b["position"]["coordinate"])
        self.assertEqual(len(restored.events),2)

    def test_wrong_full_scale_source_digest_refused(self):
        nav=fake_source("qwen-27b")
        nav.index["index_sha256"]="f"*64
        with self.assertRaisesRegex(ValueError,"exact observed"):
            ground("qwen-27b",nav.index)

    def test_identity_switch_after_session_start_refused(self):
        nav,g,s=self.new("qwen-27b")
        nav.index["revision"]="a"*40
        with self.assertRaisesRegex(ValueError,"Mismatched"):
            s.operate(nav,intention(g),expected_head=s.head)

    def test_unsupported_faux_llm_view_refused(self):
        nav,g,s=self.new("deepseek-v41-flash")
        with self.assertRaisesRegex(ValueError,"Invalid trained"):
            intention(g,views=("full_pretrained_text",))

    def test_missing_qwen_trained_projection_refused_no_commit(self):
        nav,g,s=self.new("qwen-27b")
        nav.weight_map.pop("model.language_model.layers.0.linear_attn.in_proj_b.weight")
        with self.assertRaisesRegex(ValueError,"tensor missing"):
            s.operate(nav,intention(g),expected_head=s.head)
        self.assertEqual(s.events,[])

    def test_refuse_stale_intention_position_no_commit(self):
        nav,g,s=self.new("qwen-27b")
        with self.assertRaisesRegex(ValueError,"Stale"):
            s.operate(nav,intention(g),expected_head="0"*64)
        self.assertEqual(nav.range_count,0)

    def test_refuse_tampered_intent(self):
        nav,g,s=self.new("qwen-27b")
        spec=intention(g)
        spec["amplitude"]=0.99
        with self.assertRaisesRegex(ValueError,"intention changed"):
            s.operate(nav,spec,expected_head=s.head)

    def test_refuse_corrupt_checkpoint_and_source(self):
        nav,g,s=self.new("qwen-27b")
        s.operate(nav,intention(g),expected_head=s.head)
        cp=s.checkpoint()
        t=deepcopy(cp)
        t["events"][0]["new_weight_bytes"]=0
        with self.assertRaisesRegex(ValueError,"checkpoint"):
            TrainedOperationSession.restore(g,t,expected_head=cp["checkpoint_digest"])
        with self.assertRaisesRegex(ValueError,"checkpoint"):
            TrainedOperationSession.restore(g,cp,expected_head="0"*64)

    def test_no_fabricated_weight_observation_on_empty_host(self):
        nav,g,s=self.new("qwen-27b")
        with patch("experiments.model_fabric.qualified_subgraphs.observed_rows",
                   side_effect=lambda nav,*args,**kw:[.0]*48):
            # Other metadata reads still qualify; no weight-backed rows alone
            # is not enough to claim no weights. The receipt remains partial.
            r=s.operate(nav,intention(g),expected_head=s.head)
            self.assertEqual(r["trained"]["numerical_result"]["kind"],
                             "qwen_trained_delta_gate_vector")
            self.assertIn("partial",r["trained"]["scope"])

    def test_render_real_vld_dag_when_predecessors_available(self):
        try:
            import vld.navigation.dag
            import mailcpu.data_ground
        except ImportError:
            self.skipTest("Pinned E-CPU/VLD response renderer unavailable locally")
        nav,g,s=self.new("qwen-27b")
        r=s.operate(nav,intention(g,views=("text","code","tokens")),
                    expected_head=s.head,render=True)
        self.assertEqual(r["rendered"]["state"],"RENDERED")
        self.assertEqual({x["kind"] for x in r["rendered"]["projections"]},
                         {"text","code","tokens"})


if __name__=="__main__":unittest.main()
