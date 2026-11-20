"""QUALIFIED-008: architecture-directed original Qwen causal-prefix navigation.

A pinned config.json is interpretation evidence, not numerical evidence.
Its declared layers are cross-checked against a separately source-pinned,
qualified safetensors index before any operators are constructed. The
existing VL-64 GDN and VL-I GQA model-worker implementations remain
the mathematical authorities; this coordinates their exact causal values.

The scope is text-only Qwen3.5/Qwen3.8, not arbitrary neural networks.
This module is outside the 64 KiB VL-64 interpreter.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re

from experiments.model_fabric.scaled_model_fabric import QualifiedShardNavigator
from experiments.model_fabric.complete_native_layers import (
    QWEN_PARTS, QwenQualifiedLayer, LocalStatePages,
    export_qwen_causal_ground, restore_qwen_causal_ground,
)
from .ground import _digest, _hex64
from .qwen_all_heads import QwenVectorizedCompleteLayer
from .qwen_gated_attention import (
    QwenQualifiedGatedAttention, _PARTS,
    export_attention_kv, restore_attention_kv,
)
from .qwen_layer_chain import _f64_transfer_digest

PLAN_SCHEMA="vl-i.qwen-admitted-indexed-topology.v1"
CHAIN_SCHEMA="vl-i.qwen-architecture-directed-prefix.v1"
STATE_SCHEMA="vl-i.qwen-architecture-directed-checkpoint.v1"
_LAYER=re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")
_FAMILIES={"linear_attention":"GDN","full_attention":"GATED_GQA"}
_KINDS={"GDN":QWEN_PARTS,"GATED_GQA":_PARTS}


def _reject_duplicates(pairs):
    obj={}
    for key,val in pairs:
        if key in obj:
            raise ValueError("Duplicate original configuration field")
        obj[key]=val
    return obj


def _no_nan(value):
    raise ValueError("Nonfinite config value " + value)


def _positive_int(value):
    return type(value) is int and 0<value<2**30


def admit_indexed_qwen_topology(nav, raw_config, *, expected_config_sha256=None):
    """Admit a revision-pinned text-config *and* the matching tensor inventory.

    A config can declare an operator without actual tensors, or a tensor
    index can advertise something contradictory. Both cases are rejected.
    The config SHA is reported and may additionally be externally pinned.
    At this stage source SHA matching does not prove any whole-layer math.
    """
    if (not isinstance(nav,QualifiedShardNavigator) or
        not isinstance(raw_config,bytes) or
        not 0<len(raw_config)<=65536 or
        (expected_config_sha256 is not None and
         not _hex64(expected_config_sha256))):
        raise ValueError("Unqualified original config or navigator")
    raw_sha=hashlib.sha256(raw_config).hexdigest()
    if expected_config_sha256 is not None and raw_sha!=expected_config_sha256:
        raise ValueError("Original model config bytes differ from external pin")
    try:
        cfg=json.loads(raw_config.decode("utf8"),object_pairs_hook=_reject_duplicates,
                       parse_constant=_no_nan)
    except (UnicodeError,json.JSONDecodeError) as exc:
        raise ValueError("Unreadable original model configuration") from exc
    if not isinstance(cfg,dict) or cfg.get("model_type")!="qwen3_5":
        raise ValueError("Unsupported original model configuration")
    txt=cfg.get("text_config")
    if not isinstance(txt,dict) or txt.get("model_type")!="qwen3_5_text":
        raise ValueError("Missing qualified Qwen text decoder")
    types=txt.get("layer_types")
    if (not isinstance(types,list) or
        not _positive_int(txt.get("num_hidden_layers")) or
        not 1<=len(types)<=128 or
        txt["num_hidden_layers"]!=len(types) or
        any(type(t) is not str or t not in _FAMILIES for t in types)):
        raise ValueError("Unqualified original decoder layer types")
    interval=txt.get("full_attention_interval")
    if interval is not None:
        if not _positive_int(interval):
            raise ValueError("Unqualified original mixed-attention interval")
        for idx,t in enumerate(types):
            if t!=("full_attention" if (idx+1)%interval==0 else "linear_attention"):
                raise ValueError("Configuration interval contradicts indexed layers")
    fields=("hidden_size","intermediate_size","linear_num_key_heads",
            "linear_num_value_heads","linear_key_head_dim",
            "linear_value_head_dim","num_attention_heads",
            "num_key_value_heads","head_dim","max_position_embeddings")
    if any(not _positive_int(txt.get(field)) for field in fields):
        raise ValueError("Missing source-declared Qwen geometry")
    eps=txt.get("rms_norm_eps")
    if type(eps) not in (int,float) or not math.isfinite(eps) or not 0<eps<=1:
        raise ValueError("Invalid original normalization convention")
    rope=txt.get("rope_parameters")
    if not isinstance(rope,dict) or rope.get("rope_type")!="default":
        raise ValueError("Unsupported original rotary scheme")
    fraction=rope.get("partial_rotary_factor",txt.get("partial_rotary_factor"))
    theta=rope.get("rope_theta")
    if (type(fraction) not in (int,float) or not math.isfinite(fraction) or
        not 0<fraction<=1 or type(theta) not in (int,float) or
        not math.isfinite(theta) or theta<=1):
        raise ValueError("Unqualified source rotary geometry")
    rotary=txt["head_dim"]*fraction
    if not math.isfinite(rotary) or not float(rotary).is_integer():
        raise ValueError("Fractional source rotary dimension")
    rotary=int(rotary)
    if rotary%2 or rotary<2:
        raise ValueError("Rotary head must have even dimensions")
    section=rope.get("mrope_section")
    if section is not None and (
        not isinstance(section,list) or len(section)!=3 or
        any(type(x) is not int or x<0 for x in section) or
        sum(section)!=rotary//2):
        raise ValueError("Text rotary and M-RoPE sections disagree")
    if (txt["linear_num_value_heads"]%txt["linear_num_key_heads"] or
        txt["num_attention_heads"]%txt["num_key_value_heads"] or
        txt.get("hidden_act")!="silu" or
        txt.get("attention_bias",False) is not False):
        raise ValueError("Unsupported original Qwen operator semantics")

    # The separately admitted exact safetensors index, not the config,
    # establishes which trained operators actually have source tensors.
    indexed={}
    for name in nav.weight_map:
        match=_LAYER.fullmatch(name)
        if match:
            index=int(match[1])
            if index>=len(types):
                raise ValueError("Source index advertises undeclared decoder layers")
            indexed.setdefault(index,set()).add(match[2])
    if set(indexed)!=set(range(len(types))):
        raise ValueError("Decoder source tensor index has missing layers")
    layer_plan=[]
    for idx,declared in enumerate(types):
        kind=_FAMILIES[declared]
        actual=indexed[idx]
        expected=set(_KINDS[kind])
        opposite=set(_KINDS["GATED_GQA" if kind=="GDN" else "GDN"])
        if not expected.issubset(actual):
            raise ValueError("Missing trained operator tensors at indexed layer "+str(idx))
        # Shared norm/FFN tensors occur in both operator families. Reject
        # only opposite-family-specific attention tensors.
        opposite_only=opposite-expected
        if actual.intersection(opposite_only):
            raise ValueError("Source index and model topology conflict at layer "+str(idx))
        # Check *every* planned trained matrix against its authenticated
        # safetensors header before describing a 64-layer architecture as
        # admitted. Names alone are not enough: a model can advertise the
        # right operator name with an incompatible numerical shape.
        prefix=f"model.language_model.layers.{idx}."
        h=txt["hidden_size"]
        inter=txt["intermediate_size"]
        if kind=="GDN":
            kh=txt["linear_num_key_heads"]
            vh=txt["linear_num_value_heads"]
            kd=txt["linear_key_head_dim"]
            vd=txt["linear_value_head_dim"]
            mixed=2*kh*kd+vh*vd
            kernel=txt.get("linear_conv_kernel_dim")
            checks={
                "input_layernorm.weight":[h],
                "linear_attn.in_proj_qkv.weight":[mixed,h],
                "linear_attn.in_proj_z.weight":[vh*vd,h],
                "linear_attn.in_proj_a.weight":[vh,h],
                "linear_attn.in_proj_b.weight":[vh,h],
                "linear_attn.A_log":[vh],
                "linear_attn.dt_bias":[vh],
                "linear_attn.norm.weight":[vd],
                "linear_attn.out_proj.weight":[h,vh*vd],
                "post_attention_layernorm.weight":[h],
                "mlp.gate_proj.weight":[inter,h],
                "mlp.up_proj.weight":[inter,h],
                "mlp.down_proj.weight":[h,inter],
            }
        else:
            q=txt["num_attention_heads"]*txt["head_dim"]
            kv=txt["num_key_value_heads"]*txt["head_dim"]
            hd=txt["head_dim"]
            checks={
                "input_layernorm.weight":[h],
                "post_attention_layernorm.weight":[h],
                "self_attn.q_proj.weight":[2*q,h],
                "self_attn.k_proj.weight":[kv,h],
                "self_attn.v_proj.weight":[kv,h],
                "self_attn.o_proj.weight":[h,q],
                "self_attn.q_norm.weight":[hd],
                "self_attn.k_norm.weight":[hd],
                "mlp.gate_proj.weight":[inter,h],
                "mlp.up_proj.weight":[inter,h],
                "mlp.down_proj.weight":[h,inter],
            }
        for part,shape in checks.items():
            descriptor=nav.locate(prefix+part)
            if (descriptor["shape"]!=shape or
                descriptor["dtype"] not in ("BF16","F16","F32")):
                raise ValueError("Original source tensor shape contradicts indexed architecture at layer "+str(idx))
        if kind=="GDN":
            conv=nav.locate(prefix+"linear_attn.conv1d.weight")
            kernel_dim=conv["shape"][-1] if len(conv["shape"])==3 else -1
            if (conv["dtype"] not in ("BF16","F16","F32") or
                conv["shape"]!=[mixed,1,kernel_dim] or
                not 1<=kernel_dim<=64 or
                (kernel is not None and kernel!=kernel_dim)):
                raise ValueError("Original convolution shape contradicts source config")
        layer_plan.append({"layer":idx,"kind":kind})
    geom={
        "hidden":txt["hidden_size"],"intermediate":txt["intermediate_size"],
        "key_heads":txt["linear_num_key_heads"],
        "value_heads":txt["linear_num_value_heads"],
        "key_dim":txt["linear_key_head_dim"],
        "value_dim":txt["linear_value_head_dim"],
        "query_heads":txt["num_attention_heads"],
        "kv_heads":txt["num_key_value_heads"],"head_dim":txt["head_dim"],
        "rotary_dim":rotary,"rope_theta":float(theta),
        "eps":float(eps),"max_positions":txt["max_position_embeddings"],
    }
    body={
        "schema":PLAN_SCHEMA,
        "model":nav.index["model"],"revision":nav.index["revision"],
        "index_ground_digest":nav.index["ground_digest"],
        "config_sha256":raw_sha,
        "config_bytes":len(raw_config),
        "external_config_sha256_pin_supplied":expected_config_sha256 is not None,
        "layer_count":len(types),
        "layers":layer_plan,
        "geometry":geom,
        "scope":"text-only original Qwen gated DeltaNet and full-attention decoder",
    }
    return {**body,"plan_digest":_digest(PLAN_SCHEMA,body)}


def _admit_plan(nav,plan):
    if (not isinstance(nav,QualifiedShardNavigator) or
        not isinstance(plan,dict) or
        plan.get("schema")!=PLAN_SCHEMA or
        not _hex64(plan.get("plan_digest")) or
        _digest(PLAN_SCHEMA,{k:v for k,v in plan.items()
                            if k!="plan_digest"})!=plan["plan_digest"] or
        plan.get("model")!=nav.index["model"] or
        plan.get("revision")!=nav.index["revision"] or
        plan.get("index_ground_digest")!=nav.index["ground_digest"]):
        raise ValueError("Changed or unauthenticated source architecture plan")
    if (type(plan.get("layer_count")) is not int or
        not 1<=plan["layer_count"]<=128 or
        not isinstance(plan.get("layers"),list) or
        len(plan["layers"])!=plan["layer_count"] or
        any(x!={"layer":i,"kind":x.get("kind")} or
            x.get("kind") not in _KINDS
            for i,x in enumerate(plan["layers"]) if isinstance(x,dict)) or
        any(not isinstance(x,dict) for x in plan["layers"])):
        raise ValueError("Malformed indexed model plan")
    return plan


class QwenArchitectureCausalPrefix:
    """Auto-select Qwen layers 0..N-1, preserving exact causal state per kind.

    N may cross one or many hybrid boundaries. State of *every* materialized
    GDN and GQA layer is separately persisted and independently recoverable.
    No suffix is "skipped" by claiming a prefix is a complete decoder.
    """

    def __init__(self,nav,plan,*,stop_layer,layer_budget_bytes=2_000_000_000,
                 max_positions=4096):
        _admit_plan(nav,plan)
        if (type(stop_layer) is not int or not 1<=stop_layer<=plan["layer_count"] or
            type(layer_budget_bytes) is not int or layer_budget_bytes<1 or
            type(max_positions) is not int or max_positions<1 or
            max_positions>plan["geometry"]["max_positions"]):
            raise ValueError("Invalid qualified decoder-prefix extent")
        self.nav,self.plan=nav,deepcopy(plan)
        self.stop_layer=stop_layer
        self.layer_budget_bytes=layer_budget_bytes
        self.max_positions=max_positions
        g=plan["geometry"]
        self.runners=[]
        for selected in plan["layers"][:stop_layer]:
            if selected["kind"]=="GDN":
                r=QwenVectorizedCompleteLayer(nav,layer=selected["layer"],
                    key_heads=g["key_heads"],value_heads=g["value_heads"],
                    key_dim=g["key_dim"],value_dim=g["value_dim"],
                    eps=g["eps"],budget_bytes=layer_budget_bytes)
            elif selected["kind"]=="GATED_GQA":
                r=QwenQualifiedGatedAttention(nav,layer=selected["layer"],
                    query_heads=g["query_heads"],kv_heads=g["kv_heads"],
                    head_dim=g["head_dim"],rotary_dim=g["rotary_dim"],
                    rope_theta=g["rope_theta"],eps=g["eps"],
                    budget_bytes=layer_budget_bytes,max_positions=max_positions)
            else:
                raise ValueError("Unsupported unqualified model operator")
            if r.hidden!=g["hidden"]:
                raise ValueError("Source matrix width differs from admitted model topology")
            # The individual runners validate all their full matrix shapes.
            if hasattr(r,"intermediate") and r.intermediate!=g["intermediate"]:
                raise ValueError("Trained SwiGLU width contradicts admitted architecture")
            if hasattr(r,"shapes") and r.shapes["mlp.gate_proj.weight"][0]!=g["intermediate"]:
                raise ValueError("GDN trained SwiGLU width contradicts admitted architecture")
            self.runners.append(r)
        self.width=g["hidden"]
        self.position=0
        self.last_output_digest=None
        self.genesis=_digest(CHAIN_SCHEMA,{
            "plan_digest":self.plan["plan_digest"],"stop_layer":stop_layer})
        self.head=self.genesis

    def _snapshot(self):
        return [
            (r.position,deepcopy(r.convolution_history),deepcopy(r.recurrent_state))
            if isinstance(r,QwenQualifiedLayer)
            else (r.position,deepcopy(r.kv_keys),deepcopy(r.kv_values))
            for r in self.runners
        ]

    def step(self,hidden,*,token_id=None):
        # A mutable caller-owned plan must not silently alter the operation
        # ordering after construction without a fresh source admission.
        _admit_plan(self.nav,self.plan)
        if (not isinstance(hidden,(list,tuple)) or len(hidden)!=self.width or
            (type(token_id) not in (int,type(None))) or
            (token_id is not None and not 0<=token_id<2**31) or
            self.position>=self.max_positions or
            any(r.position!=self.position for r in self.runners)):
            raise ValueError("Incomplete original decoder input or causal coordinate")
        original_digest=_f64_transfer_digest(hidden)
        saved=self._snapshot()
        old_head,old_output=self.head,self.last_output_digest
        state=list(hidden)
        witnesses=[]
        try:
            for r,layer in zip(self.runners,self.plan["layers"]):
                entering=_f64_transfer_digest(state)
                output,report=r.next(state)
                if (not isinstance(output,list) or len(output)!=self.width or
                    report.get("model")!=self.plan["model"] or
                    report.get("revision")!=self.plan["revision"] or
                    report.get("layer")!=layer["layer"] or
                    report.get("position")!=self.position or
                    report.get("complete_Qwen_transformer_layer") is not True or
                    report.get("complete_output_width")!=self.width or
                    r.position!=self.position+1):
                    raise ValueError("Original trained layer failed exact receipt admission")
                exiting=_f64_transfer_digest(output)
                witness={
                    "layer":layer["layer"],"kind":layer["kind"],
                    "position":self.position,
                    "input_f64_digest":entering,
                    "output_f64_digest":exiting,
                    "output_f32_digest":report["complete_layer_output_digest"],
                    "state_digest":(
                        report["full_recurrent_state_digest"]
                        if layer["kind"]=="GDN" else
                        report["full_attention_kv_state_digest"]),
                    "qualified_ranges":report["new_qualified_ranges"],
                    "trained_weight_bytes":report["new_weight_payload_bytes"],
                    "range_chain":report["range_chain"],
                }
                if layer["kind"]=="GDN":
                    witness["convolution_digest"]=report["full_convolution_state_digest"]
                else:
                    witness["kv_positions"]=report["kv_history_positions"]
                witnesses.append(witness)
                state=output
            if (witnesses[0]["input_f64_digest"]!=original_digest or
                any(x["output_f64_digest"]!=y["input_f64_digest"]
                    for x,y in zip(witnesses,witnesses[1:]))):
                raise ValueError("Exact predecessor/successor state transfer not observed")
            body={"schema":CHAIN_SCHEMA,"plan_digest":self.plan["plan_digest"],
                "stop_layer":self.stop_layer,"position":self.position,
                "token_id":token_id,"previous":old_head,
                "layer_witnesses":witnesses}
            semantic={**body,"layer_witnesses":[{
                k:v for k,v in w.items() if k not in (
                    "range_chain","qualified_ranges","trained_weight_bytes")}
                for w in witnesses]}
            new_head=_digest(CHAIN_SCHEMA,semantic)
            self.position+=1
            self.last_output_digest=witnesses[-1]["output_f64_digest"]
            self.head=new_head
            return state,{**body,"chain_head":new_head,
                "complete_prefix_layers":self.stop_layer,
                "all_declared_backbone_layers_completed":(
                    self.stop_layer==self.plan["layer_count"]),
                "complete_decoder":False,
                "vocabulary_logits_computed":False}
        except Exception:
            for r,(position,left,right) in zip(self.runners,saved):
                r.position=position
                if isinstance(r,QwenQualifiedLayer):
                    r.convolution_history,r.recurrent_state=left,right
                else:
                    r.kv_keys,r.kv_values=left,right
            self.head,self.last_output_digest=old_head,old_output
            raise

    def checkpoint(self,store,*,input_sequence_digest):
        _admit_plan(self.nav,self.plan)
        if (not isinstance(store,LocalStatePages) or self.position<1 or
            any(r.position!=self.position for r in self.runners) or
            not _hex64(input_sequence_digest) or
            not _hex64(self.head) or
            not _hex64(self.last_output_digest)):
            raise ValueError("Incomplete original architecture checkpoint")
        parts=[]
        for r in self.runners:
            if isinstance(r,QwenQualifiedLayer):
                parts.append(export_qwen_causal_ground(
                    r,store,input_sequence_digest=input_sequence_digest))
            else:
                parts.append(export_attention_kv(
                    r,store,input_sequence_digest=input_sequence_digest))
        body={"schema":STATE_SCHEMA,
              "model":self.plan["model"],
              "revision":self.plan["revision"],
              "index_ground_digest":self.plan["index_ground_digest"],
              "config_sha256":self.plan["config_sha256"],
              "plan_digest":self.plan["plan_digest"],
              "stop_layer":self.stop_layer,
              "position":self.position,
              "input_sequence_digest":input_sequence_digest,
              "genesis":self.genesis,
              "causal_head":self.head,
              "last_output_f64_digest":self.last_output_digest,
              "layer_checkpoints":parts}
        return {**body,"checkpoint_digest":_digest(STATE_SCHEMA,body)}

    @classmethod
    def restore(cls,nav,plan,store,manifest,*,expected_digest,
                expected_input_sequence_digest,expected_plan_digest,
                layer_budget_bytes=2_000_000_000,max_positions=4096):
        if (not isinstance(manifest,dict) or
            not _hex64(expected_digest) or
            not _hex64(expected_input_sequence_digest) or
            not _hex64(expected_plan_digest) or
            manifest.get("checkpoint_digest")!=expected_digest or
            _digest(STATE_SCHEMA,{k:v for k,v in manifest.items()
                if k!="checkpoint_digest"})!=expected_digest):
            raise ValueError("Unpinned architecture checkpoint digest")
        _admit_plan(nav,plan)
        if plan["plan_digest"]!=expected_plan_digest:
            raise ValueError("Different externally pinned architecture plan")
        expected={"schema","model","revision","index_ground_digest",
            "config_sha256","plan_digest","stop_layer","position",
            "input_sequence_digest","genesis","causal_head",
            "last_output_f64_digest","layer_checkpoints","checkpoint_digest"}
        if (set(manifest)!=expected or manifest["schema"]!=STATE_SCHEMA or
            manifest["model"]!=plan["model"] or
            manifest["revision"]!=plan["revision"] or
            manifest["index_ground_digest"]!=plan["index_ground_digest"] or
            manifest["config_sha256"]!=plan["config_sha256"] or
            manifest["plan_digest"]!=plan["plan_digest"] or
            manifest["input_sequence_digest"]!=expected_input_sequence_digest or
            not _hex64(manifest["causal_head"]) or
            not _hex64(manifest["last_output_f64_digest"]) or
            type(manifest["position"]) is not int or manifest["position"]<1 or
            type(manifest["stop_layer"]) is not int or
            not 1<=manifest["stop_layer"]<=plan["layer_count"] or
            not isinstance(manifest["layer_checkpoints"],list) or
            len(manifest["layer_checkpoints"])!=manifest["stop_layer"]):
            raise ValueError("Altered model, layer state, geometry or lineage")
        chain=cls(nav,plan,stop_layer=manifest["stop_layer"],
                  layer_budget_bytes=layer_budget_bytes,max_positions=max_positions)
        if chain.genesis!=manifest["genesis"]:
            raise ValueError("Decoder prefix genesis differs")
        for r,part in zip(chain.runners,manifest["layer_checkpoints"]):
            if (not isinstance(part,dict) or
                not _hex64(part.get("checkpoint_digest"))):
                raise ValueError("Missing authoritative layer state digest")
            if isinstance(r,QwenQualifiedLayer):
                pos=restore_qwen_causal_ground(
                    r,store,part,expected_digest=part["checkpoint_digest"],
                    expected_input_sequence_digest=expected_input_sequence_digest)
            else:
                pos=restore_attention_kv(
                    r,store,part,expected_digest=part["checkpoint_digest"],
                    expected_input_sequence_digest=expected_input_sequence_digest)
            if pos!=manifest["position"]:
                raise ValueError("Different decoder layer causal positions")
        chain.position=manifest["position"]
        chain.head=manifest["causal_head"]
        chain.last_output_digest=manifest["last_output_f64_digest"]
        return chain
