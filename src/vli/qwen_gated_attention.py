"""QUALIFIED-007: full Qwen3.8 gated causal attention at the model host.

The first full-attention layer (coordinate 3) has 24 query heads, four
KV heads, 256-wide head vectors, 64 rotary dimensions, and a SwiGLU FFN.
The original pretrained source is observed through a qualified navigator.
This model-worker implementation is deliberately OUTSIDE VL-64.

The exact physical operator remains CPU/NumPy work on the worker, not
network-native arithmetic. Text-only positions use equal M-RoPE axes:
the ordinary 1D RoPE angles are then the exact text-position case.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import struct
import time

from experiments.model_fabric.complete_native_layers import (
    LocalStatePages, _f32_state_digest)
from experiments.model_fabric.qualified_subgraphs import observed_values
from experiments.model_fabric.scaled_model_fabric import QualifiedShardNavigator
from experiments.model_fabric.trained_subgraphs import centered_rms_norm

from .ground import _digest, _hex64
from .qualified_projection_host import fast_rows

ATTENTION_RECEIPT="vl-i.qualified-original-gated-attention.v1"
ATTENTION_STATE="vl-i.qualified-original-attention-kv.v1"

_PARTS=(
    "input_layernorm.weight", "post_attention_layernorm.weight",
    "self_attn.q_proj.weight", "self_attn.k_proj.weight",
    "self_attn.v_proj.weight", "self_attn.o_proj.weight",
    "self_attn.q_norm.weight", "self_attn.k_norm.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
)


def _sigmoid(x):
    return 1/(1+math.exp(-x)) if x>=0 else math.exp(x)/(1+math.exp(x))


def rotate_text_rope(values, position, rotary_dim, theta):
    """Official rotate_half form, first rotary_dim dimensions only.

    The multimodal Qwen RoPE uses three position axes; the text-only
    default repeats one token coordinate on all axes, which reduces to
    this pairwise rotation. This helper is *not* a vision/video adapter.
    """
    if (not isinstance(values,list) or not values or
        type(position) is not int or position<0 or
        type(rotary_dim) is not int or rotary_dim<2 or
        rotary_dim%2 or rotary_dim>len(values) or
        type(theta) not in (int,float) or theta<=1 or
        any(type(z) not in (int,float) or not math.isfinite(z) for z in values)):
        raise ValueError("Unqualified rotary text position")
    half=rotary_dim//2
    output=values[:]
    for i in range(half):
        angle=position*theta**(-2*i/rotary_dim)
        co,si=math.cos(angle),math.sin(angle)
        a,b=values[i],values[i+half]
        output[i]=a*co-b*si
        output[i+half]=b*co+a*si
    return output


def _kv_raw(keys,values):
    if len(keys)!=len(values):
        raise ValueError("Malformed causal KV chronology")
    flat=[]
    for key_position,value_position in zip(keys,values):
        if len(key_position)!=len(value_position):
            raise ValueError("Malformed Qwen KV heads")
        for key,value in zip(key_position,value_position):
            if len(key)!=len(value):
                raise ValueError("Mismatched KV head dimension")
            flat.extend(key)
            flat.extend(value)
    if not all(type(z) in (int,float) and math.isfinite(z) for z in flat):
        raise ValueError("Nonfinite retained full-attention KV state")
    return struct.pack("<"+str(len(flat))+"d",*flat)


def _kv_digest(raw):
    return hashlib.sha256(ATTENTION_STATE.encode()+b"\0"+raw).hexdigest()


class QwenQualifiedGatedAttention:
    """Original-weight GQA + sigmoid gate + residual SwiGLU causal layer.

    Projections read original BF16/F16/F32 weights via fast_rows; the
    existing QualifiedShardNavigator authenticates <=64KiB logical ranges.
    """
    def __init__(self, nav, *, layer=3, query_heads=24,
                 kv_heads=4, head_dim=256, rotary_dim=64,
                 rope_theta=10_000_000.0, eps=1e-6,
                 budget_bytes=2_000_000_000, max_positions=4096):
        if not isinstance(nav,QualifiedShardNavigator):
            raise ValueError("Qualified original source navigator required")
        if (type(layer) is not int or layer<0 or layer%4!=3 or
            any(type(v) is not int or v<=0 for v in
                (query_heads,kv_heads,head_dim,rotary_dim,max_positions,
                 budget_bytes)) or query_heads%kv_heads or
            rotary_dim>head_dim or rotary_dim%2 or
            not 1<=max_positions<=262144 or
            type(rope_theta) not in (int,float) or
            not math.isfinite(rope_theta) or rope_theta<=1 or
            type(eps) not in (int,float) or not 0<eps<=1):
            raise ValueError("Unsupported original Qwen full-attention geometry")
        self.nav,self.layer=nav,layer
        self.qh,self.kvh,self.hd,self.rd=query_heads,kv_heads,head_dim,rotary_dim
        self.rope_theta,self.eps=rope_theta,eps
        self.max_positions=max_positions
        self.names={part:f"model.language_model.layers.{layer}.{part}" for part in _PARTS}
        if any(name not in nav.weight_map for name in self.names.values()):
            raise ValueError("Incomplete original attention and SwiGLU trained weights")
        descs={part:nav.locate(name) for part,name in self.names.items()}
        shapes={part:info["shape"] for part,info in descs.items()}
        self.hidden=shapes["input_layernorm.weight"][0]
        self.intermediate=shapes["mlp.gate_proj.weight"][0]
        qwidth=query_heads*head_dim; kvwidth=kv_heads*head_dim
        required={
            "input_layernorm.weight":[self.hidden],
            "post_attention_layernorm.weight":[self.hidden],
            "self_attn.q_proj.weight":[2*qwidth,self.hidden],
            "self_attn.k_proj.weight":[kvwidth,self.hidden],
            "self_attn.v_proj.weight":[kvwidth,self.hidden],
            "self_attn.o_proj.weight":[self.hidden,qwidth],
            "self_attn.q_norm.weight":[head_dim],
            "self_attn.k_norm.weight":[head_dim],
            "mlp.gate_proj.weight":[self.intermediate,self.hidden],
            "mlp.up_proj.weight":[self.intermediate,self.hidden],
            "mlp.down_proj.weight":[self.hidden,self.intermediate],
        }
        dsize={"BF16":2,"F16":2,"F32":4}
        if (any(shapes[p]!=s or descs[p]["dtype"] not in dsize
                for p,s in required.items()) or
            any(len(s)==2 and len(s[0:2])!=2 for s in shapes.values())):
            raise ValueError("Original full-attention tensor shapes/dtypes differ")
        estimated=sum(math.prod(info["shape"])*dsize[info["dtype"]]
                      for info in descs.values())
        if estimated>budget_bytes:
            raise ValueError("Original complete attention layer exceeds weight budget")
        self.estimated_weight_bytes=estimated
        self.vectors={k:observed_values(nav,self.names[k]) for k in (
            "input_layernorm.weight","post_attention_layernorm.weight",
            "self_attn.q_norm.weight","self_attn.k_norm.weight")}
        self.position=0
        self.kv_keys=[]
        self.kv_values=[]

    def _projection(self, part, values):
        return fast_rows(self.nav,self.names[part],values)

    def next(self, hidden):
        if (not isinstance(hidden,(list,tuple)) or len(hidden)!=self.hidden or
            any(type(x) not in (int,float) or not math.isfinite(x) for x in hidden) or
            self.position>=self.max_positions or
            len(self.kv_keys)!=self.position or len(self.kv_values)!=self.position):
            raise ValueError("Unqualified Qwen full-attention position or input")
        start_wall=time.perf_counter()
        start_cpu=time.process_time()
        count_before,bytes_before=self.nav.range_count,self.nav.weight_bytes
        residual=list(hidden)
        normalized=centered_rms_norm(residual,self.vectors["input_layernorm.weight"],
                                     eps=self.eps)
        qgate=self._projection("self_attn.q_proj.weight",normalized)
        k=self._projection("self_attn.k_proj.weight",normalized)
        v=self._projection("self_attn.v_proj.weight",normalized)
        queries=[];gates=[]
        for head in range(self.qh):
            block=qgate[head*2*self.hd:(head+1)*2*self.hd]
            queries.append(rotate_text_rope(
                centered_rms_norm(block[:self.hd],self.vectors["self_attn.q_norm.weight"],
                                  eps=self.eps),self.position,self.rd,self.rope_theta))
            gates.append(block[self.hd:])
        keys=[rotate_text_rope(
            centered_rms_norm(k[h*self.hd:(h+1)*self.hd],
                              self.vectors["self_attn.k_norm.weight"],eps=self.eps),
            self.position,self.rd,self.rope_theta) for h in range(self.kvh)]
        values=[v[h*self.hd:(h+1)*self.hd] for h in range(self.kvh)]
        # Purely local tentative candidate: state is committed only after
        # complete original out-projection + trained FFN and finite checks.
        candidate_keys=self.kv_keys+[keys]
        candidate_values=self.kv_values+[values]
        heads=[]
        for head in range(self.qh):
            kv_head=head//(self.qh//self.kvh)
            q=queries[head]
            scores=[math.fsum(a*b for a,b in zip(q,p[kv_head]))/math.sqrt(self.hd)
                    for p in candidate_keys]
            top=max(scores)
            scale=[math.exp(x-top) for x in scores]
            total=math.fsum(scale)
            if not 0<total<math.inf:
                raise ValueError("Nonfinite original causal attention normalization")
            weights=[s/total for s in scale]
            output=[math.fsum(w*p[kv_head][channel]
                    for w,p in zip(weights,candidate_values))
                    for channel in range(self.hd)]
            heads.extend(z*_sigmoid(g) for z,g in zip(output,gates[head]))
        projected=self._projection("self_attn.o_proj.weight",heads)
        attended=[x+y for x,y in zip(residual,projected)]
        feed=centered_rms_norm(attended,self.vectors["post_attention_layernorm.weight"],
                               eps=self.eps)
        gate=self._projection("mlp.gate_proj.weight",feed)
        up=self._projection("mlp.up_proj.weight",feed)
        intermediate=[g*_sigmoid(g)*u for g,u in zip(gate,up)]
        down=self._projection("mlp.down_proj.weight",intermediate)
        result=[x+y for x,y in zip(attended,down)]
        if len(result)!=self.hidden or not all(math.isfinite(x) for x in result):
            raise ValueError("Incomplete or nonfinite original full-attention output")
        state_raw=_kv_raw(candidate_keys,candidate_values)
        # Commit after every numerical and source operation finishes.
        before=self.position
        self.kv_keys,self.kv_values=candidate_keys,candidate_values
        self.position+=1
        return result,{
            "schema":ATTENTION_RECEIPT,
            "model":self.nav.index["model"],
            "revision":self.nav.index["revision"],
            "layer":self.layer,
            "position":before,
            "complete_Qwen_transformer_layer":True,
            "attention_kind":"causal_gated_grouped_query",
            "complete_output_width":self.hidden,
            "complete_layer_output_digest":_f32_state_digest(result),
            "full_attention_kv_state_digest":_kv_digest(state_raw),
            "kv_history_positions":self.position,
            "trained_query_heads":self.qh,
            "trained_kv_heads":self.kvh,
            "new_qualified_ranges":self.nav.range_count-count_before,
            "new_weight_payload_bytes":self.nav.weight_bytes-bytes_before,
            "range_chain":self.nav.chain,
            "host_worker_cpu_seconds":time.process_time()-start_cpu,
            "host_worker_wall_seconds":time.perf_counter()-start_wall,
            "complete_original_64_layer_decoder":False,
        }


def export_attention_kv(runner,store,*,input_sequence_digest):
    """Publish bounded, source-bound causal KV pages with an external pin."""
    if (not isinstance(runner,QwenQualifiedGatedAttention) or
        not isinstance(store,LocalStatePages) or runner.position<=0 or
        runner.position!=len(runner.kv_keys) or
        runner.position!=len(runner.kv_values) or
        not _hex64(input_sequence_digest)):
        raise ValueError("Unqualified full-attention checkpoint authority")
    raw=_kv_raw(runner.kv_keys,runner.kv_values)
    pages=[store.put(raw[i:i+65536]) for i in range(0,len(raw),65536)]
    body={"schema":ATTENTION_STATE,
          "model":runner.nav.index["model"],
          "revision":runner.nav.index["revision"],
          "index_ground_digest":runner.nav.index["ground_digest"],
          "layer":runner.layer,"position":runner.position,
          "input_sequence_digest":input_sequence_digest,
          "query_heads":runner.qh,"kv_heads":runner.kvh,
          "head_dim":runner.hd,"rotary_dim":runner.rd,
          "rope_theta":runner.rope_theta,
          "kv_pages":pages,"kv_f64_bytes":len(raw),
          "kv_state_digest":_kv_digest(raw)}
    return {**body,"checkpoint_digest":_digest(ATTENTION_STATE,body)}


def restore_attention_kv(runner,store,checkpoint,*,expected_digest,
                         expected_input_sequence_digest):
    if (not isinstance(runner,QwenQualifiedGatedAttention) or
        not isinstance(store,LocalStatePages) or runner.position!=0 or
        not isinstance(checkpoint,dict) or not _hex64(expected_digest) or
        not _hex64(expected_input_sequence_digest) or
        checkpoint.get("checkpoint_digest")!=expected_digest or
        _digest(ATTENTION_STATE,{k:v for k,v in checkpoint.items()
            if k!="checkpoint_digest"})!=expected_digest):
        raise ValueError("Unpinned or modified attention KV checkpoint")
    expected={"schema","model","revision","index_ground_digest","layer",
        "position","input_sequence_digest","query_heads","kv_heads","head_dim",
        "rotary_dim","rope_theta","kv_pages","kv_f64_bytes","kv_state_digest",
        "checkpoint_digest"}
    if (set(checkpoint)!=expected or checkpoint["schema"]!=ATTENTION_STATE or
        checkpoint["model"]!=runner.nav.index["model"] or
        checkpoint["revision"]!=runner.nav.index["revision"] or
        checkpoint["index_ground_digest"]!=runner.nav.index["ground_digest"] or
        checkpoint["layer"]!=runner.layer or
        checkpoint["input_sequence_digest"]!=expected_input_sequence_digest or
        checkpoint["query_heads"]!=runner.qh or
        checkpoint["kv_heads"]!=runner.kvh or
        checkpoint["head_dim"]!=runner.hd or
        checkpoint["rotary_dim"]!=runner.rd or
        checkpoint["rope_theta"]!=runner.rope_theta or
        type(checkpoint["position"]) is not int or
        not 1<=checkpoint["position"]<=runner.max_positions or
        not isinstance(checkpoint["kv_pages"],list) or
        not checkpoint["kv_pages"]):
        raise ValueError("Wrong attention model, lineage or causal extent")
    raw=b"".join(store.get(page) for page in checkpoint["kv_pages"])
    expected_len=checkpoint["position"]*runner.kvh*runner.hd*2*8
    if (len(raw)!=expected_len or
        checkpoint["kv_f64_bytes"]!=expected_len or
        checkpoint["kv_state_digest"]!=_kv_digest(raw)):
        raise ValueError("Incomplete or corrupted attention KV pages")
    vals=struct.unpack("<"+str(expected_len//8)+"d",raw)
    if not all(math.isfinite(x) for x in vals):
        raise ValueError("Nonfinite restored attention KV values")
    index=0
    keys=[];values=[]
    for _ in range(checkpoint["position"]):
        kp=[];vp=[]
        for _ in range(runner.kvh):
            kp.append(list(vals[index:index+runner.hd]));index+=runner.hd
            vp.append(list(vals[index:index+runner.hd]));index+=runner.hd
        keys.append(kp);values.append(vp)
    runner.kv_keys,runner.kv_values=keys,values
    runner.position=checkpoint["position"]
    return runner.position
