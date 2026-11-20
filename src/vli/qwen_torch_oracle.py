"""QUALIFIED-009: independent PyTorch F64 oracle for original trained Qwen GQA.

Only the safetensors byte-range *source navigator* is shared with VL-I.
The numerical reference here is independently implemented using PyTorch
tensor operations, without VL-I's fast_rows, QwenQualifiedGatedAttention,
or its scalar/numpy projection implementations.

This is an independent *mathematical F64* oracle, not a claim that it
matches the BF16 roundoff of an official Hugging Face full-model runtime.
Weights are read from the original commit, never generated or downscaled.
The oracle is an external model-worker workload, not part of VL-64.
"""
from __future__ import annotations

import hashlib
import math
import struct

from experiments.model_fabric.scaled_model_fabric import QualifiedShardNavigator
from .ground import _hex64


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("An independent CPU PyTorch runtime is required") from exc
    return torch


def _tensor_from_raw(raw, *, dtype, expected):
    """Decode original exact safetensors bytes independently from fast_rows."""
    import numpy as np
    torch=_torch()
    if dtype=="BF16":
        if len(raw)!=expected*2:
            raise ValueError("Truncated pretrained BF16 tensor")
        integers=np.frombuffer(raw,dtype="<u2").astype("<u4")
        integers=np.left_shift(integers,16)
        numbers=integers.view("<f4")
    elif dtype=="F16":
        if len(raw)!=expected*2:
            raise ValueError("Truncated pretrained F16 tensor")
        numbers=np.frombuffer(raw,dtype="<f2")
    elif dtype=="F32":
        if len(raw)!=expected*4:
            raise ValueError("Truncated pretrained F32 tensor")
        numbers=np.frombuffer(raw,dtype="<f4")
    else:
        raise ValueError("Unsupported pretrained PyTorch reference dtype")
    if numbers.size!=expected or not np.isfinite(numbers).all():
        raise ValueError("Nonfinite or incomplete reference weights")
    return torch.from_numpy(numbers.copy()).to(dtype=torch.float64)


class OriginalQwenTorchGQA:
    """Qwen full-attention mathematical oracle with independent torch kernels.

    The caller must have separately admitted source config and tensor index.
    Causal KV state stays private and commits only after the full trained FFN.
    """

    def __init__(self, nav, plan, *, layer=7, batch_rows=64):
        if not isinstance(nav,QualifiedShardNavigator) or not isinstance(plan,dict):
            raise ValueError("Independent original Qwen reference needs qualified source")
        if (type(layer) is not int or layer<0 or
            type(batch_rows) is not int or not 1<=batch_rows<=128 or
            not _hex64(plan.get("plan_digest")) or
            plan.get("model")!=nav.index["model"] or
            plan.get("revision")!=nav.index["revision"] or
            plan.get("index_ground_digest")!=nav.index["ground_digest"] or
            not isinstance(plan.get("layers"),list) or
            layer>=len(plan["layers"]) or
            plan["layers"][layer]!={"layer":layer,"kind":"GATED_GQA"}):
            raise ValueError("Source plan does not declare this trained attention layer")
        # Confirm plan digest independently, without invoking the VL-I runner.
        from .ground import _digest
        if _digest(plan["schema"],{k:v for k,v in plan.items()
                   if k!="plan_digest"})!=plan["plan_digest"]:
            raise ValueError("Original model plan digest was altered")
        g=plan["geometry"]
        self.width=g["hidden"]
        self.intermediate=g["intermediate"]
        self.head_dim=g["head_dim"]
        self.query_heads=g["query_heads"]
        self.kv_heads=g["kv_heads"]
        self.rotary_dim=g["rotary_dim"]
        self.theta=g["rope_theta"]
        self.eps=g["eps"]
        self.nav=nav
        self.layer=layer
        self.batch_rows=batch_rows
        self.position=0
        self.history=[]
        self.tensor_prefix=f"model.language_model.layers.{layer}."
        if self.query_heads%self.kv_heads or self.rotary_dim%2:
            raise ValueError("Unqualified grouped attention or RoPE topology")
        self._norm_weights={}
        for name,extent in (
            ("input_layernorm.weight",self.width),
            ("post_attention_layernorm.weight",self.width),
            ("self_attn.q_norm.weight",self.head_dim),
            ("self_attn.k_norm.weight",self.head_dim)):
            actual=self.nav.locate(self.tensor_prefix+name)
            if actual["shape"]!=[extent]:
                raise ValueError("Incorrect independent original norm width")
            raw,_=self.nav.get_raw(self.tensor_prefix+name)
            self._norm_weights[name]=_tensor_from_raw(
                raw,dtype=actual["dtype"],expected=extent)

    def _linear(self, part, x, *, output_width):
        torch=_torch()
        name=self.tensor_prefix+part
        descriptor=self.nav.locate(name)
        width=x.numel()
        if (descriptor["shape"]!=[output_width,width] or
            descriptor["dtype"] not in ("BF16","F16","F32")):
            raise ValueError("Reference matrix has wrong certified original extent")
        rows_per_batch=min(self.batch_rows,max(1,8*1024*1024//(width*8)))
        outputs=[]
        for first in range(0,output_width,rows_per_batch):
            count=min(rows_per_batch,output_width-first)
            raw,again=self.nav.get_raw(
                name,flat_start=first*width,count=count*width)
            if again!=descriptor:
                raise ValueError("Reference observed a modified tensor descriptor")
            block=_tensor_from_raw(
                raw,dtype=descriptor["dtype"],expected=count*width
                ).reshape(count,width)
            result=torch.mv(block,x)
            if not bool(torch.isfinite(result).all()):
                raise ValueError("Nonfinite independent pretrained projection")
            outputs.append(result)
        return torch.cat(outputs)

    def _norm(self, value, name):
        torch=_torch()
        weight=self._norm_weights[name]
        if value.shape[-1]!=weight.numel():
            raise ValueError("Reference normalization geometry differs")
        # Qwen3.5 zero-centered RMSNorm (1 + learned weight).
        factor=torch.rsqrt(torch.mean(torch.square(value),dim=-1,keepdim=True)+self.eps)
        return value*factor*(1.0+weight)

    def _rotary(self, head, position):
        torch=_torch()
        d=self.rotary_dim
        i=torch.arange(d//2,dtype=torch.float64)
        frequency=torch.pow(torch.tensor(self.theta,dtype=torch.float64),-2*i/d)
        phases=position*frequency
        first=head[:d//2]
        second=head[d//2:d]
        rotated=torch.cat((
            first*torch.cos(phases)-second*torch.sin(phases),
            second*torch.cos(phases)+first*torch.sin(phases)))
        return torch.cat((rotated,head[d:]))

    def next(self, hidden):
        """Perform an independently formulated whole original GQA+FFN layer."""
        torch=_torch()
        if (not isinstance(hidden,(list,tuple)) or len(hidden)!=self.width or
            self.position!=len(self.history) or
            any(type(x) not in (int,float) or not math.isfinite(x) for x in hidden)):
            raise ValueError("Reference original hidden state or history incomplete")
        start_ranges=self.nav.range_count
        start_bytes=self.nav.weight_bytes
        x=torch.tensor(list(hidden),dtype=torch.float64)
        norm_x=self._norm(x,"input_layernorm.weight")
        hd=self.head_dim
        qh=self.query_heads
        kvh=self.kv_heads
        qraw=self._linear("self_attn.q_proj.weight",norm_x,
                          output_width=2*qh*hd).reshape(qh,2*hd)
        kraw=self._linear("self_attn.k_proj.weight",norm_x,
                          output_width=kvh*hd).reshape(kvh,hd)
        vraw=self._linear("self_attn.v_proj.weight",norm_x,
                          output_width=kvh*hd).reshape(kvh,hd)
        query=torch.stack([self._rotary(
            self._norm(qraw[h,:hd],"self_attn.q_norm.weight"),
            self.position) for h in range(qh)])
        key=torch.stack([self._rotary(
            self._norm(kraw[h],"self_attn.k_norm.weight"),
            self.position) for h in range(kvh)])
        # History is a pair of [KV-heads, dimension] tensors per token.
        all_contexts=self.history+[(key,vraw)]
        channels=[]
        attention_distributions=[]
        context_differences=[]
        for h in range(qh):
            kv_idx=h//(qh//kvh)
            history_k=torch.stack([state[0][kv_idx] for state in all_contexts])
            history_v=torch.stack([state[1][kv_idx] for state in all_contexts])
            attn=torch.softmax(
                torch.mv(history_k,query[h])/math.sqrt(hd),dim=0)
            context=attn @ history_v
            # Both Q/K selection and actual earlier V contribution are
            # independent mathematical observations; in position 0 the
            # history has one item and these quantities are degenerate.
            attention_distributions.append([float(z) for z in attn.tolist()])
            context_differences.append(float(
                torch.max(torch.abs(context-history_v[-1])).item()))
            channels.append(context*torch.sigmoid(qraw[h,hd:]))
        projected=self._linear(
            "self_attn.o_proj.weight",torch.cat(channels),
            output_width=self.width)
        attended=x+projected
        ffinput=self._norm(attended,"post_attention_layernorm.weight")
        ff_gate=self._linear("mlp.gate_proj.weight",ffinput,
                             output_width=self.intermediate)
        ff_up=self._linear("mlp.up_proj.weight",ffinput,
                           output_width=self.intermediate)
        ff_output=self._linear(
            "mlp.down_proj.weight",
            (ff_gate*torch.sigmoid(ff_gate))*ff_up,
            output_width=self.width)
        output=attended+ff_output
        if not bool(torch.isfinite(output).all()):
            raise ValueError("Reference whole pretrained attention result nonfinite")
        values=[float(z) for z in output.tolist()]
        # No state promotion before source range and all numerical stages pass.
        self.history=all_contexts
        self.position+=1
        from .qwen_layer_chain import _f64_transfer_digest
        return values,{
            "schema":"vl-i.original-qwen-independent-pytorch-f64.v1",
            "layer":self.layer,
            "position":self.position-1,
            "model":self.nav.index["model"],
            "revision":self.nav.index["revision"],
            "source_index_ground_digest":self.nav.index["ground_digest"],
            "query_heads":qh,"kv_heads":kvh,
            "causal_kv_positions":self.position,
            "attention_probabilities_per_query_head":attention_distributions,
            "max_deviation_from_uniform_attention":max(
                abs(p-1/self.position) for row in attention_distributions
                for p in row),
            "max_context_difference_from_current_token_values":max(context_differences),
            "complete_output_width":len(values),
            "torch_f64_output_digest":_f64_transfer_digest(values),
            "new_source_weight_payload_bytes":self.nav.weight_bytes-start_bytes,
            "new_qualified_64k_range_reads":self.nav.range_count-start_ranges,
            "math_runtime":"independent_torch_float64",
            "official_huggingface_bf16_runtime_parity":False,
        }
