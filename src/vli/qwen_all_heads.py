"""Unscaled Qwen trained Gated DeltaNet recurrent frontier on a model host.

This composes ALL 16 Q/K heads and ALL 48 value heads from actual qualified
weight projections. It carries 48 causal recurrent matrices and convolution
state between natural-language tokens. A single output-projection row binds
the 48-head result into a trained downstream coordinate.

It does NOT claim an entire transformer layer: the other 5119 output rows,
residual and complete SwiGLU FFN still need execution. The existing
QwenQualifiedLayer can execute those once enough host budget is available.

The runtime interprets the admitted operation; the host performs the
vectorized trained arithmetic from its own bounded source block memory.
"""
from __future__ import annotations

import math
import time

from experiments.model_fabric.architecture_ops import (
    depthwise_causal_conv_step, qwen_gates, qwen_gated_delta_step,
    qwen_gated_rms_norm, state_digest,
)
from experiments.model_fabric.complete_native_layers import QwenQualifiedLayer
from experiments.model_fabric.qualified_subgraphs import observed_values
from experiments.model_fabric.trained_subgraphs import centered_rms_norm

from .qwen_causal_head import _reference_sample
from .qualified_projection_host import fast_rows


class QwenAllHeadsRecurrentHost(QwenQualifiedLayer):
    """Model-worker implementation of 16Q/16K/48V original recurrent heads."""

    def advance_recurrent(self, hidden, *, independent=True):
        if (not isinstance(hidden,(list,tuple)) or len(hidden)!=self.hidden
            or any(type(z) not in (int,float) or not math.isfinite(z) for z in hidden)):
            raise ValueError("Unqualified first-layer original token vector")
        t0=time.perf_counter(); cpu0=time.process_time()
        begin_bytes,begin_ranges=self.nav.weight_bytes,self.nav.range_count
        p=self.names
        normalized=centered_rms_norm(hidden,self._vectors['input_layernorm.weight'],
                                     eps=self.eps)
        qkv=fast_rows(self.nav,p['linear_attn.in_proj_qkv.weight'],normalized)
        z=fast_rows(self.nav,p['linear_attn.in_proj_z.weight'],normalized)
        a=fast_rows(self.nav,p['linear_attn.in_proj_a.weight'],normalized)
        b=fast_rows(self.nav,p['linear_attn.in_proj_b.weight'],normalized)
        qkv,next_conv=depthwise_causal_conv_step(
            qkv,self._convolution,self.convolution_history,activation='silu')
        beta,decay=qwen_gates(a,b,self._vectors['linear_attn.A_log'],
                             self._vectors['linear_attn.dt_bias'])
        kdim=self.kh*self.kd
        q=[qkv[i*self.kd:(i+1)*self.kd] for i in range(self.kh)]
        k=[qkv[kdim+i*self.kd:kdim+(i+1)*self.kd]
           for i in range(self.kh)]
        v=[qkv[2*kdim+i*self.vd:2*kdim+(i+1)*self.vd]
           for i in range(self.vh)]
        earlier=self.recurrent_state
        raw_recurrent,next_delta=qwen_gated_delta_step(
            q,k,v,decay,beta,self.recurrent_state)
        norm_weight=self._vectors['linear_attn.norm.weight']
        values=[]
        for i in range(self.vh):
            head=qwen_gated_rms_norm(
                raw_recurrent[i],z[i*self.vd:(i+1)*self.vd],
                norm_weight,eps=self.eps)
            values.extend(head)
        expected=self.vh*self.vd
        if len(values)!=expected or not all(math.isfinite(t) for t in values):
            raise ValueError("Original pretrained all-head GDN output is incomplete")
        first_row=fast_rows(self.nav,p['linear_attn.out_proj.weight'],values,
                            first=0,count=1)[0]
        reference_errors=[]
        if independent:
            ref=observed_values(self.nav,p['linear_attn.out_proj.weight'],
                                start=0,count=expected)
            independent_row=math.fsum(w*t for w,t in zip(ref,values))
            reference_errors.append(abs(first_row-independent_row))
            for head in (0,self.vh-1):
                group=head//(self.vh//self.kh)
                for channel in (0,self.vd-1):
                    reference_errors.append(abs(raw_recurrent[head][channel]-
                        _reference_sample(q[group],k[group],v[head],
                          decay[head],beta[head],
                          earlier[head] if earlier is not None else None,channel)))
        max_error=max(reference_errors) if reference_errors else None
        if (independent and
            (max_error is None or not math.isfinite(max_error)
             or max_error>2e-7)):
            raise ValueError("Independent full-head reference mismatch")
        # Commit only after all requests and numerical references complete.
        self.convolution_history=next_conv
        self.recurrent_state=next_delta
        old_position=self.position
        self.position+=1
        return {
            "schema":"vl-i.qwen-all-heads-recurrent.v1",
            "model":self.nav.index['model'],
            "revision":self.nav.index['revision'],
            "layer":self.layer,
            "position":old_position,
            "trained_query_heads":self.kh,
            "trained_key_heads":self.kh,
            "trained_value_heads":self.vh,
            "full_recurrent_output_width":len(values),
            "all_head_output_digest":state_digest(values),
            "output_projection_row0":round(first_row,12),
            "full_recurrent_state_digest":state_digest(next_delta),
            "full_convolution_state_digest":state_digest(next_conv),
            "max_independent_reference_error":max_error,
            "new_weight_payload_bytes":self.nav.weight_bytes-begin_bytes,
            "new_qualified_ranges":self.nav.range_count-begin_ranges,
            "range_chain":self.nav.chain,
            "worker_wall_seconds":time.perf_counter()-t0,
            "worker_cpu_seconds":time.process_time()-cpu0,
            "complete_Qwen_transformer_layer":False,
            "scope":"all original Qwen GDN recurrent heads and first output row; FFN and remaining output rows not executed",
        }
