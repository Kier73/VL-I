"""INTERACT-004: one actual Qwen gated-delta value head across causal tokens.

Receives real 5120-wide trained embedding rows (from the pinned-text Ground),
not a synthetic vector. Executes the original first-layer Q/K/V projections,
depthwise convolution, A/B gates, recurrent delta, gated norm and *one*
qualified output-projection row's selected-head contribution. The full
48-head layer and FFN remain required for complete model inference.
"""
from __future__ import annotations

from copy import deepcopy
import math

from .ground import _digest, _hex64, _require
from .large_model_comparison import SOURCES
from .operative_model import _check_ground

SCHEMA="vl-i.qwen-causal-head.v1"
CHECKPOINT="vl-i.qwen-causal-head-checkpoint.v1"
MAX_STEPS=32


def _reference_sample(q,k,v,decay,beta,prior,c):
    """Separate scalar expansion of one recurrent output channel."""
    dk=len(q)
    def norm(x):
        factor=1/math.sqrt(math.fsum(z*z for z in x)+1e-6)
        return [z*factor for z in x]
    qn=[z/math.sqrt(dk) for z in norm(q)]
    kn=norm(k)
    fading=math.exp(decay)
    pre=[(prior[j][c] if prior is not None else 0.)*fading
         for j in range(dk)]
    prediction=math.fsum(kn[j]*pre[j] for j in range(dk))
    residual=beta*(v[c]-prediction)
    return math.fsum(qn[j]*(pre[j]+kn[j]*residual) for j in range(dk))


class QwenCausalHead:
    """One genuine head; small host-state interpreter, not a complete layer."""
    def __init__(self,trained_ground,*,head=0):
        _check_ground(trained_ground)
        _require(type(trained_ground) is dict
                 and trained_ground.get("route")=="qwen-27b"
                 and trained_ground.get("model")==SOURCES["qwen-27b"]["model"]
                 and trained_ground.get("revision")==SOURCES["qwen-27b"]["revision"]
                 and type(head) is int and 0<=head<48,
                 "Qwen causal trained head not qualified")
        self.g=deepcopy(trained_ground)
        self.head_id=head
        self.position=0
        self.history=None
        self.state=None
        self.genesis=_digest(SCHEMA,{"model_ground":trained_ground["ground_digest"],
                                    "head_id":head})
        self.chain=self.genesis

    def next(self,nav,hidden,row_witness,*,expected_head):
        from experiments.model_fabric.qualified_subgraphs import (
             observed_values,observed_rows)
        from experiments.model_fabric.trained_subgraphs import centered_rms_norm
        from experiments.model_fabric.architecture_ops import (
             depthwise_causal_conv_step, qwen_gates, qwen_gated_delta_step,
             qwen_gated_rms_norm, state_digest)
        _require(expected_head==self.chain and self.position<MAX_STEPS,
                 "Stale selected-head causal predecessor")
        _require(type(hidden) is list and len(hidden)==5120
                 and all(type(z) in (float,int) and math.isfinite(z) for z in hidden)
                 and type(row_witness) is dict
                 and row_witness.get("tensor")==
                   "model.language_model.embed_tokens.weight"
                 and _hex64(row_witness.get("embedding_sha256"))
                 and type(row_witness.get("token_id")) is int,
                 "Incomplete real pretrained Qwen token embedding input")
        s=SOURCES["qwen-27b"]
        _require(nav.index["model"]==s["model"]
                 and nav.index["revision"]==s["revision"]
                 and nav.index["index_sha256"]==s["index_sha256"]
                 and nav.index["ground_digest"]==s["index_ground_digest"],
                 "Changed pretrained checkpoint in selected Qwen head")
        p="model.language_model.layers.0."
        names={
          "norm":p+"input_layernorm.weight",
          "qkv":p+"linear_attn.in_proj_qkv.weight",
          "z":p+"linear_attn.in_proj_z.weight",
          "a":p+"linear_attn.in_proj_a.weight",
          "b":p+"linear_attn.in_proj_b.weight",
          "conv":p+"linear_attn.conv1d.weight",
          "alog":p+"linear_attn.A_log",
          "dt":p+"linear_attn.dt_bias",
          "gamma":p+"linear_attn.norm.weight",
          "out":p+"linear_attn.out_proj.weight",
        }
        _require(all(x in nav.weight_map for x in names.values()),
                 "Missing trained Qwen head weight relation")
        d=nav.locate(names["qkv"])
        _require(d["shape"]==[10240,5120],
                 "Qwen original Gated DeltaNet projection geometry differs")
        _require(nav.locate(names["conv"])["shape"]==[10240,1,4],
                 "Qwen original causal convolution geometry differs")
        _require(nav.locate(names["out"])["shape"]==[5120,6144],
                 "Qwen original trained output-projection geometry differs")
        observed_before=nav.range_count
        norm=observed_values(nav,names["norm"])
        _require(len(norm)==5120,"Qwen native input norm width differs")
        x=centered_rms_norm(hidden,norm)
        qk=self.head_id//3
        offsets=(qk*128,2048+qk*128,4096+self.head_id*128)
        raw_qkv=[observed_rows(nav,names["qkv"],x,first=o,count=128)
                 for o in offsets]
        kernel=[]
        for start in offsets:
            fragment=observed_values(nav,names["conv"],start=start*4,count=512)
            kernel.extend([fragment[c*4:(c+1)*4] for c in range(128)])
        packed=[v for group in raw_qkv for v in group]
        activated,history=depthwise_causal_conv_step(
            packed,kernel,self.history,activation="silu")
        a=observed_rows(nav,names["a"],x,first=self.head_id,count=1)
        b=observed_rows(nav,names["b"],x,first=self.head_id,count=1)
        logs=observed_values(nav,names["alog"],start=self.head_id,count=1)
        dt=observed_values(nav,names["dt"],start=self.head_id,count=1)
        beta,decay=qwen_gates(a,b,logs,dt)
        q,k,v=(activated[0:128],activated[128:256],activated[256:384])
        updated,state=qwen_gated_delta_step([q],[k],[v],decay,beta,
                  state=[self.state] if self.state is not None else None)
        samples=(0,127)
        errors=[abs(updated[0][c]-_reference_sample(
                   q,k,v,decay[0],beta[0],self.state,c)) for c in samples]
        error=max(errors)
        _require(math.isfinite(error) and error<=2e-8,
                 "Independent trained causal recurrence disagrees")
        gamma=observed_values(nav,names["gamma"])
        gate=observed_rows(nav,names["z"],x,
                           first=self.head_id*128,count=128)
        head_output=qwen_gated_rms_norm(updated[0],gate,gamma)
        projection=observed_values(nav,names["out"],
                                  start=self.head_id*128,count=128)
        partial=math.fsum(w*z for w,z in zip(head_output,projection))
        _require(math.isfinite(partial) and nav.range_count>observed_before,
                 "Trained head output not executable")
        history_digest=state_digest(history)
        state_digest_value=state_digest(state)
        result={"schema":SCHEMA,"model":s["model"],"revision":s["revision"],
                "position":self.position,"head":self.head_id,
                "token_id":row_witness["token_id"],
                "embedding_sha256":row_witness["embedding_sha256"],
                "output_row0_selected_head_contribution":round(partial,12),
                "head_beta":round(beta[0],12),
                "head_log_decay":round(decay[0],12),
                "independent_recurrence_max_abs_error":error,
                "causal_convolution_history_digest":history_digest,
                "causal_recurrent_state_digest":state_digest_value,
                "new_weight_ranges":nav.range_count-observed_before,
                "range_chain":nav.chain,
                "scope":"one trained recurrent value head at true first-layer causal input; not complete layer"}
        _require(result["new_weight_ranges"]>0,
                 "Qwen head had no acquired trained bytes")
        result["observation_digest"]=_digest(SCHEMA,result)
        next_head=_digest(SCHEMA,{"previous":self.chain,
                        "observation_digest":result["observation_digest"]})
        # Full atomic publication only after all numerical checks and
        # qualified source ranges have succeeded.
        self.state=state[0];self.history=history
        self.position+=1;self.chain=next_head
        return result

    def checkpoint(self):
        from experiments.model_fabric.architecture_ops import state_digest
        payload={"schema":CHECKPOINT,"ground_digest":self.g["ground_digest"],
                 "head_id":self.head_id,"position":self.position,
                 "genesis":self.genesis,"chain":self.chain,
                 "history":self.history,"state":self.state,
                 "history_state_digest":state_digest(self.history) if self.history is not None else None,
                 "recurrent_state_digest":state_digest(self.state) if self.state is not None else None}
        return {**payload,"checkpoint_digest":_digest(CHECKPOINT,payload)}

    @classmethod
    def restore(cls,ground,checkpoint,*,expected_checkpoint_digest):
        from experiments.model_fabric.architecture_ops import state_digest
        _require(type(checkpoint) is dict and checkpoint.get("schema")==CHECKPOINT
                 and _hex64(expected_checkpoint_digest)
                 and checkpoint.get("checkpoint_digest")==expected_checkpoint_digest
                 and _digest(CHECKPOINT,{k:v for k,v in checkpoint.items()
                          if k!="checkpoint_digest"})==expected_checkpoint_digest
                 and checkpoint.get("ground_digest")==ground["ground_digest"]
                 and type(checkpoint.get("position")) is int
                 and 0<=checkpoint["position"]<=MAX_STEPS,
                 "Missing or changed external Qwen causal checkpoint")
        s=cls(ground,head=checkpoint["head_id"])
        _require(checkpoint["genesis"]==s.genesis
                 and _hex64(checkpoint.get("chain")),
                 "Wrong Qwen causal checkpoint genesis/head")
        history=checkpoint["history"];state=checkpoint["state"]
        if checkpoint["position"]>0:
            _require(type(history) is list and 1<=len(history)<=3
                     and all(type(x) is list and len(x)==384 for x in history)
                     and type(state) is list and len(state)==128
                     and all(type(x) is list and len(x)==128 for x in state)
                     and state_digest(history)==checkpoint["history_state_digest"]
                     and state_digest(state)==checkpoint["recurrent_state_digest"],
                     "Corrupt Qwen trained recurrent/convolution state")
        else:
            _require(history is None and state is None,"Uncommitted trained state")
        s.position=checkpoint["position"]
        s.history=deepcopy(history);s.state=deepcopy(state)
        s.chain=checkpoint["chain"]
        return s
