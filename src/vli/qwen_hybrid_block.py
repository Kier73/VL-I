"""QUALIFIED-007: one complete hybrid Qwen block (GDN/GDN/GDN/GQA).

Strict source-bound transition: every layer's exact f64 output becomes
the successor's input. Both GDN recurrent matrices and the causal
attention KV timeline are preserved in independently pinned pages.
This worker-side coordinator does not modify VL-64.
"""
from __future__ import annotations

from copy import deepcopy

from experiments.model_fabric.complete_native_layers import (
    QwenQualifiedLayer, LocalStatePages,
    export_qwen_causal_ground,restore_qwen_causal_ground,
)
from .ground import _digest,_hex64
from .qwen_layer_chain import _f64_transfer_digest
from .qwen_gated_attention import (
    QwenQualifiedGatedAttention,export_attention_kv,restore_attention_kv,
)

SCHEMA="vl-i.qwen-first-hybrid-block.v1"
CHECKPOINT="vl-i.qwen-first-hybrid-block-checkpoint.v1"


class QwenQualifiedHybridBlock:
    """Three GDN and one gated full-attention layers in original order."""

    def __init__(self,runners):
        if (not isinstance(runners,(list,tuple)) or len(runners)!=4 or
            any(not isinstance(r,QwenQualifiedLayer) for r in runners[:3]) or
            not isinstance(runners[3],QwenQualifiedGatedAttention)):
            raise ValueError("First hybrid block requires 3 complete GDN and one GQA layer")
        self.runners=list(runners)
        if [r.layer for r in self.runners]!=[0,1,2,3]:
            raise ValueError("Wrong Qwen original hybrid order")
        first=self.runners[0]
        if (any(r.position!=0 or r.hidden!=first.hidden or
                r.nav.index["model"]!=first.nav.index["model"] or
                r.nav.index["revision"]!=first.nav.index["revision"] or
                r.nav.index["ground_digest"]!=first.nav.index["ground_digest"]
                for r in self.runners) or
            any(r.kh!=first.kh or r.vh!=first.vh or
                r.kd!=first.kd or r.vd!=first.vd for r in self.runners[:3])):
            raise ValueError("Incompatible original model, width, layer or causal genesis")
        self.model=first.nav.index["model"]
        self.revision=first.nav.index["revision"]
        self.index_ground_digest=first.nav.index["ground_digest"]
        self.width=first.hidden
        self.position=0
        self.layers=[0,1,2,3]
        self.genesis=_digest(SCHEMA,{"model":self.model,"revision":self.revision,
            "index_ground_digest":self.index_ground_digest,"layers":self.layers})
        self.head=self.genesis
        self.last_output_digest=None

    def _snapshots(self):
        return [(r.position,deepcopy(r.convolution_history),
                 deepcopy(r.recurrent_state)) if isinstance(r,QwenQualifiedLayer)
                else (r.position,deepcopy(r.kv_keys),deepcopy(r.kv_values))
                for r in self.runners]

    def step(self,hidden,*,token_id=None):
        """Atomic semantics; source observations themselves are immutable."""
        if (not isinstance(hidden,(list,tuple)) or len(hidden)!=self.width or
            type(token_id) not in (int,type(None)) or
            (token_id is not None and not 0<=token_id<2**31) or
            any(r.position!=self.position for r in self.runners)):
            raise ValueError("Unqualified original hybrid input or causal position")
        first_digest=_f64_transfer_digest(hidden)
        before_state=self._snapshots()
        old_head,old_output=self.head,self.last_output_digest
        current=list(hidden);witnesses=[]
        try:
            for r in self.runners:
                entering=_f64_transfer_digest(current)
                out,receipt=r.next(current)
                if (not isinstance(out,list) or len(out)!=self.width or
                    receipt.get("model")!=self.model or
                    receipt.get("revision")!=self.revision or
                    receipt.get("layer")!=r.layer or
                    receipt.get("position")!=self.position or
                    receipt.get("complete_output_width")!=self.width or
                    receipt.get("complete_Qwen_transformer_layer") is not True or
                    r.position!=self.position+1):
                    raise ValueError("Incomplete original hybrid trained-layer receipt")
                exiting=_f64_transfer_digest(out)
                common={
                    "layer":r.layer,"position":self.position,
                    "kind":("GDN" if isinstance(r,QwenQualifiedLayer) else "GATED_GQA"),
                    "input_f64_digest":entering,"output_f64_digest":exiting,
                    "output_f32_digest":receipt["complete_layer_output_digest"],
                    "trained_weight_bytes":receipt["new_weight_payload_bytes"],
                    "qualified_ranges":receipt["new_qualified_ranges"],
                    "range_chain":receipt["range_chain"],
                    "causal_state_digest":(
                        receipt["full_recurrent_state_digest"]
                        if isinstance(r,QwenQualifiedLayer) else
                        receipt["full_attention_kv_state_digest"]),
                }
                if isinstance(r,QwenQualifiedLayer):
                    common["convolution_digest"]=receipt["full_convolution_state_digest"]
                else:
                    common["kv_history_positions"]=receipt["kv_history_positions"]
                witnesses.append(common)
                current=out
            if (witnesses[0]["input_f64_digest"]!=first_digest or
                any(a["output_f64_digest"]!=b["input_f64_digest"]
                    for a,b in zip(witnesses,witnesses[1:]))):
                raise ValueError("Causal predecessor output did not become successor input")
            body={
                "schema":SCHEMA,"model":self.model,"revision":self.revision,
                "index_ground_digest":self.index_ground_digest,
                "position":self.position,"original_token_id":token_id,
                "previous":old_head,"layer_witnesses":witnesses}
            semantic={**body,"layer_witnesses":[{
                k:v for k,v in w.items() if k not in (
                    "trained_weight_bytes","qualified_ranges","range_chain")}
                for w in witnesses]}
            next_head=_digest(SCHEMA,semantic)
            self.position+=1
            self.head=next_head
            self.last_output_digest=witnesses[-1]["output_f64_digest"]
            return current,{**body,"chain_head":next_head,
                "all_four_original_layers_completed":True,
                "original_decoder_completed":False,
                "vocabulary_logits_computed":False}
        except Exception:
            for r,(position,left,right) in zip(self.runners,before_state):
                r.position=position
                if isinstance(r,QwenQualifiedLayer):
                    r.convolution_history,r.recurrent_state=left,right
                else:
                    r.kv_keys,r.kv_values=left,right
            self.head,self.last_output_digest=old_head,old_output
            raise

    def checkpoint(self,store,*,input_sequence_digest):
        """Bind four independent causal state receipts to one external head."""
        if (not isinstance(store,LocalStatePages) or
            not _hex64(input_sequence_digest) or self.position<1 or
            not _hex64(self.head) or
            not _hex64(self.last_output_digest) or
            any(r.position!=self.position for r in self.runners)):
            raise ValueError("Unqualified first hybrid block checkpoint")
        parts=[export_qwen_causal_ground(r,store,
            input_sequence_digest=input_sequence_digest) for r in self.runners[:3]]
        parts.append(export_attention_kv(self.runners[3],store,
                    input_sequence_digest=input_sequence_digest))
        body={
            "schema":CHECKPOINT,"model":self.model,"revision":self.revision,
            "index_ground_digest":self.index_ground_digest,"layers":self.layers,
            "position":self.position,"genesis":self.genesis,
            "input_sequence_digest":input_sequence_digest,
            "causal_head":self.head,
            "last_output_f64_digest":self.last_output_digest,
            "layer_checkpoints":parts}
        return {**body,"checkpoint_digest":_digest(CHECKPOINT,body)}

    @classmethod
    def restore(cls,runners,store,manifest,*,expected_digest,
                expected_input_sequence_digest):
        if (not isinstance(manifest,dict) or not _hex64(expected_digest) or
            not _hex64(expected_input_sequence_digest) or
            manifest.get("checkpoint_digest")!=expected_digest or
            _digest(CHECKPOINT,{k:v for k,v in manifest.items()
                 if k!="checkpoint_digest"})!=expected_digest):
            raise ValueError("Modified or unpinned first hybrid block checkpoint")
        chain=cls(runners)
        expected={"schema","model","revision","index_ground_digest",
            "layers","position","genesis","input_sequence_digest",
            "causal_head","last_output_f64_digest",
            "layer_checkpoints","checkpoint_digest"}
        if (set(manifest)!=expected or manifest["schema"]!=CHECKPOINT or
            manifest["model"]!=chain.model or
            manifest["revision"]!=chain.revision or
            manifest["index_ground_digest"]!=chain.index_ground_digest or
            manifest["layers"]!=chain.layers or
            manifest["genesis"]!=chain.genesis or
            manifest["input_sequence_digest"]!=expected_input_sequence_digest or
            not _hex64(manifest["causal_head"]) or
            not _hex64(manifest["last_output_f64_digest"]) or
            type(manifest["position"]) is not int or manifest["position"]<1 or
            not isinstance(manifest["layer_checkpoints"],list) or
            len(manifest["layer_checkpoints"])!=4):
            raise ValueError("Wrong source, layer, lineage or causal head")
        for r,part in zip(chain.runners[:3],manifest["layer_checkpoints"][:3]):
            if restore_qwen_causal_ground(r,store,part,
                 expected_digest=part["checkpoint_digest"],
                 expected_input_sequence_digest=expected_input_sequence_digest
                 )!=manifest["position"]:
                raise ValueError("GDN recovery causal coordinate differed")
        if restore_attention_kv(chain.runners[3],store,
                manifest["layer_checkpoints"][3],
                expected_digest=manifest["layer_checkpoints"][3]["checkpoint_digest"],
                expected_input_sequence_digest=expected_input_sequence_digest
                )!=manifest["position"]:
            raise ValueError("Attention recovery causal coordinate differed")
        chain.position=manifest["position"]
        chain.head=manifest["causal_head"]
        chain.last_output_digest=manifest["last_output_f64_digest"]
        return chain
