"""QUALIFIED-006: causal composition of original-width trained GDN layers.

This is a VL-I model-worker coordinator, not part of the VL-64 core.
Every layer receives the *actual* output of its predecessor, with a
bit-exact F64 transfer witness. Observation-history advances on reads;
the *semantic* token transition is committed only when all layers pass.

This adapter handles GDN layers only. It does not implement original Qwen's
full-attention layers, final norm, vocabulary head, or generated text.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import math
import struct

from experiments.model_fabric.complete_native_layers import (
    QwenQualifiedLayer, LocalStatePages,
    export_qwen_causal_ground, restore_qwen_causal_ground,
)

from .ground import _digest, _hex64

SCHEMA = "vl-i.qwen-causal-gdn-chain.v1"
CHECKPOINT = "vl-i.qwen-causal-gdn-chain-checkpoint.v1"


def _f64_transfer_digest(values):
    """Commit to exactly the Python double values passed between layers."""
    if (not isinstance(values,(tuple,list)) or not values or
            any(type(x) not in (float,int) or not math.isfinite(x) for x in values)):
        raise ValueError("Unqualified nonfinite or incomplete causal hidden state")
    data=struct.pack("<"+str(len(values))+"d",*(float(x) for x in values))
    return sha256(b"vl-i.qwen-exact-f64-transfer.v1\0"+data).hexdigest()


class QwenCausalGDNChain:
    """Compose contiguous qualified complete GDN layers, with atomic token state.

    Existing QwenQualifiedLayer executors own all trained arithmetic and
    authenticated range observations. In particular, this coordinator must
    not substitute one independent token embedding at every layer.
    """
    def __init__(self, runners):
        if (not isinstance(runners,(list,tuple)) or len(runners)<2
                or any(not isinstance(r,QwenQualifiedLayer) for r in runners)):
            raise ValueError("At least two complete qualified GDN layers required")
        self.runners=list(runners)
        reference=self.runners[0]
        self.layers=[r.layer for r in self.runners]
        if (self.layers!=list(range(self.layers[0],self.layers[0]+len(self.layers)))
            or any(r.hidden!=reference.hidden or r.kh!=reference.kh
                   or r.vh!=reference.vh or r.kd!=reference.kd or r.vd!=reference.vd
                   or r.position!=0 or r.nav.index["model"]!=reference.nav.index["model"]
                   or r.nav.index["revision"]!=reference.nav.index["revision"]
                   or r.nav.index["ground_digest"]!=reference.nav.index["ground_digest"]
                   for r in self.runners)):
            raise ValueError("Changed model Ground, geometry, order, or causal genesis")
        self.model=reference.nav.index["model"]
        self.revision=reference.nav.index["revision"]
        self.index_ground_digest=reference.nav.index["ground_digest"]
        self.width=reference.hidden
        self.position=0
        self.last_output_digest=None
        self.genesis=_digest(SCHEMA,{"model":self.model,"revision":self.revision,
            "index_ground_digest":self.index_ground_digest,"layers":self.layers})
        self.head=self.genesis

    def step(self, hidden, *, token_id=None):
        """Run one genuine causal position through all consecutive trained layers.

        The caller must separately establish that the first input is an
        admitted original token embedding; this generic coordinator verifies
        the exact downstream transfer and commits no partial chain on error.
        """
        if (type(token_id) not in (int,type(None)) or
            (token_id is not None and not 0<=token_id<2**31) or
            not isinstance(hidden,(tuple,list)) or len(hidden)!=self.width):
            raise ValueError("Incomplete causal chain input or original token ID")
        first_digest=_f64_transfer_digest(hidden)
        if any(r.position!=self.position for r in self.runners):
            raise ValueError("Inconsistent trained-layer causal positions")
        # A downstream failure must not commit any upstream recurrent state.
        saved=[(r.position,deepcopy(r.convolution_history),
                deepcopy(r.recurrent_state)) for r in self.runners]
        old_head,old_output=self.head,self.last_output_digest
        current=list(hidden)
        witnesses=[]
        try:
            for runner in self.runners:
                before=_f64_transfer_digest(current)
                output,report=runner.next(current)
                if (not isinstance(output,list) or len(output)!=self.width or
                    report.get("layer")!=runner.layer or
                    report.get("position")!=self.position or
                    report.get("complete_Qwen_transformer_layer") is not True or
                    report.get("complete_output_width")!=self.width or
                    report.get("model")!=self.model or
                    report.get("revision")!=self.revision or
                    runner.position!=self.position+1):
                    raise ValueError("Incomplete or contradictory qualified layer receipt")
                after=_f64_transfer_digest(output)
                witnesses.append({
                    "layer":runner.layer,"position":self.position,
                    "input_f64_digest":before,"output_f64_digest":after,
                    "complete_output_width":self.width,
                    "output_f32_digest":report["complete_layer_output_digest"],
                    "recurrent_state_digest":report["full_recurrent_state_digest"],
                    "convolution_state_digest":report["full_convolution_state_digest"],
                    "qualified_ranges":report["new_qualified_ranges"],
                    "trained_weight_bytes":report["new_weight_payload_bytes"],
                    "range_chain":report["range_chain"],
                })
                current=output  # The *same* numerical values, not a chosen vector.
            if (witnesses[0]["input_f64_digest"]!=first_digest or
                any(a["output_f64_digest"]!=b["input_f64_digest"]
                    for a,b in zip(witnesses,witnesses[1:]))):
                raise ValueError("Predecessor output does not bind successor input")
            body={"schema":SCHEMA,"model":self.model,"revision":self.revision,
                "index_ground_digest":self.index_ground_digest,
                "position":self.position,"original_token_id":token_id,
                "previous":old_head,"layer_witnesses":witnesses}
            next_head=_digest(SCHEMA,body)
            self.position+=1
            self.head=next_head
            self.last_output_digest=witnesses[-1]["output_f64_digest"]
            return current,{**body,"chain_head":next_head,
                            "all_layers_completed":True,
                            "original_decoder_completed":False,
                            "vocabulary_logits_computed":False}
        except Exception:
            for runner,(position,conv,delta) in zip(self.runners,saved):
                runner.position=position
                runner.convolution_history=conv
                runner.recurrent_state=delta
            self.head=old_head
            self.last_output_digest=old_output
            raise

    def checkpoint(self,store,*,input_sequence_digest):
        """Publish separate per-layer state pages and one externally pinned head."""
        if (not isinstance(store,LocalStatePages) or
            not _hex64(input_sequence_digest) or self.position<1 or
            not _hex64(self.head) or not _hex64(self.last_output_digest) or
            any(r.position!=self.position for r in self.runners)):
            raise ValueError("Incomplete qualified causal chain checkpoint")
        states=[export_qwen_causal_ground(
            r,store,input_sequence_digest=input_sequence_digest)
            for r in self.runners]
        body={"schema":CHECKPOINT,"model":self.model,"revision":self.revision,
              "index_ground_digest":self.index_ground_digest,
              "layers":self.layers,"position":self.position,
              "input_sequence_digest":input_sequence_digest,
              "genesis":self.genesis,"causal_head":self.head,
              "last_output_f64_digest":self.last_output_digest,
              "layer_checkpoints":states}
        return {**body,"checkpoint_digest":_digest(CHECKPOINT,body)}

    @classmethod
    def restore(cls,runners,store,manifest,*,expected_digest,
                expected_input_sequence_digest):
        """Rehome into fresh layer executors only with an external manifest pin."""
        if (not isinstance(manifest,dict) or not _hex64(expected_digest) or
            not _hex64(expected_input_sequence_digest) or
            manifest.get("checkpoint_digest")!=expected_digest or
            _digest(CHECKPOINT,{k:v for k,v in manifest.items()
                if k!="checkpoint_digest"})!=expected_digest):
            raise ValueError("Unpinned or altered causal layer-chain checkpoint")
        chain=cls(runners)
        if (set(manifest)!={"schema","model","revision","index_ground_digest",
             "layers","position","input_sequence_digest","genesis",
             "causal_head","last_output_f64_digest","layer_checkpoints",
             "checkpoint_digest"} or manifest["schema"]!=CHECKPOINT or
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
            len(manifest["layer_checkpoints"])!=len(chain.runners)):
            raise ValueError("Wrong trained layer-chain state, model, or lineage")
        for runner,part in zip(chain.runners,manifest["layer_checkpoints"]):
            restored=restore_qwen_causal_ground(
                runner,store,part,expected_digest=part["checkpoint_digest"],
                expected_input_sequence_digest=expected_input_sequence_digest)
            if restored!=manifest["position"]:
                raise ValueError("Causal layer-chain recovery position mismatch")
        chain.position=manifest["position"]
        chain.head=manifest["causal_head"]
        chain.last_output_digest=manifest["last_output_f64_digest"]
        return chain
