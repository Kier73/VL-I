"""INTERACT-003: observed pretrained token *embedding* -> trained relation -> result.

No synthetic hidden vector is supplied to this Ground. A pinned original-size
checkpoint provides the complete embedding row for an explicit model token ID.
The Ground then executes the native Qwen layer-0 gate relation or a DeepSeek
embedding-fed routing/expert probe. The DeepSeek router's true forward input
requires attention/mHC transitions; do not mistake this probe for a valid
DeepSeek prefill/decoder output. No tokenizer claim is made for arbitrary IDs.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import math
import struct

from .ground import _digest, _hex64, _require
from .operative_model import (
    _check_ground, _qwen, _deepseek, _compose, _views,
    _canonical, ground as trained_ground,
)
from .large_model_comparison import SOURCES

INTENT = "vl-i.trained-token-embedding-intention.v1"
POSITION = "vl-i.trained-token-embedding-position.v1"
EVENT = "vl-i.trained-token-embedding-event.v1"
CHECKPOINT = "vl-i.trained-token-embedding-checkpoint.v1"
MAX_EVENTS = 32
EMBEDDING_TENSORS = {
    "qwen-27b": "model.language_model.embed_tokens.weight",
    "deepseek-v41-flash": "embed.weight",
}
# Embedding-to-gate at Qwen layer 0 is an authentic first-layer input relation.
# DeepSeek's MoE is downstream of attention/mHC; its embedding-fed routing
# is a controlled trained probe, not the official model's causal hidden state.
SCOPE = {
    "qwen-27b": "genuine_initial_token_embedding_to_qwen_layer0_gate_partial",
    "deepseek-v41-flash": "trained_embedding_fed_router_probe_not_true_moe_causal_input",
}


def token_intention(g, token_id: int, *,
                    views=("text", "page", "code", "json", "tokens")):
    """Address a token ID without claiming that natural language was tokenized."""
    _check_ground(g)
    _require(type(token_id) is int and 0 <= token_id <= 2**31-1
             and type(views) in (tuple, list) and 1 <= len(views) <= 5
             and len(set(views)) == len(views)
             and set(views).issubset({"text","page","code","json","tokens"}),
             "Unqualified explicit trained token ID or output views")
    body = {"schema": INTENT, "ground_digest":g["ground_digest"],
            "token_id":token_id, "views":list(views),
            "embedding_tensor":EMBEDDING_TENSORS[g["route"]],
            "input_scope":"explicit_token_id_not_verified_text_tokenization"}
    return {**body, "intent_digest":_digest(INTENT,body)}


def _admit_intent(g, intent):
    _require(type(intent) is dict and intent.get("schema")==INTENT and
             _hex64(intent.get("intent_digest")) and
             intent["intent_digest"]==_digest(INTENT,
                 {k:v for k,v in intent.items() if k!="intent_digest"}),
             "Trained token intention altered")
    _require(intent==token_intention(g,intent.get("token_id"),
                     views=intent.get("views")),
             "Trained token intention is not re-admitted")


def token_position(g,intent,*,previous_head):
    _admit_intent(g,intent)
    _require(_hex64(previous_head),"Missing trusted token causal predecessor")
    body={"schema":POSITION,"ground_digest":g["ground_digest"],
          "intent_digest":intent["intent_digest"],"previous_head":previous_head,
          "embedding_tensor":intent["embedding_tensor"],
          "operation":g["operator"],
          "scope":"state_address_before_token_row_or_trained_operator_observed"}
    root=_digest(POSITION,body)
    return {**body,"coordinate":"state://vli/"+root,"position_digest":root}


def _decode_and_independently_check(raw, dtype):
    """Different BF16/F16/F32 paths, with independent numeric bit decoding."""
    from experiments.model_fabric.qualified_subgraphs import _decode_float
    values=_decode_float(raw,dtype)
    if dtype=="BF16":
        _require(len(raw)%2==0,"Misaligned BF16 token row")
        # A separate exponent+mantissa bit conversion (not the native decoder).
        check=[struct.unpack("<f",struct.pack("<I",int(word)<<16))[0]
               for (word,) in struct.iter_unpack("<H",raw)]
    elif dtype=="F16":
        _require(len(raw)%2==0,"Misaligned F16 token row")
        check=[float(v[0]) for v in struct.iter_unpack("<e",raw)]
    elif dtype=="F32":
        _require(len(raw)%4==0,"Misaligned F32 token row")
        check=[float(v[0]) for v in struct.iter_unpack("<f",raw)]
    else:
        raise ValueError("Token embedding quantization has no qualified decoder")
    _require(len(check)==len(values) and
             all(math.isfinite(v) and math.isfinite(r) and abs(v-r)<=1e-7
                 for v,r in zip(values,check)),
             "Trained token row failed independent numeric decoding")
    return values


def observe_token_row(nav,g,intent):
    """Read exactly ONE trained embedding row through admitted 64 KiB ranges."""
    _admit_intent(g,intent)
    name=intent["embedding_tensor"]
    _require(name in nav.weight_map, "Original trained embedding tensor is absent")
    d=nav.locate(name)
    _require(type(d.get("shape")) is list and len(d["shape"])==2 and
             all(type(v) is int and v>0 for v in d["shape"]) and
             d.get("dtype") in ("BF16","F16","F32"),
             "Native trained embedding shape/dtype not admitted")
    vocab,width=d["shape"]; token_id=intent["token_id"]
    _require(token_id<vocab and width<=32768 and width>=128,
             "Token ID or trained embedding width exceeds original checkpoint")
    if g["route"]=="qwen-27b":
        _require(width==5120,"Qwen initial embedding hidden width differs")
    else:
        desc=nav.locate("layers.0.ffn.gate.weight")
        _require(type(desc.get("shape")) is list and
                 len(desc["shape"])==2 and desc["shape"]==[384,width],
                 "DeepSeek learned embedding and selected-router widths differ")
    before=nav.range_count;bytes_before=nav.weight_bytes
    raw,raw_desc=nav.get_raw(name,flat_start=token_id*width,count=width)
    _require(raw_desc["shape"]==d["shape"] and raw_desc["dtype"]==d["dtype"],
             "Embedding row provenance changed during acquisition")
    vector=_decode_and_independently_check(raw,d["dtype"])
    _require(len(vector)==width and nav.range_count>before and
             nav.weight_bytes-bytes_before==len(raw) and _hex64(nav.chain),
             "No complete qualified trained token embedding row acquired")
    witness={
        "tensor":name,"shape":d["shape"],"dtype":d["dtype"],
        "token_id":token_id,"embedding_bytes":len(raw),
        "embedding_sha256":sha256(raw).hexdigest(),
        "range_chain_head":nav.chain,"new_ranges":nav.range_count-before,
        "scope":"actual_original_checkpoint_token_embedding_row",
    }
    return vector,witness


class TokenEmbeddingSession:
    """Causal results, not an LLM dialogue: real token-row then trained math."""
    def __init__(self,g,*,observer="vli-token-observer"):
        _check_ground(g)
        _require(type(observer) is str and 1<=len(observer)<=80,
                 "Missing qualified interpreter observer")
        self.g=deepcopy(g);self.observer=observer
        self.genesis=_digest(CHECKPOINT,{"ground":g["ground_digest"],
                                         "observer":observer})
        self.head=self.genesis;self.events=[]

    def operate(self,nav,intent,*,expected_head,render=False):
        from experiments.model_fabric.scaled_model_fabric import QualifiedShardNavigator
        _require(expected_head==self.head and len(self.events)<MAX_EVENTS,
                 "Stale token-position causal state")
        _admit_intent(self.g,intent)
        _require(isinstance(nav,QualifiedShardNavigator)
                 and nav.index.get("model")==self.g["model"]
                 and nav.index.get("revision")==self.g["revision"]
                 and nav.index.get("index_sha256")==self.g["index_sha256"]
                 and nav.index.get("ground_digest")==self.g["index_ground_digest"],
                 "Changed actual model checkpoint or qualified observer")
        pos=token_position(self.g,intent,previous_head=self.head)
        start_bytes=nav.weight_bytes;start_ranges=nav.range_count
        vector,embedding=observe_token_row(nav,self.g,intent)
        result=(_qwen(nav,None,hidden=vector) if self.g["route"]=="qwen-27b"
                else _deepseek(nav,None,hidden=vector))
        _require(nav.weight_bytes>start_bytes+embedding["embedding_bytes"]
                 and nav.range_count>start_ranges+embedding["new_ranges"],
                 "Trained operator did not read actual pretrained parameters")
        result["input_scope"]=SCOPE[self.g["route"]]
        record={"schema":"vl-i.real-token-trained-result.v1",
                "model":self.g["model"],"revision":self.g["revision"],
                "coordinate":pos["coordinate"],
                "position_digest":pos["position_digest"],
                "ground_digest":self.g["ground_digest"],
                "intent_digest":intent["intent_digest"],
                "embedding_witness":embedding,
                "numerical_result":result,
                "new_weight_bytes":nav.weight_bytes-start_bytes,
                "new_ranges":nav.range_count-start_ranges,
                "range_chain_head":nav.chain,
                "pretrained_full_decoder_completed":False,
                "native_tokenizer_applied":False,
                "causal_scope":SCOPE[self.g["route"]]}
        record["observation_digest"]=_digest(record["schema"],record)
        view=_views(record,intent)
        rendered=None
        if render:
            from .response_state import (
                SourceSlot,ResponseSourceGround,ResponseFormation,
                compile_intention as render_intention,
            )
            names=tuple(k for k in intent["views"]
                        if k in ("text","page","code","tokens"))
            _require(names,"No qualified rendered output types")
            slots=tuple(SourceSlot(k,k,_canonical(view[k]) if k=="tokens"
                         else view[k].encode("utf-8")) for k in names)
            source=ResponseSourceGround(slots)
            typed=render_intention("Render trained token-embedding operation",
                    slots=names,observer=self.observer)
            frame=ResponseFormation(source,typed)
            for _ in range(20):
                if frame.dag.is_complete:break
                frame.advance(max_nodes=16)
            _require(frame.dag.is_complete,"Token-result renderer incomplete")
            rendered=frame.observe()
        event={"schema":EVENT,"sequence":len(self.events),"previous":self.head,
               "ground_digest":self.g["ground_digest"],"intent":deepcopy(intent),
               "position_digest":pos["position_digest"],
               "embedding_sha256":embedding["embedding_sha256"],
               "observation_digest":record["observation_digest"],
               "new_weight_bytes":record["new_weight_bytes"],
               "range_chain_head":nav.chain,
               "render_digest":rendered["observation_digest"] if rendered else None}
        event["event_digest"]=_digest(EVENT,event)
        self.head=event["event_digest"];self.events.append(event)
        return {"coordinate":pos["coordinate"],"trained":record,
                "constructed":_compose(result),"views":view,
                "rendered":rendered,"event":deepcopy(event)}

    def checkpoint(self):
        body={"schema":CHECKPOINT,"ground_digest":self.g["ground_digest"],
              "observer":self.observer,"genesis":self.genesis,"head":self.head,
              "events":deepcopy(self.events)}
        return {**body,"checkpoint_digest":_digest(CHECKPOINT,body)}

    @classmethod
    def restore(cls,g,checkpoint,*,expected_head):
        _check_ground(g)
        _require(type(checkpoint) is dict and checkpoint.get("schema")==CHECKPOINT
                 and _hex64(expected_head)
                 and checkpoint.get("checkpoint_digest")==expected_head
                 and _digest(CHECKPOINT,{k:v for k,v in checkpoint.items()
                      if k!="checkpoint_digest"})==expected_head
                 and checkpoint.get("ground_digest")==g["ground_digest"]
                 and type(checkpoint.get("events")) is list
                 and len(checkpoint["events"])<=MAX_EVENTS,
                 "Wrong or modified externally pinned token checkpoint")
        s=cls(g,observer=checkpoint["observer"])
        _require(s.genesis==checkpoint.get("genesis"),
                 "Different token-embedding Ground genesis")
        for i,event in enumerate(checkpoint["events"]):
            _require(type(event) is dict
                     and event.get("schema")==EVENT
                     and event.get("sequence")==i
                     and event.get("previous")==s.head
                     and event.get("ground_digest")==g["ground_digest"]
                     and _hex64(event.get("embedding_sha256"))
                     and _hex64(event.get("observation_digest"))
                     and _hex64(event.get("range_chain_head"))
                     and type(event.get("new_weight_bytes")) is int
                     and event["new_weight_bytes"]>0
                     and event.get("event_digest")==_digest(EVENT,
                         {k:v for k,v in event.items() if k!="event_digest"}),
                     "Token-trained event hash/causal chain differs")
            _admit_intent(g,event["intent"])
            p=token_position(g,event["intent"],previous_head=s.head)
            _require(p["position_digest"]==event["position_digest"],
                     "Token-trained state-coordinate changed")
            s.events.append(deepcopy(event));s.head=event["event_digest"]
        _require(s.head==checkpoint.get("head"),
                 "Restored trained-token causal head differs")
        return s
