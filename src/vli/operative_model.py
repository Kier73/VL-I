"""INTERACT-002: construct observable results from *executed* trained tensor laws.

The Ground is an addressable programme before the result exists. Two native
Qwen/DeepSeek relations use qualified source ranges and perform trained math
from caller-selected analytic hidden inputs. The result can then become a
renderable state through the existing E-CPU/VLD response DAG.

These are real pretrained *partial operations*, not full language inference,
checkpoint reconstruction, model-metadata prediction or trusted remote math.
"""
from __future__ import annotations

import math
from hashlib import sha256
from copy import deepcopy
from typing import Any

from .ground import _digest, _require, _hex64
from .large_model_comparison import SOURCES

SCHEMA = "vl-i.operative-trained-ground.v1"
INTENT = "vl-i.operative-trained-intention.v1"
POSITION = "vl-i.operative-trained-position.v1"
RESULT = "vl-i.operative-trained-observation.v1"
EVENT = "vl-i.operative-trained-event.v1"
CHECKPOINT = "vl-i.operative-trained-checkpoint.v1"
VIEWS = frozenset(("text", "page", "code", "json", "tokens"))
MAX_EVENTS = 24


def _canonical(v: Any) -> bytes:
    from .ground import _canonical
    return _canonical(v)


def ground(route: str, index: dict):
    """Bind an already admitted full-sized original checkpoint index."""
    _require(route in SOURCES and isinstance(index, dict),
             "Unknown trained operator/model index")
    s = SOURCES[route]
    _require(index.get("model") == s["model"]
             and index.get("revision") == s["revision"]
             and index.get("index_sha256") == s["index_sha256"]
             and index.get("ground_digest") == s["index_ground_digest"]
             and index.get("tensor_count") == s["tensors"]
             and index.get("shard_count") == s["shards"]
             and index.get("advertised_payload_bytes") == s["advertised_bytes"],
             "Source is not the exact observed full-scale trained model")
    operator = ("layer0_native_delta_gates_48"
                if route == "qwen-27b" else
                "layer0_native_router_384_and_six_fp4_expert_fragments")
    rec = {
        "schema": SCHEMA, "route": route, "model": s["model"],
        "revision": s["revision"], "index_sha256": s["index_sha256"],
        "index_ground_digest": s["index_ground_digest"],
        "model_advertised_bytes": s["advertised_bytes"],
        "model_layers": s["layers"], "operator": operator,
        "source_scope": "pinned_checkpoint_not_provider_chat_alias",
        "operation_scope": "partial_trained_numerical_subgraph_not_full_inference",
    }
    rec["ground_digest"] = _digest(SCHEMA, rec)
    return rec


def _check_ground(g):
    _require(isinstance(g, dict) and g.get("schema") == SCHEMA
             and isinstance(g.get("route"), str)
             and g.get("route") in SOURCES
             and g == {**{k:v for k,v in ground(g["route"], {
                  "model": SOURCES[g["route"]]["model"],
                  "revision": SOURCES[g["route"]]["revision"],
                  "index_sha256": SOURCES[g["route"]]["index_sha256"],
                  "ground_digest": SOURCES[g["route"]]["index_ground_digest"],
                  "tensor_count": SOURCES[g["route"]]["tensors"],
                  "shard_count": SOURCES[g["route"]]["shards"],
                  "advertised_payload_bytes": SOURCES[g["route"]]["advertised_bytes"],
             }).items()}},
             "Trained operative Ground changed")
    return g


def intention(g, *, waveform="sin", frequency=0.017, amplitude=0.125,
              phase=0.0, views=("text","page","code","json","tokens")):
    """Observer changes the hidden-state input *before* trained evaluation.

    A compact procedural input recipe, not an array of 5,120 stored numbers.
    This is not a tokenizer: selected analytic vectors are not real prompt
    embeddings and cannot substantiate paragraph-generation claims.
    """
    _check_ground(g)
    _require(waveform in ("sin", "cos")
             and type(frequency) in (float,int) and math.isfinite(frequency)
             and 0 < frequency <= 1
             and type(amplitude) in (float,int) and math.isfinite(amplitude)
             and 0 < amplitude <= 1
             and type(phase) in (float,int) and math.isfinite(phase)
             and abs(phase) <= 6.283185307179586
             and isinstance(views, (list,tuple)) and 1 <= len(views) <= len(VIEWS)
             and len(set(views)) == len(views) and set(views).issubset(VIEWS),
             "Invalid trained numerical intention")
    body={"schema": INTENT, "ground_digest":g["ground_digest"],
          "waveform": waveform, "frequency":float(frequency),
          "amplitude":float(amplitude), "phase":float(phase),
          "views":list(views),
          "scope":"observer-selected_synthetic_hidden_not_tokenized_prompt"}
    body["intent_digest"] = _digest(INTENT,body)
    return body


def _check_intention(g, spec):
    _require(isinstance(spec,dict) and spec.get("schema")==INTENT
             and _hex64(spec.get("intent_digest"))
             and spec["intent_digest"] == _digest(INTENT,
                {k:v for k,v in spec.items() if k!="intent_digest"}),
             "Trained operation intention changed")
    _require(spec == intention(g,waveform=spec["waveform"],
           frequency=spec["frequency"],amplitude=spec["amplitude"],
           phase=spec["phase"],views=spec["views"]),
           "Trained intention does not re-admit")


def position(g, spec, *, previous_head):
    _check_ground(g);_check_intention(g,spec)
    _require(_hex64(previous_head),"No externally qualified causal predecessor")
    body={"schema":POSITION,"ground_digest":g["ground_digest"],
          "intent_digest":spec["intent_digest"],"previous_head":previous_head,
          "operator":g["operator"],
          "scope":"coordinate_exists_before_trained_arithmetic_and_output_rendering"}
    d=_digest(POSITION,body)
    return {**body,"position_digest":d,"coordinate":"state://vli/"+d}


def _input(spec, width):
    _require(type(width) is int and 1 <= width <= 32768,
             "Trained layer hidden width not admitted")
    wave = math.sin if spec["waveform"] == "sin" else math.cos
    return [spec["amplitude"]*wave((j+1)*spec["frequency"]+spec["phase"])
            for j in range(width)]


def _sigmoid(value):
    if value >= 0:
        return 1.0/(1.0+math.exp(-value))
    e=math.exp(value)
    return e/(1.0+e)


def _qwen(nav,spec):
    from experiments.model_fabric.qualified_subgraphs import observed_rows, observed_values
    from experiments.model_fabric.trained_subgraphs import centered_rms_norm
    from experiments.model_fabric.architecture_ops import qwen_gates
    p="model.language_model.layers.0."
    names=[p+k for k in (
        "input_layernorm.weight","linear_attn.in_proj_a.weight",
        "linear_attn.in_proj_b.weight","linear_attn.A_log","linear_attn.dt_bias")]
    _require(all(n in nav.weight_map for n in names),
             "Qwen trained gate tensor missing")
    n,an,bn,alog,dt=names
    a_desc=nav.locate(an); b_desc=nav.locate(bn)
    _require(a_desc["shape"]==[48,5120] and b_desc["shape"]==[48,5120],
             "Original 48-head Qwen gate topology differs")
    x=_input(spec,5120)
    normalized=centered_rms_norm(x,observed_values(nav,n))
    a=observed_rows(nav,an,normalized,first=0,count=48)
    b=observed_rows(nav,bn,normalized,first=0,count=48)
    logs=observed_values(nav,alog); dts=observed_values(nav,dt)
    _require(len(logs)==len(dts)==48,"Trained gate parameter geometry differs")
    beta,decay=qwen_gates(a,b,logs,dts)
    # Independent expression from trained projected scalar inputs (no
    # qwen_gates call). This checks the native activation relation.
    b_reference=[_sigmoid(z) for z in b]
    d_reference=[-math.exp(k)*(max(t+u,0.)+
                 math.log1p(math.exp(-abs(t+u))))
                 for t,u,k in zip(a,dts,logs)]
    error=max(abs(x-y) for arr,ref in ((beta,b_reference),(decay,d_reference))
              for x,y in zip(arr,ref))
    _require(math.isfinite(error) and error <= 2e-8,
             "Qwen trained gate algebra mismatches independent realization")
    return {"kind":"qwen_trained_delta_gate_vector",
            "gated_delta_heads":48,
            "beta":[round(v,12) for v in beta],
            "log_decay":[round(v,12) for v in decay],
            "mean_beta":round(math.fsum(beta)/48,12),
            "reference_max_abs_error":error,
            "scope":"48 real trained Qwen layer-0 gate scalars; no recurrent QKV or full decoder"}


def _deepseek(nav,spec):
    from experiments.model_fabric.qualified_subgraphs import (
        observed_rows,observed_values,_mx_fp4_row)
    from experiments.model_fabric.architecture_ops import deepseek_sqrtsoftplus_route
    p="layers.0.ffn.gate."
    wn=p+"weight"; bn=p+"bias"
    _require(wn in nav.weight_map and bn in nav.weight_map,
             "Original DeepSeek trained routing tensors absent")
    desc=nav.locate(wn)
    _require(len(desc["shape"])==2 and desc["shape"][0]==384
             and 1 <= desc["shape"][1] <= 32768,
             "Original DeepSeek expert-router geometry differs")
    x=_input(spec,desc["shape"][1])
    bias=observed_values(nav,bn)
    _require(len(bias)==384,"Incomplete trained router bias")
    logits=observed_rows(nav,wn,x,count=384)
    route=deepseek_sqrtsoftplus_route(logits,bias,topk=6)
    selected=route["selected_experts"]
    _require(len(selected)==len(set(selected))==6
             and all(type(i) is int and 0<=i<384 for i in selected),
             "Trained router did not select six unique experts")
    # Independent top-six selector and normalized scores.
    scores=[math.sqrt(max(y,0)+math.log1p(math.exp(-abs(y))))
            for y in logits]
    ids=sorted(range(384),key=lambda j:(-(scores[j]+bias[j]),j))[:6]
    denom=math.fsum(scores[i] for i in ids)
    reference=[1.5*scores[i]/denom for i in ids]
    _require(ids==selected and max(abs(x-y) for x,y in
             zip(reference,route["routing_weights"]))<=2e-8,
             "Independent trained routing relation differs")
    entries=[];errors=[]
    for eid,weight in zip(selected,route["routing_weights"]):
        prefix=f"layers.0.ffn.experts.{eid}."
        gate_name=prefix+"w1.weight"
        up_name=prefix+"w3.weight"
        down_name=prefix+"w2.weight"
        d=nav.locate(gate_name)
        packed=d["dtype"]=="I8"
        width=d["shape"][1]*(2 if packed else 1)
        _require(d["shape"][0] > 0 and width >= 32
                 and width%32==0 and width<=32768,
                 "Selected FP4 expert is not shape-admitted")
        eh=_input(spec,width)
        gate,eg=_mx_fp4_row(nav,gate_name,eh)
        up,eu=_mx_fp4_row(nav,up_name,eh)
        activation=gate*_sigmoid(gate)*up
        down_desc=nav.locate(down_name)
        down_cols=down_desc["shape"][1]*(2 if down_desc["dtype"]=="I8" else 1)
        _require(down_cols>0 and down_cols<=65536,
                 "Selected downstream FP4 extent differs")
        partial,ep=_mx_fp4_row(nav,down_name,[activation]+[0.0]*(down_cols-1))
        entries.append({"expert":eid,"routing_weight":round(weight,12),
                        "gate0":round(gate,12),"up0":round(up,12),
                        "down0_channel0_only":round(partial,12)})
        errors.extend((eg,eu,ep))
    contribution=math.fsum(e["routing_weight"]*e["down0_channel0_only"]
                           for e in entries)
    _require(all(math.isfinite(z) for z in (contribution,*errors)),
             "Nonfinite trained selected-expert relation")
    return {"kind":"deepseek_selected_expert_partial_contribution",
            "router_expert_count":384,"selected_experts":selected,
            "expert_slices":entries,
            "weighted_partial_output_channel0":round(contribution,12),
            "reference_max_abs_error":max(errors),
            "scope":"trained router + six selected first-neuron FP4 slices; not full MoE output"}


def _compose(result):
    if result["kind"]=="qwen_trained_delta_gate_vector":
        return {"heads":len(result["beta"]),
                "mean_beta":round(math.fsum(result["beta"])/len(result["beta"]),12),
                "beta_above_half":sum(z>0.5 for z in result["beta"])}
    return {"selected":list(result["selected_experts"]),
            "weighted_partial":round(math.fsum(
               e["routing_weight"]*e["down0_channel0_only"]
               for e in result["expert_slices"]),12)}


def _program():
    """A standalone pure construction law; no observed output constants copied."""
    return '''# VL-I: generated operation over *qualified trained numerical observations*.
# This recomposes observed model outputs; it does not run the whole model.
from math import fsum
def construct(observation):
    if observation["kind"] == "qwen_trained_delta_gate_vector":
        gates = observation["beta"]
        return {"heads":len(gates),
                "mean_beta":round(fsum(gates)/len(gates),12),
                "beta_above_half":sum(z>0.5 for z in gates)}
    if observation["kind"] == "deepseek_selected_expert_partial_contribution":
        entries = observation["expert_slices"]
        return {"selected":list(observation["selected_experts"]),
                "weighted_partial":round(fsum(
                    e["routing_weight"]*e["down0_channel0_only"]
                    for e in entries),12)}
    raise ValueError("Unknown trained operation")
'''


def _views(record,spec):
    observation=record["numerical_result"]
    derived=_compose(observation)
    page=("# Operative trained Ground\n\n"
          + "Source: \`"+record["model"]+"\`\n\n"
          + "Position: \`"+record["coordinate"]+"\`\n\n"
          + "## Computed trained relation\n\n"
          + "\`\`\`json\n"+_canonical(derived).decode("utf-8")+"\n\`\`\`\n\n"
          + "> Partial pretrained numerical operation; not full-language inference.\n")
    text=("Executed "+record["model"]+" trained "+observation["kind"]+
          " at "+record["coordinate"]+"; result: "+
          _canonical(derived).decode("utf-8")+
          ". No complete pretrained decoder output is claimed.")
    payload={"text":text,"page":page,"code":_program(),
             "json":_canonical({"trained_observation":observation,
                  "constructed":derived}).decode("utf-8"),
             "tokens":list(text.encode("utf-8"))}
    return {kind:payload[kind] for kind in spec["views"]}


class TrainedOperationSession:
    """Causal, observer-directed interaction with the trained Ground.

    The interpreter retains only Ground/intent/result receipts. Actual trained
    weights remain at the externally qualified range host. A restored session
    validates history integrity, not remote mathematical equivalence.
    """
    def __init__(self,g,*,observer="vli-operator"):
        _check_ground(g)
        _require(type(observer) is str and 1<=len(observer)<=80,
                 "Invalid trained operator observer")
        self.g=deepcopy(g);self.observer=observer
        self.genesis=_digest(CHECKPOINT,
                     {"ground_digest":g["ground_digest"],"observer":observer})
        self.head=self.genesis;self.events=[]

    def operate(self,nav,spec,*,expected_head,render=False):
        from experiments.model_fabric.scaled_model_fabric import QualifiedShardNavigator
        _require(expected_head==self.head and len(self.events)<MAX_EVENTS,
                 "Stale trained operator causal head")
        _check_intention(self.g,spec)
        _require(isinstance(nav,QualifiedShardNavigator)
                 and nav.index["model"]==self.g["model"]
                 and nav.index["revision"]==self.g["revision"]
                 and nav.index["index_sha256"]==self.g["index_sha256"]
                 and nav.index["ground_digest"]==self.g["index_ground_digest"],
                 "Mismatched pretrained observer-host source")
        p=position(self.g,spec,previous_head=self.head)
        earlier_ranges=nav.range_count; earlier_bytes=nav.weight_bytes
        result=_qwen(nav,spec) if self.g["route"]=="qwen-27b" else _deepseek(nav,spec)
        _require(nav.range_count>earlier_ranges and nav.weight_bytes>earlier_bytes
                 and _hex64(nav.chain),
                 "No qualified pretrained tensor weights executed")
        # Construct numerical and renderable output using the trained result,
        # never a table of model names or an assumed language-model answer.
        record={"schema":RESULT, "coordinate":p["coordinate"],
                "position_digest":p["position_digest"],
                "model":self.g["model"],"revision":self.g["revision"],
                "ground_digest":self.g["ground_digest"],
                "intent_digest":spec["intent_digest"],
                "numerical_result":result,
                "new_weight_bytes":nav.weight_bytes-earlier_bytes,
                "new_qualified_ranges":nav.range_count-earlier_ranges,
                "range_chain_head":nav.chain,
                "scope":"executed_pretrained_partial_operation_from_qualified_bytes"}
        record["observation_digest"]=_digest(RESULT,record)
        views=_views(record,spec)
        frame=None
        if render:
            from .response_state import (
                SourceSlot,ResponseSourceGround,ResponseFormation,
                compile_intention as render_intent)
            slots=tuple(SourceSlot(k,k,
                _canonical(views[k]) if k=="tokens" else
                views[k].encode("utf-8"))
                for k in spec["views"] if k in ("text","page","code","tokens"))
            _require(slots,"No requested DAG-renderable view")
            source=ResponseSourceGround(slots)
            intent=render_intent("Construct output from executed trained numerical Ground",
                  slots=tuple(x.name for x in slots),observer="vli-operator")
            form=ResponseFormation(source,intent)
            for _ in range(20):
                if form.dag.is_complete:break
                form.advance(max_nodes=16)
            _require(form.dag.is_complete,"Constructive Ground rendering incomplete")
            frame=form.observe()
        event={"schema":EVENT,"sequence":len(self.events),
               "previous":self.head,
               "ground_digest":self.g["ground_digest"],
               "intent":deepcopy(spec),
               "position_digest":p["position_digest"],
               "observation_digest":record["observation_digest"],
               "new_weight_bytes":record["new_weight_bytes"],
               "range_chain_head":nav.chain,
               "render_observation_digest":frame["observation_digest"] if frame else None,
               "scope":"trained_numeric_then_constructive_render"}
        event["event_digest"]=_digest(EVENT,event)
        self.head=event["event_digest"]
        self.events.append(event)
        return {"position":p,"trained":record,"constructed":_compose(result),
                "views":views,"rendered":frame,"event":deepcopy(event)}

    def checkpoint(self):
        obj={"schema":CHECKPOINT,"ground_digest":self.g["ground_digest"],
             "genesis":self.genesis,"observer":self.observer,
             "head":self.head,"events":deepcopy(self.events)}
        obj["checkpoint_digest"]=_digest(CHECKPOINT,obj)
        return obj

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
                 "Missing pinned or qualified trained-op checkpoint")
        session=cls(g,observer=checkpoint["observer"])
        _require(checkpoint.get("genesis")==session.genesis,
                 "Wrong trained-op causal genesis")
        for idx,event in enumerate(checkpoint["events"]):
            _require(type(event) is dict and
                     event.get("schema")==EVENT and
                     event.get("sequence")==idx and
                     event.get("previous")==session.head and
                     event.get("ground_digest")==g["ground_digest"]
                     and _hex64(event.get("range_chain_head"))
                     and type(event.get("new_weight_bytes")) is int
                     and event["new_weight_bytes"]>0
                     and _hex64(event.get("observation_digest"))
                     and event.get("event_digest")==_digest(EVENT,
                        {k:v for k,v in event.items() if k!="event_digest"}),
                     "Trained-op event chain has changed")
            _check_intention(g,event["intent"])
            expected_position=position(g,event["intent"],previous_head=session.head)
            _require(event["position_digest"]==expected_position["position_digest"],
                     "Causal trained-op intent position changed")
            session.events.append(deepcopy(event))
            session.head=event["event_digest"]
        _require(session.head==checkpoint.get("head"),
                 "Recovered trained-operation history differs")
        return session
