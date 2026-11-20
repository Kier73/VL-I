"""SHAPE-006: exact, evidence-bound trained computation as a navigable state.

A pretrained operator's SOURCE BYTES were retained in SHAPE-005. Here the
OUTPUT of a complete, explicitly bounded trained Qwen RMS normalization is
also retained as an original E-CPU ComputationalObject. An identical,
source-pinned, intention-pinned observation skips all original trained
arithmetic AND qualified pretrained range requests; a distinct intention
computes a new output from the true original weights.

The original VL-64 centered_rms_norm computes all 5120 values on live Qwen.
This is a trained partial operator and NOT a decoder, token prediction or
whole-model speed comparison. Only a trusted original model index can confer
'original_pretrained' status; synthetic fixtures are visibly marked.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from time import perf_counter_ns
import inspect
import json
import math
import struct
import sys

from .ground import _digest, _hex64, _require
from .operative_model import _input
from .pretrained_range_memory import QualifiedPretrainedRangeMemory
from .large_model_comparison import SOURCES

SCHEMA="vl-i.verified-trained-result.v1"
INTENT="vl-i.verified-trained-result-intention.v1"
OPERATION="vl-i.original-qwen-centered-rmsnorm.v1"
CHECKPOINT="vl-i.verified-trained-result-checkpoint.v1"
EVENT="vl-i.verified-trained-result-event.v1"
POSITION="vl-i.verified-trained-result-position.v1"
TENSOR="model.language_model.layers.0.input_layernorm.weight"
MAX_RESULTS=32


def _encode(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),
                      ensure_ascii=False,allow_nan=False).encode("utf-8")


def _fingerprint():
    from experiments.model_fabric.trained_subgraphs import centered_rms_norm
    from experiments.model_fabric.qualified_subgraphs import _decode_float
    return _digest(OPERATION,{
        "native_vl64_math":inspect.getsource(centered_rms_norm),
        "native_vl64_BF16_reader":inspect.getsource(_decode_float),
        "vli_input_generator":inspect.getsource(_input),
        "precision":"python_binary64",
        "python_runtime":list(sys.version_info[:3]),
        "operator":OPERATION,
        "independent_algebra":"mean(square(x)); rsqrt(mean+eps); x*scale*(1+weight)",
        "eps":1e-6,
    })


def intention(*,prompt="Observe trained normalization",waveform="sin",
              frequency=.017,amplitude=.125,phase=0.,
              views=("text","code","tokens")):
    _require(type(prompt) is str and 1<=len(prompt)<=512
             and waveform in ("sin","cos")
             and type(frequency) in (float,int)
             and 0<frequency<=1 and math.isfinite(frequency)
             and type(amplitude) in (float,int)
             and 0<amplitude<=1 and math.isfinite(amplitude)
             and type(phase) in (float,int)
             and abs(phase)<=math.pi*2 and math.isfinite(phase)
             and type(views) in (tuple,list) and views
             and len(views)==len(set(views))
             and set(views).issubset({"text","code","tokens","json"}),
             "Invalid observer-selected trained input intention")
    body={"schema":INTENT,"prompt":prompt,"waveform":waveform,
          "frequency":float(frequency),"amplitude":float(amplitude),
          "phase":float(phase),"views":list(views),
          "input_scope":"intention_conditioned_analytic_hidden_not_tokenized_language"}
    body["intent_digest"]=_digest(INTENT,body)
    return body


def _admit_intention(spec):
    _require(type(spec) is dict and spec.get("schema")==INTENT
             and _hex64(spec.get("intent_digest"))
             and spec==intention(
                 prompt=spec["prompt"],waveform=spec["waveform"],
                 frequency=spec["frequency"],amplitude=spec["amplitude"],
                 phase=spec["phase"],views=spec["views"]),
             "Intent bytes/inputs modified after source qualification")


class VerifiedTrainedResultGround:
    """One source-pinned operator result bank, with causal positions per view.

    The E-CPU object bytes, source range evidence, and interpreter identity
    independently qualify a replay. A caller-provided expected source head
    is REQUIRED before every replay, including warm hits; this bank never
    upgrades its own hash to an external authentication authority.
    """

    def __init__(self,source:QualifiedPretrainedRangeMemory, *,
                 allow_synthetic_fixture=False,observer="vl-i-trained-observer"):
        from mailcpu.structural_memory_pages import StructuralMemoryPageBacking
        from mailcpu.computational_objects import ComputationalObjectStore
        _require(isinstance(source,QualifiedPretrainedRangeMemory)
                 and type(observer) is str and 1<=len(observer)<=80,
                 "Unknown original pretrained observer-host")
        original=SOURCES["qwen-27b"]
        exact=(source.model==original["model"]
               and source.revision==original["revision"]
               and source.index_sha256==original["index_sha256"]
               and source.index_ground_digest==original["index_ground_digest"])
        _require(exact or allow_synthetic_fixture,
                 "Source is not the original pinned full-scale Qwen checkpoint")
        _require(source.model==original["model"] and
                 source.revision==original["revision"],
                 "Unadmitted Qwen model or version")
        self.source=source
        self.original_pretrained=exact
        self.observer=observer
        self.implementation_fingerprint=_fingerprint()
        self.source_root=_digest(SCHEMA,{
            "model":source.model,"revision":source.revision,
            "index_ground":source.index_ground_digest,
            "index_sha256":source.index_sha256,
            "operator":OPERATION,"implementation":self.implementation_fingerprint,
            "source_claim":"original_pretrained" if exact else "synthetic_fixture",
        })
        self._backing=StructuralMemoryPageBacking()
        self._store=ComputationalObjectStore(
            self._backing,chunk_size=4096,max_object_bytes=65536)
        self.records={}
        self.head=_digest(CHECKPOINT,{
            "source_root":self.source_root,"observer":observer,
            "genesis":"independent_causal_observer_position"})
        self.events=[]
        self.trained_evaluations=0
        self.reused_trained_evaluations=0
        self.trained_math_ns=0
        self.cas_read_ns=0
        self.cas_store_ns=0

    def _admit_impl(self):
        _require(_fingerprint()==self.implementation_fingerprint,
                 "Native trained operator or input interpreter has changed")

    def _position(self,spec,head):
        body={"schema":POSITION,"source_root":self.source_root,
              "intention":spec["intent_digest"],
              "observer":self.observer,"previous_head":head,
              "operator":OPERATION}
        value=_digest(POSITION,body)
        return {"coordinate":"state://vli/"+value,
                "position_digest":value}

    def _compute(self,spec):
        from experiments.model_fabric.scaled_model_fabric import QualifiedShardNavigator
        from experiments.model_fabric.qualified_subgraphs import _decode_float
        from experiments.model_fabric.trained_subgraphs import centered_rms_norm
        start=perf_counter_ns()
        nav=QualifiedShardNavigator(
            model=self.source.model,revision=self.source.revision,
            index_raw=self._source_index_raw(),
            range_host=self.source,approved_hosts=self.source.approved_hosts)
        descriptor=nav.locate(TENSOR)
        _require(descriptor["shape"]==(
                     [5120] if self.original_pretrained else [64])
                 and descriptor["dtype"] in ("BF16","F32"),
                 "Unqualified actual trained normalization geometry")
        raw,typed=nav.get_raw(TENSOR)
        weights=_decode_float(raw,typed["dtype"])
        _require(len(weights)==len(descriptor["shape"]) or
                 len(weights)==descriptor["shape"][0],
                 "Original normalized weights incomplete")
        width=descriptor["shape"][0]
        x=_input(spec,width)
        input_hash=sha256(struct.pack("<"+"d"*width,*x)).hexdigest()
        numeric_start=perf_counter_ns()
        result=centered_rms_norm(x,weights)
        # The second computation does not call original VL-64 operator and
        # algebraically checks every output coordinate.
        factor=1.0/math.sqrt(math.fsum(v*v for v in x)/width+1e-6)
        independent=[z*factor*(1+w) for z,w in zip(x,weights)]
        error=max(abs(a-b) for a,b in zip(result,independent))
        _require(error<=2e-10 and all(math.isfinite(z) for z in result),
                 "Independent original trained normalization differs")
        vector=struct.pack("<"+"d"*width,*result)
        self.trained_math_ns+=perf_counter_ns()-numeric_start
        self.trained_evaluations+=1
        _require(nav.weight_bytes>=len(raw) and nav.range_count>0,
                 "No original trained bytes observed")
        source_memory=self.source.checkpoint()
        receipt={
            "source_root":self.source_root,
            "source_snapshot_digest":source_memory["checkpoint_digest"],
            "original_source":self.original_pretrained,
            "operator":OPERATION,
            "implementation_fingerprint":self.implementation_fingerprint,
            "intent_digest":spec["intent_digest"],
            "input_sha256":input_hash,
            "trained_weight_sha256":sha256(raw).hexdigest(),
            "native_range_chain_head":nav.chain,
            "native_range_count":nav.range_count,
            "native_weight_bytes":nav.weight_bytes,
            "output_sha256":sha256(vector).hexdigest(),
            "width":width,
            "max_independent_numeric_error":error,
            "verified_native_math":"VL64_centered_rms_norm",
        }
        receipt["receipt_digest"]=_digest(SCHEMA,receipt)
        _require(len(self.records)<MAX_RESULTS,
                 "Trained computation memory quota exceeded")
        t=perf_counter_ns()
        object_manifest=self._store.put_bytes(vector)
        _require(self._store.verify(object_manifest),
                 "ECPU trained-result object was not correctly stored")
        self.cas_store_ns+=perf_counter_ns()-t
        key=spec["intent_digest"]
        self.records[key]={
            "intent":deepcopy(spec),
            "receipt":receipt,
            "vector_manifest":object_manifest.to_dict(),
        }
        return vector,receipt,"NEW_TRAINED_ARITHMETIC",perf_counter_ns()-start

    def _source_index_raw(self):
        # Source index byte identity was admitted by its original observer host.
        # SHAPE-005 retains model fields; original bytes must be explicitly
        # supplied as a trusted source-snapshot member, not inferred from a hash.
        _require(hasattr(self,"_index_raw") and
                 sha256(self._index_raw).hexdigest()==self.source.index_sha256,
                 "Pinned model index bytes missing from trained interpreter")
        return self._index_raw

    def bind_source_index(self,index_raw:bytes):
        from experiments.model_fabric.scaled_model_fabric import admit_index
        _require(type(index_raw) is bytes and
                 sha256(index_raw).hexdigest()==self.source.index_sha256,
                 "Original source index byte identity not admitted")
        ref,_=admit_index(index_raw,model=self.source.model,
                          revision=self.source.revision)
        _require(ref["ground_digest"]==self.source.index_ground_digest,
                 "Trained source Ground index did not requalify")
        self._index_raw=bytes(index_raw)
        return self

    def _retrieve(self,spec):
        from mailcpu.computational_objects import ComputationalObjectManifest
        record=self.records[spec["intent_digest"]]
        _require(record["intent"]==spec
                 and record["receipt"]["source_root"]==self.source_root
                 and record["receipt"]["implementation_fingerprint"]==
                     self.implementation_fingerprint
                 and record["receipt"]["operator"]==OPERATION
                 and record["receipt"]["original_source"]==
                     self.original_pretrained
                 and record["receipt"]["intent_digest"]==spec["intent_digest"]
                 and _digest(SCHEMA,{
                     k:v for k,v in record["receipt"].items()
                     if k!="receipt_digest"})==record["receipt"]["receipt_digest"],
                 "Retained trained result lacks full input/source/operator proof")
        # A source-range checkpoint must still be admitted and unchanged.
        current=self.source.checkpoint()["checkpoint_digest"]
        _require(current==record["receipt"]["source_snapshot_digest"],
                 "Original pretrained source evidence changed under retained result")
        m=ComputationalObjectManifest.from_dict(record["vector_manifest"])
        t=perf_counter_ns()
        vector=self._store.read_bytes(m,max_bytes=65536)
        self.cas_read_ns+=perf_counter_ns()-t
        _require(len(vector)==record["receipt"]["width"]*8
                 and sha256(vector).hexdigest()==record["receipt"]["output_sha256"],
                 "Retained numerical result differs from independently verified source")
        self.reused_trained_evaluations+=1
        return vector,record["receipt"],"RETAINED_TRAINED_RESULT_NO_ARITHMETIC",0

    def observe(self,spec,*,expected_head):
        _admit_intention(spec);self._admit_impl()
        _require(expected_head==self.head and _hex64(expected_head)
                 and len(self.events)<80,
                 "Stale or unadmitted observer causality")
        position=self._position(spec,self.head)
        before=self.source.metrics()
        if spec["intent_digest"] in self.records:
            vector,receipt,disposition,math_time=self._retrieve(spec)
        else:
            vector,receipt,disposition,math_time=self._compute(spec)
        after=self.source.metrics()
        width=receipt["width"]
        values=struct.unpack("<"+"d"*width,vector)
        summary={
            "width":width,
            "output_sha256":receipt["output_sha256"],
            "mean_output":round(math.fsum(values)/width,12),
            "first_eight":[round(v,12) for v in values[:8]],
            "last_eight":[round(v,12) for v in values[-8:]],
            "independent_numeric_error":receipt["max_independent_numeric_error"],
        }
        base={"source_root":self.source_root,
              "source_model":self.source.model,
              "position":position["coordinate"],
              "intention":spec["prompt"],
              "computed":summary,
              "disposition":disposition,
              "original_pretrained":self.original_pretrained,
              "scope":"actual_trained_norm_result_not_language_model_decoding"}
        pretty=json.dumps(base,sort_keys=True,ensure_ascii=False)
        text="Verified original Qwen trained normalization: "+pretty
        code=("def interpret_trained_result(summary):\n"
              "    return {'width':summary['width'],"
              " 'output_sha256':summary['output_sha256'],"
              " 'mean_output':summary['mean_output']}\n")
        all_views={"text":text,"code":code,"tokens":list(text.encode("utf-8")),
                   "json":json.dumps(base,sort_keys=True,separators=(",",":"))}
        views={v:all_views[v] for v in spec["views"]}
        event={
            "schema":EVENT,"previous_head":self.head,
            "position_digest":position["position_digest"],
            "source_root":self.source_root,
            "intent_digest":spec["intent_digest"],
            "verified_result_receipt":receipt["receipt_digest"],
            "output_sha256":receipt["output_sha256"],
            "result_disposition":disposition,
            "new_external_source_requests":after["fresh_upstream_requests"]-
                                           before["fresh_upstream_requests"],
            "new_original_source_bytes":after["fresh_upstream_payload_bytes"]-
                                        before["fresh_upstream_payload_bytes"],
            "new_trained_arithmetic_evaluations":int(
                disposition=="NEW_TRAINED_ARITHMETIC"),
        }
        event["event_digest"]=_digest(EVENT,event)
        self.events.append(event)
        self.head=event["event_digest"]
        return {"position":position,"result":summary,"views":views,
                "event":deepcopy(event),"result_disposition":disposition,
                "original_observation_receipt":deepcopy(receipt),
                "trained_compute_ns":math_time}

    def checkpoint(self):
        _require(hasattr(self,"_index_raw"),
                 "Cannot publish trained result without pinned model index bytes")
        body={
            "schema":CHECKPOINT,
            "source_root":self.source_root,
            "observer":self.observer,
            "original_pretrained":self.original_pretrained,
            "implementation_fingerprint":self.implementation_fingerprint,
            "index_sha256":self.source.index_sha256,
            "source_checkpoint":self.source.checkpoint(),
            "result_pages":self._backing.snapshot(),
            "records":[deepcopy(self.records[k]) for k in sorted(self.records)],
            "head":self.head,
            "events":deepcopy(self.events),
        }
        return {**body,"checkpoint_digest":_digest(CHECKPOINT,body)}

    @classmethod
    def restore(cls,*,index_raw:bytes,snapshot:dict,expected_head:str):
        from mailcpu.structural_memory_pages import StructuralMemoryPageBacking
        from mailcpu.computational_objects import (
            ComputationalObjectStore,ComputationalObjectManifest)
        _require(type(snapshot) is dict and _hex64(expected_head)
                 and snapshot.get("schema")==CHECKPOINT
                 and snapshot.get("checkpoint_digest")==expected_head
                 and _digest(CHECKPOINT,{
                     k:v for k,v in snapshot.items()
                     if k!="checkpoint_digest"})==expected_head,
                 "Missing external pinned trained-result checkpoint")
        source_part=snapshot["source_checkpoint"]
        source=QualifiedPretrainedRangeMemory.restore(
            index_raw=index_raw,checkpoint=source_part,
            expected_head=source_part["checkpoint_digest"],upstream_host=None)
        result=cls(source,allow_synthetic_fixture=not snapshot["original_pretrained"],
                   observer=snapshot["observer"]).bind_source_index(index_raw)
        _require(result.source_root==snapshot["source_root"]
                 and result.implementation_fingerprint==
                     snapshot["implementation_fingerprint"]
                 and source.index_sha256==snapshot["index_sha256"]
                 and result.original_pretrained==snapshot["original_pretrained"],
                 "Original source/implementation differs across recovery")
        result._backing=StructuralMemoryPageBacking.from_snapshot(
            snapshot["result_pages"])
        result._store=ComputationalObjectStore(
            result._backing,chunk_size=4096,max_object_bytes=65536)
        _require(type(snapshot["records"]) is list
                 and len(snapshot["records"])<=MAX_RESULTS,
                 "Too many or missing retained trained results")
        for row in snapshot["records"]:
            _require(type(row) is dict and
                     set(row)=={"intent","receipt","vector_manifest"},
                     "Malformed original trained result record")
            spec=row["intent"];_admit_intention(spec)
            key=spec["intent_digest"]
            _require(key not in result.records,
                     "Duplicate trained computation Ground")
            result.records[key]=deepcopy(row)
            # Restore now, not on first navigation: fail-closed checkpoint.
            result._retrieve(spec)
        result.reused_trained_evaluations=0
        _require(type(snapshot["events"]) is list
                 and len(snapshot["events"])<=80,
                 "Invalid causal history")
        previous=_digest(CHECKPOINT,{
            "source_root":result.source_root,
            "observer":result.observer,"genesis":"independent_causal_observer_position"})
        for event in snapshot["events"]:
            _require(type(event) is dict and event.get("schema")==EVENT
                     and event.get("previous_head")==previous
                     and event.get("source_root")==result.source_root
                     and _hex64(event.get("event_digest"))
                     and _digest(EVENT,{k:v for k,v in event.items()
                                       if k!="event_digest"})==
                           event["event_digest"],
                     "Changed previous trained observation chain")
            previous=event["event_digest"]
        _require(snapshot["head"]==previous,
                 "Recovered trained observation head diverges")
        result.head=previous;result.events=deepcopy(snapshot["events"])
        return result

    def metrics(self):
        return {
            "source":self.source.metrics(),
            "verified_trained_evaluations":self.trained_evaluations,
            "reused_trained_evaluations_without_arithmetic":
                self.reused_trained_evaluations,
            "tracked_original_operator_math_ns":self.trained_math_ns,
            "ECPU_result_CAS_store_ns":self.cas_store_ns,
            "ECPU_result_CAS_read_ns":self.cas_read_ns,
            "retained_trained_result_positions":len(self.records),
            "original_pretrained_checkpoint":self.original_pretrained,
            "scope":"one_pure_original_centered_rmsnorm_not_full_pretrained_decoder",
        }
