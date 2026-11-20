"""SHAPE-007: observer-directed generative trained-operator Ground for UNSEEN inputs.

This is a mathematical *family law*, NOT a cache of previously observed
outputs. The input is VL-I's existing analytic sin/cos hidden-state recipe.
A finite trigonometric orbit identity replaces the original O(width)
sum-of-squares normalization with O(1) energy formation. Any requested
coordinate is rendered by one local trigonometric evaluation and an exact
original trained weight gain, read from a verified E-CPU object range.

We **do not** claim that an expression matching a few samples is proved
universally. Candidate laws are screened against witnessed original VL-64
outputs and challenged on genuinely unseen inputs independently afterward.
There is no full decoder, learned language model or token inference here.
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
from .trained_result_memory import intention,_admit_intention,TENSOR
from .pretrained_range_memory import QualifiedPretrainedRangeMemory
from .large_model_comparison import SOURCES

SCHEMA="vl-i.shape007-generative-operator-ground.v1"
PROOF="vl-i.shape007-finite-orbit-law-witness.v1"
POS="vl-i.shape007-observer-position.v1"
OBS="vl-i.shape007-generated-observation.v1"
SNAP="vl-i.shape007-generative-ground-checkpoint.v1"
ALLOWED_LAWS=("constant_unit","equidistributed","finite_orbit_identity")
MAX_POSITIONS=5120

def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),
                      allow_nan=False,ensure_ascii=False).encode("utf-8")


def _finite_orbit_energy(wave,frequency,amplitude,phase,width):
    """Mean squared analytic input without enumerating the hidden vector.

    For theta_j = (j+1)*omega+phase, j=0,...,n-1:
      sum cos(2 theta_j)/n
       = sin(n*omega)/(n*sin(omega))*cos((n+1)*omega+2*phase)
      mean sin^2 = (1 - mean cos(2 theta))/2;
      mean cos^2 = (1 + mean cos(2 theta))/2.
    Not a general rule for arbitrary token-derived hidden states.
    """
    ratio=math.sin(width*frequency)/(width*math.sin(frequency))
    signed= -1 if wave=="sin" else 1
    energy=.5*amplitude*amplitude*(
        1+signed*ratio*math.cos((width+1)*frequency+2*phase))
    _require(math.isfinite(energy) and energy>=0,
             "Unstable or inadmissible closed-orbit input energy")
    return energy


def _energy(candidate,spec,width):
    if candidate=="constant_unit":return 1.
    if candidate=="equidistributed":return spec["amplitude"]**2/2
    _require(candidate=="finite_orbit_identity","Unknown proposed law")
    return _finite_orbit_energy(spec["waveform"],spec["frequency"],
                                spec["amplitude"],spec["phase"],width)


def _element(spec,index):
    angle=(index+1)*spec["frequency"]+spec["phase"]
    return spec["amplitude"]*(
        math.sin(angle) if spec["waveform"]=="sin" else math.cos(angle))


def _fingerprint():
    from experiments.model_fabric.trained_subgraphs import centered_rms_norm
    from experiments.model_fabric.qualified_subgraphs import _decode_float
    return _digest(SCHEMA,{
        "analytic_law":inspect.getsource(_finite_orbit_energy),
        "analytic_coordinate":inspect.getsource(_element),
        "original_native_operator":inspect.getsource(centered_rms_norm),
        "original_BF16_reader":inspect.getsource(_decode_float),
        "original_VLI_input_interpreter":inspect.getsource(_input),
        "python":list(sys.version_info[:3]),
        "arithmetic":"python_binary64; numeric_tolerance_not_bitwise_parity",
        "epsilon":1e-6,
    })


def _domain(spec,width):
    _admit_intention(spec)
    _require(width in (64,5120)
             and spec["waveform"] in ("sin","cos")
             and 0.00001<=spec["frequency"]<=0.9
             and 0.04<=spec["amplitude"]<=0.75
             and abs(spec["phase"])<=math.pi,
             "This generative law is not qualified for the requested input family")


class GenerativeTrainedOperatorGround:
    """A source-qualified *function Ground*, not one saved answer."""

    def __init__(self,source:QualifiedPretrainedRangeMemory,index_raw:bytes, *,
                 allow_synthetic_fixture=False,observer="vl-i-law-observer"):
        from experiments.model_fabric.scaled_model_fabric import QualifiedShardNavigator
        from experiments.model_fabric.qualified_subgraphs import _decode_float
        from experiments.model_fabric.trained_subgraphs import centered_rms_norm
        from mailcpu.structural_memory_pages import StructuralMemoryPageBacking
        from mailcpu.computational_objects import ComputationalObjectStore

        original=SOURCES["qwen-27b"]
        is_original=(source.model==original["model"]
                     and source.revision==original["revision"]
                     and source.index_sha256==original["index_sha256"]
                     and source.index_ground_digest==original["index_ground_digest"])
        _require(isinstance(source,QualifiedPretrainedRangeMemory)
                 and type(index_raw) is bytes
                 and sha256(index_raw).hexdigest()==source.index_sha256
                 and (is_original or allow_synthetic_fixture)
                 and source.model==original["model"]
                 and source.revision==original["revision"]
                 and type(observer) is str and 1<=len(observer)<=80,
                 "Original trained source authority missing or synthetic witness mislabeled")
        self.source=source;self.index_raw=bytes(index_raw)
        self.observer=observer;self.original_pretrained=is_original
        self.implementation=_fingerprint()
        self.source_root=_digest(SCHEMA,{
            "model":source.model,"revision":source.revision,
            "index_sha256":source.index_sha256,
            "index_ground_digest":source.index_ground_digest,
            "original":is_original,
            "implementation":self.implementation,
            "operator":"original_VL64_centered_RMSNorm",
        })
        nav=QualifiedShardNavigator(
            model=source.model,revision=source.revision,index_raw=index_raw,
            range_host=source,approved_hosts=source.approved_hosts)
        d=nav.locate(TENSOR)
        _require(d["shape"]==([5120] if is_original else [64])
                 and d["dtype"] in ("BF16","F32"),
                 "Original trained operator does not have admitted geometry")
        self.width=d["shape"][0]
        raw,typed=nav.get_raw(TENSOR)
        weights=_decode_float(raw,typed["dtype"])
        _require(len(weights)==self.width
                 and all(math.isfinite(x) for x in weights),
                 "Incomplete or nonfinite trained weight Ground")
        self.original_weight_sha256=sha256(raw).hexdigest()
        self.source_snapshot_digest=source.checkpoint()["checkpoint_digest"]
        # Store generalizable gains, not a table of input-specific results.
        gains=struct.pack("<"+"d"*self.width,*(1+w for w in weights))
        self._backing=StructuralMemoryPageBacking()
        self._store=ComputationalObjectStore(
            self._backing,chunk_size=4096,max_object_bytes=65536)
        m=self._store.put_bytes(gains)
        _require(self._store.verify(m),"E-CPU generative coefficient Ground rejected")
        self.gain_manifest=m.to_dict()
        self.gain_sha256=sha256(gains).hexdigest()
        self.training_original_calls=0
        self.training_ns=0
        self.generated_calls=0
        self.generated_coordinate_evaluations=0
        self.native_heldout_calls=0
        self.events=[]
        self.head=_digest(POS,{"source":self.source_root,
                               "observer":observer,"genesis":"unseen_state_law"})
        # These are training witnesses; held-out evaluation must be performed
        # *after* generation, outside the candidate-selection path.
        witnesses=[
            intention(waveform="sin",frequency=.0002,amplitude=.25,phase=.6),
            intention(waveform="cos",frequency=.0003,amplitude=.4,phase=-.8),
            intention(waveform="sin",frequency=.025,amplitude=.33,phase=.15),
            intention(waveform="cos",frequency=.4,amplitude=.12,phase=1.7),
        ]
        scores={x:0. for x in ALLOWED_LAWS}
        start=perf_counter_ns()
        for spec in witnesses:
            _domain(spec,self.width)
            x=_input(spec,self.width)
            reference=centered_rms_norm(x,weights)
            self.training_original_calls+=1
            # compare all coordinates, not just a few cherry-picked indices
            for candidate in ALLOWED_LAWS:
                factor=1/math.sqrt(_energy(candidate,spec,self.width)+1e-6)
                err=max(abs(_element(spec,i)*(1+weights[i])*factor-y)
                        for i,y in enumerate(reference))
                scores[candidate]=max(scores[candidate],err)
        self.training_ns=perf_counter_ns()-start
        accepted=[key for key in ALLOWED_LAWS if scores[key]<=3e-9]
        _require(accepted==["finite_orbit_identity"],
                 "Original observed operator did not discriminate a unique valid generative law")
        self.chosen_law=accepted[0]
        self.witness={
            "schema":PROOF,"source_root":self.source_root,
            "source_snapshot_digest":self.source_snapshot_digest,
            "weight_bytes_sha256":self.original_weight_sha256,
            "gain_manifest_digest":m.manifest_digest,
            "gain_bytes_sha256":self.gain_sha256,
            "width":self.width,
            "candidate_max_abs_errors":scores,
            "admitted_law":self.chosen_law,
            "witness_intentions":[s["intent_digest"] for s in witnesses],
            "original_native_witness_runs":self.training_original_calls,
            "status":"source_matched_witnesses_not_universal_kernel_proof",
        }
        self.witness["witness_digest"]=_digest(PROOF,self.witness)
        self.ground_digest=_digest(SCHEMA,{
            "source":self.source_root,
            "witness":self.witness["witness_digest"],
            "gain":self.gain_manifest["manifest_digest"],
            "law":self.chosen_law})
        self.bootstrap_ns=0

    def _admit(self):
        _require(self.implementation==_fingerprint()
                 and self.source.checkpoint()["checkpoint_digest"]==
                     self.source_snapshot_digest
                 and self.witness["witness_digest"]==_digest(PROOF,{
                     k:v for k,v in self.witness.items() if k!="witness_digest"})
                 and self.witness["admitted_law"]=="finite_orbit_identity"
                 and self.witness["source_root"]==self.source_root
                 and self.witness["gain_bytes_sha256"]==self.gain_sha256,
                 "Generative law, pretrained source or original witnesses changed")

    def observe(self,spec,indices,*,expected_head):
        from mailcpu.computational_objects import ComputationalObjectManifest
        _domain(spec,self.width);self._admit()
        _require(expected_head==self.head and _hex64(expected_head)
                 and type(indices) in (list,tuple)
                 and 1<=len(indices)<=MAX_POSITIONS
                 and len(set(indices))==len(indices)
                 and all(type(i) is int and 0<=i<self.width for i in indices),
                 "Source coordinate, observer projection or causal head not admitted")
        wanted=tuple(indices)
        basis={"schema":POS,"ground":self.ground_digest,
               "observer":self.observer,"intent":spec["intent_digest"],
               "indices":list(wanted),"causal_head":self.head}
        coord=_digest(POS,basis)
        position={"coordinate":"state://vli/"+coord,
                  "position_digest":coord,"scope":list(wanted)}
        first=perf_counter_ns()
        factor=1/math.sqrt(_energy(self.chosen_law,spec,self.width)+1e-6)
        manifest=ComputationalObjectManifest.from_dict(self.gain_manifest)
        values=[];pages=set()
        for i in wanted:
            # Read only relevant E-CPU stored coefficient pages; the entire
            # 5,120-dimensional vector is never materialized per query.
            chunk=(i*8)//manifest.chunk_size
            gain=struct.unpack("<d",self._store.read_range(manifest,i*8,8))[0]
            pages.add(chunk)
            values.append(_element(spec,i)*factor*gain)
        _require(all(math.isfinite(z) for z in values),
                 "Generated observer result has nonfinite values")
        bits=struct.pack("<"+"d"*len(values),*values)
        work_ns=perf_counter_ns()-first
        self.generated_calls+=1
        self.generated_coordinate_evaluations+=len(values)
        event={
            "schema":OBS,"ground":self.ground_digest,
            "position":position["coordinate"],"previous":self.head,
            "intention":spec["intent_digest"],
            "observer_indices":list(wanted),
            "result_sha256":sha256(bits).hexdigest(),
            "generated_coordinates":len(wanted),
            "source_range_requests":0,
            "original_native_operator_evaluations":0,
            "actual_coefficient_pages_used":len(pages),
            "method":"finite_orbit_sufficient_statistic_then_sparse_ECPU_gain_observation",
        }
        event["event_digest"]=_digest(OBS,event)
        self.events.append(event);self.head=event["event_digest"]
        return {"position":position,"values":values,
                "event":deepcopy(event),"work_ns":work_ns,
                "source_scope":"original_pretrained" if self.original_pretrained else
                    "synthetic_fixture",
                "chosen_law":self.chosen_law}

    def native_reference(self,spec,indices):
        """Independent HELD-OUT oracle, intentionally called after observe()."""
        from experiments.model_fabric.scaled_model_fabric import QualifiedShardNavigator
        from experiments.model_fabric.qualified_subgraphs import _decode_float
        from experiments.model_fabric.trained_subgraphs import centered_rms_norm
        _domain(spec,self.width)
        nav=QualifiedShardNavigator(model=self.source.model,
               revision=self.source.revision,index_raw=self.index_raw,
               range_host=self.source,approved_hosts=self.source.approved_hosts)
        raw,typed=nav.get_raw(TENSOR)
        x=_input(spec,self.width)
        full=centered_rms_norm(x,_decode_float(raw,typed["dtype"]))
        self.native_heldout_calls+=1
        return [full[i] for i in indices]

    def checkpoint(self):
        body={
            "schema":SNAP,"source_root":self.source_root,
            "observer":self.observer,"original_pretrained":self.original_pretrained,
            "implementation":self.implementation,
            "width":self.width,
            "source_checkpoint":self.source.checkpoint(),
            "gain_backing":self._backing.snapshot(),
            "gain_manifest":deepcopy(self.gain_manifest),
            "gain_sha256":self.gain_sha256,
            "original_weight_sha256":self.original_weight_sha256,
            "witness":deepcopy(self.witness),
            "law":self.chosen_law,"ground_digest":self.ground_digest,
            "events":deepcopy(self.events),"head":self.head,
        }
        body["checkpoint_digest"]=_digest(SNAP,body)
        return body

    @classmethod
    def restore(cls,index_raw,saved,*,expected_head):
        from experiments.model_fabric.scaled_model_fabric import QualifiedShardNavigator
        from mailcpu.structural_memory_pages import StructuralMemoryPageBacking
        from mailcpu.computational_objects import ComputationalObjectStore
        from experiments.model_fabric.qualified_subgraphs import _decode_float
        _require(type(saved) is dict and _hex64(expected_head)
                 and saved.get("schema")==SNAP
                 and saved.get("checkpoint_digest")==expected_head
                 and _digest(SNAP,{k:v for k,v in saved.items()
                                   if k!="checkpoint_digest"})==expected_head,
                 "Missing external generative Ground checkpoint pin")
        s=saved["source_checkpoint"]
        source=QualifiedPretrainedRangeMemory.restore(
            index_raw=index_raw,checkpoint=s,
            expected_head=s["checkpoint_digest"],upstream_host=None)
        obj=cls.__new__(cls)
        obj.source=source;obj.index_raw=bytes(index_raw)
        obj.observer=saved["observer"];obj.original_pretrained=saved["original_pretrained"]
        obj.implementation=saved["implementation"]
        obj.source_root=saved["source_root"];obj.width=saved["width"]
        obj.source_snapshot_digest=s["checkpoint_digest"]
        obj.original_weight_sha256=saved["original_weight_sha256"]
        obj.gain_sha256=saved["gain_sha256"]
        obj.gain_manifest=deepcopy(saved["gain_manifest"])
        obj.witness=deepcopy(saved["witness"]);obj.chosen_law=saved["law"]
        obj.ground_digest=saved["ground_digest"]
        obj._backing=StructuralMemoryPageBacking.from_snapshot(saved["gain_backing"])
        obj._store=ComputationalObjectStore(
            obj._backing,chunk_size=4096,max_object_bytes=65536)
        obj.training_original_calls=0;obj.training_ns=0
        obj.generated_calls=0;obj.generated_coordinate_evaluations=0
        obj.native_heldout_calls=0
        obj.events=deepcopy(saved["events"]);obj.head=saved["head"]
        expected_source=_digest(SCHEMA,{
            "model":source.model,"revision":source.revision,
            "index_sha256":source.index_sha256,
            "index_ground_digest":source.index_ground_digest,
            "original":obj.original_pretrained,
            "implementation":obj.implementation,
            "operator":"original_VL64_centered_RMSNorm"})
        expected_ground=_digest(SCHEMA,{
            "source":obj.source_root,
            "witness":obj.witness["witness_digest"],
            "gain":obj.gain_manifest["manifest_digest"],
            "law":obj.chosen_law})
        _require(obj.source_root==expected_source
                 and obj.ground_digest==expected_ground
                 and obj.implementation==_fingerprint()
                 and obj.width in (64,5120),
                 "New interpreter/source differs from retained generative law")
        original=SOURCES["qwen-27b"]
        is_original=(source.model==original["model"] and
                     source.revision==original["revision"] and
                     source.index_sha256==original["index_sha256"] and
                     source.index_ground_digest==original["index_ground_digest"])
        _require(obj.original_pretrained==is_original,
                 "Retained original-model status does not requalify")
        obj._admit()
        # Reconstruct exact weight-derived coefficient Ground from true
        # source-observed bytes, not from an untrusted checkpoint assertion.
        nav=QualifiedShardNavigator(model=source.model,revision=source.revision,
                    index_raw=index_raw,range_host=source,
                    approved_hosts=source.approved_hosts)
        desc=nav.locate(TENSOR)
        _require(desc["shape"]==[obj.width],"Recovered trained source width changed")
        raw,dtype=nav.get_raw(TENSOR)
        _require(sha256(raw).hexdigest()==obj.original_weight_sha256,
                 "Recovered original trained weight bytes changed")
        original_weights=_decode_float(raw,dtype["dtype"])
        expected_gains=struct.pack("<"+"d"*obj.width,
                                   *(1+w for w in original_weights))
        from mailcpu.computational_objects import ComputationalObjectManifest
        manifest=ComputationalObjectManifest.from_dict(obj.gain_manifest)
        actual_gains=obj._store.read_bytes(manifest,max_bytes=65536)
        _require(actual_gains==expected_gains and
                 sha256(actual_gains).hexdigest()==obj.gain_sha256,
                 "Recovered generative coefficients were not derived from source")
        prev=_digest(POS,{"source":obj.source_root,
                          "observer":obj.observer,"genesis":"unseen_state_law"})
        for event in obj.events:
            _require(type(event) is dict and event.get("schema")==OBS
                     and event.get("previous")==prev
                     and event.get("ground")==obj.ground_digest
                     and event.get("source_range_requests")==0
                     and event.get("original_native_operator_evaluations")==0
                     and _hex64(event.get("event_digest"))
                     and _digest(OBS,{k:v for k,v in event.items()
                                    if k!="event_digest"})==event["event_digest"],
                     "Generative event chain changed")
            prev=event["event_digest"]
        _require(prev==obj.head,"Restored generative causal head differs")
        return obj
