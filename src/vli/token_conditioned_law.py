"""SHAPE-008: learn source-qualified norm energy laws from REAL token rows.

Unlike SHAPE-007's supplied sinusoidal identity, the coefficients here are
inferred from original VL-64 operator observations on affine mixtures of TWO
source-qualified pretrained token embedding rows. A bounded family of
constant/affine/quadratic proposals is refuted using independently observed
native counterexamples. A proven algebraic property of an affine span then
qualifies an accepted quadratic *on that span only*.

The result is an observer-position Ground that forms unseen token-conditioned
outputs without rerunning the original native operator, while refusing tokens
outside the witnessed pair or nonlinear/attention-derived hidden states.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from time import perf_counter_ns
import math
import struct

from .ground import _digest, _hex64, _require
from .large_model_comparison import SOURCES
from .operative_model import ground as original_model_ground
from .token_embedding_ground import (
    token_intention, observe_token_row, _decode_and_independently_check)
from .trained_result_memory import TENSOR
from .pretrained_range_memory import QualifiedPretrainedRangeMemory

SCHEMA="vl-i.shape008-token-conditioned-law.v1"
POS="vl-i.shape008-token-conditioned-position.v1"
EVENT="vl-i.shape008-token-conditioned-observation.v1"
SNAP="vl-i.shape008-token-conditioned-checkpoint.v1"
PROOF="vl-i.shape008-inferred-polynomial-certificate.v1"
EMBEDDING="model.language_model.embed_tokens.weight"
CANDIDATES=("constant","affine","quadratic")
EPS=1e-6
TOL=2e-9


def _bits(values):
    return struct.pack("<"+"d"*len(values),*values)


def _fit_polynomial(observed):
    """Fit degrees 0/1/2 from three REAL operator energy observations."""
    e0,ehalf,e1=observed
    a=2*(e1-2*ehalf+e0)
    b=e1-e0-a
    return {
        "constant":(0.,0.,e0),
        "affine":(0.,e1-e0,e0),
        "quadratic":(a,b,e0),
    }


def _value(coef,t):
    return (coef[0]*t+coef[1])*t+coef[2]


def _source_identity(source,original):
    return _digest(SCHEMA,{
        "model":source.model,"revision":source.revision,
        "index_sha256":source.index_sha256,
        "index_ground_digest":source.index_ground_digest,
        "real_original_checkpoint":original,
        "operator":"original_VL64_centered_rms_norm",
        "input":"affine_span_of_two_qualified_original_token_embeddings",
        "epsilon":EPS,
    })


def _admit_t(t):
    _require(type(t) in (int,float) and math.isfinite(t)
             and 0<=t<=1, "Intended token-mixture coordinate outside admitted span")
    return float(t)


class TokenConditionedLawGround:
    """Learn a model-specific operator law from token-conditioned observations."""

    def __init__(self, source, index_raw, *, token_ids=(63204,628),
                 synthetic_fixture=False, observer="vli-token-morphology"):
        from experiments.model_fabric.scaled_model_fabric import QualifiedShardNavigator
        from experiments.model_fabric.qualified_subgraphs import _decode_float
        from experiments.model_fabric.trained_subgraphs import centered_rms_norm
        from mailcpu.structural_memory_pages import StructuralMemoryPageBacking
        from mailcpu.computational_objects import ComputationalObjectStore

        cfg=SOURCES["qwen-27b"]
        original=(source.model==cfg["model"]
                  and source.revision==cfg["revision"]
                  and source.index_sha256==cfg["index_sha256"]
                  and source.index_ground_digest==cfg["index_ground_digest"])
        _require(isinstance(source,QualifiedPretrainedRangeMemory)
                 and type(index_raw) is bytes and
                 sha256(index_raw).hexdigest()==source.index_sha256
                 and (original or synthetic_fixture)
                 and source.model==cfg["model"] and source.revision==cfg["revision"]
                 and type(token_ids) is tuple and len(token_ids)==2
                 and all(type(x) is int and x>=0 for x in token_ids)
                 and token_ids[0]!=token_ids[1]
                 and type(observer) is str and 1<=len(observer)<=80,
                 "Unqualified original token source or proposed token pair")
        self.source=source;self.index_raw=bytes(index_raw)
        self.token_ids=token_ids;self.observer=observer
        self.original_pretrained=original
        self.source_root=_source_identity(source,original)
        nav=QualifiedShardNavigator(
            model=source.model,revision=source.revision,index_raw=index_raw,
            range_host=source,approved_hosts=source.approved_hosts)
        desc=nav.locate(EMBEDDING)
        _require(desc["dtype"] in ("BF16","F16","F32")
                 and len(desc["shape"])==2
                 and desc["shape"][1]==(5120 if original else 64)
                 and desc["shape"][0]>max(token_ids),
                 "Embedding geometry does not match admitted original source")
        self.width=desc["shape"][1]
        self.embedding_witnesses=[]
        embeddings=[]
        if original:
            original_ground=original_model_ground("qwen-27b",nav.index)
        for token in token_ids:
            if original:
                # The original INTERACT-003 reader verifies native token row
                # decoding and all original pretrained byte-range provenance.
                vec,proof=observe_token_row(
                    nav,original_ground,token_intention(original_ground,token))
            else:
                raw,d=nav.get_raw(EMBEDDING,flat_start=token*self.width,
                                  count=self.width)
                vec=_decode_and_independently_check(raw,d["dtype"])
                proof={"token_id":token,"embedding_sha256":sha256(raw).hexdigest(),
                       "embedding_bytes":len(raw),"range_chain_head":nav.chain,
                       "scope":"synthetic_control_not_original_token_evidence"}
            _require(len(vec)==self.width and all(math.isfinite(v) for v in vec),
                     "Original token input has missing/nonfinite coordinates")
            embeddings.append(vec);self.embedding_witnesses.append(proof)
        raw_norm,norm_desc=nav.get_raw(TENSOR)
        _require(norm_desc["shape"]==[self.width],
                 "Missing actual trained normalization tensor")
        weights=_decode_float(raw_norm,norm_desc["dtype"])
        self.original_norm_weight_sha256=sha256(raw_norm).hexdigest()
        gains=[1+w for w in weights]
        self.source_checkpoint_head=source.checkpoint()["checkpoint_digest"]

        self.backing=StructuralMemoryPageBacking()
        self.store=ComputationalObjectStore(
            self.backing,chunk_size=4096,max_object_bytes=65536)
        self.manifests={}
        self.byte_hashes={}
        for name,vector in (("token_left",embeddings[0]),
                            ("token_right",embeddings[1]),("gains",gains)):
            raw=_bits(vector)
            manifest=self.store.put_bytes(raw)
            _require(self.store.verify(manifest),
                     "Original E-CPU trained coefficient object not verified")
            self.manifests[name]=manifest.to_dict()
            self.byte_hashes[name]=sha256(raw).hexdigest()

        # The target output is measured on actual pretrained weights.
        # "Learning" is performed on native outputs, not by secretly
        # computing the energy coefficients from input dot products.
        self.training_calls=0
        self.training_ns=0
        def native_at(t):
            t=_admit_t(t)
            x=[(1-t)*l+t*r for l,r in zip(*embeddings)]
            start=perf_counter_ns()
            y=centered_rms_norm(x,weights)
            self.training_ns+=perf_counter_ns()-start
            self.training_calls+=1
            # Infer energy from source-observed OUTPUT, not from an
            # input energy loop; use multiple coordinates for stability.
            projected=[v*g for v,g in zip(x,gains)]
            numerator=math.fsum(u*v for u,v in zip(projected,y))
            denominator=math.fsum(u*u for u in projected)
            _require(denominator>1e-18 and numerator>0,
                     "Observed original normalization cannot identify energy")
            factor=numerator/denominator
            e=1/(factor*factor)-EPS
            _require(math.isfinite(e) and e>0,
                     "Original observed normalization yielded unstable energy")
            return x,y,e

        samples=[native_at(t) for t in (0.,.5,1.)]
        candidate=_fit_polynomial([s[2] for s in samples])
        challenges=[]
        for t in (.25,.75):
            x,y,e=native_at(t)
            challenges.append((t,x,y,e))
        errors={}
        for name,coef in candidate.items():
            worst=0.
            for t,x,y,e in challenges:
                inferred_energy=_value(coef,t)
                if inferred_energy<=0:
                    worst=float("inf");break
                factor=1/math.sqrt(inferred_energy+EPS)
                worst=max(worst,max(
                    abs(xi*gi*factor-yi)
                    for xi,gi,yi in zip(x,gains,y)))
            errors[name]=worst
        admitted=[name for name in CANDIDATES if errors[name]<=TOL]
        _require(admitted==["quadratic"],
                 "Candidate law not uniquely selected by actual native counterexamples")
        self.coefficients=candidate["quadratic"]
        self.chosen_law="quadratic"
        # An independent algebraic domain-wide validation of the law.
        # Affine span square-energy has *exact polynomial degree <=2*,
        # so three identified values determine the whole span (subject to
        # numerical rounding). This is a mathematical proof of *the form*,
        # not autonomous discovery of a general neural identity.
        left,right=embeddings
        delta=[r-l for l,r in zip(left,right)]
        analytic=(
            math.fsum(d*d for d in delta)/self.width,
            2*math.fsum(x*d for x,d in zip(left,delta))/self.width,
            math.fsum(x*x for x in left)/self.width,
        )
        _require(max(abs(x-y) for x,y in zip(self.coefficients,analytic))
                 <=5e-10,
                 "Inferred operator law contradicts independent source-span algebra")
        self.proof={
            "schema":PROOF,"source_root":self.source_root,
            "source_checkpoint_head":self.source_checkpoint_head,
            "token_ids":list(token_ids),
            "embedding_witnesses":deepcopy(self.embedding_witnesses),
            "trained_norm_sha256":self.original_norm_weight_sha256,
            "ECPU_objects":{k:v["manifest_digest"] for k,v in self.manifests.items()},
            "polynomial_coefficients":list(self.coefficients),
            "candidate_challenge_errors":errors,
            "admitted_degree":2,
            "training_mix_points":[0.,.5,1.],
            "counterexample_mix_points":[.25,.75],
            "native_reference_calls":self.training_calls,
            "claim":"affine_span_degree2_energy_derived_from_actual_operator_observations",
        }
        self.proof["proof_digest"]=_digest(PROOF,self.proof)
        self.ground_digest=_digest(SCHEMA,{
            "source_root":self.source_root,
            "proof":self.proof["proof_digest"]})
        self.events=[]
        self.head=_digest(POS,{"ground":self.ground_digest,
                               "observer":observer,"genesis":"token_span_ground"})
        self.generation_calls=0
        self.native_heldout_calls=0

    def _read(self,name,indices):
        from mailcpu.computational_objects import ComputationalObjectManifest
        m=ComputationalObjectManifest.from_dict(self.manifests[name])
        # Read/verify each touched E-CPU page only once per output request.
        pages={}
        values=[]
        for i in indices:
            page=i*8//m.chunk_size
            if page not in pages:
                off=page*m.chunk_size
                pages[page]=self.store.read_range(
                    m,off,min(m.chunk_size,m.size_bytes-off))
            chunk=pages[page];offset=i*8-page*m.chunk_size
            values.append(struct.unpack("<d",chunk[offset:offset+8])[0])
        return values,len(pages)

    def _evaluate(self,t,indices):
        left,p0=self._read("token_left",indices)
        right,p1=self._read("token_right",indices)
        gain,pg=self._read("gains",indices)
        e=_value(self.coefficients,t)
        _require(math.isfinite(e) and e>0,"Learned span energy is not admitted")
        scale=1/math.sqrt(e+EPS)
        output=[((1-t)*a+t*b)*g*scale
                for a,b,g in zip(left,right,gain)]
        _require(all(math.isfinite(y) for y in output),
                 "Generated trained output contains nonfinite values")
        return output,p0+p1+pg

    def _admit(self):
        _require(self.source.checkpoint()["checkpoint_digest"]==
                     self.source_checkpoint_head
                 and self.proof["source_root"]==self.source_root
                 and self.proof["source_checkpoint_head"]==self.source_checkpoint_head
                 and self.proof["proof_digest"]==_digest(PROOF,{
                     k:v for k,v in self.proof.items() if k!="proof_digest"})
                 and self.proof["admitted_degree"]==2
                 and self.chosen_law=="quadratic",
                 "Original source or native-derived operator law changed")

    def generate(self,t,indices,*,expected_head):
        t=_admit_t(t);self._admit()
        _require(type(indices) in (tuple,list) and 1<=len(indices)<=self.width
                 and all(type(i) is int and 0<=i<self.width for i in indices)
                 and len(set(indices))==len(indices)
                 and _hex64(expected_head) and expected_head==self.head
                 and len(self.events)<256,
                 "Unqualified token-mixture intention or stale causal state")
        ix=tuple(indices)
        body={"schema":POS,"ground":self.ground_digest,
              "observer":self.observer,"mixture":t,
              "token_ids":list(self.token_ids),
              "indices":list(ix),"previous":self.head}
        position="state://vli/"+_digest(POS,body)
        start=perf_counter_ns()
        values,page_reads=self._evaluate(t,ix)
        compute_ns=perf_counter_ns()-start
        event={"schema":EVENT,"previous":self.head,
               "coordinate":position,"ground":self.ground_digest,
               "mixture":t,"indices":list(ix),
               "output_sha256":sha256(_bits(values)).hexdigest(),
               "original_native_operator_calls":0,
               "original_host_range_calls":0,
               "ECPU_source_pages_verified":page_reads}
        event["event_digest"]=_digest(EVENT,event)
        self.events.append(event);self.head=event["event_digest"]
        self.generation_calls+=1
        return {"coordinate":position,"values":values,
                "event":deepcopy(event),"duration_ns":compute_ns,
                "source_kind":"original_pretrained" if self.original_pretrained
                    else "synthetic_fixture"}

    def native_reference(self,t,indices):
        """HELD OUT: original trained source operator invoked after generation."""
        from experiments.model_fabric.scaled_model_fabric import QualifiedShardNavigator
        from experiments.model_fabric.qualified_subgraphs import _decode_float
        from experiments.model_fabric.trained_subgraphs import centered_rms_norm
        t=_admit_t(t)
        nav=QualifiedShardNavigator(
            model=self.source.model,revision=self.source.revision,
            index_raw=self.index_raw,range_host=self.source,
            approved_hosts=self.source.approved_hosts)
        raw_norm,d=nav.get_raw(TENSOR)
        from mailcpu.computational_objects import ComputationalObjectManifest
        def complete(name):
            manifest=ComputationalObjectManifest.from_dict(self.manifests[name])
            data=self.store.read_bytes(manifest,max_bytes=65536)
            return struct.unpack("<"+"d"*self.width,data)
        left=complete("token_left");right=complete("token_right")
        x=[(1-t)*a+t*b for a,b in zip(left,right)]
        y=centered_rms_norm(x,_decode_float(raw_norm,d["dtype"]))
        self.native_heldout_calls+=1
        return [y[i] for i in indices]

    def checkpoint(self):
        self._admit()
        body={
            "schema":SNAP,"source_root":self.source_root,
            "source_checkpoint":self.source.checkpoint(),
            "token_ids":list(self.token_ids),"width":self.width,
            "observer":self.observer,
            "original_pretrained":self.original_pretrained,
            "original_norm_weight_sha256":self.original_norm_weight_sha256,
            "ECPU_backing":self.backing.snapshot(),
            "ECPU_manifests":deepcopy(self.manifests),
            "ECPU_hashes":dict(self.byte_hashes),
            "coefficients":list(self.coefficients),
            "proof":deepcopy(self.proof),
            "ground_digest":self.ground_digest,
            "events":deepcopy(self.events),
            "head":self.head}
        body["checkpoint_digest"]=_digest(SNAP,body)
        return body

    @classmethod
    def restore(cls,index_raw,snapshot,*,expected_head):
        from mailcpu.structural_memory_pages import StructuralMemoryPageBacking
        from mailcpu.computational_objects import ComputationalObjectStore
        from mailcpu.computational_objects import ComputationalObjectManifest
        from experiments.model_fabric.scaled_model_fabric import QualifiedShardNavigator
        from experiments.model_fabric.qualified_subgraphs import _decode_float
        _require(type(snapshot) is dict and _hex64(expected_head)
                 and snapshot.get("schema")==SNAP
                 and snapshot.get("checkpoint_digest")==expected_head
                 and _digest(SNAP,{k:v for k,v in snapshot.items()
                                  if k!="checkpoint_digest"})==expected_head,
                 "Externally pinned token-law checkpoint absent or modified")
        src=snapshot["source_checkpoint"]
        source=QualifiedPretrainedRangeMemory.restore(
            index_raw=index_raw,checkpoint=src,
            expected_head=src["checkpoint_digest"],upstream_host=None)
        obj=cls.__new__(cls)
        obj.source=source;obj.index_raw=bytes(index_raw)
        obj.token_ids=tuple(snapshot["token_ids"]);obj.width=snapshot["width"]
        obj.observer=snapshot["observer"]
        obj.original_pretrained=snapshot["original_pretrained"]
        obj.original_norm_weight_sha256=snapshot["original_norm_weight_sha256"]
        obj.source_root=snapshot["source_root"]
        obj.source_checkpoint_head=src["checkpoint_digest"]
        obj.manifests=deepcopy(snapshot["ECPU_manifests"])
        obj.byte_hashes=dict(snapshot["ECPU_hashes"])
        obj.backing=StructuralMemoryPageBacking.from_snapshot(snapshot["ECPU_backing"])
        obj.store=ComputationalObjectStore(
            obj.backing,chunk_size=4096,max_object_bytes=65536)
        obj.coefficients=tuple(snapshot["coefficients"])
        obj.proof=deepcopy(snapshot["proof"]);obj.chosen_law="quadratic"
        obj.ground_digest=snapshot["ground_digest"]
        obj.events=deepcopy(snapshot["events"]);obj.head=snapshot["head"]
        obj.training_calls=0;obj.training_ns=0
        obj.generation_calls=0;obj.native_heldout_calls=0
        original=SOURCES["qwen-27b"]
        is_real=(source.model==original["model"]
                 and source.revision==original["revision"]
                 and source.index_sha256==original["index_sha256"]
                 and source.index_ground_digest==original["index_ground_digest"])
        _require(obj.original_pretrained==is_real and
                 obj.source_root==_source_identity(source,is_real)
                 and obj.width==(5120 if is_real else 64)
                 and len(obj.token_ids)==2 and obj.token_ids[0]!=obj.token_ids[1]
                 and obj.proof["token_ids"]==list(obj.token_ids)
                 and obj.proof["polynomial_coefficients"]==list(obj.coefficients)
                 and obj.ground_digest==_digest(SCHEMA,{
                     "source_root":obj.source_root,
                     "proof":obj.proof["proof_digest"]}),
                 "Restored learned law source/weights/domain identity changed")
        obj._admit()
        nav=QualifiedShardNavigator(
            model=source.model,revision=source.revision,index_raw=index_raw,
            range_host=source,approved_hosts=source.approved_hosts)
        def verify_object(name,original_data):
            manifest=ComputationalObjectManifest.from_dict(obj.manifests[name])
            data=obj.store.read_bytes(manifest,max_bytes=65536)
            _require(data==original_data
                     and sha256(data).hexdigest()==obj.byte_hashes[name],
                     "E-CPU retained coefficient contradicts original pretrained bytes")
        desc=nav.locate(EMBEDDING)
        _require(desc["shape"][1]==obj.width and
                 desc["shape"][0]>max(obj.token_ids),
                 "Token Ground changed dimension after restart")
        for token,name in zip(obj.token_ids,("token_left","token_right")):
            raw,d=nav.get_raw(EMBEDDING,flat_start=token*obj.width,
                              count=obj.width)
            original_vec=_decode_and_independently_check(raw,d["dtype"])
            verify_object(name,_bits(original_vec))
        raw_norm,d=nav.get_raw(TENSOR)
        _require(sha256(raw_norm).hexdigest()==obj.original_norm_weight_sha256,
                 "Actual pretrained normalization source changed")
        weights=_decode_float(raw_norm,d["dtype"])
        verify_object("gains",_bits([1+w for w in weights]))
        def values(name):
            m=ComputationalObjectManifest.from_dict(obj.manifests[name])
            raw=obj.store.read_bytes(m,max_bytes=65536)
            return struct.unpack("<"+"d"*obj.width,raw)
        left,right=values("token_left"),values("token_right")
        delta=[r-l for l,r in zip(left,right)]
        full_coefficients=(
            math.fsum(d*d for d in delta)/obj.width,
            2*math.fsum(l*d for l,d in zip(left,delta))/obj.width,
            math.fsum(l*l for l in left)/obj.width)
        _require(max(abs(a-b) for a,b in zip(obj.coefficients,full_coefficients))
                 <=5e-10,
                 "Learned numerical law no longer matches independent original span algebra")
        prev=_digest(POS,{"ground":obj.ground_digest,
                           "observer":obj.observer,"genesis":"token_span_ground"})
        for event in obj.events:
            _require(type(event) is dict and event.get("schema")==EVENT
                     and event.get("previous")==prev
                     and event.get("ground")==obj.ground_digest
                     and event.get("original_native_operator_calls")==0
                     and event.get("original_host_range_calls")==0
                     and _hex64(event.get("event_digest"))
                     and _digest(EVENT,{k:v for k,v in event.items()
                                        if k!="event_digest"})==event["event_digest"],
                     "Original causal sequence or trained source claim modified")
            t=_admit_t(event["mixture"])
            ix=tuple(event["indices"])
            _require(len(ix)>0 and len(set(ix))==len(ix)
                     and all(type(j) is int and 0<=j<obj.width for j in ix),
                     "Recovered observer requested invalid hidden coordinates")
            basis={"schema":POS,"ground":obj.ground_digest,
                   "observer":obj.observer,"mixture":t,
                   "token_ids":list(obj.token_ids),
                   "indices":list(ix),"previous":prev}
            calculated="state://vli/"+_digest(POS,basis)
            generated,_=obj._evaluate(t,ix)
            _require(calculated==event["coordinate"]
                     and event["output_sha256"]==sha256(_bits(generated)).hexdigest(),
                     "Recovered trained semantic output cannot be regenerated")
            prev=event["event_digest"]
        _require(prev==obj.head,"Reconstructed token-state history diverges")
        return obj
