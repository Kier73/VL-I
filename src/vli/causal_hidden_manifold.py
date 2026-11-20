"""SHAPE-009: infer a generative Ground over ACTUAL original causal interlayer states.

The basis vectors are genuine complete layer-0 outputs consumed as inputs
to complete layer-1 execution in the existing QwenCausalGDNChain. They are
neither static embeddings nor invented analytic hidden-state vectors.

After three independently witnessed causal positions, an observer learns a
two-parameter polynomial law for the *separate* original layer-1 INPUT
NORMALIZATION operator over the convex span of those source states. The
basis is learned from native VL-64 output observations and counterexamples.
The chain's true layer-1 computation is NOT replaced by normalization alone.

The span contains hypothetical interpolated states, NOT new native causal
positions. A genuinely new causal state outside the proven span must refuse
generation and become a candidate for later basis refinement.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from time import perf_counter_ns
import math
import struct

from .ground import _digest, _hex64, _require
from .qwen_layer_chain import _f64_transfer_digest, SCHEMA as CHAIN
from .large_model_comparison import SOURCES

SCHEMA="vl-i.shape009-contextual-hidden-ground.v1"
PROOF="vl-i.shape009-contextual-native-law-proof.v1"
POSITION="vl-i.shape009-contextual-response-position.v1"
EVENT="vl-i.shape009-contextual-observation.v1"
CHECKPOINT="vl-i.shape009-contextual-checkpoint.v1"
NORM_NAME="model.language_model.layers.1.input_layernorm.weight"
EPS=1e-6
TOL=2e-8
CANDIDATES=("constant","affine","separable_quadratic","full_quadratic")
TRAIN=((0.,0.),(1.,0.),(0.,1.),(.5,0.),(0.,.5),(.5,.5))
CHALLENGES=((.21,.13),(.45,.22),(.12,.73))
MAX_EVENTS=100


def _bytes(values):
    return struct.pack("<"+"d"*len(values),*values)


def _native_energy(x,weights):
    """Observe original VL-64 normalized OUTPUT to infer its squared energy."""
    from experiments.model_fabric.trained_subgraphs import centered_rms_norm
    y=centered_rms_norm(x,weights)
    projected=[v*(1+w) for v,w in zip(x,weights)]
    denom=math.fsum(z*z for z in projected)
    _require(denom>1e-20,"Original trained native energy is not identifiable")
    factor=math.fsum(a*b for a,b in zip(projected,y))/denom
    energy=1/(factor*factor)-EPS
    _require(math.isfinite(energy) and energy>0,"Original native observation is invalid")
    return y,energy


def _vector(basis,s,t):
    return [a+s*(b-a)+t*(c-a) for a,b,c in zip(*basis)]


def _infer(observed):
    """Six observed native energies -> polynomial coefficient hypotheses."""
    e00,e10,e01,eh10,eh01,eh11=observed
    a=2*(e10-2*eh10+e00)
    b=2*(e01-2*eh01+e00)
    d=e10-e00-a
    e=e01-e00-b
    c=4*(eh11 - (.25*a+.25*b+.5*d+.5*e+e00))
    return {
        "constant":(0.,0.,0.,0.,0.,e00),
        "affine":(0.,0.,0.,e10-e00,e01-e00,e00),
        "separable_quadratic":(a,b,0.,d,e,e00),
        "full_quadratic":(a,b,c,d,e,e00),
    }


def _energy(coeff,s,t):
    a,b,c,d,e,f=coeff
    return a*s*s+b*t*t+c*s*t+d*s+e*t+f


def _domain(s,t):
    _require(type(s) in (int,float) and type(t) in (int,float)
             and math.isfinite(s) and math.isfinite(t)
             and s>=0 and t>=0 and s+t<=1,
             "Intention is outside source-qualified causal-state convex span")
    return float(s),float(t)


def _algebra(basis):
    x,y,z=basis
    dy=[v-u for u,v in zip(x,y)]
    dz=[v-u for u,v in zip(x,z)]
    n=len(x)
    return (
        math.fsum(v*v for v in dy)/n,
        math.fsum(v*v for v in dz)/n,
        2*math.fsum(v*w for v,w in zip(dy,dz))/n,
        2*math.fsum(v*w for v,w in zip(x,dy))/n,
        2*math.fsum(v*w for v,w in zip(x,dz))/n,
        math.fsum(v*v for v in x)/n,
    )


def _off_line(first,second,third):
    d=[b-a for a,b in zip(first,second)]
    v=[c-a for a,c in zip(first,third)]
    dd=math.fsum(x*x for x in d)
    vv=math.fsum(x*x for x in v)
    _require(dd>1e-20 and vv>1e-20,
             "Original causal states are not distinct")
    scale=math.fsum(x*y for x,y in zip(v,d))/dd
    residual=math.sqrt(math.fsum((x-scale*y)**2 for x,y in zip(v,d))/vv)
    return scale,residual


def _qualified_capture(report,expected_width,expected_position,previous,
                       *,model,revision,index_ground_digest):
    _require(type(report) is dict and
             report.get("schema")==CHAIN and
             report.get("model")==model and
             report.get("revision")==revision and
             report.get("index_ground_digest")==index_ground_digest and
             report.get("position")==expected_position and
             report.get("previous")==previous and
             report.get("all_layers_completed") is True and
             report.get("original_decoder_completed") is False and
             report.get("vocabulary_logits_computed") is False and
             type(report.get("original_token_id")) is int,
             "Missing original completed native causal chain witness")
    ws=report.get("layer_witnesses");states=report.get("receiver_local_intermediates")
    _require(type(ws) is list and len(ws)==2 and
             type(states) is list and len(states)==2 and
             ws[0].get("layer")==0 and ws[1].get("layer")==1 and
             ws[0].get("position")==expected_position and
             ws[1].get("position")==expected_position and
             ws[0].get("complete_output_width")==expected_width and
             ws[1].get("complete_output_width")==expected_width and
             len(states[0])==expected_width and
             len(states[1])==expected_width and
             all(math.isfinite(float(q)) for row in states for q in row),
             "No exact original layer0-to-layer1 hidden-state transfer")
    left=_f64_transfer_digest(states[0])
    right=_f64_transfer_digest(states[1])
    _require(left==ws[0]["output_f64_digest"]==
                 ws[1]["input_f64_digest"] and
             right==ws[1]["output_f64_digest"] and
             ws[0].get("input_f64_digest") and
             ws[1].get("recurrent_state_digest") and
             ws[1].get("convolution_state_digest"),
             "Causal original layer state contradicts next trained layer")
    semantic_ws=[{k:v for k,v in witness.items()
                   if k not in ("range_chain","qualified_ranges",
                                "trained_weight_bytes")} for witness in ws]
    semantic={
        "schema":CHAIN,"model":model,"revision":revision,
        "index_ground_digest":index_ground_digest,
        "position":expected_position,
        "original_token_id":report["original_token_id"],
        "previous":previous,"layer_witnesses":semantic_ws}
    _require(report.get("chain_head")==_digest(CHAIN,semantic),
             "Original complete trained chain head differs from source witness")
    record={
        "position":expected_position,
        "original_token_id":report["original_token_id"],
        "previous_head":previous,
        "chain_head":report["chain_head"],
        "layer0_output_f64_digest":left,
        "layer1_input_f64_digest":ws[1]["input_f64_digest"],
        "layer1_output_f64_digest":right,
        "original_layer0_range_chain":ws[0]["range_chain"],
        "original_layer1_range_chain":ws[1]["range_chain"],
        "layer0_recurrent_state_digest":ws[0]["recurrent_state_digest"],
        "layer1_recurrent_state_digest":ws[1]["recurrent_state_digest"],
        "layer1_convolution_state_digest":ws[1]["convolution_state_digest"],
        "native_complete_layers":True,
        "semantic_layer_witnesses":deepcopy(semantic_ws),
    }
    return list(states[0]),record


class CausalHiddenManifoldGround:
    """A discovered quadratic family over THREE *real causal* source states."""

    def __init__(self, *, reports, native_norm_raw, native_norm_dtype,
                 model,index_sha256,index_ground_digest,revision,
                 observer="vli-interlayer-observer"):
        from experiments.model_fabric.qualified_subgraphs import _decode_float
        from mailcpu.structural_memory_pages import StructuralMemoryPageBacking
        from mailcpu.computational_objects import ComputationalObjectStore
        _require(type(reports) is list and len(reports)==3
                 and type(model) is str and type(revision) is str
                 and _hex64(index_sha256) and _hex64(index_ground_digest)
                 and type(native_norm_raw) is bytes and
                 native_norm_dtype in ("BF16","F32","F16")
                 and type(observer) is str and 1<=len(observer)<=80,
                 "Missing source-qualified original causal states or trained norm")
        original=SOURCES["qwen-27b"]
        self.original_pretrained=(
            model==original["model"] and revision==original["revision"] and
            index_sha256==original["index_sha256"] and
            index_ground_digest==original["index_ground_digest"])
        self.width=5120 if self.original_pretrained else len(
            reports[0].get("receiver_local_intermediates",[[]])[0])
        _require(1<self.width<=5120 and
                 len(native_norm_raw)==(self.width*
                    (2 if native_norm_dtype in ("BF16","F16") else 4)),
                 "Original causal and normalization geometry is inconsistent")
        self.model=model;self.revision=revision
        self.index_sha256=index_sha256
        self.index_ground_digest=index_ground_digest
        self.observer=observer
        self.norm_dtype=native_norm_dtype
        self.native_norm_sha256=sha256(native_norm_raw).hexdigest()
        weights=_decode_float(native_norm_raw,native_norm_dtype)
        _require(len(weights)==self.width and
                 all(math.isfinite(w) for w in weights),
                 "Unqualified native pretrained normalization weights")
        self.norm_weights=list(weights)
        previous=_digest(CHAIN,{
            "model":model,"revision":revision,
            "index_ground_digest":index_ground_digest,"layers":[0,1]})
        basis=[];records=[]
        for pos,report in enumerate(reports):
            hidden,rec=_qualified_capture(
                report,self.width,pos,previous,
                model=model,revision=revision,
                index_ground_digest=index_ground_digest)
            basis.append(hidden);records.append(rec)
            previous=rec["chain_head"]
        self.recorded_causal_state_witnesses=records
        self.initial_chain_head=records[0]["previous_head"]
        self.final_chain_head=records[-1]["chain_head"]
        self.prior_line_parameter,self.off_line_residual=_off_line(*basis)
        _require(self.off_line_residual>1e-7,
                 "Third original causal position lies on prior one-dimensional span")
        self.source_root=_digest(SCHEMA,{
            "model":model,"revision":revision,"index_sha256":index_sha256,
            "index_ground_digest":index_ground_digest,
            "original_pretrained":self.original_pretrained,
            "causal_positions":records,"native_layer1_norm_sha256":
                self.native_norm_sha256,
            "native_layer1_norm_dtype":native_norm_dtype,
            "observed_basis_width":self.width})
        self.backing=StructuralMemoryPageBacking()
        self.store=ComputationalObjectStore(self.backing,chunk_size=4096,
                                              max_object_bytes=65536)
        self.objects={};self.object_hashes={}
        named=[("causal_0",_bytes(basis[0])),
               ("causal_1",_bytes(basis[1])),
               ("causal_2",_bytes(basis[2])),
               ("norm_raw",native_norm_raw),
               ("gain",_bytes([1+w for w in weights]))]
        for name,payload in named:
            m=self.store.put_bytes(payload)
            _require(self.store.verify(m),"Original ECPU causal basis object rejected")
            self.objects[name]=m.to_dict()
            self.object_hashes[name]=sha256(payload).hexdigest()
        self.native_training_calls=0
        self.native_training_ns=0
        observed={}
        def witness(s,t):
            x=_vector(basis,s,t)
            begin=perf_counter_ns()
            y,energy=_native_energy(x,weights)
            self.native_training_ns+=perf_counter_ns()-begin
            self.native_training_calls+=1
            return x,y,energy
        for s,t in TRAIN:
            observed[(s,t)]=witness(s,t)
        candidates=_infer([observed[p][2] for p in TRAIN])
        scores={}
        for name,coeff in candidates.items():
            err=0.
            for s,t in CHALLENGES:
                x,y,_=witness(s,t) if name==CANDIDATES[0] else observed[(s,t)]
                observed[(s,t)]=(x,y,0)
                e=_energy(coeff,s,t)
                if e<=0:
                    err=float("inf");break
                factor=1/math.sqrt(e+EPS)
                err=max(err,max(abs(z*(1+w)*factor -native)
                                for z,w,native in zip(x,weights,y)))
            scores[name]=err
        _require(scores["full_quadratic"]<=TOL and
                 all(scores[name]>TOL for name in CANDIDATES[:-1]),
                 "Original contextual observations did not distinguish law family")
        self.coefficients=tuple(candidates["full_quadratic"])
        independent=_algebra(basis)
        _require(max(abs(x-y) for x,y in
                     zip(self.coefficients,independent))<=3e-9,
                 "Observed trained law fails independent causal-basis algebra")
        self.proof={
            "schema":PROOF,"source_root":self.source_root,
            "inferred_coefficients":list(self.coefficients),
            "candidate_max_error":scores,
            "training_probes":[list(x) for x in TRAIN],
            "counterexamples":[list(x) for x in CHALLENGES],
            "original_native_norm_calls":self.native_training_calls,
            "native_law":"original_VL64_centered_rms_norm_layer1_input",
            "admitted_family":"quadratic_with_cross_term",
            "off_prior_line_relative_residual":self.off_line_residual,
            "qualifier":"observed_native_outputs_plus_independent_affine_span_identity",
        }
        self.proof["proof_digest"]=_digest(PROOF,self.proof)
        self.ground_digest=_digest(SCHEMA,{
            "source":self.source_root,"proof":self.proof["proof_digest"],
            "objects":{k:v["manifest_digest"] for k,v in self.objects.items()}})
        self.events=[]
        self.head=_digest(POSITION,{"ground":self.ground_digest,
                                   "observer":self.observer,
                                   "genesis":"original_causal_state_manifold"})
        self.generations=0
        self.native_heldout_calls=0

    def _read(self,name,indices):
        from mailcpu.computational_objects import ComputationalObjectManifest
        m=ComputationalObjectManifest.from_dict(self.objects[name])
        pages={};values=[]
        for i in indices:
            page=(i*8)//m.chunk_size
            if page not in pages:
                begin=page*m.chunk_size
                pages[page]=self.store.read_range(
                    m,begin,min(m.chunk_size,m.size_bytes-begin))
            off=i*8-page*m.chunk_size
            values.append(struct.unpack("<d",pages[page][off:off+8])[0])
        return values,len(pages)

    def _form(self,s,t,indices):
        a,pa=self._read("causal_0",indices)
        b,pb=self._read("causal_1",indices)
        c,pc=self._read("causal_2",indices)
        gain,pg=self._read("gain",indices)
        energy=_energy(self.coefficients,s,t)
        _require(energy>0 and math.isfinite(energy),
                 "Qualified causal manifold generated invalid energy")
        scale=1/math.sqrt(energy+EPS)
        result=[(u+s*(v-u)+t*(w-u))*g*scale
                for u,v,w,g in zip(a,b,c,gain)]
        _require(all(math.isfinite(q) for q in result),
                 "Original trained observer output is not finite")
        return result,pa+pb+pc+pg

    def generate(self,s,t,indices,*,expected_head):
        s,t=_domain(s,t)
        _require(type(indices) in (tuple,list) and
                 1<=len(indices)<=self.width and
                 len(indices)==len(set(indices)) and
                 all(type(i) is int and 0<=i<self.width for i in indices)
                 and _hex64(expected_head) and expected_head==self.head and
                 len(self.events)<MAX_EVENTS,
                 "Causal observer projection or externally expected head invalid")
        ix=tuple(indices)
        basis={"schema":POSITION,"ground":self.ground_digest,
               "observer":self.observer,"causal_source_head":self.final_chain_head,
               "mixture":[s,t],"indices":list(ix),"previous":self.head}
        pos="state://vli/"+_digest(POSITION,basis)
        start=perf_counter_ns()
        values,pages=self._form(s,t,ix)
        duration=perf_counter_ns()-start
        event={
            "schema":EVENT,"previous":self.head,
            "position":pos,"ground":self.ground_digest,
            "mixture":[s,t],"indices":list(ix),
            "output_sha256":sha256(_bytes(values)).hexdigest(),
            "original_native_trained_calls_during_generation":0,
            "original_source_requests_during_generation":0,
            "ECPU_coefficient_pages_verified":pages,
            "scope":"hypothetical_convex_interpolation_of_real_causal_states",
        }
        event["event_digest"]=_digest(EVENT,event)
        self.events.append(event);self.head=event["event_digest"]
        self.generations+=1
        return {"position":pos,"values":values,"duration_ns":duration,
                "event":deepcopy(event),"original_pretrained_basis":
                    self.original_pretrained}

    def native_reference(self,s,t,indices):
        from experiments.model_fabric.trained_subgraphs import centered_rms_norm
        s,t=_domain(s,t)
        from mailcpu.computational_objects import ComputationalObjectManifest
        vectors=[]
        for name in ("causal_0","causal_1","causal_2"):
            m=ComputationalObjectManifest.from_dict(self.objects[name])
            raw=self.store.read_bytes(m,max_bytes=65536)
            vectors.append(struct.unpack("<"+"d"*self.width,raw))
        values=_vector(vectors,s,t)
        out=centered_rms_norm(values,self.norm_weights)
        self.native_heldout_calls+=1
        return [out[i] for i in indices]

    def off_manifold(self,hidden):
        """Qualify a future TRUE causal state as a counterexample, not a hit."""
        _require(type(hidden) in (list,tuple) and len(hidden)==self.width and
                 all(type(v) in (int,float) and math.isfinite(v) for v in hidden),
                 "Unknown new original-layer contextual state")
        from mailcpu.computational_objects import ComputationalObjectManifest
        basis=[]
        for name in ("causal_0","causal_1","causal_2"):
            m=ComputationalObjectManifest.from_dict(self.objects[name])
            raw=self.store.read_bytes(m,max_bytes=65536)
            basis.append(struct.unpack("<"+"d"*self.width,raw))
        u=[b-a for a,b in zip(basis[0],basis[1])]
        v=[c-a for a,c in zip(basis[0],basis[2])]
        x=[h-a for h,a in zip(hidden,basis[0])]
        uu=math.fsum(z*z for z in u);vv=math.fsum(z*z for z in v)
        uv=math.fsum(a*b for a,b in zip(u,v))
        ux=math.fsum(a*b for a,b in zip(u,x))
        vx=math.fsum(a*b for a,b in zip(v,x))
        det=uu*vv-uv*uv
        _require(det>1e-24,"Causal basis is not rank two")
        s=(ux*vv-vx*uv)/det
        t=(vx*uu-ux*uv)/det
        residual=math.sqrt(math.fsum(
            (z-s*a-t*b)**2 for z,a,b in zip(x,u,v))/
            max(math.fsum(z*z for z in x),1e-30))
        return {"fit_parameters":[s,t],
                "relative_residual":residual,
                "inside_declared_convex_hull":(
                    s>=0 and t>=0 and s+t<=1 and residual<=1e-8),
                "source_qualification":"unverified_new_causal_source_until_receipt_pinned",
                "next_action":"refuse_and_observe_new_original_causal_position"}

    def checkpoint(self):
        body={
            "schema":CHECKPOINT,
            "model":self.model,"revision":self.revision,
            "index_sha256":self.index_sha256,
            "index_ground_digest":self.index_ground_digest,
            "original_pretrained":self.original_pretrained,
            "observer":self.observer,"width":self.width,
            "native_norm_dtype":self.norm_dtype,
            "native_norm_sha256":self.native_norm_sha256,
            "source_root":self.source_root,
            "original_causal_receipts":deepcopy(self.recorded_causal_state_witnesses),
            "ECPU_backing":self.backing.snapshot(),
            "ECPU_objects":deepcopy(self.objects),
            "ECPU_hashes":dict(self.object_hashes),
            "coefficients":list(self.coefficients),
            "proof":deepcopy(self.proof),
            "ground_digest":self.ground_digest,
            "events":deepcopy(self.events),"head":self.head,
        }
        return {**body,"checkpoint_digest":_digest(CHECKPOINT,body)}

    @classmethod
    def restore(cls,saved,*,expected_head):
        from mailcpu.structural_memory_pages import StructuralMemoryPageBacking
        from mailcpu.computational_objects import (
            ComputationalObjectStore,ComputationalObjectManifest)
        from experiments.model_fabric.qualified_subgraphs import _decode_float
        _require(type(saved) is dict and _hex64(expected_head)
                 and saved.get("schema")==CHECKPOINT
                 and saved.get("checkpoint_digest")==expected_head
                 and _digest(CHECKPOINT,{k:v for k,v in saved.items()
                                        if k!="checkpoint_digest"})==expected_head,
                 "Original causal Ground externally pinned head absent or altered")
        obj=cls.__new__(cls)
        obj.model=saved["model"];obj.revision=saved["revision"]
        obj.index_sha256=saved["index_sha256"]
        obj.index_ground_digest=saved["index_ground_digest"]
        obj.original_pretrained=saved["original_pretrained"]
        obj.observer=saved["observer"];obj.width=saved["width"]
        obj.norm_dtype=saved["native_norm_dtype"]
        obj.native_norm_sha256=saved["native_norm_sha256"]
        obj.source_root=saved["source_root"]
        obj.recorded_causal_state_witnesses=deepcopy(
            saved["original_causal_receipts"])
        obj.objects=deepcopy(saved["ECPU_objects"])
        obj.object_hashes=dict(saved["ECPU_hashes"])
        obj.backing=StructuralMemoryPageBacking.from_snapshot(saved["ECPU_backing"])
        obj.store=ComputationalObjectStore(
            obj.backing,chunk_size=4096,max_object_bytes=65536)
        obj.coefficients=tuple(saved["coefficients"])
        obj.proof=deepcopy(saved["proof"])
        obj.ground_digest=saved["ground_digest"]
        obj.events=deepcopy(saved["events"]);obj.head=saved["head"]
        obj.generations=0;obj.native_heldout_calls=0
        obj.native_training_calls=0;obj.native_training_ns=0
        original=SOURCES["qwen-27b"]
        _require(obj.original_pretrained==(
                 obj.model==original["model"] and
                 obj.revision==original["revision"] and
                 obj.index_sha256==original["index_sha256"] and
                 obj.index_ground_digest==original["index_ground_digest"]) and
                 obj.width==(5120 if obj.original_pretrained else 3),
                 "Changed model source scale or falsely declared original pretrained")
        expected=_digest(CHAIN,{
            "model":obj.model,"revision":obj.revision,
            "index_ground_digest":obj.index_ground_digest,
            "layers":[0,1]})
        _require(len(obj.recorded_causal_state_witnesses)==3,
                 "Missing source-trained causal predecessor states")
        basis=[]
        for pos,rec in enumerate(obj.recorded_causal_state_witnesses):
            _require(rec["position"]==pos and rec["previous_head"]==expected
                     and rec["native_complete_layers"] is True
                     and rec["layer0_output_f64_digest"]==
                         rec["layer1_input_f64_digest"],
                     "Stored observed causal layer transfer or order changed")
            witnesses=rec["semantic_layer_witnesses"]
            _require(type(witnesses) is list and len(witnesses)==2
                     and [w["layer"] for w in witnesses]==[0,1]
                     and all(w["position"]==pos and
                             w["complete_output_width"]==obj.width
                             for w in witnesses)
                     and witnesses[0]["output_f64_digest"]==
                         rec["layer0_output_f64_digest"]
                     and witnesses[1]["input_f64_digest"]==
                         rec["layer1_input_f64_digest"]
                     and witnesses[1]["output_f64_digest"]==
                         rec["layer1_output_f64_digest"]
                     and witnesses[0]["recurrent_state_digest"]==
                         rec["layer0_recurrent_state_digest"]
                     and witnesses[1]["recurrent_state_digest"]==
                         rec["layer1_recurrent_state_digest"]
                     and witnesses[1]["convolution_state_digest"]==
                         rec["layer1_convolution_state_digest"],
                     "Source semantic layer receipts no longer agree")
            chain_semantic={
                "schema":CHAIN,"model":obj.model,"revision":obj.revision,
                "index_ground_digest":obj.index_ground_digest,
                "position":pos,"original_token_id":rec["original_token_id"],
                "previous":expected,"layer_witnesses":witnesses}
            _require(_digest(CHAIN,chain_semantic)==rec["chain_head"],
                     "Original native layer causal head no longer reconstructs")
            expected=rec["chain_head"]
            m=ComputationalObjectManifest.from_dict(obj.objects[f"causal_{pos}"])
            raw=obj.store.read_bytes(m,max_bytes=65536)
            _require(len(raw)==obj.width*8
                     and sha256(raw).hexdigest()==
                         obj.object_hashes[f"causal_{pos}"] and
                     _f64_transfer_digest(struct.unpack(
                         "<"+"d"*obj.width,raw))==
                           rec["layer0_output_f64_digest"],
                     "Original causal hidden state does not match ECPU pages or native chain")
            basis.append(struct.unpack("<"+"d"*obj.width,raw))
        obj.initial_chain_head=obj.recorded_causal_state_witnesses[0]["previous_head"]
        obj.final_chain_head=expected
        obj.prior_line_parameter,obj.off_line_residual=_off_line(*basis)
        _require(obj.off_line_residual>1e-7,
                 "Recovered original causal positions no longer span a plane")
        m=ComputationalObjectManifest.from_dict(obj.objects["norm_raw"])
        native_raw=obj.store.read_bytes(m,max_bytes=65536)
        _require(sha256(native_raw).hexdigest()==obj.native_norm_sha256==
                     obj.object_hashes["norm_raw"],
                 "Original native trained weight bytes differ from retained source")
        obj.norm_weights=_decode_float(native_raw,obj.norm_dtype)
        gains=_bytes([1+w for w in obj.norm_weights])
        m=ComputationalObjectManifest.from_dict(obj.objects["gain"])
        _require(obj.store.read_bytes(m,max_bytes=65536)==gains and
                 sha256(gains).hexdigest()==obj.object_hashes["gain"],
                 "Original trained coefficient Ground no longer matches source weight")
        for name in obj.objects:
            m=ComputationalObjectManifest.from_dict(obj.objects[name])
            raw=obj.store.read_bytes(m,max_bytes=65536)
            _require(sha256(raw).hexdigest()==obj.object_hashes[name],
                     "Original observed ECPU CAS object was modified")
        expected_source=_digest(SCHEMA,{
            "model":obj.model,"revision":obj.revision,
            "index_sha256":obj.index_sha256,
            "index_ground_digest":obj.index_ground_digest,
            "original_pretrained":obj.original_pretrained,
            "causal_positions":obj.recorded_causal_state_witnesses,
            "native_layer1_norm_sha256":obj.native_norm_sha256,
            "native_layer1_norm_dtype":obj.norm_dtype,
            "observed_basis_width":obj.width})
        _require(expected_source==obj.source_root
                 and obj.proof["source_root"]==obj.source_root
                 and obj.proof["proof_digest"]==_digest(PROOF,{
                     k:v for k,v in obj.proof.items() if k!="proof_digest"})
                 and obj.proof["inferred_coefficients"]==list(obj.coefficients)
                 and obj.proof["admitted_family"]==
                     "quadratic_with_cross_term"
                 and obj.proof["off_prior_line_relative_residual"]==
                     obj.off_line_residual
                 and max(abs(a-b) for a,b in
                         zip(_algebra(basis),obj.coefficients))<=3e-9
                 and obj.ground_digest==_digest(SCHEMA,{
                     "source":obj.source_root,
                     "proof":obj.proof["proof_digest"],
                     "objects":{k:v["manifest_digest"]
                                for k,v in obj.objects.items()}}),
                 "Original contextual law or independently checked algebra changed")
        prev=_digest(POSITION,{"ground":obj.ground_digest,
                                "observer":obj.observer,
                                "genesis":"original_causal_state_manifold"})
        for event in obj.events:
            _require(type(event) is dict and event.get("schema")==EVENT
                     and event.get("previous")==prev
                     and event.get("ground")==obj.ground_digest
                     and event.get("original_native_trained_calls_during_generation")==0
                     and event.get("original_source_requests_during_generation")==0
                     and _hex64(event.get("event_digest"))
                     and _digest(EVENT,{k:v for k,v in event.items()
                                      if k!="event_digest"})==event["event_digest"],
                     "Recovered causal observer event chain forged")
            s,t=_domain(*event["mixture"])
            ix=event["indices"]
            _require(type(ix) is list and ix and len(set(ix))==len(ix)
                     and all(type(i) is int and 0<=i<obj.width for i in ix),
                     "Previously requested original observer output invalid")
            position=_digest(POSITION,{
               "schema":POSITION,"ground":obj.ground_digest,
               "observer":obj.observer,
               "causal_source_head":obj.final_chain_head,
               "mixture":[s,t],"indices":ix,"previous":prev})
            values,_=obj._form(s,t,ix)
            _require(event["position"]=="state://vli/"+position
                     and sha256(_bytes(values)).hexdigest()==event["output_sha256"],
                     "Saved causal response position or generated data cannot be reproduced")
            prev=event["event_digest"]
        _require(prev==obj.head,
                 "Source-free reconstructed learned causal Ground head differs")
        return obj
