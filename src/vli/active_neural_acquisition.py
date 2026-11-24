"""PBG-007: active attention over observed predictive laws and qualified neural sources.

Two different acts of attention are preserved:
  (1) exact source-FREE discrimination between candidate weighted automata,
      followed by a separately qualified source trace measurement;
  (2) choice of ONE not-yet-executed native causal input path from a finite
      receiver-declared catalogue using an explicit novelty/cost heuristic,
      followed by one genuinely executed Qwen-shaped causal trajectory.

Neither unobserved native outputs nor physical-model probabilities are
invented. Competing forecasts are hypotheses, never source certificates.
The native acquisition is a receiver-local synthetic-weight fixture unless a
separately qualified original model/embedding provider is installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from hashlib import sha256
from collections import deque
import math
import struct

from .source_hankel_ground import (LearnedGround, QueryAtlas, HankelLearner,
                                   rank, rowmat, digest, encode, MAX_WORD)

SCHEMA = 'vli.pbg007.active-source-observation.v1'
MAX_CANDIDATES = 8
MAX_PATHS = 32
MAX_INPUT_MAGNITUDE = 1e6


def _predicted(z,terminal):
    return sum((x*y for x,y in zip(z,terminal)),Q())


def distinguish(hypothesis_a: LearnedGround, hypothesis_b: LearnedGround,
                *, max_depth=MAX_WORD):
    """Source-free shortest exact witness for two learned rational automata.

    Linear-span closure proves all-word equality BETWEEN TWO CANDIDATES, not
    truth about an external neural source. BFS gives a shortest witness.
    If a requested cap is reached before span closure, no certificate issues.
    """
    if (not isinstance(hypothesis_a,LearnedGround) or
            not isinstance(hypothesis_b,LearnedGround) or
            type(max_depth) is not int or not 0<=max_depth<=MAX_WORD):
        raise ValueError('Requires two bounded predictive candidates')
    hypothesis_a.check_identity(); hypothesis_b.check_identity()
    if hypothesis_a.alphabet != hypothesis_b.alphabet:
        raise ValueError('Prediction alphabets differ')
    queue=deque([((),hypothesis_a.initial,hypothesis_b.initial)])
    span=[]; exhausted=False
    while queue:
        word,a,b=queue.popleft()
        joint=tuple(a)+tuple(b)
        if rank(span+[joint])==len(span):
            continue
        span.append(joint)
        pa=_predicted(a,hypothesis_a.terminal)
        pb=_predicted(b,hypothesis_b.terminal)
        if pa!=pb:
            return {'status':'CANDIDATE_DISTINCTION','word':word,
                    'prediction_a':pa,'prediction_b':pb,
                    'reachable_rank':len(span),'source_queries':0}
        if len(word)>=max_depth:
            exhausted=True
            continue
        for op,ka,kb in zip(hypothesis_a.alphabet,
                            hypothesis_a.transforms,hypothesis_b.transforms):
            queue.append((word+(op,),rowmat(a,ka),rowmat(b,kb)))
    return {'status':('UNRESOLVED_WITHIN_DEPTH' if exhausted else
                       'CANDIDATES_EQUIVALENT_ALL_WORDS'),
            'word':None,'reachable_rank':len(span),'source_queries':0}


@dataclass(frozen=True)
class Candidate:
    """Hypothetical prediction, not authority to assert a source law."""
    name: str
    ground: LearnedGround
    weight: Q

    def __post_init__(self):
        if (type(self.name) is not str or not 1<=len(self.name)<=64 or
                not isinstance(self.ground,LearnedGround)):
            raise ValueError('Invalid predictive hypothesis')
        self.ground.check_identity()
        if type(self.weight) not in (int,Q) or Q(self.weight)<=0:
            raise ValueError('Expected exact positive hypothesis weight')
        object.__setattr__(self,'weight',Q(self.weight))


@dataclass(frozen=True)
class SourceProbePlan:
    source_digest: str
    epoch_digest: str
    nonce: str
    max_depth: int
    word: tuple[str,...]
    pair: tuple[str,str]
    forecasts: tuple[Q,Q]
    cost: Q
    score: Q
    plan_digest: str


class ActiveTraceObserver:
    """Select one discriminating source query under exact rational costs.

    A verified QueryAtlas.reader (e.g. PBG-006 receiver-qualified SAB/VGI)
    supplies the source evidence. The controller may NEVER treat its candidate
    hypotheses as verified merely because their digests parse correctly.
    """
    def __init__(self,atlas: QueryAtlas,learner: HankelLearner,candidates,
                 *,costs=None):
        if (not isinstance(atlas,QueryAtlas) or not isinstance(learner,HankelLearner)
                or learner.atlas is not atlas):
            raise ValueError('A current source-observation atlas is required')
        candidates=tuple(candidates)
        if (not 2<=len(candidates)<=MAX_CANDIDATES or
            any(not isinstance(x,Candidate) for x in candidates) or
            len({x.name for x in candidates})!=len(candidates) or
            any(x.ground.alphabet!=atlas.alphabet for x in candidates)):
            raise ValueError('Candidates must speak the same observer alphabet')
        costs=costs or {a:Q(1) for a in atlas.alphabet}
        if (set(costs)!=set(atlas.alphabet) or
                any(type(x) not in (int,Q) or Q(x)<=0 for x in costs.values())):
            raise ValueError('Positive exact costs required for each source operation')
        self.atlas=atlas
        self.learner=learner
        self.candidates=candidates
        self.costs={k:Q(v) for k,v in costs.items()}
        self.seen_nonces=set()

    @property
    def epoch(self):
        return digest({'schema':SCHEMA,'source':self.atlas.source_digest,
                       'evidence':self.atlas.evidence_digest,
                       'candidate_ground':[(x.name,x.ground.ground_digest,encode(x.weight))
                                           for x in self.candidates],
                       'current':None if self.learner.ground is None else
                                   self.learner.ground.ground_digest,
                       'costs':{k:encode(v) for k,v in self.costs.items()},
                       'seen':sorted(self.seen_nonces)})

    def propose(self,*,nonce,max_depth=MAX_WORD):
        if (type(nonce) is not str or len(nonce)!=64 or
            any(c not in '0123456789abcdef' for c in nonce) or
            nonce in self.seen_nonces):
            raise ValueError('Fresh canonical source-observation nonce required')
        options=[]
        total=sum((c.weight for c in self.candidates),Q())
        for i,a in enumerate(self.candidates):
            for b in self.candidates[i+1:]:
                found=distinguish(a.ground,b.ground,max_depth=max_depth)
                if found['status']!='CANDIDATE_DISTINCTION':
                    continue
                word=found['word']
                if word in self.atlas.cache:
                    continue
                cost=Q(1)+sum((self.costs[op] for op in word),Q())
                disagreement=abs(found['prediction_a']-found['prediction_b'])
                score=(a.weight*b.weight*disagreement)/(total*total*cost)
                options.append((score,cost,len(word),word,a.name,b.name,
                                found['prediction_a'],found['prediction_b']))
        if not options:
            raise ValueError('No new source-distinguishing candidate observation')
        # Best expected *weighted difference per declared cost*; not an
        # information-theoretic guarantee or prediction of a native output.
        options.sort(key=lambda x:(-x[0],x[1],x[2],x[3],x[4],x[5]))
        score,cost,_,word,name_a,name_b,pa,pb=options[0]
        epoch=self.epoch
        material={'schema':SCHEMA,'source':self.atlas.source_digest,
                  'epoch':epoch,'nonce':nonce,'max_depth':max_depth,
                  'word':word,'pair':(name_a,name_b),
                  'forecasts':(encode(pa),encode(pb)),
                  'cost':encode(cost),'score':encode(score)}
        return SourceProbePlan(self.atlas.source_digest,epoch,nonce,max_depth,word,
                               (name_a,name_b),(pa,pb),cost,score,digest(material))

    def execute(self,proposal):
        if not isinstance(proposal,SourceProbePlan):
            raise ValueError('A prepared active observation is required')
        # Recompute complete selection and epoch BEFORE calling any source.
        current=self.propose(nonce=proposal.nonce,max_depth=proposal.max_depth)
        if current!=proposal or proposal.epoch_digest!=self.epoch:
            raise ValueError('Stale or substituted source-observation choice')
        observed=self.atlas.probability(proposal.word)
        receipt=self.atlas.receipts[proposal.word]
        self.seen_nonces.add(proposal.nonce)
        matches=tuple(name for name,forecast in zip(proposal.pair,proposal.forecasts)
                      if observed==forecast)
        old=self.learner.ground
        refined=None
        if old is None:
            refined=self.learner.build()
        elif observed!=old.probability(proposal.word):
            refined=self.learner.counterexample(proposal.word)
        return {'schema':SCHEMA,'status':'QUALIFIED_SOURCE_PROBE',
                'source_digest':self.atlas.source_digest,
                'plan_digest':proposal.plan_digest,
                'word':proposal.word,'observed':observed,
                'source_receipt_digest':receipt,'matching_forecasts':matches,
                'old_ground_digest':None if old is None else old.ground_digest,
                'new_ground_digest':None if refined is None else refined.ground_digest,
                'new_rank':None if refined is None else refined.rank,
                'native_trained_calls_during_prediction':0,
                'native_trained_calls_during_source_trace_query':0}


def _f64(values):
    return sha256(b'vl-i.qwen-exact-f64-transfer.v1\0'+
                  struct.pack('<'+str(len(values))+'d',*values)).hexdigest()


def _validate_input_path(path,width):
    if (not isinstance(path,(tuple,list)) or not 1<=len(path)<=12):
        raise ValueError('Native causal input path must have 1-12 positions')
    out=[]
    for token,raw in path:
        if (type(token) is not int or not 0<=token<2**31 or
            not isinstance(raw,(tuple,list)) or len(raw)!=width or
            any(type(v) not in (float,int) or isinstance(v,bool) or
                not math.isfinite(v) or abs(v)>MAX_INPUT_MAGNITUDE for v in raw)):
            raise ValueError('Unqualified native input vector or label')
        out.append((token,tuple(float(v) for v in raw)))
    return tuple(out)


def _path_id(path):
    return digest([{'token_id_label':token,'f64':_f64(row)} for token,row in path])


def _distance(one,two):
    return sum(((a-b)**2 for a,b in zip(one,two)))


def _novelty(path,tapes):
    # Prediction-free experiment-design heuristic: compare ONLY admitted input
    # coordinates, never guess an unexecuted neural output or causal state.
    return min(sum((_distance(path[i][1], tape.inputs[i][1])/(i+1)
                    for i in range(min(len(path),len(tape.inputs)))))
               for tape in tapes)


@dataclass(frozen=True)
class NativeRunPlan:
    parent_digest: str
    nonce: str
    selected: tuple
    selected_digest: str
    proposal_digest: str
    novelty: float
    cost: int


def select_native_run(ensemble, candidates,*,nonce):
    """Choose one admitted but unexecuted Qwen input trajectory.

    Novelty is a declared input-space geometric heuristic, not expected output
    information. No numerical trained-layer call occurs during selection.
    """
    from .neural_trace_ground import WitnessedNeuralTraceEnsemble
    if (not isinstance(ensemble,WitnessedNeuralTraceEnsemble) or
            type(nonce) is not str or len(nonce)!=64 or
            any(c not in '0123456789abcdef' for c in nonce) or
            not all(t.verify() for t in ensemble.tapes)):
        raise ValueError('Native source identity or causal observations unqualified')
    width=len(ensemble.tapes[0].inputs[0][1])
    candidates=tuple(candidates)
    if not 1<=len(candidates)<=MAX_PATHS:
        raise ValueError('Finite receiver-selected source-action catalogue required')
    paths=tuple(_validate_input_path(p,width) for p in candidates)
    ids=tuple(_path_id(p) for p in paths)
    if len(set(ids))!=len(ids):
        raise ValueError('Duplicate native causal input proposals')
    seen={_path_id(t.inputs) for t in ensemble.tapes}
    ranked=[]
    for path,pid in zip(paths,ids):
        if pid in seen:continue
        novelty=_novelty(path,ensemble.tapes)
        if not math.isfinite(novelty):
            raise ValueError('Unbounded native novelty')
        ranked.append((novelty/len(path),novelty,len(path),pid,path))
    if not ranked:
        raise ValueError('All proposed causal runs are already observed')
    ranked.sort(key=lambda x:(-x[0],-x[1],x[2],x[3]))
    _,novelty,cost,pid,path=ranked[0]
    catalogue_digest=digest({'paths':ids,'parent':ensemble.native_witness_digest,
                             'source':ensemble.source_digest})
    proposal_digest=digest({'schema':SCHEMA,'parent':ensemble.native_witness_digest,
                            'catalogue':catalogue_digest,'nonce':nonce,
                            'selected':pid,'cost':cost,'novelty_hex':float(novelty).hex()})
    return NativeRunPlan(ensemble.native_witness_digest,nonce,path,pid,
                         proposal_digest,novelty,cost)


def acquire_native_run(ensemble,candidates,plan,chain_factory):
    """Execute ONLY the selected path, requalify full causal receipts, revise Ground.

    Fresh chain must match the original genesis/source. A changed source is
    refused. Existing tapes are never reexecuted; original Ground unchanged.
    """
    from .neural_trace_ground import (WitnessedNeuralTraceEnsemble,capture_native_tape)
    from .qwen_layer_chain import QwenCausalGDNChain
    if (not isinstance(plan,NativeRunPlan) or
            select_native_run(ensemble,candidates,nonce=plan.nonce)!=plan or
            not callable(chain_factory)):
        raise ValueError('Stale or substituted causal acquisition')
    chain=chain_factory()
    t=ensemble.tapes[0]
    if (not isinstance(chain,QwenCausalGDNChain) or chain.position!=0 or
            (chain.model,chain.revision,chain.index_ground_digest,chain.head)!=
            (t.model,t.revision,t.index_ground_digest,t.genesis)):
        raise ValueError('New native execution source differs from pinned model')
    additional=capture_native_tape(chain,plan.selected,ensemble.observer)
    revised=WitnessedNeuralTraceEnsemble((*ensemble.tapes,additional))
    if revised.native_witness_digest==ensemble.native_witness_digest:
        raise AssertionError('Native witness revision did not change')
    return revised,{'schema':SCHEMA,'status':'NATIVE_CAUSAL_SOURCE_REVISION',
                    'old_source_digest':ensemble.source_digest,
                    'new_source_digest':revised.source_digest,
                    'old_witness_digest':ensemble.native_witness_digest,
                    'new_witness_digest':revised.native_witness_digest,
                    'selected_input_digest':plan.selected_digest,
                    'source_request_digest':plan.proposal_digest,
                    'selected_native_transitions':len(plan.selected),
                    'unselected_native_transitions':0,
                    'native_calls_during_selection':0,
                    'qualified_new_positions':additional.length}


@dataclass(frozen=True)
class ObserverAttentionPlan:
    """A witness-derived sensor choice for a declared desired tape subset."""
    original_witness_digest: str
    target_tapes: tuple[int,...]
    position: int
    coordinate: int
    threshold: float
    distinction: Q
    cost: Q
    plan_digest: str


def select_observer_attention(ensemble, target_tapes, *, coordinate_costs=None):
    """Find a cheap native-output observer separating *already observed* paths.

    Every option is assessed using a verified F64 source byte read; this is
    bounded empirical discrimination, not a forecast of new trained states.
    Intention = which already executed trajectories the observer wants to
    distinguish. Exact rational fractions count observed categorical labels.
    """
    from .neural_trace_ground import WitnessedNeuralTraceEnsemble
    if not isinstance(ensemble,WitnessedNeuralTraceEnsemble) or not all(t.verify() for t in ensemble.tapes):
        raise ValueError('Qualified source observations required for attention')
    tapes=ensemble.tapes
    targets=tuple(sorted(target_tapes))
    others=tuple(i for i in range(len(tapes)) if i not in targets)
    if (not targets or not others or len(set(targets))!=len(targets) or
            any(type(i) is not int or not 0<=i<len(tapes) for i in targets)):
        raise ValueError('Attention needs nonempty, disjoint target and comparison paths')
    width=len(tapes[0].inputs[0][1])
    if coordinate_costs is None:
        coordinate_costs={i:Q(1) for i in range(width)}
    if (set(coordinate_costs)!=set(range(width)) or
            any(type(v) not in (int,Q) or Q(v)<=0 for v in coordinate_costs.values())):
        raise ValueError('Every observer coordinate requires an exact positive cost')
    options=[]
    for position in range(min(t.length for t in tapes)):
        for coordinate in range(width):
            readings=tuple(float(t.reports[position]['receiver_local_intermediates'][1]
                                  [coordinate]) for t in tapes)
            for threshold in sorted(set(readings))[1:]:
                yes_a=sum(readings[i]>=threshold for i in targets)
                yes_b=sum(readings[i]>=threshold for i in others)
                separation=abs(Q(yes_a,len(targets))-Q(yes_b,len(others)))
                if separation:
                    cost=Q(coordinate_costs[coordinate])
                    options.append((separation/cost,position,coordinate,threshold,separation,cost))
    if not options:
        raise ValueError('No qualified native observer coordinate separates the intention')
    options.sort(key=lambda x:(-x[0],x[1],x[2],x[3]))
    _,position,coordinate,threshold,separation,cost=options[0]
    material={'schema':SCHEMA,'witness':ensemble.native_witness_digest,
              'targets':targets,'position':position,'coordinate':coordinate,
              'threshold_hex':threshold.hex(),'distinction':encode(separation),
              'cost':encode(cost)}
    return ObserverAttentionPlan(ensemble.native_witness_digest,targets,
                                 position,coordinate,threshold,separation,cost,
                                 digest(material))


def apply_observer_attention(ensemble,plan,*,coordinate_costs=None):
    """Reproject already executed F64 traces into a new intention-defined Ground."""
    from .neural_trace_ground import (WitnessedNeuralTraceEnsemble,
                                      NeuralObserver,reframe_native_tape)
    if (not isinstance(plan,ObserverAttentionPlan) or
            select_observer_attention(ensemble,plan.target_tapes,
                                      coordinate_costs=coordinate_costs)!=plan):
        raise ValueError('Stale or modified observer attention')
    observer=NeuralObserver(plan.coordinate,plan.threshold)
    new=WitnessedNeuralTraceEnsemble(tuple(
        reframe_native_tape(t,observer) for t in ensemble.tapes))
    return new,{'schema':SCHEMA,'status':'OBSERVER_GROUND_REFRAMED',
                'previous_witness_digest':ensemble.native_witness_digest,
                'new_witness_digest':new.native_witness_digest,
                'previous_source_digest':ensemble.source_digest,
                'new_source_digest':new.source_digest,
                'attention_digest':plan.plan_digest,
                'original_neural_trained_calls':0,
                'source_receipts_reused':sum(t.length for t in ensemble.tapes),
                'coordinate':plan.coordinate,'position':plan.position,
                'observed_separation':encode(plan.distinction)}
