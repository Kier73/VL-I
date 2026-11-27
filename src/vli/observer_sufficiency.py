"""PBG-010: CIG-015/016/017 observer sufficiency for witnessed neural traces.

The source is exactly a bounded set of verified PBG-006 causal tapes. The
invariant is q(x)=q(y) => g(x)=g(y), restricted to those source tapes.
No inference about arbitrary unobserved neural states or physical sensors.
One-hot CNF export permits an independent CIG-017 proof on the same finite law.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from hashlib import sha256
import json
import math

SCHEMA = 'vli.pbg010.source-observer-sufficiency.v1'
MAX_TAPES = 12
MAX_FEATURES = 128
MAX_STEPS = 24


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def _digest(value):
    return sha256(_json(value)).hexdigest()


def _hex(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _exact(x):
    if type(x) not in (int,Q) or Q(x)<=0:
        raise ValueError('Positive exact rational observation cost required')
    return Q(x)


def _encode(x):
    return f'{x.numerator}/{x.denominator}'


@dataclass(frozen=True)
class SourceBit:
    """Qualified native F64 numerical value >= exact F64 threshold."""
    position: int
    coordinate: int
    threshold: float

    def __post_init__(self):
        if (type(self.position) is not int or self.position<0 or
            type(self.coordinate) is not int or self.coordinate<0 or
            type(self.threshold) not in (int,float) or not math.isfinite(self.threshold)):
            raise ValueError('Invalid F64 observer coordinate')
        object.__setattr__(self,'threshold',float(self.threshold))

    @property
    def key(self):
        return f'p{self.position}:c{self.coordinate}:t{self.threshold.hex()}'

    def observe(self,tape):
        # Native receipt qualification happens at the enclosing source boundary.
        value=tape.reports[self.position]['receiver_local_intermediates'][1][self.coordinate]
        if type(value) not in (int,float) or not math.isfinite(value):
            raise ValueError('Unqualified F64 source readout')
        return int(float(value)>=self.threshold)


@dataclass(frozen=True)
class FiniteEvidence:
    """The already-qualified source's exact binary observation table."""
    source_digest: str
    source_witness: str
    tape_witnesses: tuple[str,...]
    goal: SourceBit
    targets: tuple[int,...]
    features: tuple[SourceBit,...]
    columns: tuple[tuple[int,...],...]
    costs: tuple[Q,...]

    def __post_init__(self):
        n=len(self.targets)
        if (not _hex(self.source_digest) or not _hex(self.source_witness) or
            not 1<=n<=MAX_TAPES or len(self.tape_witnesses)!=n or
            any(not _hex(x) for x in self.tape_witnesses) or
            any(type(b) is not int or b not in (0,1) for b in self.targets) or
            not isinstance(self.goal,SourceBit) or
            len(self.features)!=len(self.columns) or
            len(self.features)!=len(self.costs) or
            len(self.features)>MAX_FEATURES or
            len({f.key for f in self.features})!=len(self.features) or
            any(not isinstance(f,SourceBit) or f.position>=self.goal.position
                for f in self.features) or
            any(len(col)!=n or any(type(b) is not int or b not in (0,1)
                for b in col) for col in self.columns) or
            any(type(c) not in (int,Q) or c<=0 for c in self.costs)):
            raise ValueError('Invalid exact finite witness table')

    @property
    def digest(self):
        return _digest({'schema':SCHEMA,'source':self.source_digest,
            'witness':self.source_witness,'tapes':self.tape_witnesses,
            'goal':self.goal.key,'targets':self.targets,
            'features':[(f.key,col,_encode(Q(cost))) for f,col,cost in
                        zip(self.features,self.columns,self.costs)]})


def from_neural(ensemble, goal, *, coordinates=None, costs=None,
                max_features=MAX_FEATURES):
    """Read only PBG-006 verified native F64 receipts; no trained execution."""
    from .neural_trace_ground import WitnessedNeuralTraceEnsemble
    if (not isinstance(ensemble,WitnessedNeuralTraceEnsemble) or
        not isinstance(goal,SourceBit) or
        not all(t.verify() for t in ensemble.tapes) or
        not 1<=len(ensemble.tapes)<=MAX_TAPES or
        goal.position>=min(t.length for t in ensemble.tapes)):
        raise ValueError('Source-qualified neural ensemble and common goal required')
    width=len(ensemble.tapes[0].inputs[0][1])
    if goal.coordinate>=width or goal.position<1:
        raise ValueError('The requested goal needs an earlier observable position')
    if coordinates is None:
        coordinates=tuple(range(min(width,16)))
    coordinates=tuple(coordinates)
    if (not 1<=len(coordinates)<=24 or len(set(coordinates))!=len(coordinates) or
        any(type(c) is not int or not 0<=c<width for c in coordinates) or
        type(max_features) is not int or not 1<=max_features<=MAX_FEATURES):
        raise ValueError('Finite declared native coordinate catalogue required')
    if costs is None:
        costs={c:Q(1) for c in coordinates}
    if set(costs)!=set(coordinates):
        raise ValueError('Every declared coordinate needs an exact cost')
    prices={c:_exact(costs[c]) for c in coordinates}
    targets=tuple(goal.observe(t) for t in ensemble.tapes)
    candidates=[]
    for p in range(goal.position):
        for c in coordinates:
            readings=[float(t.reports[p]['receiver_local_intermediates'][1][c])
                      for t in ensemble.tapes]
            # Every threshold is drawn from genuinely observed F64 values.
            # Skipping the lowest value removes constant-one predicates.
            for threshold in sorted(set(readings))[1:]:
                bit=SourceBit(p,c,threshold)
                col=tuple(bit.observe(t) for t in ensemble.tapes)
                if len(set(col))==2:
                    candidates.append((bit,col,prices[c]))
    # One canonical cheapest representative of each Boolean partition; later
    # source revisions may change the partition and therefore the catalogue.
    selected={}
    for bit,col,price in candidates:
        old=selected.get(col)
        if old is None or (price,bit.key)<(old[2],old[0].key):
            selected[col]=(bit,col,price)
    ordered=sorted(selected.values(),key=lambda x:(x[2],x[0].key))[:max_features]
    return FiniteEvidence(ensemble.source_digest,ensemble.native_witness_digest,
        tuple(t.witness_digest for t in ensemble.tapes),goal,targets,
        tuple(x[0] for x in ordered),tuple(x[1] for x in ordered),
        tuple(x[2] for x in ordered))


def _signature(table,chosen,i):
    return tuple(table.columns[j][i] for j in chosen)


def _hist(table,chosen):
    groups={}
    for i,g in enumerate(table.targets):
        key=_signature(table,chosen,i)
        hist=groups.setdefault(key,[0,0]);hist[g]+=1
    return [[list(key),a,b] for key,(a,b) in sorted(groups.items())]


def _counterexample(table,chosen):
    for i,g in enumerate(table.targets):
        for j in range(i+1,len(table.targets)):
            if g!=table.targets[j] and _signature(table,chosen,i)==_signature(table,chosen,j):
                return (i,j)
    return None


def _ambiguous_pairs(table,chosen):
    pairs=[]
    for i in range(len(table.targets)):
        for j in range(i+1,len(table.targets)):
            if table.targets[i]!=table.targets[j] and _signature(table,chosen,i)==_signature(table,chosen,j):
                pairs.append((i,j))
    return pairs


def _plan_step(table,chosen):
    pairs=_ambiguous_pairs(table,chosen)
    choices=[]
    for k,(feat,col,cost) in enumerate(zip(table.features,table.columns,table.costs)):
        if k in chosen:continue
        split=sum(col[i]!=col[j] for i,j in pairs)
        if split:
            choices.append((Q(split,1)/cost,split,cost,feat.key,k))
    if not choices:return None
    choices.sort(key=lambda x:(-x[0],-x[1],x[2],x[3]))
    return choices[0][-1],choices[0][1],choices[0][0]


def compile_language(table,max_steps=MAX_STEPS):
    """Greedy counted-fiber refinement, not an optimum global sensor policy."""
    if not isinstance(table,FiniteEvidence) or type(max_steps)is not int or not 0<=max_steps<=MAX_STEPS:
        raise ValueError('Bounded qualified finite interpretation required')
    chosen=[];events=[]
    initial=_hist(table,chosen)
    for _ in range(max_steps):
        if _counterexample(table,chosen) is None:break
        decision=_plan_step(table,chosen)
        if decision is None:break
        idx,removed,score=decision
        old=_hist(table,chosen)
        chosen.append(idx)
        new=_hist(table,chosen)
        # CIG-015 exact conservation: H(u)=H'(u,0)+H'(u,1).
        for key,a,b in old:
            children=[h for sig,*h in new if sig[:-1]==key]
            if (sum(z[0] for z in children),sum(z[1] for z in children))!=(a,b):
                raise AssertionError('Observed fiber count was not conserved')
        events.append({'feature_index':idx,'removed_opposite_pairs':removed,
                       'score':_encode(score),'histogram':new})
    counter=_counterexample(table,chosen)
    decoder=None if counter is not None else [[sig, (1 if b else 0)] for sig,a,b in _hist(table,chosen)]
    result={'schema':SCHEMA,'source':table.source_digest,
        'source_witness':table.source_witness,'evidence':table.digest,
        'goal':table.goal.key,'initial_histogram':initial,
        'steps':events,'selected':chosen,'histogram':_hist(table,chosen),
        'status':('SUFFICIENT_ON_WITNESSED_ENSEMBLE' if counter is None else
                  'MORE_RETAINED_REFINEMENT_AVAILABLE' if _plan_step(table,chosen) is not None else
                  'MISSING_OBSERVER_DISTINCTION'),
        'counterexample':(None if counter is None else {'indices':list(counter),
          'witnesses':[table.tape_witnesses[counter[0]],table.tape_witnesses[counter[1]]],
          'goal_bits':[table.targets[counter[0]],table.targets[counter[1]]]}),
        'decoder':decoder,'native_trained_calls':0,
        'scope':'only verified finite neural trace ensemble'}
    result['certificate_digest']=_digest(result)
    return result


def verify_language(table,receipt,*,max_steps=MAX_STEPS):
    """Independent pairwise audit: exhaustive source witness enumeration.

    This checker does NOT call compile_language or the producer's scorer.
    Each step is independently checked against exact partition histograms and
    a separately computed maximally cost-efficient available split.
    """
    if not isinstance(table,FiniteEvidence) or not isinstance(receipt,dict):
        raise ValueError('Finite table and source certificate required')
    core={k:v for k,v in receipt.items() if k!='certificate_digest'}
    if (_digest(core)!=receipt.get('certificate_digest') or
        core.get('schema')!=SCHEMA or core.get('source')!=table.source_digest or
        core.get('source_witness')!=table.source_witness or
        core.get('evidence')!=table.digest or core.get('goal')!=table.goal.key or
        core.get('initial_histogram')!=[[[],sum(g==0 for g in table.targets),
                                         sum(g==1 for g in table.targets)]] or
        core.get('native_trained_calls')!=0 or core.get('scope')!='only verified finite neural trace ensemble'):
        raise ValueError('Source-bound observation proof mismatch')
    # Parse and validate source-bound event records using a separate verifier path.
    chosen=[]
    steps=core.get('steps')
    if not isinstance(steps,list) or len(steps)>max_steps:
        raise ValueError('Observer proof step budget exceeded')
    for event in steps:
        idx=event.get('feature_index')
        if (type(idx) is not int or idx<0 or idx>=len(table.features) or idx in chosen):
            raise ValueError('Invalid or repeated observer feature')
        current_pairs=[(i,j) for i in range(len(table.targets)) for j in range(i+1,len(table.targets))
                       if table.targets[i]!=table.targets[j] and
                       all(table.columns[k][i]==table.columns[k][j] for k in chosen)]
        scores=[]
        for k,col in enumerate(table.columns):
            if k in chosen:continue
            n=sum(col[i]!=col[j] for i,j in current_pairs)
            if n:scores.append((-Q(n,1)/Q(table.costs[k]),-n,Q(table.costs[k]),table.features[k].key,k))
        if not scores or min(scores)[-1]!=idx:
            raise ValueError('Refinement not source-optimal for declared greedy policy')
        removed=sum(table.columns[idx][i]!=table.columns[idx][j] for i,j in current_pairs)
        chosen.append(idx)
        groups={}
        for i,target in enumerate(table.targets):
            key=tuple(table.columns[k][i] for k in chosen)
            totals=groups.setdefault(key,[0,0]);totals[target]+=1
        calculated=[[list(k),v[0],v[1]] for k,v in sorted(groups.items())]
        if (event.get('removed_opposite_pairs')!=removed or
            event.get('score')!=_encode(Q(removed,1)/Q(table.costs[idx])) or
            event.get('histogram')!=calculated):
            raise ValueError('Incorrect counted fiber refinement evidence')
    if core.get('selected')!=chosen:
        raise ValueError('Observer extension history changed')
    groups={}
    for i,target in enumerate(table.targets):
        key=tuple(table.columns[k][i] for k in chosen)
        totals=groups.setdefault(key,[0,0]);totals[target]+=1
    hist=[[list(k),v[0],v[1]] for k,v in sorted(groups.items())]
    if core.get('histogram')!=hist:
        raise ValueError('Terminal observer histogram does not match original source')
    indistinct=next(((i,j) for i in range(len(table.targets)) for j in range(i+1,len(table.targets))
             if table.targets[i]!=table.targets[j] and
             all(table.columns[k][i]==table.columns[k][j] for k in chosen)),None)
    if indistinct is None:
        expected=[[sig,1 if b else 0] for sig,a,b in hist]
        if (core.get('status')!='SUFFICIENT_ON_WITNESSED_ENSEMBLE' or
            core.get('counterexample') is not None or core.get('decoder')!=expected):
            raise ValueError('Incorrect observed-image decoder')
    else:
        pair=indistinct
        expected={'indices':list(pair),'witnesses':[table.tape_witnesses[pair[0]],table.tape_witnesses[pair[1]]],
                  'goal_bits':[table.targets[pair[0]],table.targets[pair[1]]]}
        outcome=('MORE_RETAINED_REFINEMENT_AVAILABLE' if any(
                  table.columns[k][i]!=table.columns[k][j] for k in range(len(table.features))
                  if k not in chosen for i in range(len(table.targets))
                  for j in range(i+1,len(table.targets))
                  if table.targets[i]!=table.targets[j] and all(
                    table.columns[old][i]==table.columns[old][j] for old in chosen))
                else 'MISSING_OBSERVER_DISTINCTION')
        if (core.get('status')!=outcome or
            core.get('decoder') is not None or core.get('counterexample')!=expected):
            raise ValueError('Unsupported declaration of information sufficiency')
    return {'status':'INDEPENDENTLY_VERIFIED','source':table.source_digest,
            'certificate_digest':receipt['certificate_digest'],
            'observed_tapes':len(table.targets),'selected_features':len(chosen),
            'native_trained_calls':0}


def export_cig017_source(table,chosen):
    """Actual CIG-017 code can prove the same finite source via one-hot CNF.

    One selector per observed tape; all observed feature/goal bits are imposed
    by implication. Each satisfying assignment chooses exactly one tape.
    """
    if not isinstance(table,FiniteEvidence) or any(type(i) is not int or
        not 0<=i<len(table.features) for i in chosen) or len(set(chosen))!=len(chosen):
        raise ValueError('Invalid observed-source CNF projection')
    count=len(table.targets); width=len(chosen)
    n=count+width+1
    source=[tuple(range(1,count+1))]
    source.extend((-i,-j) for i in range(1,count+1) for j in range(i+1,count+1))
    for i in range(count):
        for j,k in enumerate(chosen):
            var=count+j+1; b=table.columns[k][i]
            source.append((-i-1, var if b else -var))
        var=count+width+1
        source.append((-i-1,var if table.targets[i] else -var))
    observations=tuple((count+j+1,) for j in range(width))
    target=(count+width+1,)
    return n,tuple(source),observations,target


def suggest_next_action(table,receipt):
    """Connect the CIG admission result to PBG-009 without faking a claim."""
    verify_language(table,receipt)
    return {'schema':SCHEMA,'source':table.source_digest,
        'observation_certificate':receipt['certificate_digest'],
        'action':('REUSE_VERIFIED_GROUND' if receipt['status']=='SUFFICIENT_ON_WITNESSED_ENSEMBLE'
                  else 'REFINE_FROM_RETAINED_RECEIPTS' if receipt['status']=='MORE_RETAINED_REFINEMENT_AVAILABLE'
                  else 'EXPAND_OBSERVER_CATALOGUE_OR_PBG009_SOURCE'),
        'counterexample':receipt['counterexample'],
        'native_trained_calls':0}


def compile_for_neural(ensemble,goal,*,expected_witness,coordinates=None,costs=None,max_features=MAX_FEATURES,max_steps=MAX_STEPS):
    """Caller-pinned neural input, independent source audit and no trained calls."""
    if not _hex(expected_witness) or ensemble.native_witness_digest!=expected_witness:
        raise ValueError('Externally pinned neural source witness changed')
    table=from_neural(ensemble,goal,coordinates=coordinates,costs=costs,max_features=max_features)
    proof=compile_language(table,max_steps=max_steps)
    verify_language(table,proof,max_steps=max_steps)
    return table,proof


def verify_for_neural(ensemble,goal,receipt,*,expected_witness,coordinates=None,costs=None,max_features=MAX_FEATURES,max_steps=MAX_STEPS):
    """Reconstruct exact observed data from the original SHAPE causal receipts."""
    if not _hex(expected_witness) or ensemble.native_witness_digest!=expected_witness:
        raise ValueError('Externally pinned neural source witness changed')
    table=from_neural(ensemble,goal,coordinates=coordinates,costs=costs,max_features=max_features)
    return verify_language(table,receipt,max_steps=max_steps)
