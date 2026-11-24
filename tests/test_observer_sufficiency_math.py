from __future__ import annotations
from dataclasses import replace
from fractions import Fraction as Q
import copy
import itertools
import pytest

from vli.observer_sufficiency import (
    SourceBit,FiniteEvidence,compile_language,verify_language,
    export_cig017_source,suggest_next_action,_digest,
)

D='a'*64
H='b'*64

def corpus(targets,cols,prices=None):
    prices=prices or [Q(1)]*len(cols)
    features=tuple(SourceBit(i,0,float(i)) for i in range(len(cols)))
    return FiniteEvidence(D,H,tuple(_digest({'tape':i}) for i in range(len(targets))),
         SourceBit(len(cols)+1,0,0.),tuple(targets),features,
         tuple(tuple(c) for c in cols),tuple(prices))


def test_counted_fiber_conservation_decoder_and_zero_new_execution():
    t=corpus([0,1,1,0],[[0,0,1,1],[0,1,0,1]])
    r=compile_language(t)
    assert r['status']=='SUFFICIENT_ON_WITNESSED_ENSEMBLE'
    assert len(r['selected'])==2
    assert verify_language(t,r)['selected_features']==2
    assert sum(a+b for _,a,b in r['histogram'])==4
    assert suggest_next_action(t,r)['action']=='REUSE_VERIFIED_GROUND'
    assert r['native_trained_calls']==0


def test_nonseparable_goal_produces_source_witness_pair():
    t=corpus([0,1,1],[[0,0,1]])
    p=compile_language(t)
    assert p['status']=='MISSING_OBSERVER_DISTINCTION'
    assert p['counterexample']['indices']==[0,1]
    assert verify_language(t,p)['status']=='INDEPENDENTLY_VERIFIED'
    assert suggest_next_action(t,p)['action']=='EXPAND_OBSERVER_CATALOGUE_OR_PBG009_SOURCE'


def test_uniform_goal_is_sufficient_with_zero_features():
    t=corpus([1,1,1],[])
    p=compile_language(t)
    assert p['selected']==[] and p['decoder']==[[[],1]]
    assert verify_language(t,p)['selected_features']==0


def test_declared_cost_chooses_cheapest_separating_feature():
    t=corpus([0,1,0,1],[[0,1,0,1],[0,1,0,1]],[Q(7),Q(1)])
    p=compile_language(t)
    assert p['selected']==[1]
    assert verify_language(t,p)['selected_features']==1


def test_tamper_refused_even_if_attacker_rehashes():
    t=corpus([0,1,1,0],[[0,0,1,1],[0,1,0,1]])
    p=compile_language(t)
    wrong=copy.deepcopy(p);wrong['histogram'][0]=[wrong['histogram'][0][0],9,9]
    wrong['certificate_digest']=_digest({k:v for k,v in wrong.items() if k!='certificate_digest'})
    with pytest.raises(ValueError):verify_language(t,wrong)
    wrong=copy.deepcopy(p);wrong['selected'].reverse()
    wrong['certificate_digest']=_digest({k:v for k,v in wrong.items() if k!='certificate_digest'})
    with pytest.raises(ValueError):verify_language(t,wrong)


def test_changed_source_or_labels_refuses_old_certificate():
    t=corpus([0,1,1],[[0,1,0]])
    p=compile_language(t)
    with pytest.raises(ValueError):verify_language(replace(t,targets=(1,1,1)),p)
    with pytest.raises(ValueError):verify_language(replace(t,source_witness='c'*64),p)
    with pytest.raises(ValueError):verify_language(replace(t,source_digest='d'*64),p)


def test_bounded_refinement_reports_remaining_insufficiency():
    t=corpus([0,1,1,0],[[0,0,1,1],[0,1,0,1]])
    p=compile_language(t,max_steps=1)
    assert p['status']=='MORE_RETAINED_REFINEMENT_AVAILABLE'
    assert suggest_next_action(t,p)['action']=='REFINE_FROM_RETAINED_RECEIPTS'
    assert len(p['selected'])==1
    assert verify_language(t,p,max_steps=1)
    with pytest.raises(ValueError):verify_language(t,p,max_steps=0)


def test_cig_one_hot_source_exactly_reproduces_finite_ensemble():
    t=corpus([0,1,0],[[0,1,1],[1,1,0]])
    n,clauses,q,f=export_cig017_source(t,(0,1))
    solutions=[]
    for values in itertools.product((0,1),repeat=n):
        if all(any(values[abs(l)-1]==int(l>0) for l in clause) for clause in clauses):
            solutions.append(values)
    assert len(solutions)==len(t.targets)
    got=sorted(tuple(values[i-1] for (i,) in q)+(values[f[0]-1],) for values in solutions)
    expected=sorted(tuple(t.columns[j][i] for j in (0,1))+(t.targets[i],)
                    for i in range(len(t.targets)))
    assert got==expected


@pytest.mark.parametrize('target',list(itertools.product((0,1),repeat=4)))
def test_all_four_bit_goal_assignments_source_relative_proof(target):
    t=corpus(target,[[0,0,1,1],[0,1,0,1]])
    p=compile_language(t)
    assert p['status']=='SUFFICIENT_ON_WITNESSED_ENSEMBLE'
    assert verify_language(t,p)


def test_json_checkpoint_roundtrip_is_lossless():
    import json
    t=corpus([0,1,1,0],[[0,0,1,1],[0,1,0,1]])
    cert=json.loads(json.dumps(compile_language(t)))
    assert verify_language(t,cert)['status']=='INDEPENDENTLY_VERIFIED'
