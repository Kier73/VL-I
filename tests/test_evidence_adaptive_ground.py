"""PBG-004: exact posterior and counterexample-guided Ground revision tests.

These source-only tests use a synthetic verified event shape at the private
math kernel. Separate test_ground_revision_fabric exercises actual E-CPU/VGI.
"""
from dataclasses import replace
from fractions import Fraction as Q
from itertools import product
from types import SimpleNamespace
import copy
import pytest

from vli.evidence_adaptive_ground import (
    AdaptiveGroundCouncil, GroundExpansionRequired, QualifiedLaw, RevisionPlan,
)
from vli.intention_predictive_ground import IntentionSession, compile_intention_ground
from vli.predictive_basis_ground import _hash, _rowmat
from test_intention_predictive_ground import fixture


def source_with_accuracy(accuracy):
    """The same declared goal/actions, with an explicitly different sensor law."""
    base=fixture()
    n=len(base.initial)
    detectors=[]
    for label in ('0','1'):
        rows=[]
        for i in range(n):
            t=i//4
            weight=accuracy if t==int(label) else (1-accuracy)
            row=[Q()]*n; row[i]=weight
            rows.append(tuple(row))
        detectors.append((label,tuple(rows)))
    sensors=[(name,tuple(detectors) if name=='read-type' else outcomes,cost)
             for name,outcomes,cost in base.sensors]
    goal_sensor=[]
    for label in ('0','1'):
        rows=[]
        for i in range(n):
            row=[Q()]*n
            if int(base.goal[i])==int(label):row[i]=1
            rows.append(tuple(row))
        goal_sensor.append((label,tuple(rows)))
    sensors.append(('read-goal',tuple(goal_sensor),Q(1,100)))
    return replace(base,sensors=tuple(sensors))


def council(candidates):
    contract=compile_intention_ground(source_with_accuracy(Q(3,4)))
    return AdaptiveGroundCouncil(contract,tuple(
        QualifiedLaw.from_source(name,source_with_accuracy(acc),prior)
        for name,acc,prior in candidates))


def synthetic_source_event(c,kind,name,label='',nonce=None):
    """ONLY mathematical closure tests; no external evidence is asserted."""
    nonce=nonce or hex(len(c.events)+1)[2:].rjust(64,'0')
    prior=c.anchor.snapshot()
    shadow=IntentionSession.restore(c.anchor.ground,prior,
                                    expected_digest=prior['snapshot_digest'])
    if kind=='act':shadow.apply(name)
    else:shadow.observe(name,label)
    record={'kind':kind,'operation':name,'symbol':int(label) if kind=='sense' else None,
            'nonce':nonce,'previous':prior,'after':shadow.snapshot(),
            'intent_digest':_hash({'fixture':nonce}),
            'vgi_ledger_digest':_hash({'vgi':nonce}) if kind=='sense' else None}
    record['digest']=_hash(record)
    plan=RevisionPlan(c.epoch,SimpleNamespace())
    return plan,record


def commit_fixture(c,kind,name,label='',nonce=None):
    plan,record=synthetic_source_event(c,kind,name,label,nonce)
    result=c._apply(plan,record)
    return result,plan,record


def full_state_plan(sources,weights,beliefs,h):
    """An independently implemented 8-state Bellman oracle, not reduced K."""
    active=list(sources)
    goal=sum((weights[k]*sum((v*g for v,g in zip(beliefs[k],sources[k].goal)),Q())
             for k in active),Q())
    best=(goal,goal,Q(),'stop','')
    if h==0:return best
    for name,mat,cost in sources[active[0]].actions:
        evolved={k:_rowmat(beliefs[k],next(m for op,m,_ in sources[k].actions if op==name))
                 for k in active}
        _,p,c,_,_=full_state_plan(sources,weights,evolved,h-1)
        candidate=(p-c-cost,p,c+cost,'act',name)
        if candidate[0]>best[0]:best=candidate
    for name,outs,cost in sources[active[0]].sensors:
        p_goal,c_avg=Q(),cost
        for label,_ in outs:
            raw={k:_rowmat(beliefs[k],next(mat for n,outs2,_ in sources[k].sensors
                    if n==name for l,mat in outs2 if l==label)) for k in active}
            chance={k:sum(raw[k],Q()) for k in active}
            marginal=sum((weights[k]*chance[k] for k in active),Q())
            if not marginal:continue
            posterior={k:weights[k]*chance[k]/marginal for k in active}
            conditioned={k:tuple(x/chance[k] for x in raw[k]) if chance[k]>0 else beliefs[k]
                         for k in active}
            _,p,c,_,_=full_state_plan(sources,posterior,conditioned,h-1)
            p_goal+=marginal*p
            c_avg+=marginal*c
        option=(p_goal-c_avg,p_goal,c_avg,'sense',name)
        if option[0]>best[0]:best=option
    return best


def test_source_global_goal_ground_and_mixture_planning_matches_full_state():
    c=council((('accurate',Q(1),Q(4,5)),('inverted',Q(0),Q(1,5))))
    assert tuple(x.ground.rank for x in c.laws.values())==(4,4)
    assert c.active=='accurate'
    sources={k:source_with_accuracy(Q(1) if k=='accurate' else Q(0)) for k in c.laws}
    for h in range(3):
        actual=c.plan(h)
        expected=full_state_plan(sources,c.posterior,
                                  {k:s.initial for k,s in sources.items()},h)
        assert (actual.value,actual.success,actual.cost,actual.kind,actual.name)==expected


def test_zero_probability_counterexample_falsifies_only_incompatible_law():
    c=council((('accurate',Q(1),Q(4,5)),('inverted',Q(0),Q(1,5))))
    assert commit_fixture(c,'sense','read-type','1')[0]['status']=='REVISION_ACCEPTED'
    assert c.posterior=={'accurate':Q(4,5),'inverted':Q(1,5)}
    assert commit_fixture(c,'act','apply-1')[0]['status']=='REVISION_ACCEPTED'
    assert c.states['accurate'].goal_probability()==1
    assert c.states['inverted'].goal_probability()==0
    result,_,_=commit_fixture(c,'sense','read-goal','0')
    assert result['status']=='REVISION_ACCEPTED'
    assert result['eliminated']==['accurate']
    assert c.active=='inverted' and c.posterior['inverted']==1
    assert c.goal_probability==0
    assert c.plan(1).kind=='act' and c.plan(1).name=='apply-0'
    assert commit_fixture(c,'act','apply-0')[0]['status']=='REVISION_ACCEPTED'
    assert c.goal_probability==1


def test_surprise_with_positive_likelihood_reweights_without_false_falsification():
    c=council((('better',Q(3,4),Q(1,2)),('worse',Q(1,4),Q(1,2))))
    commit_fixture(c,'sense','read-type','1')
    commit_fixture(c,'act','apply-1')
    result,_,_=commit_fixture(c,'sense','read-goal','0')
    assert result['eliminated']==[]
    assert c.posterior=={'better':Q(1,4),'worse':Q(3,4)}
    assert c.goal_probability==0


def test_all_candidate_zero_support_retained_and_extended_without_reexecution():
    c=council((('accurate',Q(1),Q(1)),))
    commit_fixture(c,'sense','read-type','1')
    commit_fixture(c,'act','apply-1')
    old=c.epoch
    proposal,record=synthetic_source_event(c,'sense','read-goal','0')
    assert c._apply(proposal,record) is None
    assert c.epoch==old
    # The public, evidence-verified entry point sets pending, tested with
    # native E-CPU/VGI integration in test_ground_revision_fabric.py.
    c.pending=(proposal,record)
    with pytest.raises(GroundExpansionRequired):c.plan(1)
    with pytest.raises(GroundExpansionRequired):c.checkpoint()
    result=c.admit_source('inverted',source_with_accuracy(Q(0)),Q(1,4))
    assert result['status']=='GROUNDED_AFTER_EXPANSION'
    assert result['no_external_execution_repeated'] is True
    assert c.pending is None
    assert c.posterior=={'accurate':Q(0),'inverted':Q(1)}
    assert c.plan(1).name=='apply-0'
    assert len(c.events)==3


def test_checkpoint_source_free_reconstruction_and_tampered_weight_refusal():
    c=council((('accurate',Q(1),Q(4,5)),('inverted',Q(0),Q(1,5))))
    for kind,name,label in [('sense','read-type','1'),('act','apply-1',''),
                            ('sense','read-goal','0'),('act','apply-0','')]:
        commit_fixture(c,kind,name,label)
    saved=c.checkpoint()
    restored=AdaptiveGroundCouncil.restore(c.anchor.ground,tuple(c.laws.values()),saved,
                                            expected_digest=saved['checkpoint_digest'])
    assert restored.posterior==c.posterior
    assert restored.goal_probability==c.goal_probability
    assert restored.epoch==c.epoch
    assert restored.plan(2)==c.plan(2)
    forged=copy.deepcopy(saved)
    forged['candidates'][1]['mass']='9/1'
    forged['checkpoint_digest']=_hash({k:v for k,v in forged.items() if k!='checkpoint_digest'})
    with pytest.raises(ValueError,match='replay|disagrees|checkpoint'):
        AdaptiveGroundCouncil.restore(c.anchor.ground,tuple(c.laws.values()),forged,
                    expected_digest=forged['checkpoint_digest'])
    with pytest.raises(ValueError,match='Unpinned'):
        AdaptiveGroundCouncil.restore(c.anchor.ground,tuple(c.laws.values()),saved,
                    expected_digest='0'*64)


def test_cross_intention_candidate_is_refused():
    from vli.intention_predictive_ground import ControlledSource
    source=source_with_accuracy(Q(3,4))
    other=replace(source,goal=tuple(1-x for x in source.goal))
    c=council((('accurate',Q(1),Q(1)),))
    with pytest.raises(ValueError,match='intention'):
        c.admit_source('altered-goal',other,Q(1,4))
    with pytest.raises(ValueError,match='identity|contract|Candidate'):
        AdaptiveGroundCouncil(c.anchor.ground,(
            QualifiedLaw.from_source('a',source,Q(1)),
            QualifiedLaw.from_source('b',source,Q(1))))


def test_stale_revision_and_reused_nonce_are_refused_by_math_kernel():
    c=council((('accurate',Q(1),Q(4,5)),('inverted',Q(0),Q(1,5))))
    old_plan,old_record=synthetic_source_event(c,'sense','read-type','1')
    assert c._apply(old_plan,old_record) is not None
    with pytest.raises(ValueError,match='Stale'):
        c._apply(old_plan,old_record)
    new_plan,new_record=synthetic_source_event(c,'act','apply-0',nonce=old_record['nonce'])
    with pytest.raises(ValueError,match='Repeated'):
        c._apply(new_plan,new_record)


def test_nonnegative_source_rejects_bad_probabilistic_hypotheses():
    source=source_with_accuracy(Q(1))
    with pytest.raises(ValueError):
        source_with_accuracy(Q(5,4))
    with pytest.raises(ValueError,match='prior'):
        QualifiedLaw.from_source('negative',source,Q(-1))
