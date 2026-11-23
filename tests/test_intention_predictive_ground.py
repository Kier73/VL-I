"""PBG-002: exact Virtual Layer planning, observational vs physical transitions."""
from dataclasses import replace
from fractions import Fraction as Q
from itertools import product
import pytest

from vli.predictive_basis_ground import _rowmat, _hash
from vli.intention_predictive_ground import (
    ControlledSource, IntentionSession, compile_intention_ground,
    verify_intention_ground,
)


def fixture(replicas=2):
    """Two material types, two success flags, redundant physical tags."""
    n=4*replicas
    idx=lambda t,s,r:(t*2+s)*replicas+r
    initial=[Q()]*n
    # Mostly unresolved material state: 20% already successful, equally split types.
    for t in range(2):
        initial[idx(t,0,0)]=Q(2,5)
        initial[idx(t,1,0)]=Q(1,10)
    goal=[Q(int((i//replicas)%2==1)) for i in range(n)]
    actions=[]
    for act in range(2):
        matrix=[]
        for i in range(n):
            t=i//(2*replicas); r=i%replicas
            row=[Q()]*n
            row[idx(t,int(t==act),r)]=1
            matrix.append(tuple(row))
        actions.append((f"apply-{act}",tuple(matrix),Q(1,100)))
    sensors=[]
    def sense(name,yes_prob):
        out=[]
        for label in ('0','1'):
            rows=[]
            for i in range(n):
                t=i//(2*replicas)
                # '0' signal favors type 0, '1' favors type 1.
                probability= yes_prob if t==int(label) else (1-yes_prob)
                row=[Q()]*n;row[i]=probability
                rows.append(tuple(row))
            out.append((label,tuple(rows)))
        return (name,tuple(out),Q(1,20))
    sensors.append(sense("read-type",Q(3,4)))
    sensors.append(sense("uninformative-coin",Q(1,2)))
    return ControlledSource(tuple(initial),tuple(goal),tuple(actions),tuple(sensors))


def source_replay(source,word):
    state=source.initial
    for kind,name,label in word:
        if kind=='act':
            mat=next(m for a,m,c in source.actions if a==name)
            state=_rowmat(state,mat)
        else:
            outcomes=next(os for a,os,c in source.sensors if a==name)
            mat=next(m for l,m in outcomes if l==label)
            state=_rowmat(state,mat)
            p=sum(state)
            if p==0:
                raise ValueError('Impossible source observation')
            state=tuple(x/p for x in state)
    return state


def test_goal_predictive_closure_over_all_horizon_three_words():
    src=fixture()
    ground=compile_intention_ground(src)
    assert ground.rank==4 < len(src.initial)==8
    assert verify_intention_ground(src,ground)
    ops=[('act',name,'') for name,_,_ in src.actions]
    ops += [('sense',name,label) for name,outcomes,cost in src.sensors
            for label,mat in outcomes]
    for h in range(4):
        for word in product(ops,repeat=h):
            source=source_replay(src,word)
            session=IntentionSession(ground)
            for kind,name,label in word:
                if kind=='act':session.apply(name)
                else:session.observe(name,label)
            assert session.goal_probability()==sum(a*g for a,g in zip(source,src.goal))
            for name,outs,cost in src.sensors:
                oracle={label:sum(_rowmat(source,mat)) for label,mat in outs}
                assert session.sensor_distribution(name)==oracle
    assert ground.rank==4


def test_sensor_changes_knowledge_not_expected_material_goal():
    ground=compile_intention_ground(fixture())
    session=IntentionSession(ground)
    before=session.goal_probability()
    probabilities=session.sensor_distribution('read-type')
    after_expectation=Q()
    for label,p in probabilities.items():
        branch=IntentionSession(ground)
        branch.observe('read-type',label)
        after_expectation+=p*branch.goal_probability()
    assert before==after_expectation==Q(1,5)
    assert probabilities=={'0':Q(1,2),'1':Q(1,2)}
    session.apply('apply-0')
    assert session.goal_probability()==Q(1,2)


def test_goal_planner_picks_information_before_effect_and_replans():
    ground=compile_intention_ground(fixture())
    session=IntentionSession(ground)
    direct=session.plan(1)
    assert direct.kind=='act' and direct.success==Q(1,2)
    assert direct.value==Q(49,100)
    plan=session.plan(2)
    assert plan.kind=='sense' and plan.name=='read-type'
    assert plan.success==Q(3,4)
    assert plan.cost==Q(6,100)
    assert plan.value==Q(69,100)
    assert dict((label,(kind,name)) for label,p,kind,name in plan.branches)=={
        '0':('act','apply-0'), '1':('act','apply-1')}
    session.observe('read-type','1')
    assert session.plan(1).name=='apply-1'
    session.apply('apply-1')
    assert session.goal_probability()==Q(3,4)
    assert session.plan(0).kind=='stop'


def test_no_epistemic_value_from_independent_random_sensor():
    src=fixture()
    ground=compile_intention_ground(src)
    session=IntentionSession(ground)
    p=session.sensor_distribution('uninformative-coin')
    assert p=={'0':Q(1,2),'1':Q(1,2)}
    for label in p:
        x=IntentionSession(ground)
        x.observe('uninformative-coin',label)
        assert x.goal_probability()==session.goal_probability()
        assert x.plan(1).value==session.plan(1).value
    assert session.plan(2).name!='uninformative-coin'


def test_restore_tampered_and_rehashed_ground_refused():
    src=fixture()
    ground=compile_intention_ground(src)
    s=IntentionSession(ground)
    s.observe('read-type','0');s.apply('apply-0')
    snap=s.snapshot()
    restored=IntentionSession.restore(ground,snap,expected_digest=snap['snapshot_digest'])
    assert restored.state==s.state and restored.plan(2)==s.plan(2)
    forged={**snap,'state':['999/1']*ground.rank}
    forged['snapshot_digest']=_hash({k:v for k,v in forged.items() if k!='snapshot_digest'})
    with pytest.raises(ValueError,match='Altered state'):
        IntentionSession.restore(ground,forged,expected_digest=forged['snapshot_digest'])
    with pytest.raises(ValueError,match='Unpinned'):
        IntentionSession.restore(ground,snap,expected_digest='0'*64)
    ops=list(ground.operators)
    kind,name,label,mat,c=ops[0]
    rows=[list(row) for row in mat];rows[0][1]+=1
    ops[0]=(kind,name,label,tuple(tuple(row) for row in rows),c)
    corrupt=replace(ground,operators=tuple(ops))
    corrupt=replace(corrupt,ground_digest=_hash(corrupt.material()))
    with pytest.raises(ValueError,match='commutation'):
        verify_intention_ground(src,corrupt)


def test_invalid_physical_sensor_and_unqualified_ops():
    src=fixture()
    bad_sensor=list(src.sensors)
    name,outputs,cost=bad_sensor[0]
    label,first=outputs[0]
    rows=[list(row) for row in first]
    rows[0][1]+=Q(1,4)
    bad_sensor[0]=(name,((label,tuple(map(tuple,rows))),outputs[1]),cost)
    with pytest.raises(ValueError,match='Sensor must not change'):
        replace(src,sensors=tuple(bad_sensor))
    s=IntentionSession(compile_intention_ground(src))
    with pytest.raises(ValueError,match='Unqualified'):
        s.apply('other')
    with pytest.raises(ValueError,match='Unknown'):
        s.sensor_distribution('other')
    with pytest.raises(ValueError,match='Unqualified lookahead'):
        s.plan(5)


def reference_plan(source, belief, horizon):
    """Independent n-state Bellman oracle, without any quotient operations."""
    success=sum((p*g for p,g in zip(belief,source.goal)),Q())
    best=(success,success,Q(),'stop','')
    if horizon==0:
        return best
    for name,matrix,fee in source.actions:
        moved=_rowmat(belief,matrix)
        val,p,c,_,_=reference_plan(source,moved,horizon-1)
        candidate=(p-(c+fee),p,c+fee,'act',name)
        if candidate[0]>best[0]:best=candidate
    for name,outcomes,fee in source.sensors:
        p_goal,cost=Q(),fee
        for label,matrix in outcomes:
            posterior=_rowmat(belief,matrix)
            likelihood=sum(posterior)
            if likelihood:
                next_belief=tuple(x/likelihood for x in posterior)
                v,p,c,_,_=reference_plan(source,next_belief,horizon-1)
                p_goal+=likelihood*p
                cost+=likelihood*c
        candidate=(p_goal-cost,p_goal,cost,'sense',name)
        if candidate[0]>best[0]:best=candidate
    return best


def test_independent_full_state_bellman_oracle_and_intention_changes_processor():
    src=fixture()
    ground=compile_intention_ground(src)
    states=[(),(('sense','read-type','0'),),
            (('sense','read-type','1'),),
            (('act','apply-0',''),),
            (('sense','read-type','1'),('act','apply-1',''))]
    for words in states:
        state=source_replay(src,words)
        session=IntentionSession(ground)
        for kind,name,label in words:
            if kind=='act':session.apply(name)
            else:session.observe(name,label)
        for horizon in range(4):
            expected=reference_plan(src,state,horizon)
            actual=session.plan(horizon)
            assert (actual.value,actual.success,actual.cost,actual.kind,actual.name)==expected
    # A new intention can require fewer predictive coordinates on same operations.
    by_type=tuple(Q(int(i//4==0)) for i in range(8))
    different=replace(src,goal=by_type)
    smaller=compile_intention_ground(different)
    assert smaller.rank==2
    assert smaller.ground_digest!=ground.ground_digest
    with pytest.raises(ValueError,match='commutation'):
        verify_intention_ground(different,ground)
    assert IntentionSession(smaller).goal_probability()==Q(1,2)
