"""PBG-002: exact-rational goal-conditioned observation/intervention lab.

PYTHONPATH=src python -m examples.adaptive_intention_lab
All material actions below are source-model simulations, not external effects.
"""
from fractions import Fraction as Q
import json
from math import log2

from vli.intention_predictive_ground import ControlledSource, IntentionSession, compile_intention_ground


def example():
    replicas=2; n=4*replicas
    idx=lambda t,s,r:(2*t+s)*replicas+r
    initial=[Q()]*n
    for typ in (0,1):
        initial[idx(typ,0,0)]=Q(2,5)
        initial[idx(typ,1,0)]=Q(1,10)
    goal=tuple(Q((i//replicas)%2) for i in range(n))
    actions=[]
    for chosen in (0,1):
        rows=[]
        for i in range(n):
            t=i//(2*replicas);tag=i%replicas
            row=[Q()]*n
            row[idx(t,int(t==chosen),tag)]=1
            rows.append(tuple(row))
        actions.append((f'apply-{chosen}',tuple(rows),Q(1,100)))
    sensors=[]
    for name,accuracy in (('read-type',Q(3,4)),('uninformative-coin',Q(1,2))):
        outcomes=[]
        for reading in ('0','1'):
            rows=[]
            for i in range(n):
                typ=i//(2*replicas)
                probability=accuracy if typ==int(reading) else 1-accuracy
                row=[Q()]*n;row[i]=probability
                rows.append(tuple(row))
            outcomes.append((reading,tuple(rows)))
        sensors.append((name,tuple(outcomes),Q(1,20)))
    return ControlledSource(tuple(initial),goal,tuple(actions),tuple(sensors))


def run():
    source=example();ground=compile_intention_ground(source)
    s=IntentionSession(ground)
    direct=s.plan(1); adaptive=s.plan(2)
    assert (ground.rank,direct.success,adaptive.success,adaptive.value)==(4,Q(1,2),Q(3,4),Q(69,100))
    branches=[]
    for label,prob,kind,action in adaptive.branches:
        x=IntentionSession(ground)
        x.observe('read-type',label)
        plan=x.plan(1)
        assert plan.name==action
        x.apply(action)
        branches.append({'observed_type_signal':label,'probability':str(prob),
                         'chosen_intervention':action,'predicted_goal_probability':str(x.goal_probability()),
                         'reconstruction':x.snapshot()['snapshot_digest']})
    def surpris(p):
        return -log2(float(p)) if p>0 else float('inf')
    return {'experiment':'PBG-002','source_states':len(source.initial),
            'virtual_ground_rank':ground.rank, 'source_digest':source.digest,
            'ground_digest':ground.ground_digest,
            'prior_goal_probability':str(s.goal_probability()),
            'direct':{'success':str(direct.success),'expected_cost':str(direct.cost),
                      'value':str(direct.value),'goal_surprisal_bits':surpris(direct.success)},
            'adaptive':{'first_operation':adaptive.kind+':'+adaptive.name,
                        'success':str(adaptive.success),'expected_cost':str(adaptive.cost),
                        'value':str(adaptive.value),'goal_surprisal_bits':surpris(adaptive.success)},
            'observation_branches':branches,
            'scope':'exact finite controlled source; observation and intervention simulated; no external physical action or LLM inference'}


if __name__=='__main__':
    print(json.dumps(run(),indent=2,sort_keys=True))
