"""PBG-003: receiver-owned E-CPU execution and strict VGI source finalization.

Physical material here is a mutable local fixture, not authenticated hardware.
The actuator caches nonce results in memory only; it is not a durable remote
exactly-once protocol. This exercises the real imported E-CPU/VGI APIs.
"""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction as Q
import copy
import unittest

from mailcpu.substrate_bootstrap import (
    AdaptiveBootstrap, BootstrapPolicy, BootstrapRefused,
    CarrierBinding, ChannelObservation, OperationOffer, RealizerBinding,
    ResourceObservation, SubstrateObservation, digest,
)
from vli.intention_predictive_ground import IntentionSession, compile_intention_ground
from vli.goal_evidence_fabric import (
    Endpoint, INPUT_KIND, _op_name, _output_kind, prepare, execute,
    verify_evidence_record, verify_material_outcome,
)
from vli.predictive_basis_ground import _hash
from test_intention_predictive_ground import fixture

SENSORS=(Endpoint('observer-A','observer-A'),Endpoint('observer-B','observer-B'))
ACT=(Endpoint('actuator-A','actuator-A'),)


def world_and_ground(*, material_type=1, sensor_values=None):
    source=fixture()
    n=len(source.initial)
    outcomes=[]
    for label in (0,1):
        rows=[]
        for i in range(n):
            row=[Q()]*n
            if source.goal[i]==label:row[i]=1
            rows.append(tuple(row))
        outcomes.append((str(label),tuple(rows)))
    source=replace(source,sensors=source.sensors+(
        ('read-goal',tuple(outcomes),Q(1,100)),))
    ground=compile_intention_ground(source)
    world={'type':material_type,'success':0,'revision':0,
           'actions':{},'overrides':dict(sensor_values or {})}
    return ground,world


def observed_fabric(world,*,deny=(),expires=30):
    """Admit exact typed source-specific paths using native SAB-001."""
    operators=[]
    for sensor in ('read-type','read-goal'):
        for p in SENSORS:operators.append(('sense',sensor,p))
    for name in ('apply-0','apply-1'):
        operators.append(('act',name,ACT[0]))

    def sensed(name,p):
        if name=='read-goal':return world['success']
        return world['overrides'].get((name,p.source_id),world['type'])

    offers={p.locus:[] for _,_,p in operators}
    channels=[]
    for kind,name,p in operators:
        op=_op_name(kind,name,p.source_id)
        output=_output_kind(kind,name,p.source_id)
        offers[p.locus].append(OperationOffer(op,INPUT_KIND,output,1))
        channels.append(ChannelObservation(
            f"to-{kind}-{name}-{p.source_id}",p.ingress,p.locus,
            (INPUT_KIND,),1,0,expires,
            digest({'from':p.ingress,'to':p.locus,'kind':kind,'name':name})))
        channels.append(ChannelObservation(
            f"from-{kind}-{name}-{p.source_id}",p.locus,p.collector,
            (output,),1,0,expires,
            digest({'from':p.locus,'to':p.collector,'kind':kind,'name':name})))
    inventory=SubstrateObservation(tuple(
        ResourceObservation(resource,'local-fixture',0,expires,'receiver',
            digest({'locus':resource}),tuple(offers.get(resource,())))
        for resource in ('start','collector','observer-A','observer-B','actuator-A')
    ),tuple(channels))
    realizers=[]
    for kind,name,p in operators:
        op=_op_name(kind,name,p.source_id)
        def handler(value,kind=kind,name=name,p=p):
            expected={'request_digest':digest(value),'source_id':p.source_id,
                      'operation':name,'nonce':value['nonce']}
            if kind=='sense':
                return {**expected,'symbol':sensed(name,p)}
            # Local idempotency: same actuation nonce cannot change state twice.
            saved=world['actions'].get(value['nonce'])
            if saved is not None:return saved
            world['success']=int(world['type']==int(name[-1]))
            world['revision']+=1
            reply={**expected,'ack':True,'material_revision':world['revision']}
            world['actions'][value['nonce']]=reply
            return reply

        def oracle(value,result,kind=kind,name=name,p=p):
            expected={'request_digest':digest(value),'source_id':p.source_id,
                      'operation':name,'nonce':value['nonce']}
            if any(result.get(k)!=v for k,v in expected.items()):return False
            if kind=='sense':
                return result.get('symbol')==sensed(name,p)
            return (world['actions'].get(value['nonce'])==result and
                    result.get('material_revision')==world['revision'] and
                    world['success']==int(world['type']==int(name[-1])))

        realizers.append(RealizerBinding(p.locus,op,handler,
                            digest({'binding':op}),oracle))
    carriers=tuple(CarrierBinding(
        c.channel_id,lambda payload,coordinate:payload,
        digest({'byte-carrier':c.channel_id})) for c in channels)
    policy=BootstrapPolicy(
        frozenset((p.locus,_op_name(kind,name,p.source_id))
                  for kind,name,p in operators if
                  (kind,name,p.source_id) not in deny),
        frozenset(c.channel_id for c in channels),'pbg003-receiver')
    return AdaptiveBootstrap(inventory,policy,realizers,carriers)


def take(session,fabric,kind,name,endpoints,nonce,coordinate=2):
    plan=prepare(session,fabric,kind,name,endpoints,
                 nonce=nonce,coordinate=coordinate)
    return plan,execute(plan,session,fabric)


class GoalEvidenceTests(unittest.TestCase):
    def test_actual_route_observation_intervention_and_goal_readback(self):
        ground,world=world_and_ground()
        self.assertEqual(ground.rank,4)
        fabric=observed_fabric(world)
        session=IntentionSession(ground)
        self.assertEqual(session.plan(2).name,'read-type')
        prior=session.snapshot()
        sensor_plan,sensor=take(session,fabric,'sense','read-type',SENSORS,'1'*64)
        self.assertTrue(verify_evidence_record(ground,sensor_plan,fabric,sensor))
        self.assertEqual(sensor['symbol'],1)
        self.assertNotEqual(prior,session.snapshot())
        self.assertEqual(world['revision'],0)  # sensor is passive
        self.assertEqual(session.plan(1).name,'apply-1')
        action_plan,action=take(session,fabric,'act','apply-1',ACT,'2'*64)
        self.assertTrue(verify_evidence_record(ground,action_plan,fabric,action))
        self.assertEqual(world['revision'],1)
        self.assertEqual(world['success'],1)
        self.assertEqual(session.goal_probability(),Q(3,4))  # prediction only
        post_plan,post=take(session,fabric,'sense','read-goal',SENSORS,'3'*64)
        self.assertEqual(post['symbol'],1)
        self.assertEqual(session.goal_probability(),Q(1))
        outcome=verify_material_outcome(ground,action_plan,post_plan,fabric,action,post)
        self.assertEqual(outcome['status'],'OBSERVED_GOAL')
        self.assertEqual(len(post['vgi_ledger']['evidence']),2)

    def test_two_valid_but_disagreeing_readings_do_not_enter_ground(self):
        ground,world=world_and_ground(
            sensor_values={('read-type','observer-B'):0})
        session=IntentionSession(ground)
        prior=session.snapshot()
        fabric=observed_fabric(world)
        plan=prepare(session,fabric,'sense','read-type',SENSORS,
                     nonce='4'*64,coordinate=2)
        with self.assertRaisesRegex(ValueError,'VGI|quorum'):
            execute(plan,session,fabric)
        self.assertEqual(prior,session.snapshot())
        self.assertEqual(world['revision'],0)

    def test_predicted_success_and_observed_failure_are_different_states(self):
        ground,world=world_and_ground(
            material_type=0,sensor_values={
                ('read-type','observer-A'):1,('read-type','observer-B'):1})
        fabric=observed_fabric(world)
        session=IntentionSession(ground)
        take(session,fabric,'sense','read-type',SENSORS,'5'*64)
        self.assertEqual(session.plan(1).name,'apply-1')
        action_plan,action=take(session,fabric,'act','apply-1',ACT,'6'*64)
        self.assertEqual(world['success'],0)
        self.assertEqual(session.goal_probability(),Q(3,4))
        goal_plan,goal=take(session,fabric,'sense','read-goal',SENSORS,'7'*64)
        result=verify_material_outcome(ground,action_plan,goal_plan,fabric,action,goal)
        self.assertEqual(result['status'],'OBSERVED_NON_GOAL')
        self.assertEqual(session.goal_probability(),Q(0))

    def test_predictive_state_restores_before_actuator_execution(self):
        ground,world=world_and_ground()
        fabric=observed_fabric(world)
        session=IntentionSession(ground)
        take(session,fabric,'sense','read-type',SENSORS,'8'*64)
        snap=session.snapshot()
        restored=IntentionSession.restore(ground,snap,
                                         expected_digest=snap['snapshot_digest'])
        self.assertEqual(restored.plan(1),session.plan(1))
        act_plan,action=take(restored,fabric,'act','apply-1',ACT,'9'*64)
        self.assertEqual(world['revision'],1)
        self.assertTrue(verify_evidence_record(ground,act_plan,fabric,action))
        # Host-local nonce idempotence on same task, not a universal guarantee.
        again=fabric.execute(act_plan.tasks[0],act_plan.routes[0],coordinate=2)
        self.assertEqual(again['output'],action['path_reports'][0]['output'])
        self.assertEqual(world['revision'],1)
        with self.assertRaisesRegex(ValueError,'Stale'):
            execute(act_plan,restored,fabric)

    def test_rehashed_wrong_report_and_vgi_receipt_are_refused(self):
        ground,world=world_and_ground()
        fab=observed_fabric(world)
        session=IntentionSession(ground)
        plan,record=take(session,fab,'sense','read-type',SENSORS,'a'*64)
        forged=copy.deepcopy(record)
        forged['path_reports'][0]['output']['symbol']=0
        forged['path_reports'][0]['output_digest']=digest(
            forged['path_reports'][0]['output'])
        forged['path_reports'][0]['report_digest']=digest({
            k:v for k,v in forged['path_reports'][0].items()
            if k!='report_digest'})
        forged['digest']=_hash({k:v for k,v in forged.items() if k!='digest'})
        self.assertFalse(verify_evidence_record(ground,plan,fab,forged))
        forged=copy.deepcopy(record)
        forged['vgi_ledger']['policy']['context_digest']='0'*64
        forged['vgi_ledger_digest']=_hash(forged['vgi_ledger'])
        forged['digest']=_hash({k:v for k,v in forged.items() if k!='digest'})
        self.assertFalse(verify_evidence_record(ground,plan,fab,forged))
        forged=copy.deepcopy(record)
        forged['symbol']=0
        forged['digest']=_hash({k:v for k,v in forged.items() if k!='digest'})
        self.assertFalse(verify_evidence_record(ground,plan,fab,forged))

    def test_revoked_host_or_stale_request_refuses(self):
        ground,world=world_and_ground()
        session=IntentionSession(ground)
        fab=observed_fabric(world)
        plan=prepare(session,fab,'sense','read-type',SENSORS,
                     nonce='b'*64,coordinate=2)
        revoked=observed_fabric(
            world,deny={('sense','read-type','observer-B')})
        with self.assertRaisesRegex(ValueError,'Stale'):
            execute(plan,session,revoked)
        with self.assertRaises(BootstrapRefused):
            prepare(session,revoked,'sense','read-type',SENSORS,
                    nonce='b'*64,coordinate=2)
        with self.assertRaises(ValueError):
            prepare(session,fab,'act','unrecognized',ACT,
                    nonce='c'*64,coordinate=2)
        with self.assertRaises(ValueError):
            prepare(session,fab,'sense','read-type',SENSORS,
                    nonce='old-nonce',coordinate=2)

    def test_wrong_causal_link_cannot_prove_material_realization(self):
        ground,world=world_and_ground()
        fab=observed_fabric(world)
        session=IntentionSession(ground)
        _,first=take(session,fab,'sense','read-type',SENSORS,'d'*64)
        act_plan,act=take(session,fab,'act','apply-1',ACT,'e'*64)
        goal_plan,goal=take(session,fab,'sense','read-goal',SENSORS,'f'*64)
        false=copy.deepcopy(goal)
        false['previous']=first['previous']
        false['digest']=_hash({k:v for k,v in false.items() if k!='digest'})
        with self.assertRaisesRegex(ValueError,'Unqualified'):
            verify_material_outcome(ground,act_plan,goal_plan,fab,act,false)


if __name__=="__main__":
    unittest.main()
