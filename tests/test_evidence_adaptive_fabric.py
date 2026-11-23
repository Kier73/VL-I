"""PBG-004: source-law revision from actual PBG-003 E-CPU/VGI evidence.

This imports PBG-003's genuine receiver-local SAB-001 and VGI fixtures,
including both receiver observers and the local material-effect handler.
"""
from __future__ import annotations

from fractions import Fraction as Q
import copy
import unittest

from mailcpu.substrate_bootstrap import BootstrapRefused
from vli.evidence_adaptive_ground import (
    AdaptiveGroundCouncil, GroundExpansionRequired, QualifiedLaw,
)
from vli.goal_evidence_fabric import Endpoint, execute
from vli.intention_predictive_ground import IntentionSession
from vli.predictive_basis_ground import _hash
from test_goal_evidence_fabric import world_and_ground, observed_fabric
from test_evidence_adaptive_ground import source_with_accuracy

SENSORS=(Endpoint('observer-A','observer-A'),Endpoint('observer-B','observer-B'))
ACT=(Endpoint('actuator-A','actuator-A'),)


def setup_case(*,two_laws=True,conflict=False):
    overrides={('read-type','observer-A'):1,
               ('read-type','observer-B'):0 if conflict else 1}
    anchor,world=world_and_ground(material_type=0,sensor_values=overrides)
    laws=[QualifiedLaw.from_source('reported-faithful',source_with_accuracy(Q(1)),Q(4,5))]
    if two_laws:
        laws.append(QualifiedLaw.from_source('sensor-inverted',source_with_accuracy(Q(0)),Q(1,5)))
    return AdaptiveGroundCouncil(anchor,tuple(laws)),world,observed_fabric(world)


def perform(council,runtime,kind,name,who,nonce):
    plan=council.propose(runtime,kind,name,who,nonce=nonce,coordinate=2)
    return plan,council.dispatch(plan,runtime)


class AdaptiveFabricIntegration(unittest.TestCase):
    def test_genuine_qualified_counterexample_changes_ground_without_reexecuting_effect(self):
        council,world,runtime=setup_case()
        before_rank=sum(law.ground.rank for law in council.laws.values())
        self.assertEqual(before_rank,8)
        observe_plan,observation=perform(council,runtime,'sense','read-type',SENSORS,'1'*64)
        self.assertEqual(observation['status'],'REVISION_ACCEPTED')
        self.assertEqual(council.active,'reported-faithful')
        self.assertEqual(len(world['actions']),0)
        self.assertEqual(council.plan(1).name,'apply-1')
        action_plan,action=perform(council,runtime,'act','apply-1',ACT,'2'*64)
        self.assertEqual(action['status'],'REVISION_ACCEPTED')
        self.assertEqual(world['revision'],1)
        self.assertEqual(world['success'],0)
        self.assertEqual(council.posterior,{'reported-faithful':Q(4,5),'sensor-inverted':Q(1,5)})
        read_plan,outcome=perform(council,runtime,'sense','read-goal',SENSORS,'3'*64)
        self.assertEqual(outcome['status'],'REVISION_ACCEPTED')
        self.assertEqual(outcome['eliminated'],['reported-faithful'])
        self.assertEqual(council.posterior['sensor-inverted'],1)
        self.assertEqual(council.goal_probability,0)
        self.assertEqual(council.plan(1).name,'apply-0')
        self.assertEqual(world['revision'],1)  # Evidence did not repeat effect.
        result_plan,new_action=perform(council,runtime,'act','apply-0',ACT,'4'*64)
        self.assertEqual(new_action['status'],'REVISION_ACCEPTED')
        self.assertEqual(world['success'],1)
        self.assertEqual(world['revision'],2)
        _,goal=perform(council,runtime,'sense','read-goal',SENSORS,'5'*64)
        self.assertEqual(goal['status'],'REVISION_ACCEPTED')
        self.assertEqual(council.goal_probability,1)
        saved=council.checkpoint()
        restored=AdaptiveGroundCouncil.restore(council.anchor.ground,
                      tuple(council.laws.values()),saved,
                      expected_digest=saved['checkpoint_digest'])
        self.assertEqual(restored.epoch,council.epoch)
        self.assertEqual(restored.plan(1),council.plan(1))
        self.assertEqual(restored.posterior,council.posterior)
        with self.assertRaises(ValueError):
            council.dispatch(action_plan,runtime)  # Stale after revision.

    def test_pending_impossible_evidence_preserved_until_new_law_is_admitted(self):
        council,world,runtime=setup_case(two_laws=False)
        perform(council,runtime,'sense','read-type',SENSORS,'6'*64)
        perform(council,runtime,'act','apply-1',ACT,'7'*64)
        action_count=world['revision']
        plan,result=perform(council,runtime,'sense','read-goal',SENSORS,'8'*64)
        self.assertEqual(result['status'],'EXPANSION_REQUIRED')
        self.assertEqual(council.pending[1]['digest'],result['evidence_digest'])
        self.assertEqual(world['revision'],action_count)
        with self.assertRaises(GroundExpansionRequired):
            council.propose(runtime,'act','apply-0',ACT,nonce='9'*64,coordinate=2)
        proposed=council.admit_source('sensor-inverted',source_with_accuracy(Q(0)),Q(1,5))
        self.assertEqual(proposed['status'],'GROUNDED_AFTER_EXPANSION')
        self.assertTrue(proposed['no_external_execution_repeated'])
        self.assertEqual(world['revision'],action_count)
        self.assertEqual(council.active,'sensor-inverted')
        self.assertEqual(council.plan(1).name,'apply-0')
        self.assertIsNone(council.pending)
        self.assertEqual(council.events[-1]['evidence_digest'],result['evidence_digest'])

    def test_vgi_conflict_does_not_change_posterior_or_admit_new_ground(self):
        council,world,runtime=setup_case(conflict=True)
        before=council.epoch
        with self.assertRaisesRegex(ValueError,'VGI|quorum'):
            perform(council,runtime,'sense','read-type',SENSORS,'a'*64)
        self.assertEqual(council.epoch,before)
        self.assertEqual(council.posterior['reported-faithful'],Q(4,5))
        self.assertEqual(world['revision'],0)

    def test_rehashed_fake_native_or_vgi_receipt_cannot_revive_wrong_law(self):
        council,world,runtime=setup_case()
        proposal=council.propose(runtime,'sense','read-type',SENSORS,
                                nonce='b'*64,coordinate=2)
        shadow=IntentionSession.restore(council.anchor.ground,proposal.scheduled.previous,
                     expected_digest=proposal.scheduled.previous['snapshot_digest'])
        real=execute(proposal.scheduled,shadow,runtime)
        forged=copy.deepcopy(real)
        forged['symbol']=0
        forged['digest']=_hash({k:v for k,v in forged.items() if k!='digest'})
        before=council.epoch
        with self.assertRaisesRegex(ValueError,'qualified'):
            council.accept(proposal,forged,runtime)
        self.assertEqual(council.epoch,before)
        self.assertEqual(council.accept(proposal,real,runtime)['status'],'REVISION_ACCEPTED')
        with self.assertRaises(ValueError):council.accept(proposal,real,runtime)

    def test_revoked_host_policy_blocks_old_plan_without_rewriting_ground(self):
        council,world,runtime=setup_case()
        plan=council.propose(runtime,'sense','read-type',SENSORS,
                             nonce='c'*64,coordinate=2)
        revoked=observed_fabric(world,deny={('sense','read-type','observer-B')})
        before=council.epoch
        with self.assertRaises((BootstrapRefused,ValueError)):
            council.dispatch(plan,revoked)
        self.assertEqual(council.epoch,before)
        self.assertEqual(world['revision'],0)


if __name__=='__main__':
    unittest.main()
