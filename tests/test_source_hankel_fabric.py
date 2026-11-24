"""PBG-005: native E-CPU SAB-001 + VGI two-source trace queries.

Source kernels are in receiver-installed closures, not in the Hankel learner.
The E-CPU host bindings are receiver-local; no remote TLS or hardware here.
"""
from __future__ import annotations
from fractions import Fraction as Q
from itertools import product
import unittest

from mailcpu.substrate_bootstrap import (
    AdaptiveBootstrap,BootstrapPolicy,BootstrapRefused,
    CarrierBinding,ChannelObservation,OperationOffer,RealizerBinding,
    ResourceObservation,SubstrateObservation,digest as ecpu_hash,
)
from vli.source_hankel_ground import (
    QueryAtlas,HankelLearner,HankelObserverSession,certify_source_equivalence,
    verify_source_closure,source_trace_probability,encode,
)
from vli.source_hankel_fabric import (
    NativeSourceReader,ProbeEndpoint,operation_id,output_kind,INPUT,
)
from test_source_hankel_ground import delayed_source,alphabet

PEERS=(ProbeEndpoint('host-A','host-A'),ProbeEndpoint('host-B','host-B'))


def admitted_sources(source, *,wrong_source=None,deny=(),expiration=20):
    """Two distinct receiver-bound hosted source readers and byte carriers."""
    resources=[];channels=[];bindings=[]
    for locus in ('start','collector','host-A','host-B'):
        offers=() if not locus.startswith('host-') else (
            OperationOffer(operation_id(locus),INPUT,output_kind(locus),2),)
        resources.append(ResourceObservation(locus,'receiver-source-probe',0,expiration,
                        'receiver',ecpu_hash({'source':locus}),offers))
    for peer in PEERS:
        channels.append(ChannelObservation('send-'+peer.locus,
            'start',peer.locus,(INPUT,),1,0,expiration,
            ecpu_hash({'src':'start','dst':peer.locus}),max_payload_bytes=65536))
        channels.append(ChannelObservation('recv-'+peer.locus,
            peer.locus,'collector',(output_kind(peer.source_id),),1,0,expiration,
            ecpu_hash({'src':peer.locus,'dst':'collector'}),max_payload_bytes=65536))
        def reader(value,peer=peer):
            current=wrong_source if (wrong_source is not None and peer.source_id=='host-B') else source
            values=[encode(source_trace_probability(current,tuple(w)))
                    for w in value['words']]
            return {'request_digest':ecpu_hash(value),
                    'source_id':peer.source_id,
                    'source_digest':value['source_digest'],
                    'question_digest':value['question_digest'],
                    'values':values}
        def oracle(value,result,peer=peer):
            expected={'request_digest':ecpu_hash(value),
                      'source_id':peer.source_id,'source_digest':source.digest,
                      'question_digest':value['question_digest']}
            if not isinstance(result,dict) or any(result.get(k)!=v for k,v in expected.items()):
                return False
            # Independent receiver oracle checks source-specific result when
            # honest; the test's wrong_source exercises only shape checks for
            # the compromised peer to make VGI's quorum essential.
            if wrong_source is not None and peer.source_id=='host-B':
                return isinstance(result.get('values'),list)
            return result.get('values')==[
                encode(source_trace_probability(source,tuple(w))) for w in value['words']]
        bindings.append(RealizerBinding(peer.locus,operation_id(peer.source_id),
                                        reader,ecpu_hash({'handler':peer.source_id}),oracle))
    inventory=SubstrateObservation(tuple(resources),tuple(channels))
    policy=BootstrapPolicy(frozenset((p.locus,operation_id(p.source_id)) for p in PEERS
                         if p.source_id not in deny),
                           frozenset(c.channel_id for c in channels),
                           'pbg005-receiver')
    carrier=tuple(CarrierBinding(c.channel_id,lambda data,coordinate:data,
                         ecpu_hash({'carrier':c.channel_id})) for c in channels)
    return AdaptiveBootstrap(inventory,policy,bindings,carrier)


class SourceFabricTests(unittest.TestCase):
    def test_native_source_learner_enlarges_ground_after_qualified_counterexample(self):
        source=delayed_source(2)
        fab=admitted_sources(source)
        reader=NativeSourceReader(fab,PEERS,source.digest,alphabet(),batch_size=32,
                                   nonce_factory=lambda:'1'*64)
        atlas=QueryAtlas(alphabet(),source.digest,reader,max_queries=1300)
        learner=HankelLearner(atlas)
        original=learner.build()
        self.assertEqual(original.rank,1)
        witness=certify_source_equivalence(source,original)
        self.assertEqual(witness['status'],'COUNTEREXAMPLE')
        new=learner.counterexample(witness['word'])
        self.assertEqual(new.rank,3)
        self.assertTrue(verify_source_closure(source,new))
        self.assertEqual(certify_source_equivalence(source,new)['status'],'EQUIVALENT_ALL_WORDS')
        self.assertGreater(len(reader.lineage),0)
        self.assertTrue(all(x['quorum']=='AUTHORIZED' for x in reader.lineage))
        self.assertTrue(all(x['source_ids']==('host-A','host-B') for x in reader.lineage))
        self.assertTrue(all(x['word_count']<=32 for x in reader.lineage))
        self.assertEqual(atlas.calls,3) # counterexample expansion requires separate Hankel and shifted-Hankel reads
        self.assertEqual(sum(x['word_count'] for x in reader.lineage),len(atlas.cache))
        session=HankelObserverSession(new,source)
        self.assertEqual(session.mass('sense:goal:1'),0)
        session.apply('step')
        self.assertEqual(session.mass('sense:goal:1'),0)
        session.apply('step')
        self.assertEqual(session.mass('sense:goal:1'),1)
        self.assertEqual(session.observe('goal','1'),1)
        self.assertEqual(session.snapshot()['ground'],new.ground_digest)

    def test_bad_peer_reading_conflicts_before_ground_admission(self):
        source=delayed_source(2)
        bad=delayed_source(3)
        fab=admitted_sources(source,wrong_source=bad)
        reader=NativeSourceReader(fab,PEERS,source.digest,alphabet(),batch_size=32,
                                   nonce_factory=lambda:'2'*64)
        # Both hosts agree for short words; the fourth action reveals the liar.
        with self.assertRaisesRegex(ValueError,'disagree|quorum'):
            reader((('act:step','act:step','sense:goal:1'),))

    def test_revoked_or_expired_source_host_refuses_before_source_measurement(self):
        source=delayed_source()
        fab=admitted_sources(source,deny=('host-B',))
        reader=NativeSourceReader(fab,PEERS,source.digest,alphabet(),
                                   nonce_factory=lambda:'3'*64)
        with self.assertRaises(BootstrapRefused):reader(((),))
        fab=admitted_sources(source,expiration=1)
        reader=NativeSourceReader(fab,PEERS,source.digest,alphabet(),coordinate=5,
                                   nonce_factory=lambda:'4'*64)
        with self.assertRaises(BootstrapRefused):reader(((),))

    def test_pbg004_pending_qualified_counterexample_admits_new_source_derived_law(self):
        """Full PBG-004 material event -> source query -> newly admitted law."""
        from test_evidence_adaptive_fabric import (
            setup_case,perform,SENSORS as WORLD_SENSORS,ACT as WORLD_ACT,
        )
        from test_evidence_adaptive_ground import source_with_accuracy
        from vli.evidence_adaptive_ground import GroundExpansionRequired
        council,world,physical=setup_case(two_laws=False)
        perform(council,physical,'sense','read-type',WORLD_SENSORS,'a'*64)
        perform(council,physical,'act','apply-1',WORLD_ACT,'b'*64)
        before_action_revision=world['revision']
        _,outcome=perform(council,physical,'sense','read-goal',WORLD_SENSORS,'c'*64)
        self.assertEqual(outcome['status'],'EXPANSION_REQUIRED')
        with self.assertRaises(GroundExpansionRequired):council.plan(1)
        source=source_with_accuracy(Q(0))
        symbol_alphabet=tuple('act:'+name for name,_,_ in source.actions)+tuple(
            'sense:'+name+':'+label for name,outs,_ in source.sensors for label,m in outs)
        reader=NativeSourceReader(admitted_sources(source),PEERS,source.digest,
                                  symbol_alphabet,batch_size=128)
        atlas=QueryAtlas(symbol_alphabet,source.digest,reader,max_queries=3500)
        learner=HankelLearner(atlas)
        candidate=learner.build()
        for _ in range(len(source.initial)):
            proof=certify_source_equivalence(source,candidate)
            if proof['status']=='EQUIVALENT_ALL_WORDS':break
            candidate=learner.counterexample(proof['word'])
        else:self.fail('The observed law failed exact all-word qualification')
        self.assertEqual(world['revision'],before_action_revision)
        admitted=council.admit_source('source-derived-inverted',source,Q(1,5))
        self.assertEqual(admitted['status'],'GROUNDED_AFTER_EXPANSION')
        self.assertTrue(admitted['no_external_execution_repeated'])
        self.assertEqual(world['revision'],before_action_revision)
        self.assertEqual(council.plan(1).name,'apply-0')
        # A second, independently source-derived interpreter can reconstruct
        # council's complete evidence history without rerunning the actuator.
        replay=HankelObserverSession(candidate,source)
        replay.replay_events(council.events)
        self.assertEqual(replay.goal_probability(),council.goal_probability)
        self.assertEqual(replay.goal_probability(),0)
        replay.apply('apply-0')
        self.assertEqual(replay.goal_probability(),1)
        self.assertEqual(world['revision'],before_action_revision)

    def test_prior_history_replays_in_source_derived_ground_without_material_reexecution(self):
        from test_evidence_adaptive_ground import source_with_accuracy
        from vli.intention_predictive_ground import IntentionSession,compile_intention_ground
        source=source_with_accuracy(Q(0))
        events=('act:apply-0','act:apply-1')
        symbol_alphabet=tuple('act:'+name for name,_,_ in source.actions)+tuple(
            'sense:'+name+':'+label for name,outs,_ in source.sensors for label,m in outs)
        fab=admitted_sources(source)
        reader=NativeSourceReader(fab,PEERS,source.digest,symbol_alphabet,batch_size=128)
        atlas=QueryAtlas(symbol_alphabet,source.digest,reader,max_queries=3500)
        learner=HankelLearner(atlas)
        candidate=learner.build()
        ranks=[candidate.rank]
        for _ in range(len(source.initial)):
            proof=certify_source_equivalence(source,candidate)
            if proof['status']=='EQUIVALENT_ALL_WORDS':break
            candidate=learner.counterexample(proof['word'])
            ranks.append(candidate.rank)
        else: self.fail('Source rank did not close after bounded counterexamples')
        session=HankelObserverSession(candidate,source)
        past=[{'kind':'sense','operation':'read-type','label':'1'},
              {'kind':'act','operation':'apply-1','label':''},
              {'kind':'sense','operation':'read-goal','label':'0'}]
        session.replay_events(past)
        self.assertEqual(session.goal_probability(),0)
        session.apply('apply-0')
        self.assertEqual(session.goal_probability(),1)
        oracle=IntentionSession(compile_intention_ground(source))
        for x in past:
            if x['kind']=='sense': oracle.observe(x['operation'],x['label'])
            else: oracle.apply(x['operation'])
        oracle.apply('apply-0')
        self.assertEqual(oracle.goal_probability(),session.goal_probability())
        self.assertLessEqual(candidate.rank,len(source.initial))
        self.assertGreaterEqual(candidate.rank,2)
        self.assertTrue(reader.lineage)

if __name__=='__main__':unittest.main()