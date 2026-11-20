"""TIME-003: model grammar construction, evidence-preserving migration and refusal."""
from copy import deepcopy
from hashlib import sha256
import unittest

from vli.ground import _digest
from vli.temporal_intelligence import TemporalIntelligenceGround
from vli.live_temporal_intelligence import LiveTemporalObserver
from vli.evolving_environment_language import (
    CAPSULE, MIGRATION, ANSWER, EvolvingEnvironmentLanguage, _inference,
    _interpret, _reference, certify_language_migration,
    verify_environment_language, verify_environment_answer,
    verify_language_migration,
)


def feed(g, values, *, host="lab-A", first=0):
    for i,value in enumerate(values,first):
        t=f"{i}/1"
        v=f"{value}/1"
        raw=f"{g.identity}|{t}|{v}".encode()
        g.observe(t,[v],source_digest=sha256(raw).hexdigest(),
                  host=host,expected_head=g.head)


def capsule(g):
    return EvolvingEnvironmentLanguage(g).current(expected_head=g.head)


class AdaptiveLanguageTests(unittest.TestCase):
    def test_affine_language_source_qualified(self):
        g=TemporalIntelligenceGround("lang:affine")
        feed(g,[1,3,5,7])
        p=capsule(g)
        self.assertEqual(p["family"],"affine")
        self.assertEqual(p["coefficients"],["1/1","2/1"])
        self.assertIn("ADD",p["vocabulary"])
        self.assertNotIn("POW",p["vocabulary"])
        self.assertEqual(p["training_samples"],2)
        self.assertEqual(p["heldout_samples"],2)
        self.assertTrue(verify_environment_language(g.checkpoint(),p,
            expected_head=g.head,expected_identity=g.identity))
        answer=EvolvingEnvironmentLanguage(g).navigate(
            p,"3/1",expected_head=g.head)
        self.assertEqual(answer["value"],"7/1")
        self.assertEqual(answer["status"],"RECORDED_SOURCE_COORDINATE")
        self.assertTrue(verify_environment_answer(g.checkpoint(),p,answer,
            expected_head=g.head,expected_identity=g.identity))

    def test_quadratic_environment_constructs_new_opcode(self):
        g=TemporalIntelligenceGround("lang:quadratic")
        feed(g,[0,1,4])
        self.assertIsNone(capsule(g))
        feed(g,[9],first=3)
        p=capsule(g)
        self.assertEqual(p["family"],"quadratic")
        self.assertIn("CHOOSE2",p["vocabulary"])
        self.assertEqual(p["heldout_samples"],1)
        a=EvolvingEnvironmentLanguage(g).navigate(
            p,"10/1",expected_head=g.head,allow_projection=True)
        self.assertEqual(a["value"],"100/1")
        self.assertEqual(a["status"],"UNVERIFIED_FUTURE_PROJECTION")
        self.assertTrue(verify_environment_answer(g.checkpoint(),p,a,
            expected_head=g.head,expected_identity=g.identity))

    def test_geometric_environment_constructs_power_opcode(self):
        g=TemporalIntelligenceGround("lang:geometric")
        feed(g,[2,6,18,54])
        p=capsule(g)
        self.assertEqual(p["family"],"geometric")
        self.assertIn("POW",p["vocabulary"])
        out=EvolvingEnvironmentLanguage(g).navigate(
            p,"8/1",expected_head=g.head,allow_projection=True)
        self.assertEqual(out["value"],f"{2*3**8}/1")

    def test_fibonacci_environment_constructs_recurrence_and_jumps(self):
        g=TemporalIntelligenceGround("lang:recurrence")
        feed(g,[1,1,2,3])
        self.assertIsNone(capsule(g))
        feed(g,[5,8],first=4)
        p=capsule(g)
        self.assertEqual(p["family"],"recurrence2")
        self.assertEqual(p["coefficients"],["1/1","1/1","1/1","1/1"])
        self.assertIn("RECURRENCE2",p["vocabulary"])
        self.assertGreaterEqual(p["heldout_samples"],1)
        far=EvolvingEnvironmentLanguage(g).navigate(
            p,"200/1",expected_head=g.head,allow_projection=True)
        self.assertEqual(far["value"],str(_reference("recurrence2",p["coefficients"],200).numerator)+"/1")
        self.assertLess(far["counted_operations"],160)
        self.assertEqual(far["expanded_source_transitions"],0)
        self.assertTrue(verify_environment_answer(g.checkpoint(),p,far,
            expected_head=g.head,expected_identity=g.identity))

    def test_only_source_interpretations_not_self_authenticating(self):
        g=TemporalIntelligenceGround("lang:heldout")
        feed(g,[2,6])
        self.assertIsNone(capsule(g))
        feed(g,[18],first=2)
        p=capsule(g)
        self.assertEqual(p["claim"],"EMPIRICAL_FIT_WITH_HELDOUT_SAMPLE_NOT_A_UNIVERSAL_LAW")
        with self.assertRaises(ValueError):
            EvolvingEnvironmentLanguage(g).navigate(p,"4/1",expected_head=g.head)
        with self.assertRaises(ValueError):
            EvolvingEnvironmentLanguage(g).navigate(p,"2/1",
                expected_head="a"*64)

    def test_qualified_linear_to_quadratic_language_preserves_history(self):
        g=TemporalIntelligenceGround("lang:changing-world")
        feed(g,[1,3,5,7])
        before=g.checkpoint()
        p_old=capsule(g)
        feed(g,[100,101,104,109],first=4,host="lab-B")
        after=g.checkpoint()
        p_new=capsule(g)
        self.assertEqual([s["family"] for s in EvolvingEnvironmentLanguage(g).segments()],
                         ["affine","quadratic"])
        self.assertEqual(p_old["family"],"affine")
        self.assertEqual(p_new["family"],"quadratic")
        self.assertEqual(p_new["epoch_first"],4)
        migration=certify_language_migration(
            before,p_old,after,p_new,
            before_head=before["head"],after_head=after["head"],identity=g.identity)
        self.assertEqual(migration["preserved_claim"],
                         "PREVIOUSLY_OBSERVED_SAMPLE_SCOPE_ONLY")
        self.assertEqual(migration["future_behavioral_equivalence"],
                         "NOT_ESTABLISHED")
        self.assertEqual(migration["retained_source_events"],4)
        self.assertTrue(verify_language_migration(
            before,p_old,after,p_new,migration,
            before_head=before["head"],after_head=after["head"],identity=g.identity))
        self.assertTrue(verify_environment_language(before,p_old,
            expected_head=before["head"],expected_identity=g.identity))
        self.assertTrue(verify_environment_language(after,p_new,
            expected_head=after["head"],expected_identity=g.identity))

    def test_migration_refuses_history_rewrite_and_rollback(self):
        g=TemporalIntelligenceGround("lang:tamper-migration")
        feed(g,[1,3,5,7])
        old=g.checkpoint()
        op=capsule(g)
        feed(g,[100,101,104,109],first=4)
        newer=g.checkpoint()
        np=capsule(g)
        with self.assertRaises(ValueError):
            certify_language_migration(newer,np,old,op,
                before_head=newer["head"],after_head=old["head"],identity=g.identity)
        forged=deepcopy(newer)
        forged["events"][1]["values"]=["999/1"]
        forged["checkpoint_digest"]=_digest(forged["schema"],{
            k:v for k,v in forged.items() if k!="checkpoint_digest"})
        with self.assertRaises(ValueError):
            certify_language_migration(old,op,forged,np,
                before_head=old["head"],after_head=newer["head"],identity=g.identity)

    def test_forged_program_even_rehashed_is_independently_rejected(self):
        g=TemporalIntelligenceGround("lang:forgery")
        feed(g,[1,1,2,3,5])
        p=capsule(g)
        f=deepcopy(p)
        f["program"][0]="ADD"
        f["capsule_digest"]=_digest(CAPSULE,{
            k:v for k,v in f.items() if k!="capsule_digest"})
        with self.assertRaises(ValueError):
            verify_environment_language(g.checkpoint(),f,
                expected_head=g.head,expected_identity=g.identity)
        a=EvolvingEnvironmentLanguage(g).navigate(
            p,"10/1",expected_head=g.head,allow_projection=True)
        a["value"]="999/1"
        a["answer_digest"]=_digest(ANSWER,{
            k:v for k,v in a.items() if k!="answer_digest"})
        with self.assertRaises(ValueError):
            verify_environment_answer(g.checkpoint(),p,a,
                expected_head=g.head,expected_identity=g.identity)

    def test_unresolved_environment_refuses_unearned_intelligence(self):
        g=TemporalIntelligenceGround("lang:outside-family")
        feed(g,[0,1,3,7,16,35])
        self.assertIsNone(capsule(g))
        self.assertTrue(any(s["status"]=="UNQUALIFIED"
                            for s in EvolvingEnvironmentLanguage(g).segments()))
        self.assertEqual(len(g.events),6)

    def test_irregular_real_time_samples_not_assumed_evenly_spaced(self):
        g=TemporalIntelligenceGround("lang:uneven-source")
        for t,x in (("0/1","1/1"),("1/2","2/1"),("2/1","5/1"),
                    ("5/2","10/1")):
            g.observe(t,[x],source_digest="a"*64,
                      host="sensor",expected_head=g.head)
        self.assertIsNone(capsule(g))
        self.assertEqual(len(g.events),4)

    def test_runtime_live_observer_and_language_are_same_source_identity(self):
        g=TemporalIntelligenceGround("lang:live-integration")
        live=LiveTemporalObserver(g,host="real-live-process")
        learner=EvolvingEnvironmentLanguage(g)
        histories=[]
        for i,x in enumerate([1,1,2,3,5,8]):
            datum={"schema":"vl-i.source-sample.v1",
                   "sequence":len(g.events),"time":f"{i}/1",
                   "values":[f"{x}/1"],
                   "source_digest":sha256(f"sensor:{i}:{x}".encode()).hexdigest()}
            decision=live.ingest(datum)
            histories.append((decision["identity"],learner.current(expected_head=g.head)))
        self.assertEqual(len({x[0] for x in histories}),1)
        self.assertTrue(all(x[1] is None for x in histories[:4]))
        self.assertEqual(histories[4][1]["family"],"recurrence2")
        self.assertEqual(live.accepted_in_lifetime,6)
        self.assertIs(learner.ground,live.ground)
        self.assertEqual(len(g.events),6)
        self.assertEqual(histories[-1][1]["heldout_samples"],2)

    def test_old_receipt_is_not_a_new_lineage_authority(self):
        g=TemporalIntelligenceGround("lang:stale")
        feed(g,[0,2,4])
        p=capsule(g)
        old_head=g.head
        feed(g,[6],first=3)
        with self.assertRaises(ValueError):
            EvolvingEnvironmentLanguage(g).navigate(
                p,"2/1",expected_head=old_head)
        self.assertNotEqual(p["capsule_digest"],capsule(g)["capsule_digest"])

    def test_cross_host_restore_recovers_qualified_interpreter(self):
        g=TemporalIntelligenceGround("lang:portable")
        feed(g,[1,1,2,3,5])
        snap=g.checkpoint()
        first=capsule(g)
        restored=TemporalIntelligenceGround.restore(
            snap,expected_head=g.head,expected_identity=g.identity)
        learner=EvolvingEnvironmentLanguage(restored)
        self.assertEqual(learner.current(expected_head=restored.head),first)
        feed(restored,[8],host="new-host",first=5)
        self.assertEqual(capsule(restored)["family"],"recurrence2")
        self.assertEqual(restored.identity,g.identity)


if __name__=="__main__":
    unittest.main()
