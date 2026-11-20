"""SHAPE-003: real IFS black-box learner + counterexamples -> VLD neural Grounds.

The candidate constructor can see only the IFS membership ledger, never the
reference transition table. The external finite source verifier independently
finds shortest behavioral counterexamples and qualifies the final candidate.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from vli.learned_morphology import (
    BlackBoxObserver, LearnedObserverGround, LearnedMorphologyConversation,
    independent_finite_verdict,
)
from partial_observation import FrozenOracle, recover

RAW=(Path(__file__).resolve().parents[1] /
     "experiments/shape002/morphology_model.json")


class TestObservedCounterexampleLearning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=RAW.read_bytes()
        cls.phase=LearnedObserverGround(cls.raw,("phase",))
        cls.micro=LearnedObserverGround(cls.raw,("micro",))
        cls.full=LearnedObserverGround(cls.raw,("phase","micro"))

    def test_three_qualifications_are_truly_learned_not_supplied_transitions(self):
        for model,states in ((self.phase,3),(self.micro,4),(self.full,12)):
            with self.subTest(channels=model.channels):
                self.assertTrue(model.verdict["exact"])
                self.assertEqual(len(model.machine.representatives),states)
                self.assertEqual(len(model.neural_ground.neural.hidden_bias),states*3)
                self.assertEqual(model.certificate["candidate_count"],
                                 len(model.revision_history))
                self.assertTrue(model.check(
                    expected_certificate_digest=model.certificate["proof_digest"]))
                self.assertNotEqual(model.machine.fingerprint,
                                    model.reference.fingerprint)
                self.assertEqual(model.ifs_checkpoint["ground"]["global_equivalence"],
                                 "not_established")
                self.assertEqual(model.ifs_checkpoint["ground"]["status"],
                                 "provisional_finitely_challenged")

    def test_underfit_initial_hypothesis_is_falsified_with_real_counterexample(self):
        for model in (self.phase,self.micro,self.full):
            with self.subTest(channels=model.channels):
                self.assertFalse(model.revision_history[0]["exact"])
                self.assertEqual(model.revision_history[0]["candidate_states"],1)
                self.assertTrue(model.revision_history[0]["counterexample"])
                self.assertTrue(model.revision_history[-1]["exact"])
                self.assertGreater(len(model.revision_history),1)
                self.assertGreater(model.oracle.external_calls,0)
                self.assertEqual(len(model.ledger["witnesses"]),
                                 model.oracle.external_calls)

    def test_whole_declared_domain_and_multiturn_sequences_equal_hidden_source(self):
        for model in (self.phase,self.micro,self.full):
            ref=model.reference
            pos=tuple(ref.channels.index(c) for c in model.channels)
            for seq in (
                ("inspect",),("advance",),("advance","inspect"),
                ("advance","advance","reset","advance"),
                ("advance",)*11+("inspect",),
                ("advance",)*20+("reset","advance"),
            ):
                s=ref.initial_state
                expected=[]
                for action in seq:
                    value=ref.step(s,action)
                    s=value.target
                    expected.append(tuple(value.output[i] for i in pos))
                predicted=model.machine.run(seq)
                self.assertEqual(predicted,tuple(expected))
                self.assertEqual(
                    tuple(model.neural_ground.generated(
                        f"q{model.machine.state_after(seq[:i]):03d}",a)[1]
                        for i,a in enumerate(seq)),tuple(expected))

    def test_source_free_ifs_ledger_recovery_is_real_not_a_new_oracle_read(self):
        model=self.phase
        a=model.ifs_checkpoint
        restored,frozen=recover(
            a,expected_checkpoint_digest=a["digest"],
            expected_ledger_head=model.oracle.head,
            expected_source_root=model.source.source_root)
        self.assertEqual(restored.document(),model.machine.document())
        self.assertEqual(frozen.external_calls,0)
        self.assertEqual(frozen.query(model.oracle.witnesses[-1].path),
                         model.oracle.witnesses[-1].outputs)

    def test_partial_candidate_cannot_claim_complete_model(self):
        with self.assertRaises(ValueError):
            LearnedObserverGround(self.raw,("phase",),max_revisions=1)
        with self.assertRaises(ValueError):
            LearnedObserverGround(self.raw,("phase",),max_membership=8,
                                  max_revisions=32)

    def test_source_change_revisions_change_certified_model_root(self):
        source=json.loads(self.raw)
        item=next(i for i in source["observations"]
                  if i["source"]=="p1m3" and i["action"]=="inspect")
        item["outcomes"][0]["output"][0]=7
        changed=LearnedObserverGround(
            json.dumps(source,sort_keys=True).encode("utf-8"),("phase",))
        self.assertTrue(changed.verdict["exact"])
        self.assertGreater(len(changed.machine.representatives),3)
        self.assertNotEqual(changed.manifest["ground_digest"],
                            self.phase.manifest["ground_digest"])
        self.assertNotEqual(changed.source_digest,self.phase.source_digest)

    def test_unseen_observer_channel_not_admitted_by_previous_quotient(self):
        phase=self.phase
        self.assertEqual(phase.channels,("phase",))
        with self.assertRaises(ValueError):
            LearnedObserverGround(self.raw,("temperature",))
        self.assertNotIn("micro",phase.neural_ground.system.channels)

    def test_ledger_tamper_refused_even_when_external_head_mismatched(self):
        model=self.phase
        saved=copy.deepcopy(model.ifs_checkpoint)
        saved["ledger"]["witnesses"][0]["outputs"][0][0]=777
        with self.assertRaises(Exception):
            recover(saved,expected_checkpoint_digest=model.ifs_checkpoint["digest"],
                    expected_ledger_head=model.oracle.head,
                    expected_source_root=model.source.source_root)

    def test_qualified_ground_tamper_and_expected_certificate_refusal(self):
        m=self.phase
        with self.assertRaises(ValueError):
            m.check(expected_certificate_digest="0"*64)
        old=m.manifest["generated_weights_digest"]
        try:
            m.manifest["generated_weights_digest"]="0"*64
            with self.assertRaises(ValueError):
                m.check(expected_certificate_digest=m.certificate["proof_digest"])
        finally:
            m.manifest["generated_weights_digest"]=old

    def test_finite_reference_conformance_cost_is_explicit(self):
        for m in (self.phase,self.micro,self.full):
            self.assertGreater(m.verdict["checked_product_edges"],0)
            self.assertGreater(m.learning_ns,0)
            self.assertEqual(m.oracle.head,m.certificate["ifs_ledger_head"])
            self.assertGreaterEqual(m.source.direct_transition_steps,
                                    m.oracle.external_calls)


class TestCausalShapeRevision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw=RAW.read_bytes()
        cls.phase=LearnedObserverGround(raw,("phase",))
        cls.full=LearnedObserverGround(raw,("phase","micro"))
        cls.micro=LearnedObserverGround(raw,("micro",))

    def test_change_observer_without_teleporting_or_erasing_hidden_causal_state(self):
        s=LearnedMorphologyConversation((self.phase,self.micro,self.full))
        a=s.respond(self.phase,prompt="Show only phase",action="advance")
        b=s.respond(self.micro,prompt="Now what microstate?",action="inspect")
        c=s.respond(self.full,prompt="Reveal both",action="advance")
        self.assertEqual(a["event"]["observed_values"],[1])
        self.assertEqual(b["event"]["observed_values"],[1])
        self.assertEqual(c["event"]["observed_values"],[2,2])
        self.assertEqual(s.history,("advance","inspect","advance"))
        self.assertEqual(len({x["event"]["position"] for x in (a,b,c)}),3)
        self.assertTrue(all(
            len(x["event"]["position"])==76 for x in (a,b,c)))
        for x in (a,b,c):
            view=x["response"]["projections"]
            self.assertEqual([v["kind"] for v in view],["text","code","tokens"])
            self.assertEqual(bytes(view[-1]["content"]).decode(),view[0]["content"])

    def test_restore_from_witnessed_learned_models_and_continue(self):
        s=LearnedMorphologyConversation((self.phase,self.full))
        s.respond(self.phase,prompt="Phase",action="advance",views=("text",))
        s.respond(self.full,prompt="Both",action="advance",views=("code",))
        saved=s.checkpoint()
        r=LearnedMorphologyConversation.restore(
            (self.phase,self.full),saved,expected_head=saved["checkpoint_digest"])
        self.assertEqual(r.checkpoint(),saved)
        self.assertEqual(r.history,s.history)
        a=s.respond(self.phase,prompt="Phase again",action="inspect",views=("page",))
        b=r.respond(self.phase,prompt="Phase again",action="inspect",views=("page",))
        self.assertEqual(a["event"],b["event"])

    def test_forged_history_and_cross_source_restore_refused(self):
        s=LearnedMorphologyConversation((self.phase,self.full))
        s.respond(self.phase,prompt="Phase",action="advance")
        saved=s.checkpoint()
        altered=copy.deepcopy(saved)
        altered["events"][0]["action"]="reset"
        with self.assertRaises(ValueError):
            LearnedMorphologyConversation.restore(
                (self.phase,self.full),altered,
                expected_head=saved["checkpoint_digest"])
        with self.assertRaises(ValueError):
            LearnedMorphologyConversation.restore(
                (self.phase,self.full),saved,
                expected_head="0"*64)

    def test_state_position_precedes_new_generated_content(self):
        s=LearnedMorphologyConversation((self.phase,self.full))
        item=s.respond(self.full,prompt="Render a new program",action="advance")
        outputs=item["response"]["projections"]
        self.assertIn("def transition",outputs[1]["content"])
        self.assertEqual(item["response"]["neural_steps_executed_this_formation"],1)
        self.assertIn("phase=1",outputs[0]["content"])


from vli.learned_morphology import LearnedObserverIntentionRouter
from vli.intent_topology import TopologyWitness


def lens_witnesses():
    return (
        TopologyWitness("report only the phase observation", ("phase",)),
        TopologyWitness("show the current phase state", ("phase",)),
        TopologyWitness("report only the micro observation", ("micro",)),
        TopologyWitness("show the current micro state", ("micro",)),
        TopologyWitness("report both phase and micro observations", ("phase","micro")),
        TopologyWitness("show the complete state with both channels", ("phase","micro")),
    )


class TestLearnedModelPromptIntention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw=RAW.read_bytes()
        cls.phase=LearnedObserverGround(raw,("phase",))
        cls.micro=LearnedObserverGround(raw,("micro",))
        cls.full=LearnedObserverGround(raw,("phase","micro"))
        cls.witnesses=lens_witnesses()

    def session(self):
        return LearnedMorphologyConversation((self.phase,self.micro,self.full))

    def test_prompt_navigates_from_black_box_learned_shape_to_new_response(self):
        session=self.session()
        router=LearnedObserverIntentionRouter(session)
        p=router.propose(self.witnesses,"report only the phase observation")
        self.assertEqual(p["status"],"PROPOSED_NEEDS_OBSERVER_APPROVAL")
        self.assertEqual(p["ecpu_intention"]["ecpu_prediction"]["evidence_tier"],"E0")
        self.assertEqual(session.history,())
        chosen=router.accept(self.witnesses,p,
                             expected_proposal_head=p["proposal_digest"])
        self.assertIs(chosen,self.phase)
        out=router.accept_and_respond(
            self.witnesses,p,expected_proposal_head=p["proposal_digest"],
            action="advance")
        self.assertEqual(out["event"]["observed_values"],[1])
        self.assertEqual(session.history,("advance",))

    def test_new_prompt_requalifies_finer_neural_model_with_same_causal_history(self):
        session=self.session()
        router=LearnedObserverIntentionRouter(session)
        p=router.propose(self.witnesses,"report only the phase observation")
        router.accept_and_respond(self.witnesses,p,
                                  expected_proposal_head=p["proposal_digest"],
                                  action="advance",views=("text",))
        broader=router.propose(self.witnesses,
                               "report both phase and micro observations")
        picked=router.accept(self.witnesses,broader,
                             expected_proposal_head=broader["proposal_digest"])
        self.assertIs(picked,self.full)
        out=router.accept_and_respond(
            self.witnesses,broader,
            expected_proposal_head=broader["proposal_digest"],
            action="inspect",views=("text","code","tokens"))
        self.assertEqual(out["event"]["observed_values"],[1,1])
        self.assertEqual(session.history,("advance","inspect"))
        self.assertNotEqual(out["event"]["position"],p["proposal_digest"])

    def test_stale_accepted_intention_cannot_skip_causal_change(self):
        session=self.session()
        router=LearnedObserverIntentionRouter(session)
        p=router.propose(self.witnesses,"report only the phase observation")
        session.respond(self.micro,prompt="micro change",action="advance")
        head=session.head
        with self.assertRaises(ValueError):
            router.accept(self.witnesses,p,
                          expected_proposal_head=p["proposal_digest"])
        self.assertEqual(session.head,head)
        self.assertEqual(len(session.events),1)

    def test_tampered_or_uncertain_intention_does_not_form_state(self):
        session=self.session()
        router=LearnedObserverIntentionRouter(session)
        p=router.propose(self.witnesses,"report only the phase observation")
        forged=copy.deepcopy(p)
        forged["observed_ledger_heads"][0]="0"*64
        with self.assertRaises(ValueError):
            router.accept(self.witnesses,forged,
                          expected_proposal_head=p["proposal_digest"])
        with self.assertRaises(ValueError):
            router.accept(self.witnesses,p,
                          expected_proposal_head="0"*64)
        uncertain=router.propose(
            self.witnesses,"unseen ambiguous request",
            min_confidence=0.999999,min_margin=0.999999)
        self.assertEqual(uncertain["status"],"INSUFFICIENT_INTENT_EVIDENCE")
        with self.assertRaises(ValueError):
            router.accept(self.witnesses,uncertain,
                          expected_proposal_head=uncertain["proposal_digest"])
        self.assertEqual(session.events,[])

if __name__=="__main__":
    unittest.main()
