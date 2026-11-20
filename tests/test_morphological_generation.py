"""SHAPE-002: two intentions reshape one model under exact all-action observation proofs.

Verifies VLD predictive quotient, independently checked Mealy congruence,
new VLD neural parameters, VL-I code/text/tokens, cold/warm cost evidence,
counterexamples, causal morphology switching and independently replayed state.
"""
from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
import unittest

from vli.morphological_generation import (
    MorphologySourceGround, ObserverMorphology, MorphologySession,
    select_qualified_morphology,
)

FIXTURE=(Path(__file__).resolve().parents[1] /
         "experiments/shape002/morphology_model.json")


class TestMalleableComputationalShape(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=MorphologySourceGround(FIXTURE.read_bytes())
        cls.phase=cls.source.observer(("phase",))
        cls.micro=cls.source.observer(("micro",))
        cls.full=cls.source.observer(("phase","micro"))

    def test_same_model_compiles_three_distinct_observer_shaped_neural_networks(self):
        s=self.source
        self.assertEqual(len(s.system.states),12)
        self.assertEqual(len(s.system.actions),3)
        self.assertEqual(s.manifest["source_bytes"],len(FIXTURE.read_bytes()))
        self.assertEqual(len(s.ecpu_chunk_digests),1)
        self.assertEqual([x.manifest["qualified_state_dimension"] for x in
                          (self.phase,self.micro,self.full)],[3,4,12])
        self.assertEqual([x.manifest["generated_hidden_units"] for x in
                          (self.phase,self.micro,self.full)],[9,12,36])
        self.assertEqual(len({x.generated_model.weights_digest for x in
                              (self.phase,self.micro,self.full)}),3)

    def test_vld_partition_and_mealy_proof_check_every_original_cell(self):
        for shaped in (self.phase,self.micro,self.full):
            with self.subTest(channels=shaped.channels):
                self.assertEqual(shaped.proof["checked_original_cells"],36)
                self.assertEqual(len(shaped.proof["state_to_class"]),12)
                self.assertTrue(shaped.generated_model.verify())
                for state in self.source.system.states:
                    for action in self.source.system.actions:
                        shaped.compare_full_cell(state,action)

    def test_equivalence_holds_for_all_length_three_action_histories(self):
        system=self.source.system
        for shape in (self.phase,self.micro):
            for origin in system.states:
                for word in itertools.product(system.actions,repeat=3):
                    s=origin
                    q=shape.class_map[s]
                    for a in word:
                        full=system.step(s,a)
                        qtarget,qout=shape.generated_model.generated(q,a)
                        self.assertEqual(qtarget,shape.class_map[full.target])
                        self.assertEqual(qout,tuple(full.output[i] for i in shape.positions))
                        q=qtarget
                        s=full.target

    def test_coordinate_located_before_actual_generating_neural_step(self):
        p=self.phase.position(
            state="p1m3",action="advance",prompt="Show only the phase",
            observer="phase-observer",causal_head="a"*64,
            expected_proof_digest=self.phase.proof["proof_digest"])
        self.assertTrue(p["coordinate"].startswith("state://vli/"))
        self.assertEqual(len(p["coordinate"]),76)
        self.assertNotIn("observed_channel_values",p)
        result=self.phase.observe(p,expected_proof_digest=self.phase.proof["proof_digest"])
        self.assertEqual(result["coordinate"],p["coordinate"])
        self.assertEqual(result["requested_channels"],["phase"])
        self.assertEqual(result["observed_channel_values"],[2])
        self.assertEqual(result["generated_neural_steps"],1)
        self.assertEqual([x["kind"] for x in result["generated_views"]],
                         ["text","code","tokens"])
        self.assertIn("def transition",result["generated_views"][1]["content"])
        self.assertEqual(bytes(result["generated_views"][2]["content"]).decode("utf-8"),
                         result["generated_views"][0]["content"])

    def test_different_observers_locate_distinct_states_and_values(self):
        kwargs=dict(state="p1m3",action="advance",prompt="Observe",
                    observer="same",causal_head="b"*64)
        addresses=[]
        readings=[]
        for shape in (self.phase,self.micro,self.full):
            pos=shape.position(**kwargs,
                expected_proof_digest=shape.proof["proof_digest"])
            out=shape.observe(pos,expected_proof_digest=shape.proof["proof_digest"],
                              views=("text",))
            addresses.append(pos["coordinate"])
            readings.append(out["observed_channel_values"])
        self.assertEqual(readings,[[2],[0],[2,0]])
        self.assertEqual(len(set(addresses)),3)
        self.assertTrue(all(len(x)==76 for x in addresses))

    def test_rejects_micro_output_requested_from_phase_quotient(self):
        with self.assertRaises(ValueError):
            self.phase.position(
                state="p1m3",action="advance",prompt="micro please",observer="x",
                causal_head="0"*64,
                expected_proof_digest=self.micro.proof["proof_digest"])
        wrong=select_qualified_morphology(
            self.source,(self.phase,self.micro,self.full),
            requested_channels=("micro",),expected_queries=30,
            warm_sample_count=24)
        self.assertEqual(wrong["options"][0]["status"],
                         "REFUSED_INSUFFICIENT_OBSERVER_SCOPE")
        self.assertEqual(wrong["options"][1]["status"],"QUALIFIED")
        self.assertEqual(wrong["options"][2]["status"],"QUALIFIED")
        self.assertIn(wrong["warm_selection_channels"],[["micro"],["phase","micro"]])

    def test_selection_distinguishes_cold_and_warm_cost(self):
        report=select_qualified_morphology(
            self.source,(self.phase,self.micro,self.full),
            requested_channels=("phase",),expected_queries=200,
            warm_sample_count=72)
        self.assertEqual(report["options"][1]["status"],
                         "REFUSED_INSUFFICIENT_OBSERVER_SCOPE")
        candidate=[x for x in report["options"] if x["status"]=="QUALIFIED"]
        self.assertEqual(len(candidate),2)
        self.assertEqual(report["cold_selection_channels"],
                         min(candidate,key=lambda x:(
                             x["estimated_total_cold_ns"],
                             x["realized_state_dimension"]))["channels"])
        for option in candidate:
            self.assertGreater(option["preparation_ns"],0)
            self.assertGreater(option["neural_and_local_oracle_ns_per_query"],0)
            self.assertEqual(option["estimated_total_cold_ns"],
                             option["preparation_ns"]+
                             option["estimated_total_warm_ns"])

    def test_mutating_qualified_certificate_refuses_even_when_rehashed(self):
        from vli.ground import _digest
        original=copy.deepcopy(self.phase.proof)
        try:
            self.phase.proof["projected_transition_cells"][0]["outcomes"][0]["output"]=[99]
            self.phase.proof["proof_digest"]=_digest(
                self.phase.proof["schema"],
                {k:v for k,v in self.phase.proof.items() if k!="proof_digest"})
            with self.assertRaises(ValueError):
                self.phase.position(
                    state="p1m3",action="inspect",prompt="Inspect",observer="x",
                    causal_head="a"*64,expected_proof_digest=original["proof_digest"])
        finally:
            self.phase.proof=original

    def test_rehashed_coordinate_tamper_refuses_different_semantic_identity(self):
        p=self.micro.position(
            state="p2m1",action="inspect",prompt="Inspect",observer="micro",
            causal_head="1"*64,expected_proof_digest=self.micro.proof["proof_digest"])
        bad=copy.deepcopy(p)
        bad["source_state"]="p2m2"
        with self.assertRaises(ValueError):
            self.micro.observe(bad,expected_proof_digest=self.micro.proof["proof_digest"])
        with self.assertRaises(ValueError):
            self.phase.observe(p,expected_proof_digest=self.phase.proof["proof_digest"])

    def test_injected_hidden_state_distinction_refines_the_projection(self):
        # The phase-observer used to merge four microstates. A changed model
        # emits one micro-dependent PHASE value. VLD must split equivalence
        # classes rather than naively reuse the old 3-state quotient.
        raw=json.loads(FIXTURE.read_text())
        item=next(x for x in raw["observations"]
                  if x["source"]=="p1m3" and x["action"]=="inspect")
        item["outcomes"][0]["output"][0]=7
        changed=MorphologySourceGround(
            json.dumps(raw,sort_keys=True).encode("utf-8"))
        refined=changed.observer(("phase",))
        self.assertGreater(refined.manifest["qualified_state_dimension"],3)
        self.assertNotEqual(refined.manifest["ground_digest"],
                            self.phase.manifest["ground_digest"])
        for s in changed.system.states:
            for a in changed.system.actions:
                refined.compare_full_cell(s,a)
        with self.assertRaises(ValueError):
            MorphologySession(changed).step(self.phase,action="inspect",prompt="Observe")

    def test_original_source_byte_change_invalidates_full_ground(self):
        raw=json.loads(FIXTURE.read_text())
        raw["metadata"]["purpose"]="revised by another observer"
        changed=MorphologySourceGround(json.dumps(raw,sort_keys=True).encode())
        self.assertNotEqual(changed.manifest["ground_digest"],
                            self.source.manifest["ground_digest"])

    def test_observer_lenses_require_true_declared_output_channels(self):
        with self.assertRaises(ValueError):
            self.source.observer(("temperature",))
        with self.assertRaises(ValueError):
            self.source.observer(("phase","phase"))
        with self.assertRaises(ValueError):
            self.source.observer(())


class TestCausalShapeChanges(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=MorphologySourceGround(FIXTURE.read_bytes())
        cls.phase=cls.source.observer(("phase",))
        cls.micro=cls.source.observer(("micro",))
        cls.full=cls.source.observer(("phase","micro"))

    def test_broader_intention_recovers_information_not_in_previous_projection(self):
        conversation=MorphologySession(self.source)
        first=conversation.step(self.phase,action="advance",
                                prompt="Only tell me the phase")
        self.assertEqual(first["observation"]["observed_channel_values"],[1])
        self.assertEqual(conversation.state,"p1m1")
        second=conversation.step(self.micro,action="inspect",
                                 prompt="Now tell me the hidden microstate")
        self.assertEqual(second["observation"]["observed_channel_values"],[1])
        self.assertEqual(conversation.state,"p1m1")
        third=conversation.step(self.full,action="advance",
                                prompt="Now show the complete state")
        self.assertEqual(third["observation"]["observed_channel_values"],[2,2])
        self.assertEqual(conversation.state,"p2m2")
        self.assertEqual(len({x["position"]["coordinate"] for x in
                              (first,second,third)}),3)
        self.assertEqual(first["turn"]["previous"],conversation.turns[0]["previous"])
        self.assertEqual(second["turn"]["previous"],first["turn"]["turn_digest"])
        self.assertEqual(third["turn"]["previous"],second["turn"]["turn_digest"])

    def test_recover_full_state_then_continue_with_novel_intention(self):
        session=MorphologySession(self.source)
        session.step(self.phase,action="advance",prompt="Phase only",views=("text",))
        session.step(self.micro,action="advance",prompt="Micro only",views=("code",))
        saved=session.checkpoint()
        restored=MorphologySession.restore(
            self.source,(self.phase,self.micro,self.full),saved,
            expected_checkpoint_head=saved["checkpoint_digest"])
        self.assertEqual(restored.checkpoint(),saved)
        self.assertEqual(restored.state,"p2m2")
        a=session.step(self.full,action="inspect",prompt="Full state",views=("text",))
        b=restored.step(self.full,action="inspect",prompt="Full state",views=("text",))
        self.assertEqual(a["turn"],b["turn"])
        self.assertEqual(a["observation"]["observation_digest"],
                         b["observation"]["observation_digest"])

    def test_recovery_refuses_wrong_head_or_changed_causal_step(self):
        session=MorphologySession(self.source)
        session.step(self.phase,action="advance",prompt="Phase",views=("text",))
        checkpoint=session.checkpoint()
        with self.assertRaises(ValueError):
            MorphologySession.restore(self.source,(self.phase,self.full),
                                      checkpoint,expected_checkpoint_head="0"*64)
        tampered=copy.deepcopy(checkpoint)
        tampered["turns"][0]["action"]="reset"
        with self.assertRaises(ValueError):
            MorphologySession.restore(
                self.source,(self.phase,self.full),tampered,
                expected_checkpoint_head=checkpoint["checkpoint_digest"])


from vli.morphological_intent import MorphologyIntentionRouter
from vli.intent_topology import TopologyWitness


def observer_witnesses():
    return (
        TopologyWitness("report only the phase observation",("phase",)),
        TopologyWitness("show the current phase state",("phase",)),
        TopologyWitness("report only the micro observation",("micro",)),
        TopologyWitness("show the current micro state",("micro",)),
        TopologyWitness("report both phase and micro observations",("phase","micro")),
        TopologyWitness("show the complete state with both channels",("phase","micro")),
    )


class TestPromptToObserverMorphology(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ground=MorphologySourceGround(FIXTURE.read_bytes())
        cls.witnesses=observer_witnesses()

    def test_phase_prompt_discovers_qualified_malleable_shape(self):
        session=MorphologySession(self.ground)
        router=MorphologyIntentionRouter(self.ground)
        proposal=router.propose(
            session,self.witnesses,"report only the phase observation",
            min_confidence=0.35,min_margin=0.001)
        self.assertEqual(proposal["status"],"PROPOSED_NEEDS_OBSERVER_APPROVAL")
        self.assertEqual(session.state,"p0m0")
        qualified=router.accept(
            session,self.witnesses,proposal,
            expected_proposal_head=proposal["proposal_digest"])
        self.assertEqual(qualified,("phase",))
        shaped=self.ground.observer(qualified)
        reading=session.step(shaped,action="advance",
                             prompt="report only the phase observation")
        self.assertEqual(reading["observation"]["observed_channel_values"],[1])
        self.assertEqual(shaped.manifest["qualified_state_dimension"],3)

    def test_new_prompt_can_select_finer_shape_with_memory_preserved(self):
        session=MorphologySession(self.ground)
        router=MorphologyIntentionRouter(self.ground)
        witnesses=self.witnesses
        def request(prompt):
            p=router.propose(session,witnesses,prompt,
                             min_confidence=0.35,min_margin=0.001)
            return router.accept(session,witnesses,p,
                                 expected_proposal_head=p["proposal_digest"])
        narrow=request("report only the phase observation")
        session.step(self.ground.observer(narrow),action="advance",prompt="phase")
        complete=request("report both phase and micro observations")
        self.assertEqual(complete,("phase","micro"))
        result=session.step(self.ground.observer(complete),action="inspect",
                            prompt="Now inspect the full state")
        self.assertEqual(result["observation"]["observed_channel_values"],[1,1])

    def test_stale_morphology_proposal_rejected_after_source_state_transition(self):
        session=MorphologySession(self.ground)
        router=MorphologyIntentionRouter(self.ground)
        p=router.propose(session,self.witnesses,
                         "report only the micro observation",
                         min_confidence=0.35,min_margin=0.001)
        session.step(self.ground.observer(("phase",)),action="advance",
                     prompt="Move before accepting")
        with self.assertRaises(ValueError):
            router.accept(session,self.witnesses,p,
                          expected_proposal_head=p["proposal_digest"])

    def test_uncertain_intent_refuses_without_mutation(self):
        session=MorphologySession(self.ground)
        router=MorphologyIntentionRouter(self.ground)
        p=router.propose(session,self.witnesses,
                         "incomprehensible unobserved unknown purpose",
                         min_confidence=0.999999,min_margin=0.999999)
        self.assertEqual(p["status"],"INSUFFICIENT_INTENT_EVIDENCE")
        with self.assertRaises(ValueError):
            router.accept(session,self.witnesses,p,
                          expected_proposal_head=p["proposal_digest"])
        self.assertEqual(session.state,"p0m0")
        self.assertEqual(session.turns,[])

    def test_forged_causal_observer_head_and_model_source_rejected(self):
        session=MorphologySession(self.ground)
        router=MorphologyIntentionRouter(self.ground)
        p=router.propose(session,self.witnesses,
                         "report only the micro observation",
                         min_confidence=0.35,min_margin=0.001)
        changed=copy.deepcopy(p)
        changed["full_source_state"]="p2m3"
        with self.assertRaises(ValueError):
            router.accept(session,self.witnesses,changed,
                          expected_proposal_head=p["proposal_digest"])
        wrong=copy.deepcopy(p)
        wrong["original_model_ground_digest"]="0"*64
        with self.assertRaises(ValueError):
            router.accept(session,self.witnesses,wrong,
                          expected_proposal_head=p["proposal_digest"])
        with self.assertRaises(ValueError):
            router.accept(session,self.witnesses,p,
                          expected_proposal_head="0"*64)

if __name__=="__main__":
    unittest.main()
