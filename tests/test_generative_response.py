"""COORD-002: actual VLD descriptive model -> generated neural weights and response.

Tests exercise real VLD whole-domain neural synthesis, E-CPU source-byte
observation, VLD's real DAG rendering, causal recovery, and counterexamples.
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path
import unittest

from vli.generative_response import (
    GeneratedModelGround, GenerativeFormation, GenerativeConversation,
)


FIXTURE = (Path(__file__).resolve().parents[1] /
           "experiments/coord002/described_model.json")


def make_ground():
    return GeneratedModelGround(FIXTURE.read_bytes())


class TestDescriptionNativeModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = make_ground()

    def test_description_generated_neural_parameters_without_weights(self):
        g = self.model
        self.assertTrue(g.verify())
        self.assertEqual(g.manifest["description_bytes"], len(FIXTURE.read_bytes()))
        self.assertEqual(g.manifest["source"],
                         "exact_supplied_description_bytes_no_pretrained_weight_payload")
        self.assertEqual(g.manifest["generated_hidden_units"], 9)
        self.assertEqual(len(g.neural.hidden_weight), 9)
        self.assertEqual(len(g.neural.next_state_weight), 3)
        self.assertEqual(len(g.neural.channel_weight), 2)
        self.assertEqual(len(g.ecpu_chunk_digests), 1)
        self.assertEqual(g.system.initial_state, "idle")

    def test_entire_neural_domain_matches_independent_vld_transition_system(self):
        g = self.model
        checked = 0
        for state in g.system.states:
            for action in g.system.actions:
                with self.subTest(state=state, action=action):
                    direct = g.reference(state, action)
                    neural = g.generated(state, action)
                    self.assertEqual(direct, neural)
                    checked += 1
        self.assertEqual(checked, 9)
        self.assertEqual(g.generated("idle","inspect"),("observing",(1,2)))
        self.assertEqual(g.generated("observing","advance"),("formed",(4,5)))
        self.assertEqual(g.generated("formed","reset"),("idle",(0,2)))

    def test_new_executable_python_source_generated_from_law(self):
        g = self.model
        source = g.generated_program()
        ast.parse(source)
        namespace = {"__builtins__": {}}
        exec(compile(source, "<derived-ground-program>", "exec"), namespace)
        fn = namespace["transition"]
        for state in g.system.states:
            for action in g.system.actions:
                self.assertEqual(fn(state, action), g.reference(state, action))
        with self.assertRaises(KeyError):
            fn("unadmitted", "inspect")
        self.assertIn("('observing', 'advance')", source)
        self.assertNotIn("import torch", source)

    def test_generator_forms_position_before_computing_any_output(self):
        g = self.model
        formation = GenerativeFormation(
            g,state="idle",action="inspect",
            prompt="Please inspect the current state and write the model's code.",
            history_head="a"*64)
        before=formation.observe()
        self.assertEqual(before["phase"], "LOCATED")
        self.assertIsNone(before["step_receipt"])
        self.assertEqual(before["projections"], [])
        self.assertEqual(before["neural_steps_executed_this_formation"],0)
        address=before["coordinate"]
        self.assertTrue(address.startswith("state://vli/"))
        self.assertEqual(len(address), 76)
        outlined=formation.advance()
        self.assertEqual(outlined["coordinate"],address)
        self.assertEqual(outlined["phase"],"FORMING")
        self.assertEqual(outlined["neural_steps_executed_this_formation"],1)
        finished=formation.finish()
        self.assertEqual(finished["coordinate"],address)
        self.assertEqual(finished["phase"],"RENDERED")
        self.assertEqual(finished["step_receipt"]["target_state"],"observing")
        self.assertEqual(finished["step_receipt"]["channel_output"],[1,2])
        views={v["kind"]:v for v in finished["projections"]}
        self.assertEqual(set(views),{"text","page","code","tokens"})
        self.assertIn("observing", views["text"]["content"])
        self.assertIn("phase=1, signal=2", views["text"]["content"])
        self.assertIn("('idle', 'inspect')", views["code"]["content"])
        self.assertIn("Generated neural weights", views["page"]["content"])
        self.assertEqual(bytes(views["tokens"]["content"]).decode("utf-8"),
                         views["text"]["content"])

    def test_generated_content_was_not_supplied_as_literal_description(self):
        g = self.model
        form=GenerativeFormation(g,state="observing",action="advance",
                                 prompt="Generate executable transition code for this state",
                                 history_head="a"*64)
        out=form.finish()
        text=next(x["content"] for x in out["projections"] if x["kind"]=="text")
        code=next(x["content"] for x in out["projections"] if x["kind"]=="code")
        self.assertNotIn(text, FIXTURE.read_text())
        self.assertNotIn("def transition", FIXTURE.read_text())
        self.assertIn("def transition", code)
        self.assertEqual(out["step_receipt"]["target_state"],"formed")

    def test_state_position_is_contextual_not_numerical(self):
        g=self.model
        def position(prompt, head, action):
            return GenerativeFormation(
                g,state="idle",action=action,prompt=prompt,history_head=head).coordinate
        a=position("State inspection","a"*64,"inspect")
        self.assertNotEqual(a,position("Write model code","a"*64,"inspect"))
        self.assertNotEqual(a,position("State inspection","b"*64,"inspect"))
        self.assertNotEqual(a,position("State inspection","a"*64,"advance"))
        self.assertTrue(all(len(x)==76 for x in [
            a,position("Write model code","a"*64,"inspect"),
            position("State inspection","a"*64,"advance")]))

    def test_unrequested_generation_views_are_not_materialized(self):
        g=self.model
        f=GenerativeFormation(g,state="idle",action="advance",
                              prompt="Describe the target state",history_head="f"*64,
                              views=("text",))
        out=f.finish()
        self.assertEqual(len(out["projections"]),1)
        self.assertEqual(out["projections"][0]["kind"],"text")
        inner=out["inner_observation"]
        self.assertEqual(inner["required_slots"],["text"])
        self.assertEqual(inner["measured_source_bytes_materialized_this_process"],
                         len(out["projections"][0]["content"].encode("utf-8")))

    def test_tools_are_only_inert_proposals(self):
        f=GenerativeFormation(self.model,state="formed",action="inspect",
                              prompt="Offer a read-only state inspection tool",
                              history_head="a"*64,views=("tool_call",))
        final=f.finish()
        tool=final["projections"][0]
        self.assertEqual(tool["status"],"INERT_PROPOSAL")
        self.assertEqual(tool["content"]["tool"],"inspect_state")
        self.assertEqual(tool["content"]["arguments"]["state"],"formed")
        self.assertTrue(tool["content"]["arguments"]["read_only"])

    def test_new_model_from_changed_metadata_makes_new_weights_and_location(self):
        import json
        source=json.loads(FIXTURE.read_bytes())
        item=next(x for x in source["observations"]
                  if x["source"]=="idle" and x["action"]=="inspect")
        item["outcomes"][0]["target"]="formed"
        item["outcomes"][0]["output"]=[7,7]
        other=GeneratedModelGround(json.dumps(source,sort_keys=True).encode("utf-8"))
        self.assertNotEqual(other.weights_digest,self.model.weights_digest)
        self.assertNotEqual(other.manifest["ground_digest"],
                            self.model.manifest["ground_digest"])
        self.assertEqual(other.generated("idle","inspect"),("formed",(7,7)))
        old=GenerativeFormation(self.model,state="idle",action="inspect",
                                prompt="Inspect",history_head="a"*64)
        new=GenerativeFormation(other,state="idle",action="inspect",
                                prompt="Inspect",history_head="a"*64)
        self.assertNotEqual(old.coordinate,new.coordinate)

    def test_negative_source_incomplete_ambiguous_or_invalid(self):
        import json
        source=json.loads(FIXTURE.read_bytes())
        source["observations"].pop()
        with self.assertRaises((ValueError,RuntimeError)):
            GeneratedModelGround(json.dumps(source).encode())
        source=json.loads(FIXTURE.read_bytes())
        source["observations"][0]["outcomes"].append(
            {"target":"idle","output":[0,0]})
        with self.assertRaises((ValueError,RuntimeError)):
            GeneratedModelGround(json.dumps(source).encode())
        source=json.loads(FIXTURE.read_bytes())
        source["observations"][0]["outcomes"][0]["output"]=[1000,3]
        with self.assertRaises((ValueError,RuntimeError)):
            GeneratedModelGround(json.dumps(source).encode())
        with self.assertRaises(ValueError):
            GeneratedModelGround(b'{"format":"fake","format":"fake"}')

    def test_unadmitted_intention_refuses(self):
        with self.assertRaises(ValueError):
            GenerativeFormation(self.model,state="idle",action="delete_everything",
                                prompt="Go",history_head="a"*64)
        with self.assertRaises(ValueError):
            GenerativeFormation(self.model,state="unknown",action="inspect",
                                prompt="Go",history_head="a"*64)
        with self.assertRaises(ValueError):
            GenerativeFormation(self.model,state="idle",action="inspect",
                                prompt="Go",history_head="invalid")
        with self.assertRaises(ValueError):
            GenerativeFormation(self.model,state="idle",action="inspect",
                                prompt="Go",history_head="a"*64,
                                views=("text","unadmitted"))

    def test_neural_output_and_independent_oracle_are_actually_compared(self):
        g=self.model
        original=g.neural
        class Forged:
            def step_neural(self, state, action):
                return "idle", (7,7)
        g.neural=Forged()
        try:
            f=GenerativeFormation(g,state="idle",action="inspect",
                                  prompt="Inspect",history_head="a"*64)
            with self.assertRaises(ValueError):
                f.advance()
        finally:
            g.neural=original


class TestGenerativeConversation(unittest.TestCase):
    def setUp(self):
        self.g=make_ground()

    def test_multi_turn_transitions_are_new_state_positions_with_memory(self):
        conversation=GenerativeConversation(self.g)
        first=conversation.respond("Inspect the starting Ground","inspect")
        second=conversation.respond("Now advance the model and report channels","advance")
        third=conversation.respond("Please reset the described system","reset")
        self.assertEqual([x["turn"]["to_state"] for x in (first,second,third)],
                         ["observing","formed","idle"])
        self.assertEqual(conversation.state,"idle")
        self.assertNotEqual(first["turn"]["coordinate"], second["turn"]["coordinate"])
        self.assertEqual(first["turn"]["previous"],conversation.genesis)
        self.assertEqual(second["turn"]["previous"],first["turn"]["turn_digest"])
        self.assertEqual(third["turn"]["previous"],second["turn"]["turn_digest"])
        self.assertEqual(len(conversation.events),3)

    def test_cross_process_style_checkpoint_replay_and_next_turn(self):
        c=GenerativeConversation(self.g)
        c.respond("Find the observed state","inspect",views=("text","tokens"))
        c.respond("Now continue to the next state","advance",views=("text",))
        saved=c.checkpoint()
        fresh=GenerativeConversation.restore(self.g, saved,
                                              expected_head=saved["checkpoint_digest"])
        self.assertEqual(fresh.state,c.state)
        self.assertEqual(fresh.head,c.head)
        self.assertEqual(fresh.checkpoint(),saved)
        a=c.respond("Now inspect after forming","inspect",views=("page",))
        b=fresh.respond("Now inspect after forming","inspect",views=("page",))
        self.assertEqual(a["turn"],b["turn"])
        self.assertEqual(a["response"]["projections"],
                         b["response"]["projections"])

    def test_checkpoint_tamper_and_external_head_refusal(self):
        c=GenerativeConversation(self.g)
        c.respond("Inspect","inspect")
        saved=c.checkpoint()
        with self.assertRaises(ValueError):
            GenerativeConversation.restore(self.g,saved,
                                           expected_head="0"*64)
        altered=copy.deepcopy(saved)
        altered["events"][0]["action"]="reset"
        with self.assertRaises(ValueError):
            GenerativeConversation.restore(self.g,altered,
                                           expected_head=saved["checkpoint_digest"])

    def test_disagreement_with_model_metadata_invalidates_recovery(self):
        import json
        c=GenerativeConversation(self.g)
        c.respond("Inspect","inspect")
        saved=c.checkpoint()
        src=json.loads(FIXTURE.read_bytes())
        src["observations"][0]["outcomes"][0]["target"]="formed"
        other=GeneratedModelGround(json.dumps(src).encode())
        with self.assertRaises(ValueError):
            GenerativeConversation.restore(other,saved,
                                           expected_head=saved["checkpoint_digest"])


if __name__=="__main__":
    unittest.main()
