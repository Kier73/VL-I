"""Response coordinates are positions of renderable states, not scalar values.

The real VLD DAG is used for progressive causal realization and checkpoint
verification; E-CPU DataGround validates exact requested bytes.
"""
import copy
import hashlib
import json
import unittest

from vli.response_state import (
    ResponseSourceGround, ResponseFormation, SourceSlot,
    compile_intention, locate_response_state
)


def source():
    return ResponseSourceGround((
        SourceSlot("reference", "text", b"The available source states three observations."),
        SourceSlot("answer", "text", b"The interpreted response follows the admitted evidence.",
                   dependencies=("reference",)),
        SourceSlot("program", "code", b"def read_state(x):\n    return x['value']\n",
                   dependencies=("reference",)),
        SourceSlot("page", "page", b"# Virtual page\n\nA rendered state."),
        SourceSlot("action", "tool_call", b'{"tool":"inspect_state","arguments":{"scope":"read"}}',
                   dependencies=("answer",)),
        SourceSlot("tokens", "tokens", b"[44,12,8]"),
    ))


class TestResponseTopology(unittest.TestCase):
    def setUp(self):
        self.source = source()

    def intent(self, slots=("answer",), history=()):
        return compile_intention(
            "Render the state needed to answer this observer",
            slots=tuple(slots), observer="test-observer", history=tuple(history))

    def test_coordinate_is_opaque_position_not_scalar(self):
        intent = self.intent(("answer","program"))
        position = locate_response_state(self.source, intent)
        self.assertTrue(position["coordinate"].startswith("state://vli/"))
        self.assertEqual(len(position["coordinate"]), len("state://vli/") + 64)
        self.assertEqual(position["selected_slots"], ["answer", "program"])
        self.assertEqual(position["required_slots"], ["reference","answer","program"])
        self.assertNotIn("position_numeric", position)

    def test_intention_collapses_topology_without_all_other_views(self):
        a = locate_response_state(self.source, self.intent(("answer",)))
        b = locate_response_state(self.source, self.intent(("page",)))
        self.assertNotEqual(a["coordinate"], b["coordinate"])
        self.assertEqual(a["required_slots"], ["reference", "answer"])
        self.assertEqual(b["required_slots"], ["page"])
        self.assertNotIn("action", a["required_slots"])
        self.assertNotIn("tokens", a["required_slots"])

    def test_progressive_refinement_forming_into_rendered_state(self):
        f = ResponseFormation(self.source, self.intent(("answer", "program", "action")))
        initial = f.observe()
        self.assertEqual(initial["state"], "FORMING")
        self.assertFalse(initial["outline_ready"])
        self.assertEqual(initial["projections"], [])
        constant = initial["coordinate"]
        first = f.advance(max_nodes=1)
        self.assertTrue(first["outline_ready"])
        self.assertEqual(first["projections"], [])
        self.assertEqual(first["coordinate"], constant)
        while not f.dag.is_complete:
            f.advance(max_nodes=8)
        final = f.observe()
        self.assertEqual(final["state"], "RENDERED")
        self.assertEqual(final["coordinate"], constant)
        self.assertEqual([p["slot"] for p in final["projections"]],
                         ["answer","program","action"])
        self.assertEqual(final["projections"][0]["kind"], "text")
        self.assertEqual(final["projections"][1]["kind"], "code")
        self.assertEqual(final["projections"][2]["kind"], "tool_call")
        self.assertEqual(final["projections"][2]["status"], "INERT_PROPOSAL")
        self.assertEqual(final["projections"][2]["content"]["tool"], "inspect_state")
        self.assertEqual(final["remaining_slots"], [])
        self.assertEqual(final["measured_source_bytes_materialized_this_process"],
                         sum(len(self.source.slots[n].content)
                             for n in ("reference","answer","program","action")))

    def test_unselected_bytes_never_enter_requested_frontier(self):
        f=ResponseFormation(self.source, self.intent(("page",)))
        while not f.dag.is_complete:
            f.advance(max_nodes=8)
        out=f.observe()
        self.assertEqual(out["required_slots"], ["page"])
        self.assertEqual(out["projections"][0]["content"], "# Virtual page\n\nA rendered state.")
        self.assertEqual(out["measured_source_bytes_materialized_this_process"],
                         len(self.source.slots["page"].content))

    def test_tokens_as_a_rendered_observation_not_a_numeric_coordinate(self):
        f = ResponseFormation(self.source, self.intent(("tokens",)))
        for _ in range(4):
            f.advance(max_nodes=4)
        out=f.observe()
        self.assertEqual(out["state"],"RENDERED")
        self.assertEqual(out["projections"][0]["content"], [44,12,8])
        self.assertTrue(out["coordinate"].startswith("state://"))

    def test_checkpoint_restores_without_repeating_committed_nodes(self):
        intent=self.intent(("answer","program"),history=(
            {"role":"user","content":"Earlier question"},
            {"role":"assistant","content":"Prior answer"}))
        f=ResponseFormation(self.source,intent)
        f.advance(max_nodes=1)
        f.advance(max_nodes=1)
        committed_before = len(f.dag.committed)
        ckp=f.checkpoint()
        restarted=ResponseFormation(self.source,intent,checkpoint=ckp,
                                    expected_head=ckp["checkpoint_digest"])
        self.assertEqual(len(restarted.dag.committed),committed_before)
        while not restarted.dag.is_complete:
            restarted.advance(max_nodes=5)
        while not f.dag.is_complete:
            f.advance(max_nodes=5)
        self.assertEqual(restarted.observe()["coordinate"], f.observe()["coordinate"])
        self.assertEqual(restarted.observe()["dag_state_digest"], f.observe()["dag_state_digest"])
        self.assertEqual([p["content"] for p in restarted.observe()["projections"]],
                         [p["content"] for p in f.observe()["projections"]])

    def test_stale_checkpoint_head_fails(self):
        it=self.intent()
        f=ResponseFormation(self.source,it)
        f.advance()
        c=f.checkpoint()
        with self.assertRaises(ValueError):
            ResponseFormation(self.source,it,checkpoint=c,expected_head="0"*64)
        forgery=copy.deepcopy(c)
        forgery["dag"]["committed"]={}
        with self.assertRaises(ValueError):
            ResponseFormation(self.source,it,checkpoint=forgery,
                              expected_head=c["checkpoint_digest"])

    def test_source_change_invalidates_position_and_checkpoint(self):
        it=self.intent()
        f=ResponseFormation(self.source,it)
        f.advance()
        saved=f.checkpoint()
        modified=ResponseSourceGround((
            SourceSlot("reference","text", b"Changed evidence"),
            SourceSlot("answer","text",b"The interpreted response follows the admitted evidence.",
                       dependencies=("reference",)),
            SourceSlot("program","code",b"def read_state(x):\n    return x['value']\n",
                       dependencies=("reference",)),
            SourceSlot("page","page",b"# Virtual page\n\nA rendered state."),
            SourceSlot("action","tool_call",b'{"tool":"inspect_state","arguments":{"scope":"read"}}',
                       dependencies=("answer",)),
            SourceSlot("tokens","tokens",b"[44,12,8]"),
        ))
        self.assertNotEqual(
            locate_response_state(modified,it)["coordinate"],
            locate_response_state(self.source,it)["coordinate"])
        with self.assertRaises(ValueError):
            ResponseFormation(modified,it,checkpoint=saved,
                              expected_head=saved["checkpoint_digest"])

    def test_wrong_history_changes_state_position(self):
        a=self.intent(("answer",),history=({"role":"user","content":"one"},))
        b=self.intent(("answer",),history=({"role":"user","content":"two"},))
        self.assertNotEqual(locate_response_state(self.source,a)["coordinate"],
                            locate_response_state(self.source,b)["coordinate"])

    def test_missing_dependency_and_cycle_refuse(self):
        with self.assertRaises(ValueError):
            ResponseSourceGround((SourceSlot("a","text",b"x",("missing",)),))
        with self.assertRaises(ValueError):
            ResponseSourceGround((SourceSlot("a","text",b"x",("b",)),
                                  SourceSlot("b","text",b"y",("a",))))
        with self.assertRaises(ValueError):
            locate_response_state(self.source,self.intent(("undefined",)))

    def test_tool_proposal_validation_and_no_invocation(self):
        with self.assertRaises(ValueError):
            SourceSlot("call","tool_call",b'{"tool":"delete","arguments":{},"tool":"again"}')
        with self.assertRaises(ValueError):
            SourceSlot("call","tool_call",b'{"tool":"rm -rf","arguments":{}}')
        f=ResponseFormation(self.source,self.intent(("action",)))
        while not f.dag.is_complete:
            f.advance()
        outcome=f.observe()
        self.assertEqual(outcome["projections"][0]["status"],"INERT_PROPOSAL")
        self.assertEqual(outcome["projections"][0]["content"]["arguments"],{"scope":"read"})

    def test_large_page_does_not_expand_coordinate_width(self):
        big=b"# Heavy page\n"+(b"A"*250_000)
        g=ResponseSourceGround((SourceSlot("large","page",big),SourceSlot("small","text",b"ok")))
        small=ResponseFormation(g,compile_intention("small", slots=("small",), observer="test"))
        address=small.position["coordinate"]
        self.assertEqual(len(address),len("state://vli/")+64)
        while not small.dag.is_complete:
            small.advance()
        self.assertEqual(small.observe()["measured_source_bytes_materialized_this_process"],2)
        huge=ResponseFormation(g,compile_intention("large",slots=("large",),observer="test"))
        while not huge.dag.is_complete:
            huge.advance()
        self.assertEqual(huge.observe()["measured_source_bytes_materialized_this_process"],len(big))
        self.assertEqual(len(huge.position["coordinate"]),len(address))
        self.assertGreater(huge.materialized_chunks,1)

    def test_invalid_intent_and_untrusted_forgery_refuse(self):
        it=self.intent()
        it["prompt"]="different request"
        with self.assertRaises(ValueError):
            locate_response_state(self.source,it)
        with self.assertRaises(ValueError):
            compile_intention("prompt",slots=("answer",),observer="!!!")
        with self.assertRaises(ValueError):
            SourceSlot("invalid","tokens",b"[1, true]")


from vli.intent_topology import (
    TopologyWitness, propose_response_topology, admit_response_topology,
)


def example_witnesses():
    return (
        TopologyWitness("render markdown document page", ("page",)),
        TopologyWitness("show the full markdown page", ("page",)),
        TopologyWitness("display python source code", ("program",)),
        TopologyWitness("give me the python program", ("program",)),
        TopologyWitness("explain the answer in text", ("answer",)),
        TopologyWitness("read the text explanation", ("answer",)),
    )


class TestWitnessConditionedTopology(unittest.TestCase):
    def setUp(self):
        self.src = source()
        self.witnesses = example_witnesses()

    def test_ecpu_prototype_proposes_state_view_from_prompt(self):
        query = "render markdown document page"
        p = propose_response_topology(self.src, self.witnesses, query,
                                      min_confidence=0.4, min_margin=0.01)
        self.assertEqual(p["status"],"PROPOSED_NEEDS_OBSERVER_APPROVAL")
        self.assertEqual(p["proposed_slots"],["page"])
        self.assertEqual(p["ecpu_prediction"]["evidence_tier"], "E0")
        typed = admit_response_topology(
            self.src, self.witnesses, p,
            expected_proposal_head=p["proposal_digest"], observer="route-observer")
        f = ResponseFormation(self.src,typed)
        while not f.dag.is_complete:
            f.advance()
        self.assertEqual([x["kind"] for x in f.observe()["projections"]],["page"])
        self.assertEqual(f.position["required_slots"],["page"])

    def test_uncertain_intention_stays_unresolved(self):
        p = propose_response_topology(
            self.src, self.witnesses, "ambiguous unknown command",
            min_confidence=0.999999, min_margin=0.999999)
        self.assertEqual(p["status"],"INSUFFICIENT_INTENT_EVIDENCE")
        self.assertEqual(p["proposed_slots"],[])
        with self.assertRaises(ValueError):
            admit_response_topology(
                self.src,self.witnesses,p,
                expected_proposal_head=p["proposal_digest"],observer="route-observer")

    def test_proposal_requires_external_acceptance_and_unchanged_witnesses(self):
        p = propose_response_topology(self.src,self.witnesses,
                                      "render markdown document page",
                                      min_confidence=0.4,min_margin=0.01)
        with self.assertRaises(ValueError):
            admit_response_topology(
                self.src,self.witnesses,p,
                expected_proposal_head="0"*64,observer="route-observer")
        changed = list(self.witnesses)
        changed[0]=TopologyWitness("render markdown document page",("program",))
        with self.assertRaises(ValueError):
            admit_response_topology(
                self.src,tuple(changed),p,
                expected_proposal_head=p["proposal_digest"],observer="route-observer")

    def test_prototype_inference_is_reproducible_on_same_ground(self):
        a = propose_response_topology(self.src,self.witnesses,
                                      "display python source code",
                                      min_confidence=0.4,min_margin=0.01)
        b = propose_response_topology(self.src,self.witnesses,
                                      "display python source code",
                                      min_confidence=0.4,min_margin=0.01)
        self.assertEqual(a,b)

if __name__=="__main__":
    unittest.main()
