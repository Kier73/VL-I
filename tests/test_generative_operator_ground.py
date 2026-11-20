"""SHAPE-007 held-out generative trained-law controls.

Native VL-64 source and operator provide reference ONLY AFTER an unseen
observation is formed. The learned law is selected using four disjoint
source observations and outputs new values for arbitrary admitted intentions.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
import unittest

from test_pretrained_range_memory import (
    fixture,FakeQualifiedRangeHost,make_memory,MODEL,REVISION)
from vli.generative_operator_ground import (
    GenerativeTrainedOperatorGround,_finite_orbit_energy)
from vli.trained_result_memory import intention
from vli.ground import _digest


class TestUnseenGenerativeLaw(unittest.TestCase):
    def setUp(self):
        self.index,self.shard,_=fixture()
        self.host=FakeQualifiedRangeHost(self.shard)
        self.source=make_memory(self.host,self.index)
        self.ground=GenerativeTrainedOperatorGround(
            self.source,self.index,allow_synthetic_fixture=True)

    def test_wrong_hypotheses_refused_and_finite_orbit_law_admitted(self):
        m=self.ground
        self.assertEqual(m.width,64)
        self.assertEqual(m.training_original_calls,4)
        self.assertEqual(m.chosen_law,"finite_orbit_identity")
        scores=m.witness["candidate_max_abs_errors"]
        self.assertLess(scores["finite_orbit_identity"],3e-9)
        self.assertGreater(scores["equidistributed"],3e-9)
        self.assertGreater(scores["constant_unit"],3e-9)
        self.assertFalse(m.original_pretrained)
        self.assertEqual(len(m.ground_digest),64)

    def test_unseen_inputs_not_retrieved_from_table_match_original_native_math(self):
        m=self.ground
        specs=(
            intention(waveform="sin",frequency=.0021,amplitude=.63,phase=.9),
            intention(waveform="cos",frequency=.2,amplitude=.33,phase=-.7),
            intention(waveform="sin",frequency=.73,amplitude=.47,phase=2.0),
            intention(waveform="cos",frequency=.00017,amplitude=.57,phase=-1.0),
        )
        chosen=(0,1,7,13,32,63)
        native_before=m.training_original_calls
        source_before=self.host.calls
        out=[]
        for spec in specs:
            r=m.observe(spec,chosen,expected_head=m.head)
            out.append(r)
            self.assertEqual(r["event"]["source_range_requests"],0)
            self.assertEqual(r["event"]["original_native_operator_evaluations"],0)
        self.assertEqual(m.training_original_calls,native_before)
        self.assertEqual(m.native_heldout_calls,0)
        self.assertEqual(self.host.calls,source_before)
        self.assertEqual(len({x["event"]["result_sha256"] for x in out}),4)
        for spec,r in zip(specs,out):
            expected=m.native_reference(spec,chosen)
            self.assertLess(max(abs(a-b) for a,b in zip(expected,r["values"])),3e-9)
        self.assertEqual(self.host.calls,source_before)

    def test_intention_proportional_sparse_projections_without_full_model_forward(self):
        m=self.ground
        spec=intention(waveform="sin",frequency=.003,amplitude=.5,phase=.6)
        one=m.observe(spec,(0,),expected_head=m.head)
        full=m.observe(spec,tuple(range(64)),expected_head=m.head)
        self.assertEqual(one["values"][0],full["values"][0])
        self.assertEqual(one["event"]["generated_coordinates"],1)
        self.assertEqual(full["event"]["generated_coordinates"],64)
        self.assertEqual(m.generated_coordinate_evaluations,65)
        self.assertEqual(one["event"]["actual_coefficient_pages_used"],1)
        self.assertEqual(m.native_heldout_calls,0)

    def test_phase_frequency_changes_form_new_states_from_same_law(self):
        m=self.ground
        spec=intention(waveform="sin",frequency=.002,phase=.2)
        changed=intention(waveform="sin",frequency=.002,phase=.7)
        first=m.observe(spec,(0,3,62),expected_head=m.head)
        second=m.observe(changed,(0,3,62),expected_head=m.head)
        self.assertNotEqual(first["position"]["coordinate"],
                            second["position"]["coordinate"])
        self.assertNotEqual(first["event"]["result_sha256"],
                            second["event"]["result_sha256"])
        self.assertEqual(m.training_original_calls,4)

    def test_finite_moment_identity_matches_true_observer_input_energy(self):
        from vli.operative_model import _input
        for width in (64,5120):
            for wave,freq,amp,phase in (
                ("sin",.00002,.3,.8),("cos",.0003,.3,-.7),
                ("sin",.4,.65,.7),("cos",.8,.1,-2.3)):
                spec=intention(waveform=wave,frequency=freq,
                               amplitude=amp,phase=phase)
                actual=math.fsum(x*x for x in _input(spec,width))/width
                exact=_finite_orbit_energy(wave,freq,amp,phase,width)
                self.assertAlmostEqual(exact,actual,delta=2e-12)

    def test_source_free_restart_can_generate_NEW_intention_not_saved_in_checkpoint(self):
        m=self.ground
        spec=intention(waveform="sin",frequency=.2,phase=.3)
        m.observe(spec,(4,19),expected_head=m.head)
        saved=m.checkpoint()
        old_calls=self.host.calls
        resumed=GenerativeTrainedOperatorGround.restore(
            self.index,saved,expected_head=saved["checkpoint_digest"])
        self.assertEqual(resumed.training_original_calls,0)
        self.assertEqual(resumed.checkpoint(),saved)
        fresh=intention(waveform="cos",frequency=.071,
                        amplitude=.25,phase=-2.2)
        result=resumed.observe(fresh,(0,1,6,31,63),
                               expected_head=resumed.head)
        self.assertEqual(self.host.calls,old_calls)
        self.assertEqual(result["event"]["original_native_operator_evaluations"],0)
        reference=m.native_reference(fresh,(0,1,6,31,63))
        self.assertLess(max(abs(a-b) for a,b in zip(reference,result["values"])),3e-9)

    def test_rehashed_forged_generative_result_is_refused_after_replay(self):
        m=self.ground
        spec=intention()
        m.observe(spec,(1,2,3),expected_head=m.head)
        saved=m.checkpoint()
        forged=deepcopy(saved)
        row=forged["events"][0]
        row["result_sha256"]="0"*64
        row["event_digest"]=_digest(
            row["schema"],{k:v for k,v in row.items() if k!="event_digest"})
        forged["head"]=row["event_digest"]
        forged["checkpoint_digest"]=_digest(
            forged["schema"],{k:v for k,v in forged.items()
                              if k!="checkpoint_digest"})
        with self.assertRaises(ValueError):
            GenerativeTrainedOperatorGround.restore(
                self.index,forged,expected_head=forged["checkpoint_digest"])

    def test_rehashed_wrong_causal_coordinate_cannot_recover(self):
        m=self.ground
        m.observe(intention(),(0,),expected_head=m.head)
        saved=m.checkpoint()
        changed=deepcopy(saved)
        changed["events"][0]["position"]="state://vli/"+"0"*64
        changed["events"][0]["event_digest"]=_digest(
            changed["events"][0]["schema"],
            {k:v for k,v in changed["events"][0].items()
             if k!="event_digest"})
        changed["head"]=changed["events"][0]["event_digest"]
        changed["checkpoint_digest"]=_digest(
            changed["schema"],{k:v for k,v in changed.items()
                               if k!="checkpoint_digest"})
        with self.assertRaises(ValueError):
            GenerativeTrainedOperatorGround.restore(
                self.index,changed,
                expected_head=changed["checkpoint_digest"])

    def test_source_change_and_tampered_coefficient_pages_refused(self):
        m=self.ground
        saved=m.checkpoint()
        changed=deepcopy(saved)
        key=next(iter(changed["gain_backing"]["local_pages"]))
        changed["gain_backing"]["local_pages"][key]="00"
        changed["checkpoint_digest"]=_digest(
            changed["schema"],{k:v for k,v in changed.items()
                               if k!="checkpoint_digest"})
        with self.assertRaises(ValueError):
            GenerativeTrainedOperatorGround.restore(
                self.index,changed,
                expected_head=changed["checkpoint_digest"])
        with self.assertRaises(ValueError):
            GenerativeTrainedOperatorGround.restore(
                b"{}",
                saved,expected_head=saved["checkpoint_digest"])

    def test_refuse_unqualified_arbitrary_hidden_inputs_and_future_semantics(self):
        m=self.ground
        with self.assertRaises(ValueError):
            m.observe(intention(frequency=.000001),(0,),
                      expected_head=m.head)
        with self.assertRaises(ValueError):
            m.observe(intention(),(-1,),expected_head=m.head)
        with self.assertRaises(ValueError):
            m.observe(intention(),(0,0),expected_head=m.head)
        with self.assertRaises(ValueError):
            m.observe(intention(),(0,),expected_head="0"*64)
        self.assertEqual(m.generated_calls,0)
        self.assertFalse(hasattr(m,"predict_next_token"))
        self.assertFalse(hasattr(m,"complete_decoder"))

    def test_disjoint_source_witness_and_heldout_input_contract(self):
        m=self.ground
        trained=set(m.witness["witness_intentions"])
        new=intention(waveform="sin",frequency=.09,
                      amplitude=.27,phase=.7)
        self.assertNotIn(new["intent_digest"],trained)
        after=m.observe(new,(2,),expected_head=m.head)
        self.assertEqual(after["event"]["original_native_operator_evaluations"],0)
        self.assertLess(abs(after["values"][0]-m.native_reference(new,(2,))[0]),
                        3e-9)


if __name__=="__main__":
    unittest.main()
