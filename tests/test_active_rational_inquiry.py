"""ACT-001: signed rational multi-observer probes and independent replay."""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vli.ground import _digest
from vli.authenticated_temporal_source import (
    _genesis, public_key_hex, sign_source_observation,
    append_signed_observation,
)
from vli.active_rational_inquiry import (
    SCHEMA, OFFER, begin_inquiry, choose_probe, accept_signed_probe,
    audit_inquiry, qualify_shared_law, verify_shared_law,
    navigate_shared_law,
)


class MultiObserverInquiryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.spec = {}
        self.seq = {"A": [1, 1, 1, 1, 1, 1],
                    "B": [1, 2, 4, 8, 16, 32]}
        for ident,cost in (("A", "1/1"), ("B", "2/1")):
            key = Ed25519PrivateKey.generate()
            pub = public_key_hex(key)
            source = "act001-source-" + ident
            self.spec[ident] = {
                "id": ident, "source": source, "key": key,
                "public_key": pub,
                "journal_path": self.root / (ident + ".jsonl"),
                "head": _genesis(source,pub), "count": 0,
                "cost": cost,
            }
            self.append(ident, 2)

    def append(self, channel, count, *, replace=None, time_override=None):
        c=self.spec[channel]
        for _ in range(count):
            idx=c["count"]
            value = self.seq[channel][idx] if replace is None else replace
            time=Fraction(idx) if time_override is None else Fraction(time_override)
            packet=sign_source_observation(
                c["key"],source=c["source"],sequence=idx,
                time=f"{time.numerator}/{time.denominator}",value=f"{value}/1")
            head,n=append_signed_observation(
                c["journal_path"],packet,
                expected_head=c["head"],expected_count=idx,
                source=c["source"],public_key=c["public_key"])
            c["head"],c["count"]=head,n

    def descriptors(self):
        return [{
            "id": c["id"], "source": c["source"],
            "public_key": c["public_key"],
            "journal_path": c["journal_path"],
            "expected_head": c["head"],
            "expected_count": c["count"], "cost": c["cost"],
        } for c in self.spec.values()]

    def external(self):
        paths={k:c["journal_path"] for k,c in self.spec.items()}
        pins={k:{"head":c["head"],"count":c["count"]}
              for k,c in self.spec.items()}
        return dict(journal_paths=paths,final_source_pins=pins)

    def identify(self):
        session=begin_inquiry(self.descriptors(),order=2)
        self.assertEqual(session["rank"],0)
        p1=choose_probe(session)
        self.assertEqual(p1["channel"],"A")
        self.assertEqual(p1["guaranteed_rank_gain"],1)
        self.assertFalse(p1["selected_using_future_value"])
        self.append("A",1)
        session=accept_signed_probe(
            session,p1,journal_path=self.spec["A"]["journal_path"],
            expected_next_head=self.spec["A"]["head"])
        self.assertEqual(session["rank"],1)
        p2=choose_probe(session)
        self.assertEqual(p2["channel"],"B")
        self.assertEqual(p2["guaranteed_rank_gain"],1)
        self.assertEqual(p2["regressor"],["2/1","1/1"])
        self.append("B",1)
        session=accept_signed_probe(
            session,p2,journal_path=self.spec["B"]["journal_path"],
            expected_next_head=self.spec["B"]["head"])
        self.assertEqual(session["rank"],2)
        return session

    def test_complementary_signed_observers_identify_two_dimensional_law(self):
        session=self.identify()
        self.append("A",2)
        self.append("B",2)
        self.assertTrue(audit_inquiry(session,**self.external()))
        law=qualify_shared_law(session,**self.external())
        self.assertEqual(law["coefficients"],["3/1","-2/1"])
        self.assertTrue(verify_shared_law(
            session,law,**self.external()))
        self.assertEqual([c["challenged_holdout_count"]
                          for c in law["channels"]],[2,2])
        self.assertEqual(navigate_shared_law(
            law,"A",100,allow_projection=True)["value"],"1/1")
        self.assertEqual(navigate_shared_law(
            law,"B",10,allow_projection=True)["value"],"1024/1")
        self.assertEqual(sum(Fraction(row["receipt"]["offer"]["cost"])
                             for row in session["accepted"]),3)
        self.assertEqual(len(session["accepted"]),2)
        with self.assertRaises(ValueError):
            navigate_shared_law(law,"B",10)

    def test_independent_rank_auditor_rejects_forged_probe_choice(self):
        session=begin_inquiry(self.descriptors(),order=2)
        offer=choose_probe(session)
        changed=deepcopy(offer)
        changed["channel"]="B"
        changed["offer_digest"]=_digest(
            OFFER,{k:v for k,v in changed.items() if k!="offer_digest"})
        self.append("A",1)
        with self.assertRaises(ValueError):
            accept_signed_probe(session,changed,
                journal_path=self.spec["A"]["journal_path"],
                expected_next_head=self.spec["A"]["head"])
        accepted=accept_signed_probe(session,offer,
            journal_path=self.spec["A"]["journal_path"],
            expected_next_head=self.spec["A"]["head"])
        self.append("B",1)
        o2=choose_probe(accepted)
        complete=accept_signed_probe(accepted,o2,
            journal_path=self.spec["B"]["journal_path"],
            expected_next_head=self.spec["B"]["head"])
        self.append("A",2)
        self.append("B",2)
        spoof=deepcopy(complete)
        spoof["accepted"][0]["value"]="200/1"
        spoof["inquiry_digest"]=_digest(
            SCHEMA,{k:v for k,v in spoof.items() if k!="inquiry_digest"})
        with self.assertRaises(ValueError):
            audit_inquiry(spoof,**self.external())

    def test_valid_signature_does_not_turn_counterexample_into_law(self):
        session=self.identify()
        self.append("A",2)
        self.append("B",1,replace=9)  # signed and genuinely source-observed
        self.append("B",1)
        self.assertTrue(audit_inquiry(session,**self.external()))
        with self.assertRaises(ValueError):
            qualify_shared_law(session,**self.external())

    def test_signed_but_time_misaligned_heldout_is_not_a_law_witness(self):
        session=self.identify()
        self.append("A",2)
        self.append("B",1,time_override=Fraction(7,2))
        self.append("B",1)
        self.assertTrue(audit_inquiry(session,**self.external()))
        with self.assertRaisesRegex(ValueError, "sampling grid"):
            qualify_shared_law(session,**self.external())

    def test_stale_publisher_head_and_source_alignment_refuse(self):
        desc=self.descriptors()
        desc[1]["expected_head"]="0"*64
        with self.assertRaises(ValueError):
            begin_inquiry(desc,order=2)
        desc=self.descriptors()
        desc[1]["cost"]="0/1"
        with self.assertRaises(ValueError):
            begin_inquiry(desc,order=2)
        session=begin_inquiry(self.descriptors(),order=2)
        probe=choose_probe(session)
        self.append(probe["channel"],1)
        with self.assertRaises(ValueError):
            accept_signed_probe(session,probe,
                journal_path=self.spec[probe["channel"]]["journal_path"],
                expected_next_head="0"*64)
        with self.assertRaises(ValueError):
            begin_inquiry(self.descriptors(),order=4)


if __name__=="__main__":
    unittest.main()
