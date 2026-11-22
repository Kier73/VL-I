"""End-to-end challenges over the actual retained checkpoint outcomes."""
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import unittest

from examples.readout_triage import choose, triage


RECEIPT = Path(__file__).resolve().parents[1] / "experiments/architecture_survey/2026-11-22-trained-readouts.json"


class TestReadoutTriage(unittest.TestCase):
    def test_all_observation_states(self):
        """For x∈{0,1}⁵, the highest-priority enabled tool must win, including the empty set."""
        for context in product((False,True),repeat=5):
            actions = ("read_model","compare_trial","report_missing","report_mismatch","report_malformed")
            enabled = [name for name, bit in zip(actions,context) if bit]
            self.assertEqual(choose(context), enabled[-1] if enabled else "finish")

    def test_every_single_trial_regression(self):
        """Flipping s_k while retaining a green summary must yield witness k for all 110 k."""
        rows = json.loads(RECEIPT.read_bytes())
        clean = triage(rows)
        self.assertEqual((clean["action"],clean["checked_trials"]), ("finish",110))
        for row in rows:
            for k, observation in enumerate(row["trials"]):
                before = observation["sign"]
                observation["sign"] = -before if before else 1
                result = triage(rows)
                observation["sign"] = before
                self.assertEqual(result["action"],"report_mismatch")
                self.assertEqual((result["witness"]["model"],result["witness"]["trial"]), (row["model"],k))

    def test_incomplete_and_inconsistent_evidence(self):
        """Missing observations are unresolved; an identity or finite-value contradiction is malformed."""
        rows = json.loads(RECEIPT.read_bytes())
        self.assertEqual(triage(rows[:-1])["action"], "report_missing")
        self.assertEqual(triage(rows + rows[:1])["action"], "report_malformed")
        for field, value, verdict in (("revision","a"*40,"report_malformed"),
                                      ("status","unresolved","report_missing"),
                                      ("trials",[],"report_malformed")):
            altered = deepcopy(rows)
            altered[0][field] = value
            self.assertEqual(triage(altered)["action"], verdict)
        for field, value in (("native_gap",float("nan")), ("matches",False), ("terms",0)):
            altered = deepcopy(rows)
            altered[0]["trials"][0][field] = value
            self.assertEqual(triage(altered)["action"], "report_malformed")
