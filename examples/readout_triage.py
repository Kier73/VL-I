"""Read-only build assistant for the cross-architecture readout experiment.

At each step the observation vector x encodes available evidence. Select the
allowed tool maximizing w_tool·x+b_tool via VL-I's exact navigated comparisons.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from .multi_model_readout import MODELS, compile_rows, navigate


# x = (another model, active trial, missing evidence, disagreement, malformed).
# An enabled higher-priority action dominates all enabled lower priorities.
WEIGHTS = {
    "finish":           (0, 0, 0, 0, 0),
    "read_model":       (1, 0, 0, 0, 0),
    "compare_trial":    (0, 2, 0, 0, 0),
    "report_missing":   (0, 0, 3, 0, 0),
    "report_mismatch":  (0, 0, 0, 4, 0),
    "report_malformed": (0, 0, 0, 0, 5),
}
CONTRASTS = {(i,j): compile_rows(wi,wj,0,0) for i,wi in WEIGHTS.items()
             for j,wj in WEIGHTS.items() if i != j}


def choose(context):
    """Inductively maintain argmax over visited rows; ties keep the first row."""
    winner = "finish"
    for candidate in WEIGHTS:
        if candidate != winner and navigate(CONTRASTS[candidate,winner], context)[0] > 0:
            winner = candidate
    return winner


def triage(rows):
    """Check every expected checkpoint and trial until a discrepancy supplies a witness."""
    panel = {name: (rev, cls, mode) for name, rev, cls, mode in MODELS}
    missing = mismatch = malformed = False
    queue = []
    index = trial = checked = 0
    counts = Counter()
    current = None
    detail = {}
    if not isinstance(rows, list) or any(not isinstance(r, dict) or not isinstance(r.get("model"), str)
                                         for r in rows):
        malformed, detail = True, {"reason": "invalid model inventory"}
    else:
        names = [r["model"] for r in rows]
        if len(set(names)) != len(names) or set(names) - panel.keys():
            malformed, detail = True, {"reason": "duplicate or unexpected model"}
        else:
            by_name = {r["model"]: r for r in rows}
            queue = [(name, by_name.get(name)) for name in panel]
    while True:
        active = current is not None and trial < len(current["trials"])
        tool = choose([index < len(queue), active, missing, mismatch, malformed])
        counts[tool] += 1
        if tool == "read_model":
            name, record = queue[index]
            index += 1
            trial = 0
            current = None
            detail = {"model": name}
            revision, cls, mode = panel[name]
            if record is None:
                missing, detail["reason"] = True, "missing checkpoint"
            elif (record.get("revision"),record.get("reference_class"),record.get("mode")) != (revision,cls,mode):
                malformed, detail["reason"] = True, "checkpoint or reference differs"
            elif record.get("status") == "unresolved":
                missing, detail["reason"] = True, record.get("reason", "unresolved")
            elif (record.get("status") not in ("agreed","disagreement")
                  or type(record.get("width_including_bias")) is not int or record["width_including_bias"] < 2
                  or not isinstance(record.get("trials"), list)
                  or len(record["trials"]) != (2 if mode == "vision" else 12)
                  or record.get("pair") != ([0,1] if mode == "vision" else [42,75])):
                malformed, detail["reason"] = True, "invalid readout contract"
            else:
                current = record
        elif tool == "compare_trial":
            observation = current["trials"][trial]
            detail = {"model": current["model"], "trial": trial}
            trial += 1
            checked += 1
            if not isinstance(observation, dict):
                malformed, detail["reason"] = True, "invalid trial"
                continue
            sign, gap, terms = (observation.get(k) for k in ("sign","native_gap","terms"))
            if (type(sign) is not int or sign not in (-1,0,1)
                    or type(gap) not in (float,int) or not math.isfinite(gap)
                    or type(terms) is not int or not 1 <= terms <= current["width_including_bias"]):
                malformed, detail["reason"] = True, "invalid numerical evidence"
                continue
            # The independent oracle verdict is sign(ℓ_i−ℓ_j), including zero.
            expected = (gap > 0) - (gap < 0)
            if sign != expected:
                mismatch = True
                detail.update(candidate_sign=sign, native_gap=gap, expected_sign=expected)
            elif observation.get("matches") is not True or (trial == len(current["trials"]) and
                                                            current["status"] != "agreed"):
                malformed, detail["reason"] = True, "summary contradicts recorded scores"
        else:
            return {"action": tool, "checked_trials": checked, "tool_calls": dict(counts),
                    "witness": {} if tool == "finish" else detail}


def main():
    """Return success iff the controller reaches finish after the full evidence trajectory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    report = triage(json.loads(args.receipt.read_bytes()))
    print(json.dumps(report, sort_keys=True))
    return int(report["action"] != "finish")


if __name__ == "__main__":
    raise SystemExit(main())
