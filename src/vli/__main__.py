"""VL-I command line.

Requires the pinned VL-64 upstream source for the manifest compiler and native
pretrained operator execution. The default inspect command reads no weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ground import load_json, lift_observation, intention_frontier, verify_frontier
from .realization import run_pinned_decoder, verify_realization
from .navigation import commit_continuation, navigate_continuation


def _save(path, record):
    raw = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path == "-":
        print(raw, end="")
    else:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(raw, encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="vli", description="Lift pinned model observations and realize qualified VL-native outputs"
    )
    sub = p.add_subparsers(dest="operation", required=True)
    i = sub.add_parser("inspect", help="Lift typed model and select an intention frontier; no weights")
    i.add_argument("--manifest", required=True)
    i.add_argument("--target", action="append", default=["vocab_logits"])
    i.add_argument("--out", default="-")

    r = sub.add_parser("run", help="Execute complete pretrained decoder with real qualified ranges")
    r.add_argument("--manifest", required=True)
    r.add_argument("--tokens", nargs="+", type=int, default=[42, 75])
    r.add_argument("--state-dir", default="artifacts/vli-001-state")
    r.add_argument("--upstream-out", default="artifacts/vli-001-upstream.json")
    r.add_argument("--out", default="artifacts/vli-001-result.json")

    v = sub.add_parser("verify", help="Verify recorded source/frontier binding (not numerical re-execution)")
    v.add_argument("--manifest", required=True)
    v.add_argument("--report", required=True)

    memo = sub.add_parser("remember", help="Retain one B2-verified exact-input next-token observation")
    memo.add_argument("--manifest", required=True)
    memo.add_argument("--report", required=True)
    memo.add_argument("--out", default="artifacts/vli-retained-continuation.json")

    nav = sub.add_parser("navigate", help="Recover the exact same token from a pinned observed state")
    nav.add_argument("--manifest", required=True)
    nav.add_argument("--retained", required=True)
    nav.add_argument("--expected-head", required=True,
                     help="Trusted SHA-256 observation digest retained outside the cache")
    nav.add_argument("--tokens", nargs="+", type=int, required=True)
    nav.add_argument("--out", default="-")

    args = p.parse_args(argv)
    observed = load_json(args.manifest)
    if args.operation == "run":
        result = run_pinned_decoder(observed, args.tokens, state_dir=args.state_dir,
                                    result_path=args.upstream_out)
        _save(args.out, result)
        return 0
    ground, plan = lift_observation(observed)
    if args.operation == "inspect":
        frontier = intention_frontier(plan, args.target)
        verify_frontier(frontier, plan)
        _save(args.out, {"ground": ground, "frontier": frontier})
        return 0
    frontier = intention_frontier(plan, ["vocab_logits"])
    verify_frontier(frontier, plan)
    if args.operation == "navigate":
        retained = load_json(args.retained)
        answer = navigate_continuation(
            ground, frontier, retained, input_token_ids=args.tokens,
            expected_observation_digest=args.expected_head)
        _save(args.out, answer)
        return 0
    report = load_json(args.report)
    if report.get("ground") != ground or report.get("frontier") != frontier:
        raise ValueError("Reconstructed Ground/frontier differs from recorded evidence")
    verify_realization(ground, frontier, report.get("realization", {}))
    if args.operation == "remember":
        record = commit_continuation(ground, frontier, report["realization"])
        _save(args.out, record)
        print("VL_I_EXPECTED_OBSERVATION_HEAD " + record["observation_digest"])
        return 0
    print("VL_I_STRUCTURAL_RECEIPT_VALID — numerical equivalence requires reference replay")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
