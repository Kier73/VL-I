"""TIME-001: derivation, drift and host-continuity demonstrator."""
from __future__ import annotations

from hashlib import sha256
import json

from vli.temporal_intelligence import (
    TemporalIntelligenceGround, verify_temporal_navigation,
    verify_temporal_processor, verify_temporal_snapshot,
)


def main():
    g = TemporalIntelligenceGround("laboratory:changing-observed-flow",
                                    observer="time-relative-vl-i")
    inputs = [(0,1),(1,3),(2,5),(3,7),(4,9),(5,30),(6,31),(7,32)]
    first_snapshot = None
    for t, x in inputs:
        if t == 5:
            first_snapshot = g.checkpoint()
            g = TemporalIntelligenceGround.restore(
                first_snapshot, expected_head=g.head,
                expected_identity=g.identity)
        host = "observer-A" if t < 5 else "observer-B"
        source = sha256(f"source-sample:{t}:{x}".encode()).hexdigest()
        g.observe(f"{t}/1",[f"{x}/1"],source_digest=source,
                  host=host,expected_head=g.head)
    snapshot = g.checkpoint()
    models = [g.compile_epoch(i,expected_head=g.head)
              for i in range(len(g.epochs))]
    for p in models:
        verify_temporal_processor(snapshot,p,expected_head=g.head,
                                  expected_identity=g.identity)
    early = g.navigate("5/2",epoch_index=0,expected_head=g.head)
    later = g.navigate("11/2",epoch_index=1,expected_head=g.head)
    for n in (early,later):
        verify_temporal_navigation(snapshot,n,expected_head=g.head,
                                   expected_identity=g.identity)
    verify_temporal_snapshot(snapshot,expected_head=g.head,
                             expected_identity=g.identity)
    result = {"schema":"vl-i.time-001-demonstration.v1",
              "ground_identity":g.identity,"latest_observed_head":g.head,
              "observations":len(g.events),"epochs":len(g.epochs),
              "rates":[p["rate"] for p in models],
              "source_hosts":sorted({e["host"] for e in g.events}),
              "environment_change_uncertainty":
                   g.epochs[1]["uncertain_change_interval"],
              "first_interpolation":early,
              "second_interpolation":later,
              "claims":{"inferred_law":"empirically supported at sampled times",
                        "interpolation":"not an observation of the world",
                        "self_interpreter":"not demonstrated",
                        "continuous_host_uptime":"not demonstrated"}}
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__ == "__main__":
    main()
