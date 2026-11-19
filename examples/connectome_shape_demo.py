"""SHAPE-001: benchmark observer-relative synthetic connectome state topology.

Measures whole-source admission, observer quotient discovery/proof, exact short
full-microstate vs smaller quotient queries, and distant-time certified query.
Uses a *real* pinned OpenWorm connectivity source to parameterize a *synthetic*
network. No biological neural dynamics or LLM cost reduction is implied.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter_ns

from vli.connectome_shape import ConnectomeObserverShape


def run(*, groups: int = 8, copies: int = 32, seed: int = 73):
    path = (Path(__file__).resolve().parents[1] / "upstream/ifs" /
            "experiments/real_connectome_ground/data/herm_full_edgelist.csv")
    original = path.read_bytes()
    shape = ConnectomeObserverShape(original, groups=groups, copies=copies,
                                   seed=seed, original_source=True)
    cost = shape.build_timings_ns.copy()
    count = 5
    trials=[]
    for source_micro,tick in ((0,5),(1,6),(17,7),(shape.synthetic.n-1,9),(8,12)):
        position = shape.position(source_micro=source_micro,
                                  logical_tick=tick, observer="lab-observer",
                                  prompt="Observe the verified future model state")
        result = shape.observe(position)
        nav = result["navigation"]
        trials.append({
            "coordinate": result["coordinate"],
            "source_microstate": source_micro, "tick":tick,
            "observer_class_dimension":shape.dimension,
            "microstate_dimension":shape.synthetic.n,
            "exact_full_micro_oracle_equal":nav["microstate_oracle_equal"],
            "quotient_sparse_edge_operations":nav["quotient_edge_operations_for_query"],
            "microstate_full_edge_visit_upper_bound":nav[
                "full_microstate_edge_visits_upper_bound"],
            "quotient_execution_ns":nav["quotient_execution_ns"],
            "microstate_reference_ns":nav["independent_microstate_reference_ns"],
            "observable_digest": result["observation_digest"],
        })
    huge = 2**40+73
    distant_position = shape.position(
        source_micro=23,logical_tick=huge,observer="lab-observer",
        prompt="Navigate to the observer's very distant computational state",
        views=("text","page"))
    far=shape.observe(distant_position)
    total_cost=sum(cost.values())
    total_nav=sum(t["quotient_execution_ns"] for t in trials)
    total_micro=sum(t["microstate_reference_ns"] for t in trials)
    return {
        "schema": "vl-i.connectome-shape-benchmark.v1",
        "source": shape.manifest,
        "source_csv_sha256": shape.source_sha256,
        "synthetic_expansion_not_original_biology": True,
        "synthetic_microstate_count": shape.synthetic.n,
        "verified_observer_quotient_dimension": shape.dimension,
        "state_dimension_ratio": shape.synthetic.n / shape.dimension,
        "preparation_ns": cost,
        "preparation_total_ns": total_cost,
        "queries": trials,
        "total_quotient_short_query_ns": total_nav,
        "total_full_micro_reference_ns": total_micro,
        "short_query_runtime_ratio_reference_over_quotient": (
            total_micro/total_nav if total_nav else None),
        "preparation_amortization_readout": (
            "one_time_proof_cost_must_be_repaid_by_qualified_future_queries"),
        "distant_observation": {
            "logical_tick":huge, "position":far["coordinate"],
            "certified_value":far["value"],
            "observer_quotient_method":far["navigation"]["method"],
            "polynomial_products":far["navigation"]["polynomial_products"],
            "full_microstate_vectors_at_query":far["navigation"][
                "source_microstate_vectors_realized_by_quotient"],
            "measured_navigation_ns":far["navigation"]["navigation_ns"],
        },
        "not_claimed": [
            "actual biological neural firing simulation or trained connectome",
            "any reduction in pretrained LLM inference time",
            "free upfront source/proof or quotient discovery",
            "guaranteed quotient for the unmodified full biological graph",
            "actual host-to-host energy reduction or network transport speedup",
            "real distant-time full-microstate brute-force oracle",
        ],
    }


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",default="artifacts/connectome-shape-observer.json")
    parser.add_argument("--groups",type=int,default=8)
    parser.add_argument("--copies",type=int,default=32)
    args=parser.parse_args(argv)
    result=run(groups=args.groups,copies=args.copies)
    output=Path(args.out)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("VL_I_CONNECTOME_SHAPE "+json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()
