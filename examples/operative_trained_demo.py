"""Run an intention-directed trained operator on the ORIGINAL large checkpoints.

No smaller language-model substitute. No model weights live in the Ground.
Public pinned HF shard ranges supply exactly selected model relation inputs.
The operator is partial trained math, NOT full inference or learned paragraphs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vli.operative_model import ground, intention, TrainedOperationSession
from vli.large_model_comparison import SOURCES


def run(route, *, render=True, two_positions=True):
    from experiments.model_fabric.scaled_model_fabric import (
        QualifiedShardNavigator,MAX_INDEX_BYTES)
    from experiments.model_fabric.scaled_model_lab import (
        _url,_fetch_metadata,pinned_hf_range_host)
    if route not in SOURCES:
        raise ValueError("Unqualified original-scale model")
    entry=SOURCES[route]
    model,revision=entry["model"],entry["revision"]
    source=_fetch_metadata(_url(model,revision,"model.safetensors.index.json"),
                           MAX_INDEX_BYTES)
    nav=QualifiedShardNavigator(model=model,revision=revision,index_raw=source,
         range_host=pinned_hf_range_host(model,revision),
         approved_hosts={"pinned-huggingface-https-range"})
    g=ground(route,nav.index)
    session=TrainedOperationSession(g,observer="vli-research")
    first=session.operate(nav,intention(g,frequency=.017,phase=0.,amplitude=.125),
                          expected_head=session.head,render=render)
    pin=session.checkpoint()
    restored=TrainedOperationSession.restore(g,pin,
                          expected_head=pin["checkpoint_digest"])
    outputs=[first]
    if two_positions:
        second=restored.operate(nav,intention(g,frequency=.019,phase=.6,amplitude=.125),
                                expected_head=restored.head,render=render)
        if second["position"]["coordinate"]==first["position"]["coordinate"]:
            raise ValueError("Different intention did not select a new state")
        outputs.append(second)
    records=[]
    for output in outputs:
        r=output["trained"]
        records.append({
            "coordinate":output["position"]["coordinate"],
            "model":r["model"],"revision":r["revision"],
            "intention_digest":r["intent_digest"],
            "computed_relation":r["numerical_result"]["kind"],
            "computed_values":r["numerical_result"],
            "constructed":output["constructed"],
            "source_weight_bytes_read":r["new_weight_bytes"],
            "qualified_range_reads":r["new_qualified_ranges"],
            "range_witness":r["range_chain_head"],
            "observation_digest":r["observation_digest"],
            "rendered_state":(output["rendered"]["state"] if output["rendered"] else None),
            "generated_code":output["views"]["code"],
        })
    return {"schema":"vl-i.operative-pretrained-interaction-lab.v1",
        "status":"REAL_PINNED_PRETRAINED_PARTIAL_OPERATIONS_EXECUTED",
        "model":model,"revision":revision,"route":route,
        "full_checkpoint_payload_advertised_bytes":entry["advertised_bytes"],
        "source_index_sha256":entry["index_sha256"],
        "state_observations":records,
        "weight_payload_bytes_stored_in_virtual_ground":0,
        "causal_checkpoint_recovered":restored.head==pin["head"],
        "new_operator_inputs_are_observer_selected":True,
        "all_layer_language_inference_completed":False,
        "scope":"real_original_checkpoint_typed_operators_to_generated_views"}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",choices=tuple(SOURCES),required=True)
    p.add_argument("--no-render",action="store_true")
    p.add_argument("--one-position",action="store_true")
    p.add_argument("--out",required=True)
    args=p.parse_args(argv)
    target=Path(args.out);target.parent.mkdir(parents=True,exist_ok=True)
    try:
        report=run(args.model,render=not args.no_render,
                   two_positions=not args.one_position)
    except (ValueError,RuntimeError,OSError,TimeoutError,ImportError) as exc:
        report={"schema":"vl-i.operative-pretrained-interaction-lab.v1",
                "status":"UNRESOLVED","route":args.model,
                "error_type":type(exc).__name__,
                "no_model_inference_or_numeric_completion_claimed":True}
    target.write_text(json.dumps(report,allow_nan=False,
                      sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print("VL_I_OPERATIVE_PRETRAINED",report["status"],report["route"])
    return 0 if report["status"].startswith("REAL_") else 1


if __name__=="__main__":
    raise SystemExit(main())
