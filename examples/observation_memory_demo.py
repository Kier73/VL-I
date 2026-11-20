"""SHAPE-004: compare exact black-box source acquisition with structural memory.

Reuse original IFS COG-008 unchanged; the observer-host transparently saves
pure source-state prefixes. Proven model, IFS ledger and generated neural
parameters must match the *uncached* baseline byte-for-byte.
"""
from __future__ import annotations

import argparse
import json
from statistics import median
from pathlib import Path
from time import perf_counter_ns

from vli.learned_morphology import LearnedObserverGround
from vli.observation_memory import StructuralPrefixMemory


def run():
    raw=(Path(__file__).resolve().parents[1]/
         "experiments/shape002/morphology_model.json").read_bytes()
    scopes=(("phase",),("micro",),("phase","micro"))
    summaries={}
    total_baseline_steps=0
    total_cached_steps=0
    t0=perf_counter_ns()
    baseline={}
    for scope in scopes:
        start=perf_counter_ns()
        g=LearnedObserverGround(raw,scope)
        elapsed=perf_counter_ns()-start
        baseline[scope]=g
        total_baseline_steps+=g.source.direct_transition_steps
        summaries["+".join(scope)]={
            "scope":list(scope),
            "uncached_original_transition_steps":g.source.direct_transition_steps,
            "uncached_observational_membership_calls":g.oracle.external_calls,
            "uncached_acquire_and_neural_compile_ns":elapsed,
            "learned_state_count":len(g.machine.representatives),
            "generated_weights_digest":g.neural_ground.weights_digest,
            "IFS_evidence_head":g.oracle.head,
        }
    t1=perf_counter_ns()
    memory=StructuralPrefixMemory(raw)
    for scope in scopes:
        start=perf_counter_ns()
        g=LearnedObserverGround(raw,scope,structural_memory=memory)
        elapsed=perf_counter_ns()-start
        total_cached_steps+=g.source.direct_transition_steps
        b=baseline[scope]
        assert g.machine.document()==b.machine.document()
        assert g.ifs_checkpoint==b.ifs_checkpoint
        assert g.manifest==b.manifest
        assert g.neural_ground.weights_digest==b.neural_ground.weights_digest
        s=summaries["+".join(scope)]
        s["cached_actual_original_transition_steps"]=g.source.direct_transition_steps
        s["cached_observational_membership_calls"]=g.oracle.external_calls
        s["cached_acquire_and_neural_compile_ns"]=elapsed
        s["independent_global_reference_qualifies_same_model"]=g.verdict==b.verdict
    t2=perf_counter_ns()
    snapshot=memory.checkpoint()
    verification_start=perf_counter_ns()
    restored=StructuralPrefixMemory.restore(
        raw,snapshot,expected_head=snapshot["checkpoint_digest"])
    snapshot_verification_ns=perf_counter_ns()-verification_start
    assert restored.checkpoint()==snapshot
    assert total_cached_steps==memory.evaluated_transitions
    assert total_baseline_steps==memory.naive_transition_steps

    # Repeat with *warm imports* and alternate orders so Python's first
    # module load does not masquerade as a structural-memory speedup.
    warm_trials=[]
    for trial in range(6):
        order=("uncached","cached") if trial%2==0 else ("cached","uncached")
        sample={"order":list(order)}
        for mode in order:
            start=perf_counter_ns()
            trial_mem=StructuralPrefixMemory(raw) if mode=="cached" else None
            trial_roots=[]
            for scope in scopes:
                m=LearnedObserverGround(
                    raw,scope,structural_memory=trial_mem)
                trial_roots.append(m.manifest["ground_digest"])
            total_ns=perf_counter_ns()-start
            sample[mode+"_ns"]=total_ns
            _expected=[baseline[scope].manifest["ground_digest"]
                       for scope in scopes]
            assert trial_roots==_expected
            if mode=="cached":
                sample["cached_actual_source_steps"]=trial_mem.evaluated_transitions
        warm_trials.append(sample)
    median_uncached=median(x["uncached_ns"] for x in warm_trials)
    median_cached=median(x["cached_ns"] for x in warm_trials)
    return {
        "schema":"vl-i.shape004-verified-prefix-memory-cost.v1",
        "original_source_sha256":memory.source_sha256,
        "source_deterministic_pure":True,
        "black_box_observer_only_accesses_projected_outputs":True,
        "IFS_observational_learner_unchanged":True,
        "original_source_state_count":len(memory.reference.states),
        "original_action_count":len(memory.reference.actions),
        "observer_morphologies":summaries,
        "total_uncached_source_transition_evaluations":total_baseline_steps,
        "total_cached_source_transition_evaluations":total_cached_steps,
        "transition_evaluations_saved":total_baseline_steps-total_cached_steps,
        "transition_evaluation_reduction_ratio":total_baseline_steps/total_cached_steps,
        "verified_prefix_nodes":memory.unique_nodes,
        "prefix_memory_evidence":memory.stats(),
        "source_snapshot_head":snapshot["checkpoint_digest"],
        "source_snapshot_bytes":len(json.dumps(snapshot,sort_keys=True)),
        "snapshot_independent_validation_transitions":restored.restoration_validation_steps,
        "snapshot_independent_validation_ns":snapshot_verification_ns,
        "whole_baseline_acquisition_and_generation_ns":t1-t0,
        "whole_cached_acquisition_and_generation_ns":t2-t1,
        "warm_order_alternated_trials":warm_trials,
        "warm_median_uncached_acquisition_ns":median_uncached,
        "warm_median_cached_acquisition_ns":median_cached,
        "warm_median_wall_ratio_uncached_over_cached":(
            median_uncached/median_cached if median_cached else None),
        "warm_benchmark_scope":"three_shape_learning_compilation_including_source_parsing_and_cache_bootstrap;_six_alternating_trials",
        "exact_generated_model_and_original_ifs_ledger_equal":True,
        "not_claimed":[
            "a reduction in pretrained language-model forward-pass cost",
            "ability to reuse effectful, nondeterministic or unauthenticated remote transitions",
            "free source acquisition or verification",
            "wall-clock speedup from a source-step reduction alone",
            "globally unique causal identity without a trusted external checkpoint pin",
        ],
    }


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",default="artifacts/shape004-observation-memory.json")
    opt=parser.parse_args(argv)
    r=run()
    target=Path(opt.out)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("VL_I_SHAPE004 "+json.dumps(r,sort_keys=True))


if __name__=="__main__":
    main()
