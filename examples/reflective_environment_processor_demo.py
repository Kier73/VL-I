"""TIME-004: two source-derived processors, each self-interpreted and verified."""
import json
from hashlib import sha256

from vli.temporal_intelligence import TemporalIntelligenceGround
from vli.live_temporal_intelligence import LiveTemporalObserver
from vli.evolving_environment_language import (
    EvolvingEnvironmentLanguage, certify_language_migration,
)
from vli.reflective_environment_processor import (
    construct_reflective_environment_processor,
    verify_reflective_environment_processor,
    certify_reflective_interpreter_succession,
    verify_reflective_interpreter_succession,
)


def main():
    ground=TemporalIntelligenceGround("lab:reflective-continuity")
    journal=[]
    agent=LiveTemporalObserver(ground,host="observer-A",
                               checkpoint_sink=journal.append)
    learner=EvolvingEnvironmentLanguage(ground)
    admission=[]
    for time,value in enumerate([1,3,5,100,101,104,109]):
        digest=sha256(f"source:{time}:{value}".encode()).hexdigest()
        agent.ingest({"schema":"vl-i.source-sample.v1",
                      "sequence":len(ground.events),
                      "time":f"{time}/1","values":[f"{value}/1"],
                      "source_digest":digest})
        current=learner.current(expected_head=ground.head)
        if current and current["family"] not in [a[1]["family"] for a in admission]:
            snapshot=ground.checkpoint()
            coord=0 if not admission else 17
            result=construct_reflective_environment_processor(
                snapshot,current,expected_head=ground.head,
                expected_identity=ground.identity,
                representation_coordinate=coord)
            assert verify_reflective_environment_processor(
                snapshot,current,result,expected_head=ground.head,
                expected_identity=ground.identity)
            admission.append((snapshot,current,result))
    assert [x[1]["family"] for x in admission]==["affine","quadratic"]
    (s0,p0,b0),(s1,p1,b1)=admission
    m=certify_language_migration(s0,p0,s1,p1,
         before_head=s0["head"],after_head=s1["head"],identity=ground.identity)
    succession=certify_reflective_interpreter_succession(
         s0,p0,b0,s1,p1,b1,m,
         old_head=s0["head"],new_head=s1["head"],identity=ground.identity)
    assert verify_reflective_interpreter_succession(
         s0,p0,b0,s1,p1,b1,m,succession,
         old_head=s0["head"],new_head=s1["head"],identity=ground.identity)
    print(json.dumps({
        "schema":"vl-i.time-004-reflective-environment-evidence.v1",
        "continuous_source_identity":ground.identity,
        "live_observations":agent.accepted_in_lifetime,
        "environment_families":[p["family"] for _,p,_ in admission],
        "constructed_task_algebras":[b["generated_algebra_digest"]
                                     for _,_,b in admission],
        "constructed_self_interpreters":[b["generated_interpreter_digest"]
                                         for _,_,b in admission],
        "primitive_bases":[b["primitive_seed"] for _,_,b in admission],
        "observed_result_values":[b["result_recorded_value"]
                                  for _,_,b in admission],
        "reflection_expression_steps":[
            [r["expression_steps"] for r in b["execution_receipts"]]
            for _,_,b in admission],
        "preserved_succession":succession,
        "source_evidence_not_fabricated":True,
        "limits":{"future_behavioral_equivalence":False,
                  "source_carries_authentic_signature":False,
                  "environment_dynamics_realized_by_reflective_processor":False,
                  "general_language_self_creation":False,
                  "subjective_consciousness_continuity":False},
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
