"""TIME-006: signed quadratic and recurrent laws through NCR-002 self-interpreter."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vli.ground import load_json
from vli.temporal_intelligence import TemporalIntelligenceGround
from vli.authenticated_temporal_source import (
    _genesis,append_signed_observation,public_key_hex,
    sign_source_observation,resume_signed_observer,
)
from vli.evolving_environment_language import EvolvingEnvironmentLanguage
from vli.numerical_reflective_ground import (
    construct_authenticated_numerical_processor,
    verify_authenticated_numerical_processor,
)


def main():
    output=[]
    secret=Ed25519PrivateKey.generate()
    pub=public_key_hex(secret)
    with tempfile.TemporaryDirectory() as temp:
        for label,values,horizon in [
            ("quadratic",[0,1,4,9],10),
            ("recurrence2",[1,1,2,3,5],10),
        ]:
            folder=Path(temp)/label
            folder.mkdir()
            journal=folder/"signed.jsonl"
            checkpoint=folder/"observer.json"
            source="time006:publisher:"+label
            head=_genesis(source,pub)
            for i,val in enumerate(values):
                signed=sign_source_observation(
                    secret,source=source,sequence=i,
                    time=f"{i}/1",value=f"{val}/1")
                head,_=append_signed_observation(
                    journal,signed,expected_head=head,expected_count=i,
                    source=source,public_key=pub)
            report=resume_signed_observer(
                journal_path=journal,checkpoint_path=checkpoint,
                source=source,public_key=pub,
                expected_source_head=head,expected_source_count=len(values),
                environment="time006:"+label,host="source-qualified-host")
            snapshot=load_json(checkpoint)
            ground=TemporalIntelligenceGround.restore(
                snapshot,expected_head=snapshot["head"],
                expected_identity=report["observer_identity"])
            capsule=EvolvingEnvironmentLanguage(ground).current(
                expected_head=ground.head)
            arguments={
                "source_journal":journal,
                "source_id":source,
                "public_key":pub,
                "expected_source_head":head,
                "expected_source_count":len(values),
                "expected_ground_head":ground.head,
                "expected_ground_identity":ground.identity,
            }
            processor=construct_authenticated_numerical_processor(
                snapshot,capsule,horizon=horizon,
                allow_projection=True,**arguments)
            assert verify_authenticated_numerical_processor(
                snapshot,capsule,processor,**arguments)
            compiled=processor["generated_semantic_processor"]
            output.append({
                "qualifying_signed_samples":len(values),
                "family":label,
                "result_time":processor["result_time"],
                "result_value":processor["result_exact_value"],
                "result_status":processor["result_evidence_status"],
                "source_head":head,
                "compiled_numeric_states":compiled["compiled_state_count"],
                "semantic_action_strides":processor["jump_strides"],
                "semantic_instruction_word":compiled["task_instruction_word"],
                "independent_reflection_costs":[r["expression_steps"]
                    for r in compiled["depth_receipts"]],
                "direct_naive_cost":compiled["naive_unrolled_direct_steps"],
                "source_qualified_generated_interpreter":
                    compiled["ncr_reflective_interpreter_digest"],
                "runtime_semantics":processor["runtime_arithmetic"],
                "numeric_processor_digest":processor["numerical_processor_digest"],
            })
    print(json.dumps({
        "schema":"vl-i.time006-exact-numeric-self-reflection-experiment.v1",
        "signed_source_not_equal_to_physical_truth":True,
        "compiled_numeric_operations_not_unbounded_host_arithmetic":True,
        "future_states_are_unverified_projections":True,
        "experiments":output,
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
