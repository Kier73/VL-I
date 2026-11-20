"""TIME-007: an intention selects a state; no intervening states are realized."""
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import tempfile

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vli.ground import load_json
from vli.authenticated_temporal_source import (
    _genesis,public_key_hex,sign_source_observation,
    append_signed_observation,resume_signed_observer,
)
from vli.temporal_intelligence import TemporalIntelligenceGround
from vli.evolving_environment_language import EvolvingEnvironmentLanguage
from vli.intention_state_space import (
    construct_intention_space,navigate_intention,verify_intention_navigation,
)
from vli.reflective_rational import reflect_intention


def main():
    key=Ed25519PrivateKey.generate()
    pub=public_key_hex(key)
    cases=(
        ("quadratic",[0,1,4,9],1000000,"1000000000000/1"),
        ("recurrence2",[1,1,2,3,5],100,"573147844013817084101/1"),
        ("geometric",[2,6,18],17,str(2*3**17)+"/1"),
    )
    report=[]
    with tempfile.TemporaryDirectory() as temp:
        for family,values,target,expected in cases:
            work=Path(temp)/family
            work.mkdir()
            log=work/"signed-source.jsonl"
            saved=work/"observer.json"
            source="time007:"+family
            head=_genesis(source,pub)
            for idx,val in enumerate(values):
                signed=sign_source_observation(
                    key,source=source,sequence=idx,
                    time=f"{idx}/1",value=f"{val}/1")
                head,_=append_signed_observation(
                    log,signed,expected_head=head,expected_count=idx,
                    source=source,public_key=pub)
            run=resume_signed_observer(
                journal_path=log,checkpoint_path=saved,
                source=source,public_key=pub,
                expected_source_head=head,expected_source_count=len(values),
                environment="time007:"+family,host="continuing-source-observer")
            snap=load_json(saved)
            g=TemporalIntelligenceGround.restore(
                snap,expected_head=snap["head"],
                expected_identity=run["observer_identity"])
            cap=EvolvingEnvironmentLanguage(g).current(expected_head=g.head)
            auth={
                "journal_path":log,"source_id":source,"public_key":pub,
                "expected_source_head":head,
                "expected_source_count":len(values),
                "expected_ground_head":g.head,
                "expected_ground_identity":g.identity,
            }
            space=construct_intention_space(snap,cap,**auth)
            result=navigate_intention(
                space,f"{target}/1",allow_projection=True)
            assert result["value"]==expected
            assert verify_intention_navigation(
                snap,cap,space,result,**auth)
            short=2 if family=="geometric" else len(values)-1
            shorter=navigate_intention(space,f"{short}/1")
            reflected=reflect_intention(
                space,short,expected_value=shorter["value"],
                depth=2 if family=="geometric" else 1)
            report.append({
                "source_family":family,
                "signed_source_count":len(values),
                "intention":{"time":f"{target}/1"},
                "mathematical_state_dimension":space["dimension"],
                "transition_matrix":space["transition_operator"],
                "retained_initial_state":space["initial_state"],
                "selected_result":result["value"],
                "evidence_status":result["evidence_status"],
                "power_compositions":result["calculation"][
                    "navigation_metrics"]["matrix_products"],
                "temporal_transitions_replayed":0,
                "trajectory_states_materialized":0,
                "reflector_scope":{"at_step":short,
                                   "receipts":reflected["reflection_receipts"]},
                "source_codebook":reflected["source_codebook"],
                "external_arithmetic_seed":reflected["declared_external_primitive_seed"],
                "signed_observation_head":space["signed_source_journal_head"],
            })
    print(json.dumps({
        "schema":"vl-i.time007.intention-state-space-experiment.v1",
        "claim":"verified_exact_model_navigation_not_physical_world_prediction",
        "future_result_is_source_observation":False,
        "model_family_supplied_as_finite_search_grammar":True,
        "arithmetic_seed_is_physical_host":True,
        "no_future_trajectory_table":True,
        "experiments":report,
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
