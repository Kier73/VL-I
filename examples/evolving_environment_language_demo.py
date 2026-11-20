"""TIME-003: a single living intelligence changes the mathematics it interprets."""
from __future__ import annotations

from hashlib import sha256
import json

from vli.live_temporal_intelligence import LiveTemporalObserver
from vli.temporal_intelligence import TemporalIntelligenceGround
from vli.evolving_environment_language import (
    EvolvingEnvironmentLanguage, certify_language_migration,
    verify_environment_language, verify_environment_answer,
    verify_language_migration,
)


def main():
    environment=TemporalIntelligenceGround(
        "laboratory:environment-demands-new-language",observer="vli-evolving")
    durable=[]
    live=LiveTemporalObserver(environment,host="persistent-observer-A",
                              checkpoint_sink=durable.append)
    learner=EvolvingEnvironmentLanguage(environment)

    # Three distinct mathematical relations, not supplied to the learner.
    data=([1,3,5,7]+[100,101,104,109]+[1,1,2,3,5])
    qualified=[]
    last_family=None
    audit=[]
    for t,value in enumerate(data):
        raw=f"source:{t}:{value}".encode()
        decision=live.ingest({
            "schema":"vl-i.source-sample.v1",
            "sequence":len(environment.events),
            "time":f"{t}/1",
            "values":[f"{value}/1"],
            "source_digest":sha256(raw).hexdigest(),
        })
        capsule=learner.current(expected_head=environment.head)
        family=capsule["family"] if capsule else None
        audit.append({"source_sequence":t,
                      "language":family or "NOT_QUALIFIED",
                      "live_sample_count":decision["samples_in_one_live_process"]})
        if capsule is not None and family!=last_family:
            snapshot=environment.checkpoint()
            verify_environment_language(snapshot,capsule,
                expected_head=environment.head,expected_identity=environment.identity)
            qualified.append((snapshot,capsule))
            last_family=family

    assert [item[1]["family"] for item in qualified]==[
        "affine","quadratic","recurrence2"]
    migrations=[]
    for (old,prior),(new,later) in zip(qualified,qualified[1:]):
        certified=certify_language_migration(old,prior,new,later,
               before_head=old["head"],after_head=new["head"],
               identity=environment.identity)
        assert verify_language_migration(old,prior,new,later,certified,
               before_head=old["head"],after_head=new["head"],
               identity=environment.identity)
        migrations.append(certified)
    final=learner.current(expected_head=environment.head)
    projected=learner.navigate(final,"40/1",
               expected_head=environment.head,allow_projection=True)
    assert verify_environment_answer(environment.checkpoint(),final,projected,
               expected_head=environment.head,expected_identity=environment.identity)
    print(json.dumps({
       "schema":"vl-i.time-003-generated-language-experiment.v1",
       "ground_identity":environment.identity,
       "observations_in_one_live_process":live.accepted_in_lifetime,
       "durable_checkpoint_acknowledgments":len(durable),
       "qualified_law_sequence":[p["family"] for _,p in qualified],
       "distinct_generated_vocabulary":[p["vocabulary"] for _,p in qualified],
       "qualified_language_digests":[p["capsule_digest"] for _,p in qualified],
       "migration_receipts":migrations,
       "far_future_navigation":projected,
       "source_events_preserved":len(environment.events),
       "timeline":audit,
       "claim_limits":{
            "independent_source_authentication":False,
            "network_quorum_recovery":False,
            "neural_trainable_primitive_generation":False,
            "bootstrap_reflective_interpreter":False,
            "physical_continuous_observation":False,
            "predicted_future_is_observed":False,
        },
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
