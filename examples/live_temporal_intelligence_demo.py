"""TIME-002: one autonomous query loop, continuous in-process learning."""
import asyncio
from fractions import Fraction
from hashlib import sha256
import json

from vli.live_temporal_intelligence import LiveTemporalObserver
from vli.temporal_intelligence import TemporalIntelligenceGround, _fraction, _encode


async def main():
    ground=TemporalIntelligenceGround("laboratory:online-adapting-flow")
    durable=[]
    agent=LiveTemporalObserver(ground,host="continuous-observer-A",
                               checkpoint_sink=durable.append)
    async def environment(question):
        t=_fraction(question["time"])
        value=1+2*t if t<=4 else 30+(t-5)
        observed=_encode(value)
        source=sha256(f"controlled-test:{question['time']}:{observed}".encode()).hexdigest()
        return {"schema":"vl-i.source-sample.v1",
                "sequence":len(ground.events),
                "time":question["time"],
                "values":[observed],
                "source_digest":source}
    decisions=await agent.run_probes(environment,samples=8,strategy="uniform",
                                     pause_seconds=0.001)
    print(json.dumps({
        "schema":"vl-i.time-002-live-laboratory.v1",
        "ground_identity":ground.identity,
        "accepted_in_same_live_process":agent.accepted_in_lifetime,
        "persisted_checkpoint_count":len(durable),
        "observed_environment_epochs":len(ground.epochs),
        "qualified_epoch_rates":[_encode(e["rate"][0])
                                 for e in ground.epochs if e["rate"] is not None],
        "changed_interpretation_at":[d["sample_time"] for d in decisions
                                      if d["changed_interpretation"]],
        "decisions":[{"time":d["sample_time"],
                      "active_status":d["active_status"],
                      "active_epoch":d["active_epoch"],
                      "uptime_ns":d["measured_process_uptime_ns"]}
                     for d in decisions],
        "claims":{
            "one_host_process_persisted_across_samples":True,
            "physical_sensor_authenticated":False,
            "observed_every_real_time_instant":False,
            "self_interpreting_neural_intelligence":False
        },
    },indent=2,sort_keys=True))


if __name__=="__main__":
    asyncio.run(main())
