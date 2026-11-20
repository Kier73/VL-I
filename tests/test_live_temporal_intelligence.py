"""TIME-002: active lifelong observer, source silence and durability boundaries."""
import asyncio
from hashlib import sha256
import unittest

from vli.live_temporal_intelligence import LiveTemporalObserver
from vli.temporal_intelligence import TemporalIntelligenceGround, _fraction


def sample(ground, time, value):
    return {"schema":"vl-i.source-sample.v1",
            "sequence":len(ground.events), "time":time,
            "values":[value], "source_digest":
                sha256(f"observed:{time}:{value}".encode()).hexdigest()}


class LivingObserverTests(unittest.IsolatedAsyncioTestCase):
    async def test_observer_actively_queries_environment_in_one_lifetime(self):
        ground=TemporalIntelligenceGround("world:piecewise-affine")
        saved=[]
        agent=LiveTemporalObserver(ground,host="live-host",checkpoint_sink=saved.append)

        async def sensor(query):
            t=_fraction(query["time"])
            x=(1+2*t if t <= 4 else 30+(t-5))
            return sample(ground,query["time"],f"{x.numerator}/{x.denominator}")

        original=id(agent.ground)
        decisions=await agent.run_probes(sensor,samples=8)
        self.assertEqual(id(agent.ground),original)
        self.assertEqual(len(saved),8)
        self.assertEqual(agent.accepted_in_lifetime,8)
        self.assertTrue(all(d["durability"]=="SINK_ACKNOWLEDGED" for d in decisions))
        self.assertEqual(decisions[2]["active_status"],"QUALIFIED_SAMPLES_ONLY")
        self.assertTrue(decisions[5]["changed_interpretation"])
        self.assertEqual(decisions[5]["active_status"],"INSUFFICIENT")
        self.assertEqual(decisions[7]["active_status"],"QUALIFIED_SAMPLES_ONLY")
        self.assertEqual([g["status"] for g in ground.epochs],
                         ["QUALIFIED_SAMPLES_ONLY","QUALIFIED_SAMPLES_ONLY"])
        self.assertEqual(ground.compile_epoch(1,expected_head=ground.head)["rate"],
                         ["1/1"])
        self.assertEqual(len({d["identity"] for d in decisions}),1)
        self.assertEqual(len({d["committed_head"] for d in decisions}),8)

    async def test_attention_refines_after_unqualified_epoch(self):
        ground=TemporalIntelligenceGround("synthetic:sparse-observer")
        agent=LiveTemporalObserver(ground,host="A")
        self.assertEqual(agent.intention(strategy="counterexample-adaptive")["time"],
                         "0/1")
        agent.ingest(sample(ground,"0/1","1/1"))
        self.assertEqual(agent.intention(strategy="counterexample-adaptive")["time"],
                         "1/2")
        agent.ingest(sample(ground,"1/2","2/1"))
        self.assertEqual(agent.intention(strategy="counterexample-adaptive")["time"],
                         "3/4")
        agent.ingest(sample(ground,"3/4","5/2"))
        self.assertEqual(ground.epochs[0]["status"],"QUALIFIED_SAMPLES_ONLY")
        self.assertEqual(agent.intention(strategy="counterexample-adaptive")["time"],
                         "1/1")

    async def test_live_stream_not_a_sequence_of_stateless_restores(self):
        g=TemporalIntelligenceGround("streamed-env")
        agent=LiveTemporalObserver(g,host="observer")
        async def incoming():
            yield {"kind":"heartbeat","time":"0/1"}
            yield {"kind":"sample",**sample(g,"1/1","2/1")}
            yield {"kind":"sample",**sample(g,"2/1","4/1")}
            yield {"kind":"sample",**sample(g,"3/1","6/1")}
        updates=[u async for u in agent.observe_stream(incoming())]
        self.assertEqual(updates[0]["schema"],"vl-i.live-source-heartbeat.v1")
        self.assertEqual(updates[0]["source_observation"],False)
        self.assertEqual(agent.heartbeats_seen,1)
        self.assertEqual(len(g.events),3)
        self.assertEqual(updates[-1]["active_status"],"QUALIFIED_SAMPLES_ONLY")
        self.assertEqual([u["samples_in_one_live_process"] for u in updates[1:]],
                         [1,2,3])

    async def test_source_silence_keeps_one_pending_read_alive(self):
        g=TemporalIntelligenceGround("slow-sensor")
        agent=LiveTemporalObserver(g,host="observer")
        source_calls=0
        async def incoming():
            nonlocal source_calls
            source_calls+=1
            await asyncio.sleep(0.04)
            yield {"kind":"sample",**sample(g,"0/1","1/1")}
        notices=[u async for u in agent.observe_stream(
            incoming(),silence_timeout_seconds=0.01)]
        self.assertEqual(source_calls,1)
        silences=[u for u in notices if u["schema"]=="vl-i.live-source-silence.v1"]
        self.assertGreaterEqual(len(silences),1)
        self.assertTrue(all(u["source_observation"] is False for u in silences))
        self.assertEqual(len(g.events),1)
        self.assertEqual(notices[-1]["source_sample_sequence"],0)
        self.assertGreater(notices[-1]["measured_process_uptime_ns"],0)

    async def test_sink_failure_poisoned_and_never_acknowledged(self):
        g=TemporalIntelligenceGround("strict-commit")
        trusted=[]
        def sink(s):
            if len(s["events"])==2:
                raise IOError("durable storage interrupted")
            trusted.append(s)
        agent=LiveTemporalObserver(g,host="A",checkpoint_sink=sink)
        ok=agent.ingest(sample(g,"0/1","1/1"))
        self.assertEqual(ok["durability"],"SINK_ACKNOWLEDGED")
        with self.assertRaises(IOError):
            agent.ingest(sample(g,"1/1","2/1"))
        self.assertTrue(agent.poisoned)
        self.assertEqual(agent.accepted_in_lifetime,1)
        with self.assertRaises(ValueError):
            agent.ingest(sample(g,"2/1","3/1"))
        pinned=trusted[-1]
        restored=TemporalIntelligenceGround.restore(pinned,
                      expected_head=pinned["head"],
                      expected_identity=pinned["identity"])
        self.assertEqual(len(restored.events),1)
        # Failed sample was NOT acknowledged; source can retry from committed pin.
        new_agent=LiveTemporalObserver(restored,host="B")
        resumed=new_agent.ingest(sample(restored,"1/1","2/1"))
        self.assertEqual(resumed["source_sample_sequence"],1)

    async def test_rehomed_live_agent_recovers_learning_not_just_a_value(self):
        g=TemporalIntelligenceGround("shared-world")
        first=LiveTemporalObserver(g,host="A")
        for i in range(3):
            first.ingest(sample(g,f"{i}/1",f"{i*4}/1"))
        snap=g.checkpoint()
        revived=TemporalIntelligenceGround.restore(snap,
                   expected_head=g.head,expected_identity=g.identity)
        second=LiveTemporalObserver(revived,host="B")
        observed=second.ingest(sample(revived,"3/1","12/1"))
        self.assertEqual(observed["active_status"],"QUALIFIED_SAMPLES_ONLY")
        self.assertEqual(observed["active_epoch"],0)
        self.assertEqual(revived.identity,g.identity)
        self.assertEqual(revived.events[-1]["host"],"B")
        self.assertEqual(second.accepted_in_lifetime,1)
        self.assertEqual(len(revived.events),4)

    async def test_duplicate_source_sample_is_not_reexecuted(self):
        g=TemporalIntelligenceGround("sequence-protect")
        agent=LiveTemporalObserver(g,host="A")
        a=sample(g,"0/1","0/1")
        agent.ingest(a)
        head=g.head
        with self.assertRaises(ValueError):
            agent.ingest(a)
        self.assertEqual(g.head,head)
        self.assertEqual(len(g.events),1)

    async def test_query_source_must_answer_exact_requested_coordinate(self):
        g=TemporalIntelligenceGround("query-safety")
        agent=LiveTemporalObserver(g,host="A")
        async def misbehaving(question):
            return sample(g,"999/1","0/1")
        with self.assertRaises(ValueError):
            await agent.run_probes(misbehaving,samples=1)
        self.assertEqual(len(g.events),0)

    async def test_predefined_bootstrap_does_not_claim_auto_self_interpretation(self):
        g=TemporalIntelligenceGround("runtime")
        agent=LiveTemporalObserver(g,host="A")
        for i in range(3):
            agent.ingest(sample(g,f"{i}/1",f"{2*i}/1"))
        self.assertIsNotNone(agent.ground.compile_epoch(0,
                               expected_head=g.head)["program"])
        self.assertEqual(agent.intention()["source_observation"],False)


if __name__=="__main__":
    unittest.main()
