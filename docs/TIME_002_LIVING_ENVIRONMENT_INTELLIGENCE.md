# TIME-002 — A living observer, not a stateless request sequence

The ambition is to let intelligence emerge from interaction with its current environment and remain **active as the environment changes**, preserving grounded knowledge without freezing its mathematical form.

TIME-001 demonstrated a portable causal history with reconstructable environment-derived flow interpreters. TIME-002 adds an **ongoing in-process observational lifecycle**: [live_temporal_intelligence.py](../src/vli/live_temporal_intelligence.py).

## Operational distinction

The same object holds its causal Ground throughout an asynchronous series of real source responses. Its active interpretation is continuously updated as each observation arrives. It does not reload a model or recreate context for each request.

~~~text
  Queryable environment / ongoing asynchronous source
                   |
          observed sample / heartbeat
                   |
       one live TemporalIntelligenceGround
         |               |
         v               v
  source qualification   continuity head
         |               |
    interpreted law     trusted sink acknowledgment
         |               |
         +-------> current decision and new inquiry
~~~

When a consistent third sample arrives, an empirical mathematical interpreter is generated. When a sample contradicts that interpreter, the current interpretation is **immediately withdrawn**, previous qualified epochs remain preserved, and active sampling continues until the next law is qualified. An unqualified new epoch cannot silently reuse the old law as if the environment remained unchanged.

## Two acquisition modes

**Active controlled experiment.** A queryable source accepts a proposed mathematical time coordinate. The LiveTemporalObserver requests a sample, the source reports the exact observed result, and the observer extends its Ground. The baseline uses uniform probing; counterexample-adaptive probing reduces the proposed interval when the active epoch is not yet qualified. The inquiry itself is not evidence.

**Passive live stream.** A long-running asynchronous source produces sample events and/or heartbeats. When nothing arrives for a bounded real wall-clock interval, the monitor emits a SOURCE_SILENT telemetry notice without fabricating any observation. The pending read is deliberately kept alive instead of being repeatedly canceled, preventing accidental duplicate effectful source invocations. Heartbeats confirm a received message but are *not* environmental samples.

A live process measures monotonic uptime separately from the environment's declared mathematical time. A timed-out source stream is not interpreted as proof of environment stability.

## Durable admission and crash semantics

An optional checkpoint sink must acknowledge every accepted source observation **before** the observer publishes its decision. In a live deployment, this sink must bind the checkpoint to an independent trusted head (for example through VLD's existing quorum/M1 recovery architecture). A sink failure poisons the current live agent; it will not acknowledge or continue after the uncommitted write. Recovery begins from the last independently trusted snapshot and allows source retry.

This is deliberately stricter than pretending an in-memory append is a durable distributed commit. The sink callback is a **dependency injection seam**, not a packaged authenticated persistence mechanism. TIME-002's own tests use in-memory sinks and verify failure/recovery behavior; it is not yet wired to an actual VLD quorum or E-CPU mTLS host.

The source's event index is required to match the accepted causal event count. This rejects naive duplicate/reordered observations; it does not constitute cross-host deduplication against a malicious source.

## Verified test scenarios

- Eight active source probes in **one uninterrupted running interpreter object**, with a stable process identity.
- Online slope qualification, a real counterexample at the change point, temporary unqualification and a new law fit without erasing the old.
- Counterexample-adaptive probe interval selection.
- Async source that sends an informational heartbeat rather than a sensor observation.
- A source-silent interval where the pending read survives and the original event is accepted only once.
- Persistence failure: poisoned live agent, no acknowledged uncommitted event, and new host recovery from the last trustworthy head.
- Continued model qualification after changing the labeled observer-host.
- Duplicate source sequence refusal and wrong-coordinate probe refusal.

Reproduce:

~~~sh
python -m unittest discover -s tests -p 'test_live_temporal_intelligence.py' -v
python -m unittest discover -s tests -p 'test_temporal_intelligence.py' -v
python -m examples.live_temporal_intelligence_demo
~~~

## Physical, mathematical and epistemic distinctions

**What this supports:** an interpreter really executing across successive observations during one runtime; new mathematical law parameters generated from the environment; history retained through changing law epochs; and evidence-aware recovery contracts.

**What it does not support yet:** literal observation of every time instant, independently authenticated real-world sensing, neural model self-training, an interpreter whose bootstrap language is generated from no supplied primitives, uninterrupted survival of a dead process, infinite event retention, or automatically operating an external actuator. Continuous rational mathematical time is not continuous physical sensing.

## Next: the intelligence should generate more of its own interpreter

The next constructive leap is a *variable grammar* rather than a variable rate. A stream containing nonlinear laws should be able to generate an admissible alternative operator family, with discriminating observations and counterexamples, instead of repeatedly failing to fit an affine relation.

Navigable-Compute's NCS-001/NCR-001 already offers algebra closure and bounded reflective interpretation; VLD supplies continuous stage, structural memory, effect semantics and quorum-fenced execution. The prudent integration is to lift TIME-002's qualified observations into those existing tools, then test whether the **required environment/observer invariants select both an expressive processor algebra and a self-interpreting representation**. Benchmarks should report source calls, law qualification work, interpretation steps, retained history size, recovery verification cost, adaptation latency and actual host runtime.

In other words, an intelligence persists not by protecting one fixed model from change, but by preserving the *verified relations* through which it can justifiably become another model.
