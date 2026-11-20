"""TIME-002: one living interpreter receives/requests observations over time.

The agent operates an in-memory Ground over an ongoing async lifespan. Accepted
source observations extend an externally pin-able causal ledger. A source-silent
interval is NOT a made-up sensor observation; an inactivity event is a local
runtime telemetry reading. No physical sensor authentication or uninterrupted
host uptime across an actual crash is claimed.
"""
from __future__ import annotations

import asyncio
from fractions import Fraction
from time import monotonic_ns
from typing import AsyncIterator, Awaitable, Callable

from .ground import _hex64, _require
from .temporal_intelligence import (
    TemporalIntelligenceGround, _encode, _fraction, MAX_OBSERVATIONS,
)

MAX_RUN_SAMPLES = MAX_OBSERVATIONS


class LiveTemporalObserver:
    """Alive across many observations, not a stateless per-request predictor.

    A supplied sink must durably pin the latest accepted Ground head outside
    the checkpoint before the caller treats the observation as committed.
    The sink can use VLD quorum/M1 or another trusted service. This object
    does not implement the source's authentication or that persistence layer.
    """

    def __init__(self, ground: TemporalIntelligenceGround, *,
                 host: str, checkpoint_sink: Callable | None = None):
        _require(isinstance(ground, TemporalIntelligenceGround)
                 and type(host) is str and 1 <= len(host) <= 100
                 and (checkpoint_sink is None or callable(checkpoint_sink)),
                 "Invalid living interpreter, observer host or persistence sink")
        self.ground = ground
        self.host = host
        self.sink = checkpoint_sink
        self.started_ns = monotonic_ns()
        self.accepted_in_lifetime = 0
        self.source_silent_intervals = 0
        self.heartbeats_seen = 0
        self.poisoned = False

    def intention(self, *, strategy="uniform") -> dict:
        """Select an exact next observation coordinate for a queryable source.

        A probe request is NOT an observation. Only a source supplying a real
        sample for that time can fulfill it. Adaptive probing refines time
        resolution after model invalidation without changing earlier evidence.
        """
        _require(strategy in ("uniform", "counterexample-adaptive"),
                 "Unadmitted environmental attention strategy")
        events = self.ground.events
        if not events:
            t = Fraction(0)
            interval = Fraction(1)
            reason = "INITIAL_ENVIRONMENT_CONTACT"
        else:
            last = _fraction(events[-1]["time"])
            interval = (last - _fraction(events[-2]["time"])
                        if len(events)>1 else Fraction(1))
            current = self.ground.epochs[-1]["status"]
            if strategy == "counterexample-adaptive" and current != "QUALIFIED_SAMPLES_ONLY":
                interval /= 2
                reason = "REFINE_UNQUALIFIED_ENVIRONMENT"
            else:
                reason = "MONITOR_CURRENT_INTERPRETATION"
            t = last + interval
        return {"schema":"vl-i.active-environment-inquiry.v1",
                "identity":self.ground.identity, "head":self.ground.head,
                "time":_encode(t), "proposed_interval":_encode(interval),
                "purpose":reason, "source_observation":False}

    def ingest(self, source_event: dict) -> dict:
        """Observe/qualify exactly one source response in the same living process.

        Sequence fencing prevents silently accepting duplicate source events.
        Sink failure poisons this process instance: never acknowledge an
        observation which was not externally persisted.
        """
        _require(not self.poisoned, "Uncommitted observation: restore trusted checkpoint")
        _require(type(source_event) is dict and set(source_event) == {
            "schema","sequence","time","values","source_digest"},
            "Malformed source observation envelope")
        _require(source_event["schema"] == "vl-i.source-sample.v1"
                 and type(source_event["sequence"]) is int
                 and source_event["sequence"] == len(self.ground.events)
                 and _hex64(source_event["source_digest"]),
                 "Reordered, duplicated or unbound source sample")
        previous_epochs = len(self.ground.epochs)
        self.ground.observe(
            source_event["time"],source_event["values"],
            source_digest=source_event["source_digest"],host=self.host,
            expected_head=self.ground.head,
        )
        snapshot = self.ground.checkpoint()
        if self.sink is not None:
            try:
                persisted = self.sink(snapshot)
                _require(persisted is not False,
                         "Trusted persistence sink refused new head")
            except Exception:
                self.poisoned = True
                raise
        self.accepted_in_lifetime += 1
        epochs = self.ground.epochs
        current = epochs[-1]
        qualified = current["status"] == "QUALIFIED_SAMPLES_ONLY"
        processor = (self.ground.compile_epoch(len(epochs)-1,
                     expected_head=self.ground.head) if qualified else None)
        return {"schema":"vl-i.live-observer-decision.v1",
                "identity":self.ground.identity,
                "committed_head":self.ground.head,
                "source_sample_sequence":source_event["sequence"],
                "sample_time":source_event["time"],
                "active_epoch":len(epochs)-1,
                "epoch_count":len(epochs),
                "changed_interpretation":len(epochs)>previous_epochs,
                "active_status":current["status"],
                "active_processor_digest":(processor["processor_digest"]
                                            if processor else None),
                "historical_epochs_preserved":len(epochs)-1,
                "samples_in_one_live_process":self.accepted_in_lifetime,
                "host":self.host,
                "measured_process_uptime_ns":monotonic_ns()-self.started_ns,
                "physical_coverage_between_samples":"NOT_OBSERVED",
                "durability":("SINK_ACKNOWLEDGED" if self.sink else
                              "IN_MEMORY_ONLY")}

    async def run_probes(
        self, sensor: Callable[[dict], Awaitable[dict]], *,
        samples: int, strategy="uniform", pause_seconds=0.0,
    ) -> list[dict]:
        """Initiate successive probes while one interpreter remains alive.

        The sensor MUST explicitly implement arbitrary-time queries. This
        contract cannot be silently used against an ordinary physical sensor.
        """
        _require(callable(sensor) and type(samples) is int
                 and 1 <= samples <= MAX_RUN_SAMPLES
                 and type(pause_seconds) in (int,float)
                 and 0 <= pause_seconds <= 60,
                 "Invalid active observation work budget")
        decisions = []
        for _ in range(samples):
            question = self.intention(strategy=strategy)
            answer = await sensor(question)
            _require(type(answer) is dict and answer.get("time") == question["time"],
                     "Source did not observe the requested environment time")
            decisions.append(self.ingest(answer))
            if pause_seconds:
                await asyncio.sleep(pause_seconds)
        return decisions

    async def observe_stream(self, source: AsyncIterator[dict], *,
                             silence_timeout_seconds=2.0,
                             max_messages: int | None = None):
        """Consume an ongoing asynchronous source; report *measured* silences.

        A timed-out read is kept alive, not canceled and retried. Thus one
        pending source read cannot accidentally become duplicate effectful
        sensor activity. The stream may naturally run without a finite end.
        """
        _require(type(silence_timeout_seconds) in (int,float)
                 and 0 < silence_timeout_seconds <= 3600
                 and (max_messages is None or
                      type(max_messages) is int and 1 <= max_messages <= MAX_RUN_SAMPLES),
                 "Unadmitted live observation schedule")
        iterator = source.__aiter__()
        pending = None
        messages = 0
        try:
            while max_messages is None or messages < max_messages:
                if pending is None:
                    pending = asyncio.create_task(anext(iterator))
                done, _ = await asyncio.wait({pending},
                            timeout=silence_timeout_seconds)
                if not done:
                    self.source_silent_intervals += 1
                    yield {"schema":"vl-i.live-source-silence.v1",
                           "identity":self.ground.identity,
                           "head":self.ground.head,
                           "source_observation":False,
                           "silence_intervals":self.source_silent_intervals,
                           "measured_process_uptime_ns":
                                  monotonic_ns()-self.started_ns,
                           "reason":"NO_NEW_SOURCE_EVENT"}
                    continue
                try:
                    event = pending.result()
                except StopAsyncIteration:
                    return
                pending = None
                messages += 1
                _require(type(event) is dict and event.get("kind") in (
                    "sample","heartbeat"), "Unknown live source message")
                if event["kind"] == "sample":
                    _require(set(event) == {
                        "kind","schema","sequence","time","values","source_digest"},
                        "Unqualified sample stream envelope")
                    yield self.ingest({k:v for k,v in event.items() if k!="kind"})
                else:
                    _require(set(event) == {"kind","time"},
                             "Heartbeat is not a source observation")
                    _fraction(event["time"])
                    self.heartbeats_seen += 1
                    yield {"schema":"vl-i.live-source-heartbeat.v1",
                           "identity":self.ground.identity,"head":self.ground.head,
                           "time":event["time"], "source_observation":False,
                           "heartbeat_count":self.heartbeats_seen,
                           "measured_process_uptime_ns":
                                  monotonic_ns()-self.started_ns}
        finally:
            if pending is not None and not pending.done():
                pending.cancel()
                try:
                    await pending
                except asyncio.CancelledError:
                    pass
