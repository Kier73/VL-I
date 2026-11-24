"""PBG-003: goal-conditioned observation, E-CPU realization and VGI finalization.

Receiver-owned E-CPU operations execute through SAB-001, and sensor values
are finalized through the actual VGI channel/evidence ledger. A native
operation acknowledgement permits a predicted state transition; a separately
qualified goal-sensor observation is necessary to report observed realization.

Scope: receiver-local software handlers and receiver policy, not remote physical
independence. Durable idempotency of an external actuator must be provided by
the actuator, not inferred from an E-CPU report digest.
"""
from __future__ import annotations
from dataclasses import dataclass
import re

from .intention_predictive_ground import IntentionGround, IntentionSession
from .predictive_basis_ground import _hash

SCHEMA = "vl-i.pbg003.goal-evidence-record.v1"
GROUND_ID = "pbg003.intention-observer"
SOURCE_STATE = "predictive-intention"
CHANNEL_ID = "pbg003.receiver-events"
PROTOCOL_ID = "pbg003.observer-event/1"
INPUT_KIND = "pbg003.request.v1"
NONCE = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class Endpoint:
    source_id: str
    locus: str
    ingress: str = "start"
    collector: str = "collector"


@dataclass(frozen=True)
class ScheduledIntent:
    ground_digest: str
    previous: dict
    nonce: str
    epoch_digest: str
    kind: str
    operation: str
    coordinate: int
    tasks: tuple
    routes: tuple
    endpoints: tuple[Endpoint, ...]
    inventory_digest: str
    policy_digest: str

    @property
    def fingerprint(self):
        return _hash({
            "schema":"vl-i.pbg003.scheduled-intent.v1",
            "ground":self.ground_digest,
            "previous":self.previous["snapshot_digest"],
            "nonce":self.nonce,"epoch":self.epoch_digest,
            "kind":self.kind,"operation":self.operation,
            "coordinate":self.coordinate,
            "inventory":self.inventory_digest,"policy":self.policy_digest,
            "endpoints":[(p.source_id,p.locus,p.ingress,p.collector) for p in self.endpoints],
            "tasks":[t.fingerprint for t in self.tasks],
            "routes":[r.fingerprint for r in self.routes],
        })


def _op_name(kind, operation, source_id):
    return f"pbg003.{kind}.{operation}.{source_id}.v1"


def _output_kind(kind, operation, source_id):
    return f"pbg003.{kind}.result.{operation}.{source_id}"


def prepare(session, runtime, kind, operation, endpoints, *, nonce, coordinate):
    """Bind current intention state and host inventory to typed SAB-001 routes."""
    from mailcpu.substrate_bootstrap import AdaptiveBootstrap, BootstrapTask
    if (not isinstance(session, IntentionSession) or
            not isinstance(runtime, AdaptiveBootstrap) or
            kind not in ("sense","act") or
            type(nonce) is not str or NONCE.fullmatch(nonce) is None or
            type(coordinate) is not int or coordinate < 0):
        raise ValueError("Unqualified intention Ground, runtime, nonce or coordinate")
    endpoints = tuple(endpoints)
    if (not endpoints or len(endpoints)!=(2 if kind=="sense" else 1) or
            any(not isinstance(p,Endpoint) for p in endpoints) or
            len({p.source_id for p in endpoints})!=len(endpoints) or
            len({p.locus for p in endpoints})!=len(endpoints)):
        raise ValueError("A sensor needs two distinct hosts; an act needs one")
    if kind=="sense":
        if operation not in session.ground.sensors():
            raise ValueError("Unknown requested sensor")
    elif operation not in dict(session.ground.actions()):
        raise ValueError("Unknown requested intervention")
    before=session.snapshot()
    epoch=_hash({
        "schema":"vl-i.pbg003.observation-epoch.v1",
        "ground":session.ground.ground_digest,
        "previous":before["snapshot_digest"],
        "kind":kind,"operation":operation,"nonce":nonce,
        "coordinate":coordinate,
        "inventory":runtime.inventory.fingerprint,
        "policy":runtime.policy.fingerprint,
    })
    tasks,plans=[],[]
    for p in endpoints:
        op=_op_name(kind,operation,p.source_id)
        output=_output_kind(kind,operation,p.source_id)
        task=BootstrapTask(
            intention_id=f"pbg003-{kind}-{p.source_id}",
            ground_digest=epoch,
            source=p.ingress,input_kind=INPUT_KIND,
            target=p.collector,output_kind=output,
            value={"epoch":epoch,"ground":session.ground.ground_digest,
                   "previous":before["snapshot_digest"],"kind":kind,
                   "operation":operation,"source_id":p.source_id,"nonce":nonce},
            require_oracle=True,max_bytes=8192,max_steps=16,max_quanta=100_000,
        )
        if any(key[1]==op and key[0]!=p.locus for key in runtime.realizers):
            raise ValueError("An operation is advertised at another locus")
        plan=runtime.plan(task,coordinate=coordinate)
        realizes=[s for s in plan.steps if s.kind=="realize"]
        if (len(realizes)!=1 or realizes[0].locus!=p.locus or
                realizes[0].operation!=op or realizes[0].produces!=output):
            raise ValueError("E-CPU operation/host route was substituted")
        tasks.append(task);plans.append(plan)
    return ScheduledIntent(session.ground.ground_digest,before,nonce,epoch,
        kind,operation,coordinate,tuple(tasks),tuple(plans),endpoints,
        runtime.inventory.fingerprint,runtime.policy.fingerprint)


def _validate_plan(session,scheduled,runtime):
    from mailcpu.substrate_bootstrap import AdaptiveBootstrap
    if (not isinstance(scheduled,ScheduledIntent) or
            not isinstance(session,IntentionSession) or
            not isinstance(runtime,AdaptiveBootstrap) or
            session.ground.ground_digest!=scheduled.ground_digest or
            session.snapshot()!=scheduled.previous or
            runtime.inventory.fingerprint!=scheduled.inventory_digest or
            runtime.policy.fingerprint!=scheduled.policy_digest):
        raise ValueError("Stale intention snapshot, Ground or E-CPU route authority")
    current=prepare(session,runtime,scheduled.kind,scheduled.operation,
                    scheduled.endpoints,nonce=scheduled.nonce,
                    coordinate=scheduled.coordinate)
    if current.fingerprint!=scheduled.fingerprint:
        raise ValueError("E-CPU operation route changed after intent formation")


def _native_reports(scheduled,runtime,ground):
    from mailcpu.substrate_bootstrap import verify_bootstrap_report,digest
    reports=[]
    for p,task,route in zip(scheduled.endpoints,scheduled.tasks,scheduled.routes):
        if runtime.plan(task,coordinate=scheduled.coordinate).fingerprint!=route.fingerprint:
            raise ValueError("E-CPU route authority changed before execution")
        report=runtime.execute(task,route,coordinate=scheduled.coordinate)
        if not verify_bootstrap_report(report,task,route):
            raise ValueError("E-CPU execution receipt failed reconstruction")
        value=report["output"]
        expected={"request_digest":digest(task.value),"source_id":p.source_id,
                  "operation":scheduled.operation,"nonce":scheduled.nonce}
        if type(value) is not dict or any(value.get(k)!=v for k,v in expected.items()):
            raise ValueError("Result belongs to another operation or intention epoch")
        if scheduled.kind=="sense":
            if (set(value)!=set(expected)|{"symbol"} or
                    type(value["symbol"]) is not int or
                    str(value["symbol"]) not in
                    {a for a,_ in ground.sensors()[scheduled.operation]}):
                raise ValueError("Unqualified declared sensor outcome")
        else:
            if (set(value)!=set(expected)|{"ack","material_revision"} or
                    value["ack"] is not True or
                    type(value["material_revision"]) is not int or
                    value["material_revision"]<1):
                raise ValueError("Unqualified material operation acknowledgement")
        reports.append(report)
    return reports


def _make_message(scheduled,index,report):
    from vgi.channels import ChannelMessage
    from vgi.encounters import EncounterProvenance
    p=scheduled.endpoints[index]
    return ChannelMessage(f"pbg003-{index}-{scheduled.nonce[:12]}",
        CHANNEL_ID,PROTOCOL_ID,"observation",
        {"status":"known","source_state":SOURCE_STATE,
         "action":scheduled.operation,"target":SOURCE_STATE,
         "output":[report["output"]["symbol"]],
         "ground_fingerprint":report["report_digest"]},
        EncounterProvenance(
            p.source_id,f"ecpu-report://{report['report_digest']}",
            collector="pbg003-receiver"))


def _qualify_observation(scheduled,reports):
    from vgi.channels import ChannelContract,ChannelExchangeLedger
    from vgi.evidence import (TransitionEvidenceLedger,TransitionEvidencePolicy,
                              TransportFrameReceipt)
    sources=tuple(p.source_id for p in scheduled.endpoints)
    ledger=TransitionEvidenceLedger(
        TransitionEvidencePolicy(GROUND_ID,scheduled.epoch_digest,
                                 sources,len(sources),True))
    contract=ChannelContract(CHANNEL_ID,PROTOCOL_ID,
        observation_schema="vgi.transition-observation/1")
    messages=[_make_message(scheduled,i,report) for i,report in enumerate(reports)]
    exchange=ChannelExchangeLedger()
    for message in messages:
        exchange=exchange.record_observation(contract,message)
    encounters=exchange.to_encounter_ledger()
    for i,message in enumerate(messages,1):
        frame=TransportFrameReceipt.capture(message,
            transport_id="receiver-local-ecpu-sab",sequence=i)
        ledger=ledger.ingest_channel_message(exchange,message.message_id,frame,encounters)
    ledger=ledger.finalize(SOURCE_STATE,scheduled.operation)
    ledger=TransitionEvidenceLedger.from_dict(ledger.to_dict())
    decision=ledger.decision(SOURCE_STATE,scheduled.operation)
    if decision.status!="AUTHORIZED":
        raise ValueError("VGI sensor quorum was not finalized AUTHORIZED")
    observed=ledger.transition_observation(SOURCE_STATE,scheduled.operation)
    if len(observed.outcomes)!=1 or len(observed.outcomes[0].output)!=1:
        raise ValueError("VGI sensor observation was not unambiguous")
    symbol=observed.outcomes[0].output[0]
    if any(r["output"]["symbol"]!=symbol for r in reports):
        raise ValueError("Conflicting sensor evidence")
    return ledger,symbol,decision.digest


def _record(scheduled,reports,after,*,ledger=None,symbol=None,decision=None):
    body={
        "schema":SCHEMA,
        "scope":"receiver-local-SAB+VGI; physical independence external",
        "ground_digest":scheduled.ground_digest,
        "intent_digest":scheduled.fingerprint,
        "previous":scheduled.previous,"after":after,
        "nonce":scheduled.nonce,"epoch_digest":scheduled.epoch_digest,
        "kind":scheduled.kind,"operation":scheduled.operation,
        "path_reports":reports,
        "vgi_ledger":ledger.to_dict() if ledger else None,
        "vgi_ledger_digest":ledger.fingerprint if ledger else None,
        "vgi_decision_digest":decision,"symbol":symbol,
    }
    return {**body,"digest":_hash(body)}


def verify_evidence_record(ground,scheduled,runtime,record):
    """Independently check native reports, VGI provenance and Ground replay."""
    try:
        from mailcpu.substrate_bootstrap import verify_bootstrap_report,digest
        from vgi.evidence import TransitionEvidenceLedger,TransportFrameReceipt
        if not isinstance(ground,IntentionGround):return False
        ground.check_identity()
        keys={"schema","scope","ground_digest","intent_digest","previous","after",
              "nonce","epoch_digest","kind","operation","path_reports","vgi_ledger",
              "vgi_ledger_digest","vgi_decision_digest","symbol","digest"}
        if (not isinstance(scheduled,ScheduledIntent) or type(record) is not dict or
                set(record)!=keys or record["schema"]!=SCHEMA or
                record["scope"]!="receiver-local-SAB+VGI; physical independence external" or
                _hash({k:v for k,v in record.items() if k!="digest"})!=record["digest"] or
                record["ground_digest"]!=ground.ground_digest or
                record["intent_digest"]!=scheduled.fingerprint or
                record["previous"]!=scheduled.previous or
                record["epoch_digest"]!=scheduled.epoch_digest or
                record["nonce"]!=scheduled.nonce or
                record["kind"]!=scheduled.kind or
                record["operation"]!=scheduled.operation or
                len(record["path_reports"])!=len(scheduled.tasks)):
            return False
        session=IntentionSession.restore(ground,scheduled.previous,
            expected_digest=scheduled.previous["snapshot_digest"])
        _validate_plan(session,scheduled,runtime)
        for p,task,route,report in zip(scheduled.endpoints,scheduled.tasks,
                                         scheduled.routes,record["path_reports"]):
            if not verify_bootstrap_report(report,task,route):return False
            value=report["output"]
            required={"request_digest":digest(task.value),"source_id":p.source_id,
                      "operation":scheduled.operation,"nonce":scheduled.nonce}
            if type(value) is not dict or any(value.get(k)!=v for k,v in required.items()):
                return False
            if scheduled.kind=="act":
                if (set(value)!=set(required)|{"ack","material_revision"} or
                        value["ack"] is not True or
                        type(value["material_revision"]) is not int or
                        value["material_revision"]<1):return False
            elif (set(value)!=set(required)|{"symbol"} or
                  type(value["symbol"]) is not int or
                  str(value["symbol"]) not in
                  {label for label,_ in ground.sensors()[scheduled.operation]}):
                return False
        if scheduled.kind=="sense":
            if type(record["vgi_ledger"]) is not dict:return False
            ledger=TransitionEvidenceLedger.from_dict(record["vgi_ledger"])
            if (ledger.fingerprint!=record["vgi_ledger_digest"] or
                    ledger.policy.ground_id!=GROUND_ID or
                    ledger.policy.context_digest!=scheduled.epoch_digest or
                    tuple(ledger.policy.allowed_sources)!=
                    tuple(sorted(p.source_id for p in scheduled.endpoints)) or
                    ledger.policy.quorum!=len(scheduled.endpoints) or
                    ledger.policy.strict is not True):return False
            decision=ledger.decision(SOURCE_STATE,scheduled.operation)
            event=ledger.transition_observation(SOURCE_STATE,scheduled.operation)
            if (decision.status!="AUTHORIZED" or
                    decision.digest!=record["vgi_decision_digest"] or
                    len(event.outcomes)!=1 or len(event.outcomes[0].output)!=1 or
                    event.outcomes[0].output[0]!=record["symbol"] or
                    any(r["output"]["symbol"]!=record["symbol"]
                        for r in record["path_reports"])):return False
            evidence=tuple(e for e in ledger.evidence
                           if e.event==(SOURCE_STATE,scheduled.operation))
            if len(evidence)!=len(scheduled.endpoints):return False
            for i,report in enumerate(record["path_reports"],1):
                msg=_make_message(scheduled,i-1,report)
                frame=TransportFrameReceipt.capture(msg,
                    transport_id="receiver-local-ecpu-sab",sequence=i)
                e=evidence[i-1]
                if (e.source!=scheduled.endpoints[i-1].source_id or
                    e.message_id!=msg.message_id or
                    e.encounter_id!=f"channel-message:{msg.message_id}" or
                    e.channel_id!=CHANNEL_ID or e.protocol_id!=PROTOCOL_ID or
                    e.transport_id!=frame.transport_id or e.frame_id!=frame.frame_id or
                    e.transport_sequence!=i or e.payload_digest!=frame.payload_digest or
                    e.source_model_fingerprint!=report["report_digest"] or
                    e.ground_id!=GROUND_ID or
                    e.context_digest!=scheduled.epoch_digest or
                    e.status!="known" or len(e.outcomes)!=1 or
                    e.outcomes[0].target!=SOURCE_STATE or
                    e.outcomes[0].output!=(record["symbol"],)):
                    return False
            session.observe(scheduled.operation,str(record["symbol"]))
        else:
            if any(record[k] is not None for k in
                   ("vgi_ledger","vgi_ledger_digest","vgi_decision_digest","symbol")):
                return False
            session.apply(scheduled.operation)
        return session.snapshot()==record["after"]
    except (ValueError,TypeError,KeyError,IndexError,AttributeError):
        return False


def execute(scheduled,session,runtime):
    """Execute receiver-admitted operations then accept a verified Ground update."""
    _validate_plan(session,scheduled,runtime)
    reports=_native_reports(scheduled,runtime,session.ground)
    proposed=IntentionSession.restore(session.ground,scheduled.previous,
        expected_digest=scheduled.previous["snapshot_digest"])
    if scheduled.kind=="sense":
        ledger,symbol,decision=_qualify_observation(scheduled,reports)
        proposed.observe(scheduled.operation,str(symbol))
        record=_record(scheduled,reports,proposed.snapshot(),
                       ledger=ledger,symbol=symbol,decision=decision)
    else:
        proposed.apply(scheduled.operation)
        record=_record(scheduled,reports,proposed.snapshot())
    if not verify_evidence_record(session.ground,scheduled,runtime,record):
        raise ValueError("Independent E-CPU/VGI/Ground audit refused")
    session.state=proposed.state
    session.history=list(proposed.history)
    return record


def verify_material_outcome(ground,action_plan,goal_plan,runtime,
                            action_record,goal_observation_record):
    """Distinguish admitted intervention from observed post-intervention goal."""
    if (not verify_evidence_record(ground,action_plan,runtime,action_record) or
            not verify_evidence_record(ground,goal_plan,runtime,goal_observation_record)):
        raise ValueError("Unqualified action or post-action evidence")
    if (action_record["kind"]!="act" or goal_observation_record["kind"]!="sense" or
            goal_observation_record["operation"]!="read-goal" or
            goal_observation_record["previous"]!=action_record["after"] or
            goal_observation_record["ground_digest"]!=action_record["ground_digest"] or
            goal_observation_record["symbol"] not in (0,1)):
        raise ValueError("Post-effect observation does not continue intervention")
    body={"schema":"vl-i.pbg003.outcome.v1",
          "status":"OBSERVED_GOAL" if goal_observation_record["symbol"]==1
                    else "OBSERVED_NON_GOAL",
          "action_digest":action_record["digest"],
          "observation_digest":goal_observation_record["digest"],
          "goal_symbol":goal_observation_record["symbol"],
          "authority_scope":"receiver-local-admitted-actions-and-sensors"}
    return {**body,"digest":_hash(body)}
