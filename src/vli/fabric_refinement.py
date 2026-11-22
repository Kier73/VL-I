"""TDL-004: observer formation over E-CPU receiver-owned executable routes.

This composes existing SAB-001 admission, TDL-003 mathematical selection and
VGI explicit evidence finalization. It has receiver-local handler/carrier
authority; authentication of physical remote origins is a separate contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns

from .problem_formed_interpreter import _hash
from .cost_aware_refinement import (
    ObservationOffer, plan_cost_aware_observations,
    observation_context, acquire_vgi_refinement, GROUND_ID, SOURCE_STATE,
)
from .cost_aware_refinement_verify import verify_cost_plan, verify_vgi_refinement_receipt

INPUT = "refinement.request.v1"
OUTPUT = "refinement.observation.v1"
SCHEMA = "vl-i.fabric-refinement.v1"
ADAPTER_SOURCE = "receiver-admitted-fabric"
CHANNEL_ID = "refinement.fabric.receipts"
PROTOCOL_ID = "refinement.fabric.receipt/1"
EPOCH_SCHEMA = "vl-i.fabric-refinement-epoch.v1"
MAX_QUANTA = 100_000


def offer_operation(offer_id: str) -> str:
    """Typed operation requested only at a receiver-admitted observation locus."""
    return "refinement.observe." + offer_id + ".v1"


def _epoch(args, runtime, old_native_state: int, coordinate: int) -> str:
    old_source, old_q, old_bundle, new_source, new_q, new_bundle = args
    if (type(coordinate) is not int or coordinate < 0
            or type(old_native_state) is not int
            or not 0 <= old_native_state < 1 << old_q.dimension):
        raise ValueError("Invalid intent state or observation coordinate")
    return _hash({
        "schema": EPOCH_SCHEMA,
        "old_source": old_source.digest,
        "new_source": new_source.digest,
        "old_interpreter": old_bundle["digest"],
        "new_interpreter": new_bundle["digest"],
        "old_native_decimal": str(old_native_state),
        "inventory": runtime.inventory.fingerprint,
        "policy": runtime.policy.fingerprint,
        "coordinate": coordinate,
    })


@dataclass(frozen=True)
class AdmittedRoute:
    """A typed task and concrete SAB-001 host/path plan for one offered row."""
    offer: ObservationOffer
    task: object
    plan: object


@dataclass(frozen=True)
class FabricRefinement:
    """Mathematical optimum together with its currently admitted E-CPU routes."""
    mathematical_plan: dict
    epoch_digest: str
    old_native_state: int
    coordinate: int
    ingress: str
    collector: str
    declared_offers: tuple[ObservationOffer, ...]
    routes: tuple[AdmittedRoute, ...]

    @property
    def fingerprint(self) -> str:
        return _hash({
            "schema": "vl-i.fabric-refinement-plan.v1",
            "math": self.mathematical_plan["digest"], "epoch": self.epoch_digest,
            "state": str(self.old_native_state), "coordinate": self.coordinate,
            "ingress": self.ingress, "collector": self.collector,
            "declared_offers": [o.encode() for o in self.declared_offers],
            "routes": [{
                "offer_id": route.offer.offer_id,
                "task_digest": route.task.fingerprint,
                "plan_digest": route.plan.fingerprint,
            } for route in self.routes],
        })


def plan_fabric_refinement(
    old_source, old_quotient, old_bundle,
    new_source, new_quotient, new_bundle,
    offers: tuple[ObservationOffer, ...],
    runtime, *, ingress: str, collector: str,
    old_native_state: int, coordinate: int,
) -> FabricRefinement:
    """Minimize total admitted path+operator quanta, not just advertised cost.

    SAB-001 intersects observed leases, resource offers, receiver allowlists,
    bound callable handlers, and bound carriers. Unroutable offers are removed
    before the TDL-003 exact row-set optimization.
    """
    from mailcpu.substrate_bootstrap import (
        AdaptiveBootstrap, BootstrapRefused, BootstrapTask,
    )
    if not isinstance(runtime, AdaptiveBootstrap):
        raise ValueError("A receiver-owned AdaptiveBootstrap runtime is required")
    if (type(offers) is not tuple or len(offers) > 12 or
            any(not isinstance(o, ObservationOffer) for o in offers) or
            len({o.offer_id for o in offers}) != len(offers)):
        raise ValueError("Bounded unique observer offers required")
    args = (old_source, old_quotient, old_bundle,
            new_source, new_quotient, new_bundle)
    epoch = _epoch(args, runtime, old_native_state, coordinate)
    available = []
    for offer in sorted(offers, key=lambda item: item.offer_id):
        if offer.row >= 1 << old_source.width:
            raise ValueError("Offered observation row outside declared source")
        # Bind each observation to one unique operation identity and locus.
        operation = offer_operation(offer.offer_id)
        if any(key[1] == operation and key[0] != offer.locus
               for key in runtime.realizers):
            raise ValueError("One offered observation has multiple local executing loci")
        task = BootstrapTask(
            intention_id="observe-" + offer.offer_id,
            ground_digest=epoch,
            source=ingress, input_kind=INPUT,
            target=collector, output_kind=OUTPUT,
            value={
                "epoch_digest": epoch,
                "offer_id": offer.offer_id,
                "row_decimal": str(offer.row),
                "old_native_decimal": str(old_native_state),
            },
            max_quanta=MAX_QUANTA, max_steps=16, max_bytes=8192,
            require_oracle=True,
        )
        try:
            route = runtime.plan(task, coordinate=coordinate)
        except BootstrapRefused:
            continue
        realizations = [step for step in route.steps if step.kind == "realize"]
        if (len(realizations) != 1
                or realizations[0].locus != offer.locus
                or realizations[0].operation != operation
                or realizations[0].quanta != offer.cost):
            # Resource claims must agree with receiver-admitted operation and
            # its declared cost. A route to another handler is not equivalent.
            continue
        available.append(AdmittedRoute(offer, task, route))
    effective = tuple(ObservationOffer(
        r.offer.offer_id, r.offer.row, r.plan.estimated_quanta, r.offer.locus
    ) for r in available)
    mathematical_plan = plan_cost_aware_observations(*args, effective)
    if not verify_cost_plan(*args, mathematical_plan):
        raise ValueError("Chosen route-cost plan failed independent mathematical audit")
    by_id = {r.offer.offer_id: r for r in available}
    selected = tuple(by_id[i] for i in mathematical_plan["selected_offer_ids"])
    return FabricRefinement(
        mathematical_plan, epoch, old_native_state, coordinate,
        ingress, collector, offers, selected
    )


def execute_fabric_refinement(
    old_source, old_quotient, old_bundle,
    new_source, new_quotient, new_bundle,
    acquisition: FabricRefinement, runtime,
) -> dict:
    """Execute admitted receiver-local paths and translate outputs to VGI.

    A single receiver adapter represents locally verified SAB reports to VGI;
    it is one policy source, not a claim of independently authenticated peers.
    """
    from mailcpu.substrate_bootstrap import (
        AdaptiveBootstrap, BootstrapRefused, verify_bootstrap_report,
        canonical, digest,
    )
    from vgi.channels import ChannelContract, ChannelExchangeLedger, ChannelMessage
    from vgi.encounters import EncounterProvenance
    from vgi.evidence import (
        TransitionEvidenceLedger, TransitionEvidencePolicy, TransportFrameReceipt,
    )
    if not isinstance(runtime, AdaptiveBootstrap) or not isinstance(acquisition, FabricRefinement):
        raise ValueError("Admitted E-CPU runtime and fabric plan required")
    args = (old_source, old_quotient, old_bundle,
            new_source, new_quotient, new_bundle)
    epoch = _epoch(args, runtime, acquisition.old_native_state, acquisition.coordinate)
    if epoch != acquisition.epoch_digest or not verify_cost_plan(*args, acquisition.mathematical_plan):
        raise ValueError("Source, inventory, policy or intent state changed")
    if tuple(r.offer.offer_id for r in acquisition.routes) != tuple(acquisition.mathematical_plan["selected_offer_ids"]):
        raise ValueError("Route selection differs from mathematical plan")

    reports = []
    for route in acquisition.routes:
        replan = runtime.plan(route.task, coordinate=acquisition.coordinate)
        if replan.fingerprint != route.plan.fingerprint:
            raise ValueError("SAB route is no longer admitted")
        start = perf_counter_ns()
        report = runtime.execute(route.task, route.plan, coordinate=acquisition.coordinate)
        elapsed = perf_counter_ns() - start
        if not verify_bootstrap_report(report, route.task, route.plan):
            raise ValueError("SAB execution receipt failed verification")
        value = report["output"]
        if (type(value) is not dict or set(value) != {
                "request_digest", "offer_id", "row_decimal", "bit"
            } or value["request_digest"] != digest(route.task.value)
            or value["offer_id"] != route.offer.offer_id
            or value["row_decimal"] != str(route.offer.row)
            or type(value["bit"]) is not int or value["bit"] not in (0, 1)):
            raise ValueError("Receiver-bound observer returned wrong typed witness")
        reports.append({
            "offer_id": route.offer.offer_id, "bit": value["bit"],
            "report": report, "elapsed_ns": elapsed,
            "application_input_bytes": len(canonical(route.task.value)),
            "application_output_bytes": len(canonical(value)),
        })

    context = observation_context(acquisition.mathematical_plan,
                                  old_native_state=acquisition.old_native_state)
    ledger = TransitionEvidenceLedger(
        TransitionEvidencePolicy(
            GROUND_ID, context, (ADAPTER_SOURCE,), 1, True
        )
    )
    contract = ChannelContract(
        CHANNEL_ID, PROTOCOL_ID, observation_schema="vgi.transition-observation/1"
    )
    messages = []
    for i, item in enumerate(reports):
        messages.append(ChannelMessage(
            "fabric-refine-" + str(i), CHANNEL_ID, PROTOCOL_ID,
            "observation", {
                "status": "known", "source_state": SOURCE_STATE,
                "action": item["offer_id"],
                "target": SOURCE_STATE, "output": [item["bit"]],
            },
            EncounterProvenance(
                ADAPTER_SOURCE, "ecpu-report://" + item["report"]["report_digest"],
                collector="receiver-admitted-fabric",
            ),
        ))
    exchange = ChannelExchangeLedger()
    for message in messages:
        exchange = exchange.record_observation(contract, message)
    encounters = exchange.to_encounter_ledger()
    for i, message in enumerate(messages):
        frame = TransportFrameReceipt.capture(
            message, transport_id="receiver-local-ecpu-sab", sequence=i + 1
        )
        ledger = ledger.ingest_channel_message(
            exchange, message.message_id, frame, encounters
        )
    for route in acquisition.routes:
        ledger = ledger.finalize(SOURCE_STATE, route.offer.offer_id)
    new_state = acquire_vgi_refinement(
        *args, acquisition.mathematical_plan,
        old_native_state=acquisition.old_native_state,
        ledger=ledger if acquisition.routes else None,
    )
    result = {
        "schema": SCHEMA,
        "fabric_plan_digest": acquisition.fingerprint,
        "epoch_digest": epoch,
        "mathematical_plan_digest": acquisition.mathematical_plan["digest"],
        "declared_total_quanta": acquisition.mathematical_plan["total_cost"],
        "measured_elapsed_ns": sum(x["elapsed_ns"] for x in reports),
        "measured_application_bytes": sum(x["application_input_bytes"] + x["application_output_bytes"] for x in reports),
        "path_reports": reports,
        "vgi_ledger": ledger.to_dict() if acquisition.routes else None,
        "vgi_ledger_digest": ledger.fingerprint if acquisition.routes else None,
        "refinement_receipt": new_state,
        "authority_scope": "receiver-owned-sab-handler/carrier+single-VGI-adapter",
    }
    result["digest"] = _hash(result)
    from .fabric_refinement_verify import verify_fabric_refinement
    if not verify_fabric_refinement(*args, acquisition, runtime, result):
        raise ValueError("Independent host-path / VGI refinement audit refused")
    return result
