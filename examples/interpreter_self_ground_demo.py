"""Reproduce the interpreter-as-Ground experiment with actual E-CPU and VLD.

Usage:
  python -m examples.interpreter_self_ground_demo --out artifacts/self-ground.json

Requires separately installed, pinned E-CPU and Virtual-Layer-Dynamics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter_ns

from mailcpu.virtual_layer import OperatorRegistry, OperatorSpec
from vld.signal import ModularRingBackend, SignalGroundSpec, VirtualGroundInterpreter

from vli.interpreter_ground import (
    derive_orbit_from_interpreter_description,
    lift_interpreter,
    load_real_interpreter_modules,
    operate_byte_ground,
    rebind_interpreter_from_ground,
    verify_byte_ground_run,
    verify_interpreter_ground,
)


def run():
    registry = OperatorRegistry()
    registry.declare(OperatorSpec(
        operator_id="vli.byte.relation", version=1,
        description="VLD observer-byte relation (declarative)",
        input_kind="object", output_kind="object",
        substrates=("cpu",), deterministic=True, effect_mode="pure"))
    backend = ModularRingBackend(8)
    interpreter = VirtualGroundInterpreter(
        SignalGroundSpec.for_backend(backend, ground_id="vli.byte-ground"), backend)
    source = load_real_interpreter_modules()
    t0 = perf_counter_ns()
    ground = lift_interpreter(source, registry, interpreter)
    lift_time_ns = perf_counter_ns()-t0
    verify_interpreter_ground(ground, source, registry, interpreter)

    declared, restored = rebind_interpreter_from_ground(
        ground, source, expected_ground_digest=ground["ground_digest"])
    data = source["VLI.InterpreterSelfGround"]
    t1 = perf_counter_ns()
    reading = operate_byte_ground(
        data, left=0, right=1, operation="ADD", observer="vl-i-self-observer",
        interpreter_ground=ground, signal_interpreter=restored)
    execution_time_ns = perf_counter_ns()-t1
    verify_byte_ground_run(reading, data, ground, restored)

    t2 = perf_counter_ns()
    distant = derive_orbit_from_interpreter_description(
        ground, source, source_name="VLI.InterpreterSelfGround",
        coordinate=1_000_000_003, observer="vl-i-temporal-observer")
    orbit_time_ns = perf_counter_ns()-t2

    return {
        "schema": "vl-i.interpreter-self-ground-experiment.v1",
        "scope": "real_interpreter_source_bytes_e-cpu_operator_contract_vld_execution_and_certified_memory",
        "source_modules": [v["name"] for v in ground["source_modules"]],
        "source_bytes_observed": ground["source_bytes_observed"],
        "ground_digest": ground["ground_digest"],
        "ecpu_operator_digest": declared.digest,
        "vld_signal_ground_fingerprint": restored.ground.fingerprint,
        "source_byte_intention": {
            "original_source_sha256": reading["source_sha256"],
            "positions": reading["input_coordinates"],
            "input_values": reading["operand_bytes"],
            "result": reading["value"],
            "independent_result": reading["oracle_value"],
            "executed_vld_signal_steps": reading["signal_steps"],
            "radiance_head": reading["transcript"]["head"],
            "radiance_event_count": len(reading["transcript"]["entries"]),
            "run_digest": reading["run_digest"],
        },
        "navigated_source_derived_memory": {
            "coordinate": distant["coordinate"],
            "value": distant["value"],
            "source_sha256": distant["source_sha256"],
            "source_bytes": distant["source_byte_values"],
            "law_family": distant["law_family"],
            "certificate_digest": distant["native_certificate_digest"],
            "native_receipt_digest": distant["native_receipt_digest"],
            "expanded_intermediate_states": distant["expanded_intermediate_states"],
            "radiance_head": distant["transcript"]["head"],
        },
        "measured_local_cost_ns": {
            "lift": lift_time_ns,
            "byte_relation_execution": execution_time_ns,
            "certify_and_navigate": orbit_time_ns,
        },
        "not_claimed": [
            "reconstruction of executable handlers from metadata alone",
            "cryptographic authentication of remote hosts",
            "neural-model token generation",
            "zero physical or interpreter computation",
        ],
    }


def main(argv=None):
    p=argparse.ArgumentParser(description="Execute ECPU/VLD-based interpreter self-Ground experiment")
    p.add_argument("--out",default="artifacts/interpreter-self-ground.json")
    args=p.parse_args(argv)
    result=run()
    output=Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n",encoding="utf-8")
    print("VL_I_INTERPRETER_SELF_GROUND",json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()
