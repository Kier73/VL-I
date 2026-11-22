"""PFI-001: source-intention -> new machine -> E-CPU receiver/checkpoint -> revised machine."""
import json

from navigable_compute.symbolic_quotient import AffineMap, SymbolicSource
from mailcpu.computation_dag import NavigableComputationDAG
from vli.problem_formed_interpreter import (
    construct, realize, create_ecpu_dag, run_one_ecpu_frontier,
    certify_succession, verify_succession,
)


def _source(outputs, n=128):
    rows = [1 << i for i in range(n)]
    rows[0] ^= 1 << 1
    return SymbolicSource(n,(AffineMap(tuple(rows),1),),tuple(outputs),
                          'specified-affine-environment-not-live-sensing')


def main():
    source = _source((1,))
    quotient, interpreter = construct(source)
    intention = (0,)
    count = 10**20 + 137
    dag = create_ecpu_dag(interpreter, initial=7, word=intention,repetitions=count)
    run_one_ecpu_frontier(source,quotient,interpreter,dag)
    # The checkpoint is an actual native E-CPU computation DAG snapshot; a
    # completely fresh Python process could load it after transferring bytes.
    restored = NavigableComputationDAG.from_snapshot(json.loads(json.dumps(dag.snapshot())))
    while not restored.is_complete:
        run_one_ecpu_frontier(source,quotient,interpreter,restored)
    result = restored.result('observe')
    source2 = _source((1,4))
    quotient2, interpreter2 = construct(source2)
    anchor = 7
    old_native = quotient.project(anchor)
    succession = certify_succession(source, quotient, interpreter, source2,
                                    quotient2,interpreter2,
                                    old_native_state=old_native,source_anchor=anchor)
    assert verify_succession(source,quotient,interpreter,source2,quotient2,
                             interpreter2,succession)
    output = {
        'schema':'vl-i.pfi-001-demo.v1','source_bits':source.width,
        'interpreter_1_registers':quotient.dimension,
        'interpreter_2_registers':quotient2.dimension,
        'interpreter_1_primitives':interpreter['primitive_set'],
        'interpreter_2_primitives':interpreter2['primitive_set'],
        'original_interpreter_digest':interpreter['digest'],
        'reformed_interpreter_digest':interpreter2['digest'],
        'logical_coordinate':str(count),
        'observed_result':result['observed_decimal'],
        'native_ecpu_complete':restored.is_complete,
        'succession_kind':succession['transition'],
        'mathematical_claim_scope':'fully-declared-binary-affine-dynamics-only',
        'host_claim_scope':'local-python-receiver-ECPU-DAG-not-real-network',
    }
    print(json.dumps(output,indent=2))
    return output


if __name__ == '__main__':
    main()
