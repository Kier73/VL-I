"""PFI-001: three genuinely separate Python processes, one retained E-CPU DAG.

Run stage=0,1,2 from three subprocesses; stage 0 writes a retained checkpoint,
subsequent stages reconstruct independently from portable mathematical source.
Nothing in this test uses sockets, TLS, network computation or remote authority.
"""
import argparse
import json
from pathlib import Path

from mailcpu.computation_dag import NavigableComputationDAG
from navigable_compute.symbolic_quotient import AffineMap, SymbolicSource
from navigable_compute.symbolic_quotient_wire import encode_symbolic, decode_symbolic
from vli.problem_formed_interpreter import (
    construct, create_ecpu_dag, run_one_ecpu_frontier,
)


def source_fixture():
    n = 128
    rows = [1 << i for i in range(n)]
    rows[0] ^= 2
    return SymbolicSource(n,(AffineMap(tuple(rows),1),),(1,),
                          'pfi001-problem-declared-dynamics-not-physical')


def one_stage(path, stage):
    if stage == 0:
        source = source_fixture()
        quotient, bundle = construct(source)
        graph = create_ecpu_dag(bundle,initial=3,word=(0,),repetitions=10**16+137)
        math_bytes = encode_symbolic(source,quotient).decode('ascii')
        envelope = {'source_mathematics': math_bytes,
                    'interpreter_digest': bundle['digest'],
                    'native_dag': graph.snapshot(),
                    'stage_completed': -1}
    else:
        envelope = json.loads(path.read_text(encoding='ascii'))
        source, quotient, _ = decode_symbolic(envelope['source_mathematics'].encode('ascii'))
        constructed_quotient, bundle = construct(source)
        if constructed_quotient.packet() != quotient.packet() or \
           envelope['interpreter_digest'] != bundle['digest'] or \
           envelope['stage_completed'] != stage - 1:
            raise ValueError('Unqualified new-process host state or stale epoch')
        graph = NavigableComputationDAG.from_snapshot(envelope['native_dag'])
    if not run_one_ecpu_frontier(source,quotient,bundle,graph):
        raise ValueError('Expected one pending native E-CPU DAG operation')
    envelope.update({'native_dag':graph.snapshot(), 'stage_completed':stage})
    if stage == 2 and not graph.is_complete:
        raise ValueError('Logical computation did not finish')
    tmp = path.with_suffix('.pending')
    tmp.write_text(json.dumps(envelope,sort_keys=True), encoding='ascii')
    tmp.replace(path)
    return {'stage':stage,'ready': [x.node_id for x in graph.ready_nodes()],
            'complete':graph.is_complete, 'interpreter_digest':bundle['digest'],
            'observed_decimal':graph.result('observe')['observed_decimal'] if graph.is_complete else None}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path',type=Path)
    parser.add_argument('stage',type=int,choices=(0,1,2))
    args = parser.parse_args()
    print(json.dumps(one_stage(args.path,args.stage),sort_keys=True))
