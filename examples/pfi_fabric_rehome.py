"""PFI-002: use a new real Python process and fresh E-CPU Hypervisor per stage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from navigable_compute.symbolic_quotient import SymbolicSource, AffineMap
from navigable_compute.symbolic_quotient_wire import encode_symbolic, decode_symbolic
from vli.problem_formed_interpreter import construct
from vli.problem_formed_fabric import FormedFabric


def source_fixture():
    n=128
    r1=[1<<i for i in range(n)]
    r1[0]=3
    r2=[1<<i for i in range(n)]
    r2[0],r2[1]=2,1
    return SymbolicSource(n,(AffineMap(tuple(r1),1),AffineMap(tuple(r2),0)),
                          (1,), 'pfi002-described-problem-not-physical')


def stage(path: Path, host: str, *, initialize=False):
    if initialize:
        source=source_fixture()
        quotient,interpreter=construct(source)
        session=FormedFabric(source,quotient,interpreter,initial=3,
                             gates_per_host=1,job='formed-host-epoch')
        state={'mathematics':encode_symbolic(source,quotient).decode('ascii'),
               'checkpoint':session.snapshot()}
    else:
        state=json.loads(path.read_text('utf-8'))
        source,quotient,_=decode_symbolic(state['mathematics'].encode('ascii'))
        fresh,interpreter=construct(source)
        if fresh.packet()!=quotient.packet():
            raise ValueError('Fresh host failed independent mathematical audit')
        session=FormedFabric.restore(source,quotient,interpreter,state['checkpoint'])
    if not session.advance(host=host):
        raise ValueError('No remaining E-CPU computation stage')
    state['checkpoint']=session.snapshot()
    new=path.with_suffix('.pending')
    new.write_text(json.dumps(state,sort_keys=True),encoding='utf-8')
    new.replace(path)
    return {'host':host,'complete':session.dag.is_complete,
            'committed_nodes':list(session.dag.committed_nodes),
            'result':session.result() if session.dag.is_complete else None}

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('path',type=Path)
    parser.add_argument('--host',required=True)
    parser.add_argument('--initialize',action='store_true')
    args=parser.parse_args()
    print(json.dumps(stage(args.path,args.host,initialize=args.initialize),
                     sort_keys=True))
