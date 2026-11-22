"""PFI-003: crash/rehome command runner. No network/TLS is implied here.

All subcommands are separate invocations, reopening fresh E-CPU quorum
journals, native Ground and local Hypervisor. A killed speculative process
may leave a correct *candidate* but cannot publish after reassignment.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from navigable_compute.symbolic_quotient import AffineMap, SymbolicSource
from vli.problem_formed_interpreter import construct
from vli.problem_formed_claims import FencedFormedProcessor, _atomic_json, _read_json, _hash_record


def problem():
    width=128
    rows=[1<<i for i in range(width)]
    rows[0]=3
    return SymbolicSource(width,(AffineMap(tuple(rows),1),), (1,),
                          'pfi003-declared-binary-affine-source')


def instance(path: Path) -> FencedFormedProcessor:
    source=problem()
    quotient, bundle=construct(source)
    return FencedFormedProcessor(path,source,quotient,bundle,
                                 initial=3,gates_per_host=2,job='formed-identity')


def action(root: Path, command: str, *, host='host-a', peer='peer-a',
           claim_id='', failure='', candidate='') -> dict:
    run=instance(root)
    if command=='dispatch':
        node=run.local_step()
        return {'node':node,'epoch':run.run.epoch}
    if command=='claim':
        ticket=run.claim(peer=peer,host=host)
        return {'claim_id':ticket.claim.claim_id,
                'epoch':ticket.claim.current_assignment.epoch,
                'root':ticket.manifest.digest}
    if command=='calculate':
        artifact=run.calculate(claim_id,peer=peer,host=host)
        _atomic_json(root/'speculative.json',artifact)
        return {'candidate_digest':artifact['digest'],
                'result_digest':artifact['output_sha256']}
    if command=='publish':
        artifact=_read_json(root / (candidate or 'speculative.json'))
        return run.publish(artifact)
    if command=='reassign':
        ticket=run.reassign(claim_id,peer=peer,host=host,
            failure_evidence_digest=failure or _hash_record({'observed':'worker-lost'}))
        return {'claim_id':ticket.claim.claim_id,
                'epoch':ticket.claim.current_assignment.epoch,
                'root':ticket.manifest.digest}
    if command=='local':
        return {'node':run.local_step()}
    if command=='reconcile':
        return run.reconcile()
    if command=='status':
        return {'complete':run.run.dag.is_complete,
                'committed':list(run.run.dag.committed_nodes),
                'result':run.result() if run.run.dag.is_complete else None}
    raise ValueError('Unknown command')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    p.add_argument('command',choices=['dispatch','claim','calculate',
                                      'publish','reassign','local','reconcile','status',
                                      'hold-after-calculate'])
    p.add_argument('--host',default='host-a')
    p.add_argument('--peer',default='peer-a')
    p.add_argument('--claim-id',default='')
    p.add_argument('--failure',default='')
    p.add_argument('--candidate',default='')
    args=p.parse_args()
    if args.command=='hold-after-calculate':
        value=action(args.root,'calculate',host=args.host,peer=args.peer,
                     claim_id=args.claim_id)
        print(json.dumps(value,sort_keys=True),flush=True)
        while True:
            time.sleep(1)
    else:
        print(json.dumps(action(args.root,args.command,host=args.host,peer=args.peer,
                    claim_id=args.claim_id,failure=args.failure,
                    candidate=args.candidate),sort_keys=True))
