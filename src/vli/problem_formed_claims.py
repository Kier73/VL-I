"""PFI-003: quorum-fenced result publication for a problem-formed processor.

Compose (do not replace) VL-I PFI-001/002, E-CPU native Virtual Layer,
OwnedNetworkFabric, computation DAG and TransferableExecutionClaimStore.
This finite prototype uses independent on-disk majority journals in ONE
shared filesystem and local subprocess workers, not physical remote quorums,
mTLS transport, or network-atomic external effects. It fences the accepted
result of a deterministic pure generated processor stage, not the physical
act of computation.
"""
from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path
from typing import Any

from .problem_formed_fabric import (FormedFabric, _frame, _hash, _canon, _decode,
                                     _apply_stage, _local_hypervisor_execute,
                                     MAX_PACKET)

SCHEMA = 'vl-i.problem-formed-quorum-stage.v1'
CANDIDATE = 'vl-i.problem-formed-candidate.v1'
MEMBERSHIP = 'pfi003-formed-processor'


def _require(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError(why)


def _hash_record(record: dict) -> str:
    return sha256(_canon(record)).hexdigest()


def _native():
    from mailcpu import (RecoveryLineageAuthority, RecoveryLineageCoordinator,
                         SeedCommitMembership, TransferableExecutionClaimStore,
                         ExecutionClaimStatus, OperatorSpec)
    return (RecoveryLineageAuthority, RecoveryLineageCoordinator,
            SeedCommitMembership, TransferableExecutionClaimStore,
            ExecutionClaimStatus, OperatorSpec)


def _atomic_json(path: Path, value: dict) -> None:
    """Local crash-consistent replacement, with fsync where supported."""
    raw = _canon(value)
    _require(len(raw) <= 2_000_000, 'Persisted PFI checkpoint exceeds budget')
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + '.staging')
    with staging.open('wb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    os.replace(staging, path)
    if os.name != 'nt':
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _read_json(path: Path) -> dict:
    raw = path.read_bytes()
    _require(len(raw) <= 2_000_000 and raw == _canon(json.loads(raw)),
             'Checkpoint must be bounded canonical JSON')
    return json.loads(raw)


def _spec():
    """The *same* pure operator contract used by PFI-002, not a new runtime."""
    _, _, _, _, _, OperatorSpec = _native()
    return OperatorSpec(
        operator_id='pfi.formed_segment', version=1,
        description='Execute source-qualified generated Boolean gate segment',
        input_kind='object', output_kind='object', substrates=('cpu',),
        required_rights=frozenset({'compute'}), max_input_bytes=MAX_PACKET,
        max_output_bytes=MAX_PACKET, base_energy=1,
        energy_per_byte=1, deterministic=True, effect_mode='pure')


class FencedFormedProcessor:
    """One persistent problem-formed pipeline and live quorum publication owner.

    Sequential owner protocol: stage claim creation, local receiver execution,
    durable candidate, majority completion, then PFI-002 DAG publication.
    If interrupted between majority completion and DAG checkpoint, reconcile
    replays the *pure* stage with fresh local authorization before advancing.
    Multiple concurrent owners require stronger global coordination than this
    filesystem demonstration; live native quorum checks fence acceptance.
    """

    def __init__(self, root: str | Path, source, quotient, bundle, *,
                 initial: int, gates_per_host: int = 4, job: str = 'pfi003-job'):
        Auth, Coordinator, Membership, Store, Status, _ = _native()
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        membership = Membership(MEMBERSHIP, 1, ('A', 'B', 'C'), 'A', 1, '0' * 64)
        authorities = {m: Auth(membership, m, self.root / 'quorum' / (m + '.jsonl'))
                       for m in membership.member_ids}
        self.coordinator = Coordinator(membership, authorities)
        self.store = Store(self.root / 'claims', self.coordinator)
        self.checkpoint = self.root / 'processor.json'
        expected = FormedFabric(source, quotient, bundle, initial=initial,
                                gates_per_host=gates_per_host, job=job)
        if self.checkpoint.exists():
            self.run = FormedFabric.restore(source, quotient, bundle,
                                            _read_json(self.checkpoint))
            _require(self.run.epoch == expected.epoch, 'Reopened wrong problem/host architecture')
        else:
            self.run = expected
            _atomic_json(self.checkpoint, self.run.snapshot())
        self.source, self.quotient, self.bundle = source, quotient, bundle
        self.spec = _spec()

    def _save(self) -> None:
        _atomic_json(self.checkpoint, self.run.snapshot())

    def _pending(self) -> tuple[int, dict, dict]:
        ready = self.run.dag.ready_nodes()
        _require(len(ready) == 1 and ready[0].node_id.startswith('step-'),
                 'No pending transferable generated stage')
        index = int(ready[0].node_id.split('-')[1])
        expected = self.run._reference_packets()
        queued = self.run.fabric.peek(self.run._receiver(index), max_size=MAX_PACKET)
        _require(queued is not None and
                 queued.source == self.run._sender(index) and
                 _decode(queued.payload) == expected[index],
                 'Missing or altered incoming E-CPU frame')
        return index, expected[index], expected[index + 1]

    def _binding(self) -> dict:
        index, before, after = self._pending()
        stage = self.run.steps[index]
        return {'parent_request_digest': _hash_record({'epoch':self.run.epoch,
                    'interpreter':self.bundle['digest'],'job':self.run.job}),
                'child_request_digest': _hash_record({'index':index,'in':before['sha256'],
                    'out':after['sha256'], 'stage':stage}),
                'address_digest': _hash_record({'operator':self.spec.digest,
                    'stage':index,'input_sha256':before['sha256'],
                    'interpreter':self.bundle['digest']}),
                'operator_spec':self.spec,
                'computational_ground_fingerprint': self.bundle['digest']}

    @staticmethod
    def _actor(peer: str, host: str) -> dict:
        _require(type(peer) is str and type(host) is str and
                 peer.isascii() and host.isascii() and
                 1 <= len(peer) <= 80 and 1 <= len(host) <= 80 and
                 all(c.isalnum() or c in '-_.' for c in peer + host),
                 'Unqualified host identity')
        return {'owner_peer_id':peer, 'owner_host_id':host,
                'implementation_fingerprint':_hash_record({'operator':'pfi.formed_segment@1',
                    'implementation':'PFI-002-checked-Boolean-gates',
                    'host':host})}

    def claim(self, *, peer: str, host: str):
        """Only initiate once at a ready stage, never replace a current owner."""
        return self.store.create(**self._binding(), **self._actor(peer, host))

    def _fresh(self, claim_id: str, peer: str, host: str):
        token = self.store.get(claim_id)
        token.verify_for_target(self.coordinator, **self._binding(),
                                **self._actor(peer,host))
        return token

    def calculate(self, claim_id: str, *, peer: str, host: str,
                  grant_compute: bool = True) -> dict:
        """Do actual receiver-local authorized work but *do not publish*."""
        ticket = self._fresh(claim_id,peer,host)
        index, before, after = self._pending()
        output, receipt = _local_hypervisor_execute(
            self.run.steps[index], before['wires'],
            epoch=self.run.epoch, index=index,host=host,
            bundle_digest=self.bundle['digest'],grant_compute=grant_compute)
        _require(output == after['wires'] and receipt['local_kernel_journal_verified'],
                 'Candidate calculation lacks independent problem-ground verification')
        body={'schema':CANDIDATE, 'claim_id':claim_id,
              'assignment_epoch':ticket.claim.current_assignment.epoch,
              'peer':peer,'host':host,'stage':index,
              'problem_epoch':self.run.epoch,'interpreter':self.bundle['digest'],
              'input_sha256':before['sha256'], 'output_sha256':_hash_record(output),
              'output':output, 'receiver':receipt}
        body['digest']=_hash_record(body)
        return body

    def reassign(self, claim_id: str, *, peer: str, host: str,
                 failure_evidence_digest: str):
        ticket=self.store.get(claim_id)
        return self.store.reassign(claim_id,
            expected_epoch=ticket.claim.current_assignment.epoch,
            **self._actor(peer,host),failure_evidence_digest=failure_evidence_digest)

    def _validate_candidate(self, candidate: dict, ticket) -> None:
        index, before, after = self._pending()
        expected={'schema':CANDIDATE,'claim_id':ticket.claim.claim_id,
                  'assignment_epoch':ticket.claim.current_assignment.epoch,
                  'peer':ticket.claim.current_assignment.owner_peer_id,
                  'host':ticket.claim.current_assignment.owner_host_id,
                  'stage':index, 'problem_epoch':self.run.epoch,
                  'interpreter':self.bundle['digest'],
                  'input_sha256':before['sha256'],
                  'output_sha256':_hash_record(after['wires']),
                  'output':after['wires']}
        _require(type(candidate) is dict and set(candidate)==set(expected)|{'receiver','digest'} and
                 all(candidate[k]==v for k,v in expected.items()) and
                 candidate.get('digest')==_hash_record({k:v for k,v in candidate.items() if k!='digest'}),
                 'Changed, stale, or wrongly bound computational candidate')
        rr=candidate['receiver']
        _require(type(rr) is dict and rr.get('host')==candidate['host'] and
                 rr.get('stage')==index and rr.get('hypervisor_action')=='ALLOW' and
                 rr.get('local_kernel_journal_verified') is True and
                 rr.get('result_sha256')==_hash_record(after['wires']),
                 'Receiver-local execution receipt projection does not match')

    def publish(self, candidate: dict) -> dict:
        """Commit current-epoch pure result through native quorum BEFORE DAG."""
        _require(type(candidate) is dict and 'claim_id' in candidate,
                 'Missing exact proposed claim')
        ticket=self.store.get(candidate['claim_id'])
        if ticket.claim.status.value == 'completed':
            return self.reconcile()
        self._validate_candidate(candidate,ticket)
        peer,host=candidate['peer'],candidate['host']
        self._fresh(candidate['claim_id'],peer,host)
        receipt_path=self.root / 'candidates' / (candidate['claim_id']+'.json')
        _atomic_json(receipt_path,candidate)  # durable before quorum commit
        finished=self.store.complete(candidate['claim_id'],
            expected_epoch=candidate['assignment_epoch'],**self._actor(peer,host),
            result_digest=candidate['output_sha256'],
            child_receipt_digest=candidate['digest'])
        _require(finished.claim.result_digest == candidate['output_sha256'],
                 'Completion changed the mathematical result')
        return self.reconcile()

    def reconcile(self) -> dict:
        """After crash between quorum commit and DAG write, re-execute pure stage."""
        index, before, after=self._pending()
        binding=self._binding()
        # Claim is identified from immutable Ground and native input, not from
        # an unchecked filename or an untrusted result packet.
        from mailcpu import TransferableExecutionClaim
        # Retrieve stage identity through typed E-CPU claim structure: validate
        # durable candidate independently of its user-supplied checksum.
        path=self.root / 'candidates'
        _require(path.is_dir(), 'No durable candidate exists for reconciling stage')
        candidates=[p for p in path.glob('*.json') if p.is_file()]
        matching=[]
        for candidate_path in candidates:
            candidate=_read_json(candidate_path)
            if candidate.get('stage')==index and candidate.get('problem_epoch')==self.run.epoch:
                matching.append(candidate)
        _require(len(matching)==1, 'Missing or ambiguous durable candidate')
        candidate=matching[0]
        token=self.store.get(candidate['claim_id'])
        token.verify_current(self.coordinator)
        self._validate_candidate(candidate,token)
        _require(token.claim.status.value=='completed' and
                 token.claim.result_digest==candidate['output_sha256'] and
                 token.claim.child_receipt_digest==candidate['digest'],
                 'Native quorum did not publish exactly this result')
        self.run.advance(host=candidate['host'],grant_compute=True)
        self._save()
        return {'status':'published','stage':index,'claim_id':candidate['claim_id'],
                'result_sha256':candidate['output_sha256'],
                'epoch':self.run.epoch,'native_quorum_generation':token.manifest.generation}

    def local_step(self) -> str:
        """Only local dispatch/finish; generated host stages require claims."""
        names=self.run.dag.ready_nodes()
        if not names:
            return 'done'
        node=names[0].node_id
        _require(node in ('dispatch','finish'),
                 'A generated host stage requires native claim ownership')
        self.run.advance(host='local-dispatch')
        self._save()
        return node

    def result(self) -> dict:
        return self.run.result()
