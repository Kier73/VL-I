"""PBG-009: E-CPU quorum-fenced acceptance of a recovered neural observation.

The source Qwen computation is *not* smuggled into a replay-safe operator.
The E-CPU transferable claim is over PURE, deterministic verification and
publication of an already acquired, fully witnessed PBG-007 causal tape.
E-CPU's live RecoveryLineageCoordinator is the sole result authority;
PBG-008's IFS SQLite owner ledger is NOT used by this adapter.

The immutable observation journal and quorum claim state are distinct durable
stores. Publication is recoverable following any cut between quorum completion
and local outbox persistence. Origin peer/receiver TLS and remote discovery are
supplied by existing E-CPU claim-status and fabric transports when installed;
this adapter does not invent a new network channel or claim independent hosts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .source_hankel_ground import digest
from .active_neural_acquisition import NativeRunPlan, acquire_native_run
from .neural_trace_ground import WitnessedNeuralTraceEnsemble
from .durable_neural_acquisition import (
    ObservationRefused, _plan_record, _seal, _atomic_once, _read,
    _tape_record, _restore_tape,
)

SCHEMA = 'vli.pbg009.ecpu-quorum-neural-observation.v1'
OPERATOR_ID = 'vli.neural-observation.verify-prepared.v1'


def verified_observation_operator():
    """E-CPU's declared pure verification contract, not neural inference."""
    from mailcpu.virtual_layer import OperatorSpec
    return OperatorSpec(
        OPERATOR_ID, 1, 'Verify an immutable source-qualified neural tape and publish its digest',
        'object', 'object', ('cpu', 'memory'), required_rights=frozenset({'compute'}),
        base_energy=1, energy_per_byte=1, max_input_bytes=1_048_576,
        max_output_bytes=1_048_576, deterministic=True, effect_mode='pure',
        structural_memory_access='read',
    )


def semantic_identity(parent, candidates, plan):
    """Stable source/task/address identities, independent of host generation."""
    request = _seal(_plan_record(parent, candidates, plan))
    return {
        'parent_request_digest': digest({'schema': SCHEMA, 'parent_witness': parent.native_witness_digest,
                                         'source': parent.source_digest}),
        'child_request_digest': request['digest'],
        'address_digest': digest({'schema': SCHEMA, 'operation': OPERATOR_ID,
                                  'input': plan.selected_digest, 'observer': parent.observer.material(),
                                  'nonce': plan.nonce}),
        'computational_ground_fingerprint': parent.source_digest,
    }


def _validate_prepared(parent, candidates, plan, record, *, claim_id, intention):
    """Independent source-native audit; no original trained layer execution."""
    if (record.get('schema') != SCHEMA or record.get('claim_id') != claim_id
        or record.get('intent_digest') != intention['digest']
        or type(record.get('epoch')) is not int or record['epoch'] < 1):
        raise ObservationRefused('Prepared native observation is not bound to original E-CPU claim')
    tape = _restore_tape(record['tape'])  # SHAPE causal receipts checked here.
    if (tape.inputs != plan.selected or tape.observer != parent.observer or
        (tape.model, tape.revision, tape.index_ground_digest, tape.genesis) !=
        (parent.tapes[0].model, parent.tapes[0].revision,
         parent.tapes[0].index_ground_digest, parent.tapes[0].genesis)):
        raise ObservationRefused('Neural source identity, causal genesis or selected input changed')
    child = WitnessedNeuralTraceEnsemble((*parent.tapes, tape))
    acquired = record['native_receipt']
    if (acquired.get('schema') != 'vli.pbg007.active-source-observation.v1'
        or acquired.get('source_request_digest') != plan.proposal_digest
        or acquired.get('old_witness_digest') != parent.native_witness_digest
        or acquired.get('new_witness_digest') != child.native_witness_digest
        or acquired.get('selected_input_digest') != plan.selected_digest
        or acquired.get('new_source_digest') != child.source_digest
        or acquired.get('old_source_digest') != parent.source_digest
        or acquired.get('selected_native_transitions') != len(plan.selected)
        or acquired.get('unselected_native_transitions') != 0
        or acquired.get('qualified_new_positions') != tape.length
        or record.get('child_witness') != child.native_witness_digest
        or record.get('child_source') != child.source_digest):
        raise ObservationRefused('Native acquisition receipt, source revision or selected operation changed')
    return child


def _receipt_commitment(prepared):
    """Bind typed verification input; independent of host ownership."""
    return digest({'schema': SCHEMA, 'kind': 'verified-causal-neural-receipt',
                   'claim_id': prepared['claim_id'],
                   'tape_witness': prepared['tape']['witness_digest'],
                   'native_acquisition': prepared['native_receipt'],
                   'intent_digest': prepared['intent_digest']})


class QuorumNeuralObservation:
    """Bind PBG-007 native source evidence to E-CPU claim-store quorum.

    `store` and `live_coordinator` must be real E-CPU objects. A remote host
    should instantiate its own verified coordinator connected to live quorum;
    an old local cached proof is not a substitute for verify_current().
    """
    def __init__(self, directory, store, live_coordinator):
        from mailcpu.transferable_execution_claim import TransferableExecutionClaimStore
        from mailcpu.recovery_lineage import RecoveryLineageCoordinator
        if (not isinstance(store, TransferableExecutionClaimStore) or
            not isinstance(live_coordinator, RecoveryLineageCoordinator)):
            raise ObservationRefused('Live E-CPU transferable store and quorum coordinator required')
        self.root = Path(directory).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = store
        self.live = live_coordinator

    def _file(self, label):
        return self.root / (label + '.json')

    def _intent(self, parent, candidates, plan):
        expected = _seal(_plan_record(parent, candidates, plan))
        recorded = _read(self._file('intent'))
        if recorded != expected:
            raise ObservationRefused('Intention or source changed after durable issue')
        return recorded

    def _pointer(self, intent):
        pointer = _read(self._file('claim'))
        if (pointer.get('schema') != SCHEMA or
            pointer.get('intent_digest') != intent['digest'] or
            type(pointer.get('claim_id')) is not str or len(pointer['claim_id']) != 64):
            raise ObservationRefused('Wrong immutable E-CPU claim pointer')
        return pointer

    def _latest(self, parent, candidates, plan):
        intent = self._intent(parent, candidates, plan)
        ptr = self._pointer(intent)
        # Never trust a cached ticket: any unrelated state transition expires it.
        bundle = self.store.get(ptr['claim_id'])
        bundle.verify_current(self.live)
        claimed = bundle.claim
        spec = verified_observation_operator()
        expected = semantic_identity(parent, candidates, plan)
        if (claimed.claim_id != ptr['claim_id'] or
            any(getattr(claimed, key) != value for key, value in expected.items()) or
            claimed.operator_spec.to_dict() != spec.to_dict()):
            raise ObservationRefused('Live quorum claim does not match exact neural intention and operator')
        return intent, bundle

    def _target(self, parent, candidates, plan, *, peer, host, implementation):
        intent, bundle = self._latest(parent, candidates, plan)
        bundle.verify_for_target(self.live, **semantic_identity(parent, candidates, plan),
                                 operator_spec=verified_observation_operator(),
                                 owner_peer_id=peer, owner_host_id=host,
                                 implementation_fingerprint=implementation)
        return intent, bundle

    def issue(self, parent, candidates, plan, *, peer, host, implementation):
        """Durably pin selected source request, then create E-CPU quorum claim."""
        intent = _seal(_plan_record(parent, candidates, plan))
        _atomic_once(self._file('intent'), intent)
        bundle = self.store.create(**semantic_identity(parent, candidates, plan),
                                   operator_spec=verified_observation_operator(),
                                   owner_peer_id=peer, owner_host_id=host,
                                   implementation_fingerprint=implementation)
        ptr = _seal({'schema': SCHEMA, 'intent_digest': intent['digest'],
                     'claim_id': bundle.claim.claim_id})
        _atomic_once(self._file('claim'), ptr)
        self._target(parent, candidates, plan, peer=peer, host=host,
                     implementation=implementation)
        return bundle

    def reassign(self, parent, candidates, plan, *, expected_epoch, peer, host,
                 implementation, failure_evidence_digest):
        """Existing E-CPU quorum transition revokes old result acceptance."""
        _intent, prior = self._latest(parent, candidates, plan)
        if prior.claim.current_assignment.epoch != expected_epoch:
            raise ObservationRefused('Stale observer assignment cannot be rehomed')
        return self.store.reassign(prior.claim.claim_id, expected_epoch=expected_epoch,
                                   owner_peer_id=peer, owner_host_id=host,
                                   implementation_fingerprint=implementation,
                                   failure_evidence_digest=failure_evidence_digest)

    def _prepared(self, parent, candidates, plan, *, intent, claim_id):
        candidates_paths = sorted(self.root.glob('prepared-*.json'))
        for path in candidates_paths:
            record = _read(path) # An invalid recorded source result fails closed.
            _validate_prepared(parent, candidates, plan, record,
                               claim_id=claim_id, intention=intent)
            return record
        return None

    def prepare(self, parent, candidates, plan, factory, *, peer, host, implementation):
        """Reuse verified earlier prepared evidence, or execute ONE selected run.

        Neural execution is outside the pure E-CPU operation. The prepared
        bytes are audited against original SHAPE witnesses before commitment.
        """
        intent, bundle = self._target(parent, candidates, plan,
                                      peer=peer, host=host, implementation=implementation)
        existing = self._prepared(parent, candidates, plan, intent=intent,
                                  claim_id=bundle.claim.claim_id)
        if existing is not None:
            return existing
        child, acquisition = acquire_native_run(parent, candidates, plan, factory)
        # Host may have been reassigned while native computation was running.
        _intent, current = self._target(parent, candidates, plan,
                                        peer=peer, host=host, implementation=implementation)
        if current.claim.current_assignment.epoch != bundle.claim.current_assignment.epoch:
            raise ObservationRefused('Owner generation changed during native source acquisition')
        tape = child.tapes[-1]
        candidate = _seal({'schema': SCHEMA, 'claim_id': bundle.claim.claim_id,
                           'intent_digest': intent['digest'],
                           'epoch': current.claim.current_assignment.epoch,
                           'tape': _tape_record(tape), 'native_receipt': acquisition,
                           'child_witness': child.native_witness_digest,
                           'child_source': child.source_digest})
        _validate_prepared(parent, candidates, plan, candidate,
                           claim_id=bundle.claim.claim_id, intention=intent)
        _atomic_once(self._file('prepared-'+str(candidate['epoch'])), candidate)
        return candidate

    def restore_prepared(self, parent, candidates, plan):
        """Rebuild verified source child with zero trained neural executions."""
        intent, bundle = self._latest(parent, candidates, plan)
        prepared = self._prepared(parent, candidates, plan, intent=intent,
                                  claim_id=bundle.claim.claim_id)
        if prepared is None:
            raise ObservationRefused('No durable qualified native observation is prepared')
        child = _validate_prepared(parent, candidates, plan, prepared,
                                   claim_id=bundle.claim.claim_id, intention=intent)
        return child, prepared

    def publish(self, parent, candidates, plan, *, peer, host, implementation):
        """Quorum completion is the *only* semantic result acceptance event."""
        intent, current = self._latest(parent, candidates, plan)
        assignment = current.claim.current_assignment
        if (assignment.owner_peer_id, assignment.owner_host_id,
            assignment.implementation_fingerprint) != (peer, host, implementation):
            raise ObservationRefused('Stale or wrong Host cannot publish verified observation')
        child, prepared = self.restore_prepared(parent, candidates, plan)
        result_digest, receipt_digest = prepared['digest'], _receipt_commitment(prepared)
        from mailcpu.transferable_execution_claim import ExecutionClaimStatus
        if current.claim.status is ExecutionClaimStatus.ASSIGNED:
            # Fresh admission immediately before quorum completion.
            self._target(parent, candidates, plan, peer=peer, host=host,
                         implementation=implementation)
            current = self.store.complete(current.claim.claim_id,
                expected_epoch=assignment.epoch, owner_peer_id=peer,
                owner_host_id=host, implementation_fingerprint=implementation,
                result_digest=result_digest, child_receipt_digest=receipt_digest)
        else:
            if (current.claim.result_digest != result_digest or
                current.claim.child_receipt_digest != receipt_digest):
                raise ObservationRefused('Completed E-CPU claim binds another observation result')
        current.verify_current(self.live)
        if (current.claim.status is not ExecutionClaimStatus.COMPLETED or
            current.claim.result_digest != result_digest or
            current.claim.child_receipt_digest != receipt_digest):
            raise ObservationRefused('E-CPU did not quorum-commit the qualified source result')
        accepted = _seal({'schema': SCHEMA, 'claim_id': current.claim.claim_id,
                          'intent_digest': intent['digest'],
                          'result_digest': result_digest,
                          'receipt_digest': receipt_digest,
                          'child_source': child.source_digest,
                          'child_witness': child.native_witness_digest,
                          'completion_generation': current.claim.completion_generation})
        _atomic_once(self._file('accepted'), accepted)
        return self.verify_publication(parent, candidates, plan)

    def verify_publication(self, parent, candidates, plan):
        """Verify latest quorum state + original causal evidence without replay."""
        intent, ticket = self._latest(parent, candidates, plan)
        from mailcpu.transferable_execution_claim import ExecutionClaimStatus
        if ticket.claim.status is not ExecutionClaimStatus.COMPLETED:
            raise ObservationRefused('No E-CPU quorum-completed observation')
        child, prepared = self.restore_prepared(parent, candidates, plan)
        accepted = _read(self._file('accepted'))
        expected = _seal({'schema': SCHEMA, 'claim_id': ticket.claim.claim_id,
                          'intent_digest': intent['digest'],
                          'result_digest': prepared['digest'],
                          'receipt_digest': _receipt_commitment(prepared),
                          'child_source': child.source_digest,
                          'child_witness': child.native_witness_digest,
                          'completion_generation': ticket.claim.completion_generation})
        if (accepted != expected or
            ticket.claim.result_digest != prepared['digest'] or
            ticket.claim.child_receipt_digest != _receipt_commitment(prepared)):
            raise ObservationRefused('Published neural observation is not live-quorum accepted')
        return {'schema': SCHEMA, 'status': 'QUORUM_NEURAL_OBSERVATION_ACCEPTED',
                'claim_id': ticket.claim.claim_id,
                'assignment_epoch': ticket.claim.current_assignment.epoch,
                'owner_peer_id': ticket.claim.current_assignment.owner_peer_id,
                'owner_host_id': ticket.claim.current_assignment.owner_host_id,
                'completion_generation': ticket.claim.completion_generation,
                'source_digest': child.source_digest,
                'native_witness_digest': child.native_witness_digest,
                'accepted_result_digest': prepared['digest'],
                'receipt_digest': ticket.claim.child_receipt_digest,
                'new_neural_calls_during_restore': 0,
                'quorum_manifest_digest': ticket.manifest.digest,
                'portable_claim_ticket_digest': ticket.digest}


def admit_authenticated_observer(client, live_coordinator, parent, candidates, plan,
                                 *, claim_id, owner_host_id, implementation_fingerprint):
    """E-CPU direct-mTLS receiver admission for a recovered observer job.

    The existing ComputationalFabricPeerClient performs CA/peer-DNS-validated
    mutual TLS and asks the owner claim-status service whether this exact
    authenticated receiving peer and logical Host still own `claim_id`.
    Independently demand current quorum/Merkle proof and entire semantic,
    address, pure operator, source Ground and implementation match here.
    The result is an admission ticket, NOT result transport or completion.
    """
    from mailcpu.computational_fabric import ComputationalFabricPeerClient
    from mailcpu.recovery_lineage import RecoveryLineageCoordinator
    if (not isinstance(client, ComputationalFabricPeerClient) or
            not isinstance(live_coordinator, RecoveryLineageCoordinator)):
        raise ObservationRefused('Authenticated E-CPU peer client and live quorum required')
    ticket = client.get_execution_claim_status(claim_id,
                                                expected_host_id=owner_host_id)
    ticket.verify_for_target(
        live_coordinator,
        **semantic_identity(parent, candidates, plan),
        operator_spec=verified_observation_operator(),
        owner_peer_id=client.local_peer_id,
        owner_host_id=owner_host_id,
        implementation_fingerprint=implementation_fingerprint,
    )
    return ticket
