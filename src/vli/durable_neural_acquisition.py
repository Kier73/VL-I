"""PBG-008: crash-recoverable, owner-fenced native observation publication.

Reuses IFS USM-007's ProofActionRouteEpochLedger + SQLiteRouteEpochLedgerStore
for durable CAS owner generations, and PBG-007 / SHAPE for native verification.
Prepared neural output is a recoverable *pure observational* artifact. The
ledger's execution_authorized event is the publication authority; its stable
ticket deduplicates the logical event across process restart. Retrying native
numerical execution may consume compute; no exactly-once side-effect promise.

Single SQLite authority on one fault domain, not remote distributed consensus
or a live mutual-TLS source-identity claim. Remote authenticated receivers can
be adapted later using existing TDL-005/006 IOS mTLS measurements.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import uuid
from typing import Any

from .source_hankel_ground import digest
from .active_neural_acquisition import NativeRunPlan, select_native_run, acquire_native_run
from .neural_trace_ground import CausalTape, NeuralObserver, WitnessedNeuralTraceEnsemble

SCHEMA = 'vli.pbg008.durable-neural-observation.v1'
RUNNER = 'pbg008.source-native-result-publisher'


class ObservationRefused(ValueError):
    """The observation or the claimed source authority is not admissible."""


def _canonical(v):
    return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf8')


def _path(path):
    return [[token,[float(x).hex() for x in values]] for token,values in path]


def _unpath(path):
    return tuple((int(token),tuple(float.fromhex(x) for x in values)) for token,values in path)


def _plan_record(parent,candidates,plan):
    if not isinstance(parent,WitnessedNeuralTraceEnsemble) or not isinstance(plan,NativeRunPlan):
        raise ObservationRefused('Missing witnessed parent or selected native run')
    if select_native_run(parent,candidates,nonce=plan.nonce)!=plan:
        raise ObservationRefused('Stale or changed native acquisition selection')
    return {'schema':SCHEMA,'parent_witness':parent.native_witness_digest,
            'parent_source':parent.source_digest,'observer':parent.observer.material(),
            'nonce':plan.nonce,'selected':_path(plan.selected),
            'selection_digest':plan.selected_digest,'proposal_digest':plan.proposal_digest,
            'catalogue':[_path(c) for c in candidates]}


def _tape_record(tape):
    if not isinstance(tape,CausalTape) or tape.verify() is not True:
        raise ObservationRefused('Unqualified new native causal tape')
    return {'model':tape.model,'revision':tape.revision,
            'index_ground_digest':tape.index_ground_digest,'genesis':tape.genesis,
            'observer':tape.observer.material(),'inputs':_path(tape.inputs),
            'reports':tape.reports,'labels':list(tape.labels),
            'witness_digest':tape.witness_digest}


def _restore_tape(body):
    observer=body['observer']
    tape=CausalTape(body['model'],body['revision'],body['index_ground_digest'],
                    body['genesis'],NeuralObserver(observer['coordinate'],
                    float.fromhex(observer['threshold_hex'])),
                    _unpath(body['inputs']),tuple(body['reports']),
                    tuple(body['labels']),body['witness_digest'])
    if tape.verify() is not True:raise ObservationRefused('Prepared causal receipt verification failed')
    return tape


def _atomic_once(path:Path, content:dict):
    """Immutable, content-bound record with fsync+atomic hardlink+directory fsync."""
    raw=_canonical(content)
    tmp=path.with_name('.tmp-'+uuid.uuid4().hex)
    fd=os.open(tmp,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    try:
        with os.fdopen(fd,'wb') as f:
            f.write(raw);f.flush();os.fsync(f.fileno())
        try:os.link(tmp,path)
        except FileExistsError:
            if path.read_bytes()!=raw:raise ObservationRefused('Conflicting durable record')
        if os.name != "nt":
            directory=os.open(str(path.parent),os.O_RDONLY)
            try:os.fsync(directory)
            finally:os.close(directory)
    finally:
        tmp.unlink(missing_ok=True)


def _read(path:Path):
    try:
        obj=json.loads(path.read_bytes())
        if _canonical(obj)!=path.read_bytes():raise ObservationRefused('Noncanonical durable record')
        if not isinstance(obj,dict) or 'digest' not in obj:
            raise ObservationRefused('Missing record digest')
        if obj['digest']!=digest({k:v for k,v in obj.items() if k!='digest'}):
            raise ObservationRefused('Durable record digest changed')
        return obj
    except (OSError,ValueError,TypeError) as e:
        raise ObservationRefused('Missing or invalid durable record') from e


def _seal(body):
    return {**body,'digest':digest(body)}


@dataclass(frozen=True)
class ActiveIdentity:
    request_digest:str
    epoch_id:str
    owner_id:str
    generation:int
    route_digest:str


class DurableNeuralObservation:
    """One PBG-007 native run; one IFS durable owner epoch and result identity.

    `proof-attention` must be importable from IFS's checkout. Its public
    SQLiteRouteEpochLedgerStore and ProofActionRouteEpochLedger are the only
    authority that may admit or authorize a publisher.
    """
    def __init__(self, directory):
        self.root=Path(directory).resolve()
        self.root.mkdir(parents=True,exist_ok=True)
        self.db=self.root/'authority.sqlite'

    def _ledger(self):
        from proof_action_route_epoch import ProofActionRouteEpochLedger
        from sqlite_route_epoch_store import SQLiteRouteEpochLedgerStore
        return ProofActionRouteEpochLedger(SQLiteRouteEpochLedgerStore(self.db))

    def _file(self,kind):
        return self.root/(kind+'.json')

    def issue(self,parent,candidates,plan,*,owner_id):
        """Persist selected observation before any source call or host handoff."""
        body=_plan_record(parent,candidates,plan)
        req=_seal(body)
        _atomic_once(self._file('intent'),req)
        ledger=self._ledger()
        route={'host_id':'observation-source-'+plan.proposal_digest[:16],
               'worker_id':'observation-publisher-'+plan.proposal_digest[:16],
               'request_channel':'vli.pbg008.request.'+plan.nonce[:8],
               'result_channel':'vli.pbg008.result.'+plan.nonce[:8],
               'policy':{'lease_ticks':4,'max_attempts':12,'window_size':4}}
        ledger.open_epoch(plan.proposal_digest,route,owner_id=owner_id)
        active=ledger.active_epoch
        if not ledger.record_request_admission(plan.proposal_digest,active['route_digest'],
                                               owner_id=owner_id,owner_generation=1,
                                               request_sequence=1,
                                               attempt_id=plan.proposal_digest,
                                               request_digest=req['digest']):
            raise ObservationRefused('Source request was not durably admitted')
        return self.identity(owner_id=owner_id,generation=1)

    def identity(self,*,owner_id,generation):
        intent=_read(self._file('intent'))
        ledger=self._ledger();active=ledger.active_epoch
        if active is None or (active['epoch_id'],active['owner_id'],active['owner_generation'])!=(
                intent['proposal_digest'],owner_id,generation):
            raise ObservationRefused('Stale or absent IFS owner generation')
        ledger.assert_active(intent['proposal_digest'],active['route_digest'],
                             owner_id=owner_id,owner_generation=generation)
        return ActiveIdentity(intent['digest'],intent['proposal_digest'],owner_id,
                              generation,active['route_digest'])

    def claim(self,*,new_owner,expected_owner,expected_generation):
        intent=_read(self._file('intent'))
        ledger=self._ledger()
        ledger.claim_epoch(intent['proposal_digest'],owner_id=new_owner,
                           expected_owner_id=expected_owner,
                           expected_owner_generation=expected_generation)
        return self.identity(owner_id=new_owner,generation=expected_generation+1)

    def _intent(self,parent,candidates,plan):
        intent=_read(self._file('intent'))
        expected=_seal(_plan_record(parent,candidates,plan))
        if intent!=expected:raise ObservationRefused('Changed parent, input catalogue or intention')
        return intent

    def prepare(self,parent,candidates,plan,chain_factory,*,owner_id,generation):
        """Execute only when needed, persist full verified result before publish."""
        self.identity(owner_id=owner_id,generation=generation)
        intent=self._intent(parent,candidates,plan)
        if self._file('prepared').exists():
            return self.restore_prepared(parent,candidates,plan)[1]
        child,receipt=acquire_native_run(parent,candidates,plan,chain_factory)
        newly=child.tapes[-1]
        record=_seal({'schema':SCHEMA,'intent_digest':intent['digest'],
                      'tape':_tape_record(newly),'native_receipt':receipt,
                      'child_witness':child.native_witness_digest,
                      'child_source':child.source_digest})
        _atomic_once(self._file('prepared'),record)
        return record

    def restore_prepared(self,parent,candidates,plan):
        """No numerical trained call: revalidate retained causal F64 receipts."""
        intent=self._intent(parent,candidates,plan)
        record=_read(self._file('prepared'))
        if record['intent_digest']!=intent['digest']:
            raise ObservationRefused('Prepared result belongs to another intention')
        new=_restore_tape(record['tape'])
        if (new.inputs!=plan.selected or new.observer!=parent.observer or
                (new.model,new.revision,new.index_ground_digest,new.genesis)!=
                (parent.tapes[0].model,parent.tapes[0].revision,
                 parent.tapes[0].index_ground_digest,parent.tapes[0].genesis)):
            raise ObservationRefused('Prepared result does not match pinned native source')
        child=WitnessedNeuralTraceEnsemble((*parent.tapes,new))
        receipt=record['native_receipt']
        if (record['child_witness']!=child.native_witness_digest or
                record['child_source']!=child.source_digest or
                receipt['schema']!='vli.pbg007.active-source-observation.v1' or
                receipt['source_request_digest']!=plan.proposal_digest or
                receipt['old_witness_digest']!=parent.native_witness_digest or
                receipt['new_witness_digest']!=child.native_witness_digest or
                receipt['selected_input_digest']!=plan.selected_digest or
                receipt['new_source_digest']!=child.source_digest or
                receipt['old_source_digest']!=parent.source_digest or
                receipt['selected_native_transitions']!=len(plan.selected) or
                receipt['qualified_new_positions']!=new.length or
                receipt['unselected_native_transitions']!=0):
            raise ObservationRefused('Prepared result does not match native acquisition')
        return child,record

    def publish(self,parent,candidates,plan,*,owner_id,generation):
        """Use IFS owner-generation CAS as the logical publication authority."""
        active=self.identity(owner_id=owner_id,generation=generation)
        child,prepared=self.restore_prepared(parent,candidates,plan)
        ledger=self._ledger()
        execution=ledger.record_execution_authorization(
            active.epoch_id,active.route_digest,owner_id=owner_id,
            owner_generation=generation,attempt_id=active.epoch_id,
            request_digest=active.request_digest,runner_id=RUNNER)
        if not execution:
            raise ObservationRefused('Fenced IFS authority refused publication')
        # Ledger CAS is the authorization linearization point. Result outbox
        # is immutable idempotent evidence, not a second consensus decision.
        pub=_seal({'schema':SCHEMA,'intent_digest':active.request_digest,
                   'prepared_digest':prepared['digest'],'execution_ticket':execution,
                   'child_witness':child.native_witness_digest,
                   'child_source':child.source_digest})
        _atomic_once(self._file('publication'),pub)
        return self.verify_publication(parent,candidates,plan)

    def verify_publication(self,parent,candidates,plan):
        """Audit immutable result against existing IFS append-only authority."""
        intent=self._intent(parent,candidates,plan)
        child,prepared=self.restore_prepared(parent,candidates,plan)
        pub=_read(self._file('publication'))
        ledger=self._ledger()
        authorized=[x for x in ledger.entries if x['event_type']=='execution_authorized'
                    and x['epoch_id']==plan.proposal_digest
                    and x['details']['request_digest']==intent['digest']
                    and x['details']['runner_id']==RUNNER]
        if (len(authorized)!=1 or
            pub['execution_ticket']!=authorized[0]['details']['execution_id'] or
            pub['intent_digest']!=intent['digest'] or
            pub['prepared_digest']!=prepared['digest'] or
            pub['child_witness']!=child.native_witness_digest or
            pub['child_source']!=child.source_digest):
            raise ObservationRefused('Publication lacks a single authoritative IFS event')
        return {'status':'RECOVERED_OBSERVATION_PUBLISHED',
                'execution_ticket':pub['execution_ticket'],
                'authority_owner':authorized[0]['details']['owner_id'],
                'owner_generation':authorized[0]['details']['owner_generation'],
                'native_result_digest':prepared['digest'],
                'source_digest':child.source_digest,
                'native_calls_during_restore':0,
                'source_revision':child.native_witness_digest,
                'publication_digest':pub['digest']}
