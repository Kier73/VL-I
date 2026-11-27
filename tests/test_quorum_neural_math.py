"""PBG-009 standard-library identity tests (native E-CPU tests run separately)."""
from copy import deepcopy
from types import SimpleNamespace

from vli.quorum_neural_observation import (
    semantic_identity, _receipt_commitment, _validate_prepared,
    ObservationRefused,
)
from vli.source_hankel_ground import digest
import vli.quorum_neural_observation as module
import pytest


class Observer:
    def material(self):
        return {'coordinate':0, 'threshold_hex':'0x0.0p+0'}


def fixture(monkeypatch):
    parent=SimpleNamespace(native_witness_digest='a'*64,source_digest='b'*64,
                           observer=Observer())
    plan=SimpleNamespace(selected_digest='c'*64,nonce='d'*64)
    request={'parent_witness':'a'*64,'source':'b'*64,'selected':['test-pinned-input'],
             'nonce':'d'*64}
    monkeypatch.setattr(module,'_plan_record',lambda parent,candidates,plan:request)
    return parent,plan,request


def test_request_identity_is_independent_of_assignment_host_and_generation(monkeypatch):
    parent,plan,request=fixture(monkeypatch)
    first=semantic_identity(parent,('run',),plan)
    second=semantic_identity(parent,('run',),plan)
    assert first==second
    assert all(len(x)==64 for x in first.values())
    assert first['computational_ground_fingerprint']==parent.source_digest
    request['selected']=['different-pinned-input']
    changed=semantic_identity(parent,('run',),plan)
    assert changed['child_request_digest']!=first['child_request_digest']
    assert changed['address_digest']==first['address_digest']


def test_a_new_observer_intention_changes_address_without_material_replay(monkeypatch):
    parent,plan,_request=fixture(monkeypatch)
    original=semantic_identity(parent,('run',),plan)
    parent.observer.material=lambda:{'coordinate':1,'threshold_hex':'0x0.0p+0'}
    new=semantic_identity(parent,('run',),plan)
    assert original['address_digest']!=new['address_digest']
    assert original['parent_request_digest']==new['parent_request_digest']


def test_typed_receipt_commitment_rejects_altered_source_evidence():
    original={'claim_id':'a'*64,'tape':{'witness_digest':'b'*64},
              'native_receipt':{'source_request_digest':'c'*64},
              'intent_digest':'d'*64}
    checked=_receipt_commitment(original)
    assert len(checked)==64
    forged=deepcopy(original);forged['native_receipt']['source_request_digest']='e'*64
    assert _receipt_commitment(forged)!=checked
    forged=deepcopy(original);forged['tape']['witness_digest']='0'*64
    assert _receipt_commitment(forged)!=checked


def test_prepared_record_must_bind_exact_claim_before_native_verification(monkeypatch):
    parent,plan,request=fixture(monkeypatch)
    intent={**request,'digest':digest(request)}
    with pytest.raises(ObservationRefused,match='claim'):
        _validate_prepared(parent,('run',),plan,
            {'schema':module.SCHEMA,'claim_id':'wrong','intent_digest':intent['digest'],
             'epoch':1,'tape':{}},claim_id='0'*64,intention=intent)
    with pytest.raises(ObservationRefused,match='claim'):
        _validate_prepared(parent,('run',),plan,
            {'schema':module.SCHEMA,'claim_id':'0'*64,'intent_digest':intent['digest'],
             'epoch':0,'tape':{}},claim_id='0'*64,intention=intent)
