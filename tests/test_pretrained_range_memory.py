"""SHAPE-005: real VL-64 source-range protocol + E-CPU immutable page objects.

Offline tests use a *fabricated local safetensors fixture*, NOT pretrained
weights. Actual original checkpoint ranges are tested by separate opt-in live
workflow on the full-size pinned checkpoint.
"""
from __future__ import annotations

import copy
import json
import struct
import unittest
from hashlib import sha256

from experiments.model_fabric.scaled_model_fabric import QualifiedShardNavigator,admit_index
from vli.pretrained_range_memory import QualifiedPretrainedRangeMemory,LOCAL_HOST
from vli.ground import _digest

MODEL="Qwen/Qwen3.8-27B"
REVISION="37561816e66b5b9153b1b64c10968c18c1e6f4a0"
SHARD="model-00001-of-00020.safetensors"
NAME="model.language_model.layers.0.input_layernorm.weight"


def fixture():
    index_raw=json.dumps({
        "metadata":{"total_size":4096},
        "weight_map":{NAME:SHARD},
    },sort_keys=True,separators=(",",":")).encode()
    values=[struct.pack("<H",0x3f80+i%3) for i in range(64)]
    payload=b"".join(values)
    header=json.dumps({NAME:{"dtype":"BF16","shape":[64],
                             "data_offsets":[0,len(payload)]}},
                      sort_keys=True,separators=(",",":")).encode()
    fake_shard=struct.pack("<Q",len(header))+header+payload
    ground,_=admit_index(index_raw,model=MODEL,revision=REVISION)
    return index_raw,fake_shard,ground


class FakeQualifiedRangeHost:
    """Deterministic synthetic *range endpoint*, with actual validated receipts."""

    def __init__(self,data,delay=0):
        self.data=data;self.calls=0;self.bytes_sent=0;self.corrupt=False
        self.delay=delay

    def __call__(self,request):
        from time import sleep
        self.calls+=1
        begin=request["byte_begin"];end=request["byte_end_exclusive"]
        if self.delay:sleep(self.delay)
        data=self.data[begin:end]
        self.bytes_sent+=len(data)
        if self.corrupt:
            data=(b"X" if data else b"") + data[1:]
        return {**request,"status":"QUALIFIED_EXACT_SHARD_RANGE",
                "observer_host":"synthetic-qualified-range-host",
                "payload":data,"sha256":sha256(data).hexdigest()}


def make_memory(src,index_raw,*,budget=400000):
    return QualifiedPretrainedRangeMemory(
        model=MODEL,revision=REVISION,index_raw=index_raw,
        upstream_host=src,
        approved_upstream_hosts={"synthetic-qualified-range-host"},
        max_resident_bytes=budget)


def navigator(index_raw,host):
    return QualifiedShardNavigator(
        model=MODEL,revision=REVISION,index_raw=index_raw,
        range_host=host,approved_hosts=host.approved_hosts)


class TestExactPretrainedRangeMemory(unittest.TestCase):
    def setUp(self):
        self.index_raw,self.fake_shard,self.index=fixture()
        self.host=FakeQualifiedRangeHost(self.fake_shard)
        self.memory=make_memory(self.host,self.index_raw)

    def test_actual_vl64_range_protocol_qualifies_original_and_locally_retained_receipts(self):
        memory=self.memory
        first=navigator(self.index_raw,memory).small_numeric_probe()
        self.assertEqual(first["status"],"WEIGHT_BACKED_NORMALIZATION_SLICE_VERIFIED")
        self.assertEqual(first["dimension"],64)
        calls=self.host.calls
        original_bytes=self.host.bytes_sent
        second=navigator(self.index_raw,memory).small_numeric_probe()
        self.assertEqual(self.host.calls,calls)
        self.assertEqual(self.host.bytes_sent,original_bytes)
        self.assertEqual(second["status"],first["status"])
        self.assertEqual(second["max_abs_error"],first["max_abs_error"])
        self.assertNotEqual(first["range_witness_chain_digest"],
                            second["range_witness_chain_digest"])
        stats=memory.metrics()
        self.assertEqual(stats["fresh_upstream_requests"],calls)
        self.assertEqual(stats["reused_local_range_reads"],calls)
        self.assertEqual(stats["fresh_upstream_payload_bytes"],original_bytes)
        self.assertEqual(stats["avoided_upstream_payload_bytes"],original_bytes)
        self.assertEqual(stats["retained_range_count"],calls)

    def test_returned_retained_range_declares_original_source_and_local_observer(self):
        nav=navigator(self.index_raw,self.memory)
        req={"model":MODEL,"revision":REVISION,
             "index_ground_digest":self.index["ground_digest"],
             "shard":SHARD,"byte_begin":0,"byte_end_exclusive":8,
             "purpose":"header"}
        fresh=self.memory(req)
        self.assertEqual(fresh["cache_disposition"],"FRESH_HOST_OBSERVATION")
        self.assertEqual(fresh["observer_host"],"synthetic-qualified-range-host")
        warm=self.memory(req)
        self.assertEqual(warm["observer_host"],LOCAL_HOST)
        self.assertEqual(warm["cache_disposition"],"PINNED_GLANCE_DERIVATION")
        self.assertEqual(warm["payload"],fresh["payload"])
        self.assertEqual(warm["original_witnesses"][0]["original_request"],req)
        self.assertEqual(warm["original_witnesses"][0]["origin_observer_host"],
                         "synthetic-qualified-range-host")
        self.assertEqual(len(warm["glance_ground_head"]),64)
        self.assertIn(LOCAL_HOST,nav.approved)

    def test_checkpoint_recovers_without_remote_source_but_cannot_invent_a_new_range(self):
        memory=self.memory
        result=navigator(self.index_raw,memory).small_numeric_probe()
        saved=memory.checkpoint()
        restored=QualifiedPretrainedRangeMemory.restore(
            index_raw=self.index_raw,checkpoint=saved,
            expected_head=saved["checkpoint_digest"],upstream_host=None)
        self.assertEqual(restored.checkpoint(),saved)
        reused=navigator(self.index_raw,restored).small_numeric_probe()
        self.assertEqual(reused["status"],result["status"])
        self.assertEqual(self.host.calls,3)
        req={"model":MODEL,"revision":REVISION,
             "index_ground_digest":self.index["ground_digest"],
             "shard":SHARD,"byte_begin":3,"byte_end_exclusive":7,
             "purpose":"tensor"}
        with self.assertRaises(ValueError):
            restored(req)

    def test_original_source_requalification_exact_match_and_drift_refusal(self):
        self.memory(navigator(self.index_raw,self.memory).index and
            {"model":MODEL,"revision":REVISION,
             "index_ground_digest":self.index["ground_digest"],
             "shard":SHARD,"byte_begin":0,"byte_end_exclusive":8,
             "purpose":"header"})
        req={"model":MODEL,"revision":REVISION,
             "index_ground_digest":self.index["ground_digest"],
             "shard":SHARD,"byte_begin":0,"byte_end_exclusive":8,
             "purpose":"header"}
        self.assertTrue(self.memory.requalify(req))
        before=self.memory.metrics()["fresh_upstream_requests"]
        self.host.corrupt=True
        with self.assertRaises(ValueError):
            self.memory.requalify(req)
        self.assertEqual(self.memory.metrics()["fresh_upstream_requests"],before)
        # Previously retained source bytes remain the original observation,
        # not a fabricated fresh observation of a now-changed host.
        self.assertEqual(self.memory(req)["payload"],self.fake_shard[:8])

    def test_cross_model_revision_and_shard_injections_refused(self):
        _,_,index=self.index_raw,self.fake_shard,self.index
        req={"model":MODEL,"revision":REVISION,
             "index_ground_digest":index["ground_digest"],
             "shard":SHARD,"byte_begin":0,"byte_end_exclusive":8,
             "purpose":"header"}
        baseline=self.host.calls
        for change in ({"model":"other/Model"},
                       {"revision":"0"*40},
                       {"index_ground_digest":"0"*64},
                       {"shard":"model-OTHER.safetensors"},
                       {"byte_begin":0,"byte_end_exclusive":65537},
                       {"purpose":"host-command"}):
            changed={**req,**change}
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    self.memory(changed)
        self.assertEqual(self.host.calls,baseline)

    def test_overbudget_refuses_before_source_request_and_does_not_publish(self):
        low=make_memory(self.host,self.index_raw,budget=7)
        req={"model":MODEL,"revision":REVISION,
             "index_ground_digest":self.index["ground_digest"],
             "shard":SHARD,"byte_begin":0,"byte_end_exclusive":8,
             "purpose":"header"}
        with self.assertRaises(ValueError):
            low(req)
        self.assertEqual(low.metrics()["retained_range_count"],0)
        self.assertEqual(self.host.calls,0)

    def test_mutated_e_c_p_u_page_fails_even_when_checkpoint_digest_recomputed(self):
        self.memory(navigator(self.index_raw,self.memory).index and
            {"model":MODEL,"revision":REVISION,
             "index_ground_digest":self.index["ground_digest"],
             "shard":SHARD,"byte_begin":0,"byte_end_exclusive":8,
             "purpose":"header"})
        saved=self.memory.checkpoint()
        changed=copy.deepcopy(saved)
        digest=next(iter(changed["backing"]["local_pages"]))
        changed["backing"]["local_pages"][digest]="00"
        changed["checkpoint_digest"]=_digest(
            changed["schema"],{k:v for k,v in changed.items()
                               if k!="checkpoint_digest"})
        with self.assertRaises(ValueError):
            QualifiedPretrainedRangeMemory.restore(
                index_raw=self.index_raw,checkpoint=changed,
                expected_head=changed["checkpoint_digest"])

    def test_mutated_original_observation_provenance_refused(self):
        self.memory(navigator(self.index_raw,self.memory).index and
            {"model":MODEL,"revision":REVISION,
             "index_ground_digest":self.index["ground_digest"],
             "shard":SHARD,"byte_begin":0,"byte_end_exclusive":8,
             "purpose":"header"})
        saved=self.memory.checkpoint()
        changed=copy.deepcopy(saved)
        changed["records"][0]["original_witness"]["origin_observer_host"]="evil-source"
        changed["checkpoint_digest"]=_digest(
            changed["schema"],{k:v for k,v in changed.items()
                               if k!="checkpoint_digest"})
        with self.assertRaises(ValueError):
            QualifiedPretrainedRangeMemory.restore(
                index_raw=self.index_raw,checkpoint=changed,
                expected_head=changed["checkpoint_digest"])

    def test_external_pin_prevents_rebinding_witnesses(self):
        navigator(self.index_raw,self.memory).small_numeric_probe()
        saved=self.memory.checkpoint()
        altered=copy.deepcopy(saved)
        altered["records"][0]["request"]["byte_begin"]=1
        with self.assertRaises(ValueError):
            QualifiedPretrainedRangeMemory.restore(
                index_raw=self.index_raw,checkpoint=altered,
                expected_head=saved["checkpoint_digest"])

    def test_returned_witness_and_saved_checkpoint_are_not_mutable_cache_authorities(self):
        req={"model":MODEL,"revision":REVISION,
             "index_ground_digest":self.index["ground_digest"],
             "shard":SHARD,"byte_begin":0,"byte_end_exclusive":8,
             "purpose":"header"}
        self.memory(req)
        first=self.memory(req)
        snapshot=self.memory.checkpoint()
        head=snapshot["checkpoint_digest"]
        first["original_witnesses"][0]["origin_observer_host"]="attacker"
        snapshot["records"][0]["original_witness"]["origin_observer_host"]="attacker"
        self.assertEqual(
            self.memory(req)["original_witnesses"][0]["origin_observer_host"],
            "synthetic-qualified-range-host")
        self.assertEqual(self.memory.checkpoint()["checkpoint_digest"],head)
        restored=QualifiedPretrainedRangeMemory.restore(
            index_raw=self.index_raw,
            checkpoint=self.memory.checkpoint(),
            expected_head=head,
            upstream_host=None)
        another=restored.checkpoint()
        another["records"][0]["original_witness"]["observed_payload_sha256"]="0"*64
        self.assertEqual(
            restored(req)["sha256"],sha256(self.fake_shard[:8]).hexdigest())

    def test_no_model_execution_inferred_from_range_memory_alone(self):
        self.assertEqual(self.memory.metrics()["fresh_upstream_requests"],0)
        self.assertEqual(self.memory.metrics()["retained_range_count"],0)
        self.assertFalse(hasattr(self.memory,"generate_token"))


if __name__=="__main__":
    unittest.main()
