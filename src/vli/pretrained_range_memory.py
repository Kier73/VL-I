"""SHAPE-005: E-CPU content-addressed memory for *observed* pretrained ranges.

Compose VL-64's original QualifiedShardNavigator request/receipt protocol
with E-CPU's original ComputationalObjectStore and StructuralMemoryPageBacking.

An observer receives exact bytes through a caller-owned qualified range host,
then publishes each byte range in E-CPU's immutable CAS. Later identical
queries navigate the retained observation; they do not ask the original host
again. The VL-64 consumer *must* see PINNED_GLANCE_DERIVATION evidence, not
a fabricated new network receipt.

This is an observation/transport-cost experiment. Reusing trained parameter
bytes is not equivalent to reusing or bypassing a neural forward pass.
"""
from __future__ import annotations

from hashlib import sha256
from time import perf_counter_ns
from typing import Any, Callable
import json
import re

from .ground import _digest, _hex64, _require

SCHEMA="vl-i.pretrained-range-observer-memory.v1"
SNAPSHOT="vl-i.pretrained-range-observer-checkpoint.v1"
LOCAL_HOST="ecpu-qualified-content-addressed-range-memory"
MAX_BYTES=16*1024*1024
MAX_ENTRIES=4096
_SHA=re.compile(r"[0-9a-f]{64}\Z")


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),
                      ensure_ascii=False,allow_nan=False).encode("utf-8")


def _source_request(req):
    """Typed exact VL-64 request. Caller must still qualify the range data."""
    from experiments.model_fabric.scaled_model_fabric import MAX_RANGE
    _require(type(req) is dict and set(req)=={
        "model","revision","index_ground_digest","shard",
        "byte_begin","byte_end_exclusive","purpose"},
        "Missing original pretrained range request fields")
    begin=req["byte_begin"];end=req["byte_end_exclusive"]
    _require(type(begin) is int and type(end) is int and
             0<=begin<end and end-begin<=MAX_RANGE and
             req["purpose"] in ("tensor","header"),
             "Unadmitted original model shard offset/purpose")
    return req


class QualifiedPretrainedRangeMemory:
    """Source-bound receiver observer with E-CPU durable byte-object storage.

    The sole source of original bytes is the supplied range_host. The
    caller's explicitly approved upstream_host_names are *trust inputs*;
    neither a checksum nor this cache authenticates a remote host.
    """

    def __init__(self, *, model:str, revision:str, index_raw:bytes,
                 upstream_host:Callable[[dict],dict]|None,
                 approved_upstream_hosts:set[str],
                 max_resident_bytes:int=MAX_BYTES):
        from experiments.model_fabric.scaled_model_fabric import admit_index
        from mailcpu.structural_memory_pages import StructuralMemoryPageBacking
        from mailcpu.computational_objects import ComputationalObjectStore

        _require(type(index_raw) is bytes and index_raw
                 and callable(upstream_host) or upstream_host is None,
                 "Missing original-shard metadata or host interface")
        index,weights=admit_index(index_raw,model=model,revision=revision)
        _require(type(approved_upstream_hosts) is set and
                 approved_upstream_hosts and
                 all(type(x) is str and x and x!=LOCAL_HOST
                     for x in approved_upstream_hosts) and
                 type(max_resident_bytes) is int and
                 1<=max_resident_bytes<=MAX_BYTES,
                 "No approved, distinct original range host or bounded memory")
        self.model=model;self.revision=revision
        self.index_ground_digest=index["ground_digest"]
        self.index_sha256=index["index_sha256"]
        self.allowed_shards=frozenset(weights.values())
        self.upstream_host=upstream_host
        self.approved_upstream_hosts=frozenset(approved_upstream_hosts)
        self.max_resident_bytes=max_resident_bytes
        self._backing=StructuralMemoryPageBacking()
        self._store=ComputationalObjectStore(
            self._backing,chunk_size=4096,max_object_bytes=65536)
        self._records={}
        self._resident=0
        self.source_requests=0
        self.source_bytes=0
        self.retained_requests=0
        self.retained_bytes=0
        self.cas_read_ns=0
        self.cas_store_ns=0
        self.upstream_ns=0

    def _request(self,req):
        _source_request(req)
        _require(req["model"]==self.model
                 and req["revision"]==self.revision
                 and req["index_ground_digest"]==self.index_ground_digest
                 and req["shard"] in self.allowed_shards,
                 "Request cannot cross model/revision/shard source identity")
        return dict(req)

    @staticmethod
    def _key(req):
        return (req["model"],req["revision"],req["index_ground_digest"],
                req["shard"],req["byte_begin"],req["byte_end_exclusive"],
                req["purpose"])

    def _qualified_upstream(self,req):
        _require(callable(self.upstream_host),
                 "No original source connected; no new trained bytes available")
        start=perf_counter_ns()
        reply=self.upstream_host(dict(req))
        self.upstream_ns+=perf_counter_ns()-start
        _require(type(reply) is dict and
                 all(reply.get(k)==v for k,v in req.items()) and
                 reply.get("status")=="QUALIFIED_EXACT_SHARD_RANGE"
                 and reply.get("observer_host") in self.approved_upstream_hosts
                 and reply.get("cache_disposition") is None,
                 "Original observer-host receipt unqualified or already cached")
        raw=reply.get("payload")
        _require(type(raw) is bytes and
                 len(raw)==req["byte_end_exclusive"]-req["byte_begin"]
                 and reply.get("sha256")==sha256(raw).hexdigest(),
                 "Original source returned altered, partial or unverified bytes")
        return raw,reply["observer_host"]

    def __call__(self,request):
        req=self._request(request)
        key=self._key(req)
        if key not in self._records:
            _require(len(self._records)<MAX_ENTRIES
                     and self._resident+(req["byte_end_exclusive"]-
                                          req["byte_begin"])<=self.max_resident_bytes,
                     "Receiver object-store memory budget exhausted")
            data,original_host=self._qualified_upstream(req)
            start=perf_counter_ns()
            manifest=self._store.put_bytes(data)
            _require(self._store.verify(manifest),
                     "E-CPU object store did not verify original observed bytes")
            self.cas_store_ns+=perf_counter_ns()-start
            witness={
                "original_request":req,
                "origin_observer_host":original_host,
                "observed_payload_sha256":sha256(data).hexdigest(),
                "observed_payload_bytes":len(data),
                "object_manifest_digest":manifest.manifest_digest,
                "observation_kind":"fresh_exact_source_bytes_not_entire_checkpoint",
            }
            record={"request":req,"manifest":manifest.to_dict(),
                    "original_witness":witness,
                    "original_witness_digest":_digest(SCHEMA,witness)}
            self._records[key]=record
            self._resident+=len(data)
            self.source_requests+=1;self.source_bytes+=len(data)
            return {**req,"status":"QUALIFIED_EXACT_SHARD_RANGE",
                    "observer_host":original_host,
                    "payload":data,"sha256":witness["observed_payload_sha256"],
                    "cache_disposition":"FRESH_HOST_OBSERVATION"}
        record=self._records[key]
        from mailcpu.computational_objects import ComputationalObjectManifest
        manifest=ComputationalObjectManifest.from_dict(record["manifest"])
        t0=perf_counter_ns()
        raw=self._store.read_bytes(manifest,max_bytes=65536)
        self.cas_read_ns+=perf_counter_ns()-t0
        witness=record["original_witness"]
        _require(sha256(raw).hexdigest()==witness["observed_payload_sha256"]
                 and manifest.manifest_digest==witness["object_manifest_digest"]
                 and witness["original_request"]==req
                 and _digest(SCHEMA,witness)==record["original_witness_digest"],
                 "Retained observation lost its original source/byte lineage")
        self.retained_requests+=1;self.retained_bytes+=len(raw)
        return {**req,"status":"QUALIFIED_EXACT_SHARD_RANGE",
                "observer_host":LOCAL_HOST,
                "payload":raw,"sha256":witness["observed_payload_sha256"],
                "cache_disposition":"PINNED_GLANCE_DERIVATION",
                "glance_ground_head":self._range_head(record),
                "original_witnesses":[witness]}

    @staticmethod
    def _range_head(record):
        return _digest(SCHEMA,{
            "original_witness_digest":record["original_witness_digest"],
            "manifest_digest":record["original_witness"]["object_manifest_digest"],
            "source_range":record["request"]})

    @property
    def approved_hosts(self):
        """For the original VL-64 navigator. Retained reads are local, not HTTP."""
        return set(self.approved_upstream_hosts)|{LOCAL_HOST}

    def requalify(self,request):
        """Independently ask original host if bytes are *still* the same.

        Does not substitute new source bytes into an already-qualified Ground;
        on mismatch it refuses and requires a fresh source epoch and lineage.
        """
        req=self._request(request)
        record=self._records.get(self._key(req))
        _require(record is not None,"Cannot requalify an unobserved range")
        data,host=self._qualified_upstream(req)
        _require(host==record["original_witness"]["origin_observer_host"]
                 and sha256(data).hexdigest()==
                     record["original_witness"]["observed_payload_sha256"],
                 "Fresh source disagrees with retained original observation")
        self.source_requests+=1;self.source_bytes+=len(data)
        return True

    def checkpoint(self):
        _require(len(self._records)<=MAX_ENTRIES
                 and self._resident<=self.max_resident_bytes,
                 "Retained source exceeds declared snapshot")
        records=[self._records[key] for key in sorted(self._records)]
        body={
            "schema":SNAPSHOT,"model":self.model,"revision":self.revision,
            "index_ground_digest":self.index_ground_digest,
            "index_sha256":self.index_sha256,
            "allowed_shards":sorted(self.allowed_shards),
            "approved_upstream_hosts":sorted(self.approved_upstream_hosts),
            "max_resident_bytes":self.max_resident_bytes,
            "records":records,"backing":self._backing.snapshot(),
            "scope":"original_observed_bytes_retained_not_current_remote_attestation",
        }
        return {**body,"checkpoint_digest":_digest(SNAPSHOT,body)}

    @classmethod
    def restore(cls,*,index_raw:bytes,checkpoint:dict,expected_head:str,
                upstream_host=None):
        from mailcpu.computational_objects import (
            ComputationalObjectManifest,ComputationalObjectStore)
        from mailcpu.structural_memory_pages import StructuralMemoryPageBacking

        _require(type(checkpoint) is dict and _hex64(expected_head)
                 and checkpoint.get("schema")==SNAPSHOT
                 and checkpoint.get("checkpoint_digest")==expected_head
                 and _digest(SNAPSHOT,{
                     k:v for k,v in checkpoint.items()
                     if k!="checkpoint_digest"})==expected_head,
                 "Missing independent checkpoint pin or altered observer range memory")
        obj=cls(model=checkpoint["model"],revision=checkpoint["revision"],
                index_raw=index_raw,upstream_host=upstream_host,
                approved_upstream_hosts=set(checkpoint["approved_upstream_hosts"]),
                max_resident_bytes=checkpoint["max_resident_bytes"])
        _require(obj.index_sha256==checkpoint["index_sha256"] and
                 obj.index_ground_digest==checkpoint["index_ground_digest"] and
                 sorted(obj.allowed_shards)==checkpoint["allowed_shards"] and
                 checkpoint["scope"]==
                 "original_observed_bytes_retained_not_current_remote_attestation" and
                 type(checkpoint["records"]) is list and
                 len(checkpoint["records"])<=MAX_ENTRIES,
                 "Original model index differs from persisted source contract")
        obj._backing=StructuralMemoryPageBacking.from_snapshot(
            checkpoint["backing"])
        obj._store=ComputationalObjectStore(
            obj._backing,chunk_size=4096,max_object_bytes=65536)
        for record in checkpoint["records"]:
            _require(type(record) is dict and
                     set(record)=={"request","manifest","original_witness",
                                  "original_witness_digest"},
                     "Malformed original range witness")
            req=obj._request(record["request"])
            key=obj._key(req)
            _require(key not in obj._records,
                     "Duplicate retained original source range")
            witness=record["original_witness"]
            manifest=ComputationalObjectManifest.from_dict(record["manifest"])
            _require(type(witness) is dict
                     and witness.get("original_request")==req
                     and witness.get("origin_observer_host") in
                         obj.approved_upstream_hosts
                     and witness.get("observed_payload_bytes")==
                         req["byte_end_exclusive"]-req["byte_begin"]
                     and witness.get("object_manifest_digest")==
                         manifest.manifest_digest
                     and record["original_witness_digest"]==_digest(SCHEMA,witness)
                     and manifest.size_bytes==witness["observed_payload_bytes"],
                     "Restored range manifest lacks qualified original source binding")
            raw=obj._store.read_bytes(manifest,max_bytes=65536)
            _require(sha256(raw).hexdigest()==witness["observed_payload_sha256"],
                     "Restored original pretrained observation bytes failed E-CPU CAS")
            obj._records[key]=record
            obj._resident+=len(raw)
            _require(obj._resident<=obj.max_resident_bytes,
                     "Persisted range memory exceeds admitted budget")
        return obj

    def metrics(self):
        return {
            "source_model":self.model,"source_revision":self.revision,
            "index_ground_digest":self.index_ground_digest,
            "retained_range_count":len(self._records),
            "retained_object_bytes":self._resident,
            "fresh_upstream_requests":self.source_requests,
            "fresh_upstream_payload_bytes":self.source_bytes,
            "reused_local_range_reads":self.retained_requests,
            "reused_local_bytes":self.retained_bytes,
            "avoided_upstream_requests":self.retained_requests,
            "avoided_upstream_payload_bytes":self.retained_bytes,
            "upstream_call_ns":self.upstream_ns,
            "ECPU_CAS_store_ns":self.cas_store_ns,
            "ECPU_CAS_read_ns":self.cas_read_ns,
            "scope":"actual_original_observed_ranges_not_full_neural_inference",
        }
