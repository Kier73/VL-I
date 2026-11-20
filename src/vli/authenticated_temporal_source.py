"""TIME-005A: externally signed source observations survive real process death.

The publisher (another process) owns the Ed25519 private key and a trusted
journal-head pin. The observer receives only the *public* key, source journal
and independently supplied expected head. Every new observation is verified
against the publisher's signature before it enters the TIME-001 Ground.
The observer writes a fsynced, atomically renamed checkpoint, and can be
killed *before* or *after* that checkpoint without losing a signed sample.

This is real process restart and physical clock sampling on one machine,
not distributed quorum, a continuously observed physical variable, proof of
source truth, or continuity of subjective experience. The private signing
key must never be supplied to an observer subprocess.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from time import monotonic_ns, time_ns

from .ground import _canonical, _hex64, _require, load_json
from .temporal_intelligence import TemporalIntelligenceGround, _fraction
from .live_temporal_intelligence import LiveTemporalObserver
from .evolving_environment_language import (
    EvolvingEnvironmentLanguage, verify_environment_language,
)

SIGNED_SCHEMA = "vl-i.time005.ed25519-observation.v1"
JOURNAL_SCHEMA = "vl-i.time005.source-journal.v1"
RUN_SCHEMA = "vl-i.time005.fenced-observer-recovery.v1"
MAX_SIGNED_EVENTS = 128
MAX_JOURNAL_BYTES = 2 * 1024 * 1024


def _cryptography():
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey, Ed25519PublicKey,
        )
        from cryptography.hazmat.primitives import serialization
        from cryptography.exceptions import InvalidSignature
    except ImportError as exc:
        raise RuntimeError(
            "Signed source experiment requires optional cryptography dependency"
        ) from exc
    return Ed25519PrivateKey, Ed25519PublicKey, serialization, InvalidSignature


def public_key_hex(private_key):
    """Publisher-only helper; observer never needs or receives private key."""
    _,_,serialization,_ = _cryptography()
    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()


def _signature_body(doc):
    _require(type(doc) is dict and set(doc) == {
        "schema","source","sequence","time","values",
        "captured_wall_ns","captured_monotonic_ns",
    }, "Malformed source-signed observation")
    _require(doc["schema"] == SIGNED_SCHEMA
             and type(doc["source"]) is str and 1 <= len(doc["source"]) <= 100
             and type(doc["sequence"]) is int and 0 <= doc["sequence"] < MAX_SIGNED_EVENTS
             and type(doc["time"]) is str
             and type(doc["values"]) is list and len(doc["values"]) == 1
             and type(doc["captured_wall_ns"]) is int and doc["captured_wall_ns"] > 0
             and type(doc["captured_monotonic_ns"]) is int
             and doc["captured_monotonic_ns"] > 0,
             "Invalid signed source envelope")
    _fraction(doc["time"])
    _fraction(doc["values"][0])
    return SIGNED_SCHEMA.encode("ascii")+b"\0"+_canonical(doc)


def sign_source_observation(private_key, *, source, sequence, time, value):
    """Publisher's actual clock is observed; value is a declared source claim.

    In the demonstration 'value' is a controlled mathematical sequence, NOT
    a measured physical quantity. The capture clocks are actual measurements.
    """
    Ed25519PrivateKey,_,_,_ = _cryptography()
    _require(isinstance(private_key,Ed25519PrivateKey),
             "Only publisher's Ed25519 private key may sign")
    doc={"schema":SIGNED_SCHEMA,"source":source,"sequence":sequence,
         "time":time,"values":[value],
         "captured_wall_ns":time_ns(),
         "captured_monotonic_ns":monotonic_ns()}
    signature=private_key.sign(_signature_body(doc)).hex()
    return {**doc,"signature":signature}


def verify_source_signature(packet, *, public_key, source):
    """Verify public-key authenticity before interpreting a sample."""
    _,Ed25519PublicKey,_,InvalidSignature=_cryptography()
    _require(type(packet) is dict and "signature" in packet
             and type(packet["signature"]) is str
             and len(packet["signature"]) == 128
             and type(public_key) is str and len(public_key) == 64,
             "Malformed signed source receipt")
    body={k:v for k,v in packet.items() if k!="signature"}
    _require(body["source"] == source, "Source identity substituted")
    raw=_signature_body(body)
    try:
        key=Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key))
        key.verify(bytes.fromhex(packet["signature"]),raw)
    except (InvalidSignature,ValueError,TypeError) as exc:
        raise ValueError("Source Ed25519 signature not admitted") from exc
    return True


def _packet_digest(packet):
    return sha256(b"TIME005.PACKET\0"+_canonical(packet)).hexdigest()


def _genesis(source,public_key):
    return sha256(b"TIME005.JOURNAL.GENESIS\0"+
                  source.encode("utf-8")+bytes.fromhex(public_key)).hexdigest()


def _extend(head,packet):
    return sha256(b"TIME005.JOURNAL.LINK\0"+_canonical({
        "previous":head,"packet":packet,
    })).hexdigest()


def _json_unique(raw):
    def unique(pairs):
        out={}
        for key,value in pairs:
            _require(key not in out,"Duplicate JSON key in signed source journal")
            out[key]=value
        return out
    def reject(value):
        raise ValueError("Nonfinite signed-source JSON constant")
    return json.loads(raw,object_pairs_hook=unique,parse_constant=reject)


def verified_source_journal(path, *, expected_head, expected_count,
                            source, public_key):
    """Rebuild signed ledger from trusted publisher head, not from file content.

    The head is an *external input*, not a journal-provided self-checksum.
    Authentication does not establish physical accuracy of signed values.
    """
    _require(_hex64(expected_head) and type(expected_count) is int
             and 0 <= expected_count <= MAX_SIGNED_EVENTS
             and type(source) is str and source
             and type(public_key) is str and len(public_key)==64,
             "Externally trusted source pin required")
    path=Path(path)
    data=path.read_bytes() if path.exists() else b""
    _require(len(data) <= MAX_JOURNAL_BYTES
             and (not data or data.endswith(b"\n")),
             "Incomplete, oversized or torn signed source journal")
    rows=data.splitlines()
    _require(len(rows)==expected_count, "Publisher-pinned source count mismatch")
    current=_genesis(source,public_key)
    result=[]
    last_clock=0
    last_time=None
    for i,raw in enumerate(rows):
        _require(len(raw)<=65536 and raw,"Invalid signed journal row")
        packet=_json_unique(raw)
        verify_source_signature(packet,public_key=public_key,source=source)
        _require(packet["sequence"]==i,
                 "Duplicate, reordered or missing publisher sequence")
        _require(packet["captured_monotonic_ns"] > last_clock,
                 "Source clock reading did not advance monotonically")
        t=_fraction(packet["time"])
        _require(last_time is None or t>last_time,
                 "Environmental coordinate did not advance monotonically")
        last_time=t
        last_clock=packet["captured_monotonic_ns"]
        current=_extend(current,packet)
        result.append(packet)
    _require(current==expected_head,"Externally pinned publisher chain disagrees")
    return result


def append_signed_observation(path,packet,*,expected_head,expected_count,
                              source,public_key):
    """Single-authority publisher appends and fsyncs one genuine observation.

    The caller retains the returned head in its own trust domain and passes it
    to later worker processes. No multi-writer lock, fsync-failure recovery
    protocol or quorum claim is made here.
    """
    path=Path(path)
    signed=verified_source_journal(path,expected_head=expected_head,
        expected_count=expected_count,source=source,public_key=public_key)
    verify_source_signature(packet,public_key=public_key,source=source)
    _require(len(signed)<MAX_SIGNED_EVENTS
             and packet["sequence"] == len(signed)
             and (not signed or packet["captured_monotonic_ns"] >
                  signed[-1]["captured_monotonic_ns"])
             and (not signed or _fraction(packet["time"]) >
                  _fraction(signed[-1]["time"])),
             "Publisher journal is not a strictly advancing one-writer stream")
    raw=_canonical(packet)+b"\n"
    _require(len(raw)<=65536,"Source receipt exceeds bounded size")
    with path.open("ab") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    return _extend(expected_head,packet),expected_count+1


def _persist_snapshot(path,snapshot):
    """Crash-consistent write; parent-directory fsync on POSIX."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,name=tempfile.mkstemp(prefix=".time005.",suffix=".tmp",dir=str(path.parent))
    try:
        with os.fdopen(descriptor,"wb") as f:
            f.write(_canonical(snapshot)+b"\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name,path)
        if os.name=="posix":
            fd=os.open(path.parent,os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _check_checkpoint_against_signed_source(snapshot,packets,source):
    """Reject rehashed forged checkpoints using independently signed receipts."""
    _require(type(snapshot) is dict and type(snapshot.get("events")) is list,
             "Checkpoint lacks source evidence")
    _require(len(snapshot["events"]) <= len(packets),
             "Checkpoint claims samples outside publisher's authorized stream")
    for i,event in enumerate(snapshot["events"]):
        packet=packets[i]
        _require(event["index"]==i
                 and event["time"]==packet["time"]
                 and event["values"]==packet["values"]
                 and event["source_digest"]==_packet_digest(packet)
                 and type(event["host"]) is str and event["host"],
                 "Checkpoint event cannot be derived from signed source history")
    return True


def resume_signed_observer(*, journal_path, checkpoint_path, source, public_key,
                           expected_source_head,expected_source_count,
                           environment,host, observer="vli-time",
                           hard_exit_before=None,hard_exit_after=None):
    """Replay signed observations at most once into a checkpointed TIME-001 Ground.

    A process can die before commit or after fsync/rename but before it
    acknowledges an observation. The next process independently verifies the
    original source journal, restores the durable Ground, then accepts only
    missing source indices; previously persisted events are NEVER reapplied.
    """
    packets=verified_source_journal(
        journal_path,expected_head=expected_source_head,
        expected_count=expected_source_count,
        source=source,public_key=public_key)
    fresh=TemporalIntelligenceGround(environment,observer=observer)
    snapshot_path=Path(checkpoint_path)
    old_events=0
    if snapshot_path.exists():
        saved=load_json(snapshot_path)
        _check_checkpoint_against_signed_source(saved,packets,source)
        ground=TemporalIntelligenceGround.restore(
            saved,expected_head=saved["head"],expected_identity=fresh.identity)
        old_events=len(ground.events)
    else:
        ground=fresh
    _require(ground.identity==fresh.identity,"Recovered observer identity changed")
    worker=LiveTemporalObserver(ground,host=host,
                    checkpoint_sink=lambda snap:_persist_snapshot(snapshot_path,snap))
    for index in range(old_events,len(packets)):
        packet=packets[index]
        if hard_exit_before==index:
            os._exit(71)  # no Python cleanup, process dies before durable effect
        worker.ingest({
            "schema":"vl-i.source-sample.v1",
            "sequence":index,"time":packet["time"],
            "values":packet["values"],
            "source_digest":_packet_digest(packet),
        })
        if hard_exit_after==index:
            os._exit(72)  # snapshot/fsync/rename complete, acknowledgment lost

    _require(len(ground.events)==len(packets),
             "Recovered source ledger is missing authenticated events")
    final_snapshot=ground.checkpoint()
    _check_checkpoint_against_signed_source(final_snapshot,packets,source)
    if packets:
        _require(snapshot_path.exists(),
                 "No acknowledged source history was durably saved")
    learner=EvolvingEnvironmentLanguage(ground)
    language=(learner.current(expected_head=ground.head)
              if ground.events else None)
    if language:
        verify_environment_language(
            final_snapshot,language,expected_head=ground.head,
            expected_identity=ground.identity)

    gaps=[packets[i]["captured_monotonic_ns"]-
          packets[i-1]["captured_monotonic_ns"]
          for i in range(1,len(packets))]
    body={
        "schema":RUN_SCHEMA,
        "observer_identity":ground.identity,
        "source":source,
        "authenticated_source_head":expected_source_head,
        "authenticated_observation_count":len(packets),
        "restored_source_count":old_events,
        "newly_admitted_in_this_process":worker.accepted_in_lifetime,
        "source_events_replayed_after_durable_commit":0,
        "source_continuity_verified":True,
        "result_ground_head":ground.head,
        "active_environment_family":language["family"] if language else None,
        "active_language_digest":language["capsule_digest"] if language else None,
        "recorded_wall_clock_capture_ns":[p["captured_wall_ns"] for p in packets],
        "measured_source_inter_sample_gaps_ns":gaps,
        "unobserved_intervals_between_samples":len(gaps),
        "host":host,"process_id":os.getpid(),
        "physical_continuity_across_crash":"NOT_OBSERVED",
        "physical_source_value_truth":"NOT_PROVEN_BY_SIGNATURE",
        "external_quorum_fencing":"NOT_IMPLEMENTED",
    }
    return body


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal",required=True)
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--source",required=True)
    parser.add_argument("--public-key",required=True)
    parser.add_argument("--head",required=True)
    parser.add_argument("--count",required=True,type=int)
    parser.add_argument("--environment",required=True)
    parser.add_argument("--host",required=True)
    parser.add_argument("--crash-before",type=int,default=None)
    parser.add_argument("--crash-after",type=int,default=None)
    args=parser.parse_args(argv)
    result=resume_signed_observer(
        journal_path=args.journal,checkpoint_path=args.checkpoint,
        source=args.source,public_key=args.public_key,
        expected_source_head=args.head,expected_source_count=args.count,
        environment=args.environment,host=args.host,
        hard_exit_before=args.crash_before,hard_exit_after=args.crash_after)
    print(json.dumps(result,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
