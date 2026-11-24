import hashlib
import json
from pathlib import Path
import struct

import pytest

from examples.architecture_dry_run import HeaderGuard, PayloadRequested


def test_payload_stop_precedes_provider_call():
    """For S=LE64(2)||{}||W, the guard may observe precisely S[0:10]."""
    class CountingSource:
        model,revision,authority = "m","r","test"
        def read_range(self,name,start,length):
            """S[a:a+n] is defined here only inside the metadata prefix."""
            assert start+length<=10
            return (struct.pack("<Q",2)+b"{}")[start:start+length],14
    source = HeaderGuard(CountingSource())
    source.read_range("shard",0,8)
    source.read_range("shard",8,2)
    with pytest.raises(PayloadRequested):
        source.read_range("shard",10,4)
    assert source.header_bytes == 10


def test_inventory_and_prior_witness_coverage():
    """Reconcile |I|=Σ status counts and descriptor keys=the set of prior observed sources."""
    root = Path(__file__).resolve().parents[1]/"experiments/architecture_survey"
    path = root/"2026-11-24-dry-run"
    inventory = (path/"inventory.jsonl").read_bytes()
    summary = json.loads((path/"summary.json").read_bytes())
    rows = [json.loads(line) for line in inventory.splitlines()]
    assert hashlib.sha256(inventory).hexdigest()==summary["inventory_sha256"]
    assert len({(r["catalogue"],r["family"],r["entry"]) for r in rows})==len(rows)==sum(summary["counts"].values())
    assert sum(r["result"]["status"]=="first_weight_requested" for r in rows)==3
    old = json.loads((root/"2026-11-22-public-descriptions.json").read_bytes())
    witnesses = [json.loads(line) for line in (path/"public-descriptors.jsonl").read_bytes().splitlines()]
    def key(r):
        """Identifier k=(repository,revision,descriptor name)."""
        return r["model"],r["revision"],r["descriptor"]
    assert {key(row) for row in witnesses}=={key(r["observation"]) for r in old["rows"] if r["status"]=="observed"}
