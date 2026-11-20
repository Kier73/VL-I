"""SHAPE-005: exact VL-64 pretrained ranges through an E-CPU observer-host CAS.

The live mode checks the ORIGINAL Qwen3.8-27B pinned model index and public
HTTPS byte ranges, executes a small independently checked pretrained norm,
then repeats it via the original VL-64 navigator using retained E-CPU bytes.
Neither action is a language-model forward pass, generated token, or a
proof that the entire model can be reconstructed from cached fragments.

Fixture mode uses a synthetic BF16 shard and a *real local TCP HTTP server*
with Range and Content-Range. It measures actual transport round trips but
IS NOT trained-model evidence or authenticated mutual-TLS delivery.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import re
import struct
from threading import Thread
from time import perf_counter_ns
from urllib.request import Request,urlopen

from experiments.model_fabric.scaled_model_fabric import (
    QualifiedShardNavigator,MAX_INDEX_BYTES,admit_index)
from vli.pretrained_range_memory import QualifiedPretrainedRangeMemory

MODEL="Qwen/Qwen3.8-27B"
REVISION="37561816e66b5b9153b1b64c10968c18c1e6f4a0"
SHARD="model-00001-of-00020.safetensors"
NAME="model.language_model.layers.0.input_layernorm.weight"


def _fixture():
    # Deliberately synthetic Qwen-*shaped* range; not from Qwen pretrained
    # weights. The real mode has separate original source integrity checks.
    weights=b"".join(struct.pack("<H",0x3f80+i%3) for i in range(64))
    header=json.dumps({NAME:{"dtype":"BF16","shape":[64],
                             "data_offsets":[0,len(weights)]}},
                      separators=(",",":"),sort_keys=True).encode()
    shard=struct.pack("<Q",len(header))+header+weights
    idx=json.dumps({"metadata":{"total_size":len(weights)},
                    "weight_map":{NAME:SHARD}},
                   separators=(",",":"),sort_keys=True).encode()
    return idx,shard


def _evaluate(index,source_host,approved,*,model=MODEL,revision=REVISION):
    memory=QualifiedPretrainedRangeMemory(
        model=model,revision=revision,index_raw=index,
        upstream_host=source_host,approved_upstream_hosts=set(approved))
    baseline_start=perf_counter_ns()
    first=QualifiedShardNavigator(
        model=model,revision=revision,index_raw=index,
        range_host=memory,approved_hosts=memory.approved_hosts).small_numeric_probe()
    first_ns=perf_counter_ns()-baseline_start
    metrics_after_first=memory.metrics()
    warm_start=perf_counter_ns()
    reused=QualifiedShardNavigator(
        model=model,revision=revision,index_raw=index,
        range_host=memory,approved_hosts=memory.approved_hosts).small_numeric_probe()
    warm_ns=perf_counter_ns()-warm_start
    _assure(first["status"]==reused["status"]==
            "WEIGHT_BACKED_NORMALIZATION_SLICE_VERIFIED",
            "Original trained operation incomplete")
    _assure(first["tensor"]==reused["tensor"] and
            first["dimension"]==reused["dimension"] and
            first["max_abs_error"]==reused["max_abs_error"],
            "Retained observer output differs from original")
    _assure(memory.source_requests==metrics_after_first["fresh_upstream_requests"]
            and memory.retained_requests==memory.source_requests,
            "Warm source unexpectedly invoked original network")
    snapshot=memory.checkpoint()
    start=perf_counter_ns()
    restart=QualifiedPretrainedRangeMemory.restore(
        index_raw=index,checkpoint=snapshot,
        expected_head=snapshot["checkpoint_digest"],upstream_host=None)
    recovery_ns=perf_counter_ns()-start
    rebuilt=QualifiedShardNavigator(
        model=model,revision=revision,index_raw=index,
        range_host=restart,approved_hosts=restart.approved_hosts).small_numeric_probe()
    _assure(rebuilt["status"]==first["status"] and
            rebuilt["max_abs_error"]==first["max_abs_error"],
            "Recovered original byte observation contradicts original operation")
    report={
        "schema":"vl-i.shape005-pretrained-range-observer-cost.v1",
        "model":model,"revision":revision,
        "source_index_sha256":sha256(index).hexdigest(),
        "source_identity_pinned_to_original_range_contract":True,
        "first_trained_numeric_probe":{
            "tensor":first["tensor"],"dimension":first["dimension"],
            "trained_weight_bytes":first["actual_weight_bytes"],
            "independent_max_abs_error":first["max_abs_error"],
            "original_range_witness_chain":first["range_witness_chain_digest"]},
        "warm_retained_numeric_probe":{
            "same_numerical_result":True,
            "local_range_witness_chain":reused["range_witness_chain_digest"],
            "fresh_HTTPS_or_loopback_source_requests":(
                memory.source_requests-metrics_after_first["fresh_upstream_requests"])},
        "source_observer_memory_metrics":memory.metrics(),
        "first_observation_wall_ns":first_ns,
        "warm_retained_observation_wall_ns":warm_ns,
        "receiver_checkpoint_digest":snapshot["checkpoint_digest"],
        "receiver_checkpoint_bytes":len(json.dumps(snapshot)),
        "receiver_independent_recovery_ns":recovery_ns,
        "receiver_restart_without_original_source":True,
        "scoped_capability":"trained_original_weight_norm_only_not_full_neural_model",
    }
    return report


def _assure(value,label):
    if not value:
        raise ValueError(label)


def loopback():
    index,shard=_fixture()
    transport={"calls":0,"payload_bytes":0,"http_requests":0}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,fmt,*args):
            pass
        def do_GET(self):
            transport["http_requests"]+=1
            rng=self.headers.get("Range","")
            m=re.fullmatch(r"bytes=(\d+)-(\d+)",rng)
            if self.path!="/"+SHARD or not m:
                self.send_error(400);return
            a,b=int(m[1]),int(m[2])
            if a>b or b>=len(shard):
                self.send_error(416);return
            payload=shard[a:b+1]
            self.send_response(206)
            self.send_header("Content-Type","application/octet-stream")
            self.send_header("Content-Range",f"bytes {a}-{b}/{len(shard)}")
            self.send_header("Content-Length",str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
    runner=Thread(target=server.serve_forever,daemon=True)
    runner.start()
    try:
        url=f"http://127.0.0.1:{server.server_port}/{SHARD}"
        def host(req):
            a=req["byte_begin"];b=req["byte_end_exclusive"]
            request=Request(url,headers={"Range":f"bytes={a}-{b-1}"},
                            method="GET")
            with urlopen(request,timeout=6) as response:
                _assure(response.status==206 and
                    response.headers.get("Content-Range")==
                      f"bytes {a}-{b-1}/{len(shard)}",
                    "TCP range receiver misreported original extent")
                raw=response.read(b-a+1)
            _assure(len(raw)==b-a,"Incomplete local TCP byte range")
            transport["calls"]+=1;transport["payload_bytes"]+=len(raw)
            return {**req,"status":"QUALIFIED_EXACT_SHARD_RANGE",
                    "observer_host":"untrusted-loopback-TCP-fixture",
                    "payload":raw,"sha256":sha256(raw).hexdigest()}
        report=_evaluate(index,host,{"untrusted-loopback-TCP-fixture"})
        report["mode"]="SYNTHETIC_SAFETENSORS_OVER_REAL_LOCAL_TCP"
        report["source_authentication"]="NONE_local_loopback_no_mutual_TLS"
        report["trained_original_checkpoint_observed"]=False
        report["actual_local_TCP_requests"]=transport["http_requests"]
        report["actual_local_TCP_payload_bytes"]=transport["payload_bytes"]
        return report
    finally:
        server.shutdown();server.server_close();runner.join(timeout=3)


def live():
    from experiments.model_fabric.scaled_model_lab import (
        _fetch_metadata,_url,pinned_hf_range_host)
    from vli.large_model_comparison import SOURCES
    entry=SOURCES["qwen-27b"]
    model=entry["model"];revision=entry["revision"]
    index=_fetch_metadata(
        _url(model,revision,"model.safetensors.index.json"),MAX_INDEX_BYTES)
    _assure(sha256(index).hexdigest()==entry["index_sha256"],
            "Original pretrained model index changed from existing VL-I source pin")
    report=_evaluate(
        index,pinned_hf_range_host(model,revision),
        {"pinned-huggingface-https-range"},model=model,revision=revision)
    report["mode"]="REAL_PINNED_ORIGINAL_PRETRAINED_RANGE_OVER_HTTPS"
    report["trained_original_checkpoint_observed"]=True
    report["full_original_model_not_downscaled"]=True
    report["full_checkpoint_payload_advertised_bytes"]=entry["advertised_bytes"]
    report["full_model_forward_pass_executed"]=False
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("loopback","live"),required=True)
    p.add_argument("--out",required=True)
    args=p.parse_args(argv)
    destination=Path(args.out)
    destination.parent.mkdir(parents=True,exist_ok=True)
    try:
        report=loopback() if args.mode=="loopback" else live()
    except (ValueError,OSError,TimeoutError,RuntimeError) as exc:
        report={"schema":"vl-i.shape005-pretrained-range-observer-cost.v1",
                "mode":args.mode,"status":"UNRESOLVED",
                "error_class":type(exc).__name__,
                "reason":str(exc)[:240],
                "trained_original_checkpoint_observed":False,
                "full_model_forward_pass_executed":False}
        destination.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
        print("VL_I_SHAPE005 "+json.dumps(report,sort_keys=True))
        return 1
    report["status"]="MEASURED"
    destination.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("VL_I_SHAPE005 "+json.dumps(report,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
