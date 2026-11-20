"""SHAPE-006: original trained Qwen operation result as an E-CPU state-position Ground.

'live' uses actual full-size pinned Qwen/Qwen3.8-27B pretrained norm.
'loopback' uses invented BF16 weights over REAL 127.0.0.1 TCP as a contract
control: NEVER present its results as trained Qwen evidence.
"""
from __future__ import annotations

import argparse
import json
from hashlib import sha256
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import re
from threading import Thread
from time import perf_counter_ns
from urllib.request import Request,urlopen

from experiments.model_fabric.scaled_model_fabric import MAX_INDEX_BYTES
from vli.pretrained_range_memory import QualifiedPretrainedRangeMemory
from vli.trained_result_memory import VerifiedTrainedResultGround,intention
from vli.large_model_comparison import SOURCES
from examples.pretrained_range_memory_demo import _fixture,MODEL,REVISION,SHARD

def _measure(index,host,approved,*,fixture=False):
    src=QualifiedPretrainedRangeMemory(
        model=MODEL,revision=REVISION,index_raw=index,
        upstream_host=host,approved_upstream_hosts=set(approved))
    m=VerifiedTrainedResultGround(
        src,allow_synthetic_fixture=fixture).bind_source_index(index)
    target=intention(
        prompt="Compute the trained normalized state from this intended waveform",
        frequency=0.017,phase=0.0,
        views=("text","code","tokens","json"))
    t0=perf_counter_ns()
    first=m.observe(target,expected_head=m.head)
    t1=perf_counter_ns()
    first_metrics=m.metrics()
    prior_source=src.metrics()
    reused=m.observe(target,expected_head=m.head)
    t2=perf_counter_ns()
    warm_metrics=m.metrics()
    assert first["result"]["output_sha256"]==reused["result"]["output_sha256"]
    assert first["event"]["new_trained_arithmetic_evaluations"]==1
    assert reused["event"]["new_trained_arithmetic_evaluations"]==0
    assert reused["event"]["new_external_source_requests"]==0
    assert m.trained_evaluations==1 and m.reused_trained_evaluations==1
    assert src.source_requests==prior_source["fresh_upstream_requests"]
    changed=intention(
        prompt="Compute the trained normalized state from a DIFFERENT waveform",
        frequency=0.018,phase=0.55,
        views=("text","code","tokens","json"))
    t3=perf_counter_ns()
    different=m.observe(changed,expected_head=m.head)
    t4=perf_counter_ns()
    assert different["event"]["new_trained_arithmetic_evaluations"]==1
    assert different["event"]["new_external_source_requests"]==0
    assert different["result"]["output_sha256"]!=first["result"]["output_sha256"]
    saved=m.checkpoint()
    t5=perf_counter_ns()
    restored=VerifiedTrainedResultGround.restore(
        index_raw=index,snapshot=saved,
        expected_head=saved["checkpoint_digest"])
    t6=perf_counter_ns()
    after=restored.observe(target,expected_head=restored.head)
    assert after["result"]["output_sha256"]==first["result"]["output_sha256"]
    assert after["event"]["new_trained_arithmetic_evaluations"]==0
    assert after["event"]["new_external_source_requests"]==0
    return {
        "schema":"vl-i.shape006-verified-trained-computation-cost.v1",
        "original_model":MODEL,"pinned_revision":REVISION,
        "pinned_source_index_sha256":sha256(index).hexdigest(),
        "actual_original_pretrained_model":m.original_pretrained,
        "source_index_original_scale_verified":m.original_pretrained,
        "trained_operator":"original_VL64_Qwen_zero_centered_RMSNorm",
        "operator_implementation_fingerprint":m.implementation_fingerprint,
        "scope":"one_original_partial_trained_operator_not_full_decoder",
        "first": {
            "position":first["position"]["coordinate"],
            "output_sha256":first["result"]["output_sha256"],
            "numeric_width":first["result"]["width"],
            "independent_max_abs_error":first["result"]["independent_numeric_error"],
            "source_read_requests":first["event"]["new_external_source_requests"],
            "source_bytes":first["event"]["new_original_source_bytes"],
            "trained_operator_executions":
                first["event"]["new_trained_arithmetic_evaluations"],
            "observation_wall_ns":t1-t0,
        },
        "repeated_same_intention":{
            "position":reused["position"]["coordinate"],
            "output_sha256":reused["result"]["output_sha256"],
            "source_read_requests":reused["event"]["new_external_source_requests"],
            "source_bytes":reused["event"]["new_original_source_bytes"],
            "trained_operator_executions":
                reused["event"]["new_trained_arithmetic_evaluations"],
            "observation_wall_ns":t2-t1,
        },
        "different_intention":{
            "position":different["position"]["coordinate"],
            "output_sha256":different["result"]["output_sha256"],
            "source_read_requests":different["event"]["new_external_source_requests"],
            "trained_operator_executions":
                different["event"]["new_trained_arithmetic_evaluations"],
            "observation_wall_ns":t4-t3,
        },
        "source_witness_metrics":warm_metrics["source"],
        "source_after_first_result":first_metrics["source"],
        "verified_result_metrics":m.metrics(),
        "checkpoint_digest":saved["checkpoint_digest"],
        "checkpoint_bytes":len(json.dumps(saved,sort_keys=True)),
        "source_free_recovery_and_qualification_ns":t6-t5,
        "restored_result_identical_without_source_or_trained_math":True,
        "distinct_new_causal_coordinates":(
            len({first["position"]["coordinate"],
                 reused["position"]["coordinate"],
                 different["position"]["coordinate"]})==3),
        "not_claimed":[
            "complete 27B language-model decoder",
            "next-token generation from header-only metadata",
            "full model parameter snapshot reconstruction",
            "zero cost to maintain verified result Ground",
            "statistically established end-to-end performance acceleration",
        ],
    }


def loopback():
    index,shard=_fixture()
    transport={"calls":0,"bytes":0}
    class Receiver(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            m=re.fullmatch(r"bytes=(\d+)-(\d+)",self.headers.get("Range",""))
            if self.path!="/"+SHARD or not m:
                self.send_error(400);return
            a,b=map(int,m.groups())
            if a>b or b>=len(shard):
                self.send_error(416);return
            data=shard[a:b+1]
            self.send_response(206)
            self.send_header("Content-Range",f"bytes {a}-{b}/{len(shard)}")
            self.send_header("Content-Length",str(len(data)))
            self.end_headers();self.wfile.write(data)
    server=ThreadingHTTPServer(("127.0.0.1",0),Receiver)
    thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        target=f"http://127.0.0.1:{server.server_port}/{SHARD}"
        def source(req):
            a=req["byte_begin"];b=req["byte_end_exclusive"]
            with urlopen(Request(
                target,headers={"Range":f"bytes={a}-{b-1}"},method="GET"),
                timeout=6) as res:
                if res.status!=206 or res.headers.get("Content-Range")!=(
                    f"bytes {a}-{b-1}/{len(shard)}"):
                    raise ValueError("Wrong actual TCP exact-range response")
                data=res.read(b-a+1)
            if len(data)!=b-a:raise ValueError("Short TCP original range")
            transport["calls"]+=1;transport["bytes"]+=len(data)
            return {**req,"status":"QUALIFIED_EXACT_SHARD_RANGE",
                    "observer_host":"synthetic-unauthenticated-loopback-TCP",
                    "payload":data,"sha256":sha256(data).hexdigest()}
        result=_measure(index,source,
                        {"synthetic-unauthenticated-loopback-TCP"},fixture=True)
        result.update({
            "mode":"SYNTHETIC_BF16_OVER_REAL_TCP",
            "original_pretrained_weights_used":False,
            "source_authentication":"NO_TLS_LOOPBACK_FIXTURE",
            "actual_loopback_TCP_requests":transport["calls"],
            "actual_loopback_TCP_payload_bytes":transport["bytes"],
        })
        return result
    finally:
        server.shutdown();server.server_close();thread.join(timeout=3)


def live():
    from experiments.model_fabric.scaled_model_lab import (
        _fetch_metadata,_url,pinned_hf_range_host)
    pinned=SOURCES["qwen-27b"]
    index=_fetch_metadata(
        _url(MODEL,REVISION,"model.safetensors.index.json"),MAX_INDEX_BYTES)
    if sha256(index).hexdigest()!=pinned["index_sha256"]:
        raise ValueError("Original source index differs from pinned pretrained source")
    result=_measure(index,pinned_hf_range_host(MODEL,REVISION),
                    {"pinned-huggingface-https-range"})
    result.update({
        "mode":"LIVE_ORIGINAL_QWEN_27B_TRAINED_NORM",
        "original_pretrained_weights_used":True,
        "full_checkpoint_advertised_bytes":pinned["advertised_bytes"],
        "complete_original_model_forward_pass":False,
    })
    return result


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("loopback","live"),required=True)
    parser.add_argument("--out",required=True)
    args=parser.parse_args(argv)
    output=Path(args.out);output.parent.mkdir(parents=True,exist_ok=True)
    try: result=loopback() if args.mode=="loopback" else live()
    except (OSError,TimeoutError,ValueError,RuntimeError) as e:
        result={"schema":"vl-i.shape006-verified-trained-computation-cost.v1",
                "mode":args.mode,"status":"UNRESOLVED",
                "exception_type":type(e).__name__,"reason":str(e)[:250],
                "original_pretrained_weights_used":False}
        output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
        print("VL_I_SHAPE006 "+json.dumps(result,sort_keys=True))
        return 1
    result["status"]="MEASURED"
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("VL_I_SHAPE006 "+json.dumps(result,sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
