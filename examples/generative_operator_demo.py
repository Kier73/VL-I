"""SHAPE-007: original-source-trained generative sufficient-statistic Ground.

The PROPOSED output is formed from the law BEFORE asking the original VL-64
trained native reference to assess held-out accuracy. No input-specific
output table is cached. Source-free restart must generate an unseen new
intention, not just read a saved numerical answer.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import json
import re
from threading import Thread
from time import perf_counter_ns
from urllib.request import Request,urlopen

from experiments.model_fabric.scaled_model_fabric import MAX_INDEX_BYTES
from examples.pretrained_range_memory_demo import _fixture,MODEL,REVISION,SHARD
from vli.pretrained_range_memory import QualifiedPretrainedRangeMemory
from vli.generative_operator_ground import GenerativeTrainedOperatorGround
from vli.trained_result_memory import intention
from vli.large_model_comparison import SOURCES


def _experiment(index,host,approved,*,fixture):
    source=QualifiedPretrainedRangeMemory(
        model=MODEL,revision=REVISION,index_raw=index,
        upstream_host=host,approved_upstream_hosts=set(approved))
    t0=perf_counter_ns()
    ground=GenerativeTrainedOperatorGround(
        source,index,allow_synthetic_fixture=fixture)
    t1=perf_counter_ns()
    width=ground.width
    sparse=(0,1,2,width//4,width//2,3*width//4,width-2,width-1)
    holdouts=[
        intention(waveform="sin",frequency=.00007,amplitude=.57,phase=.9),
        intention(waveform="cos",frequency=.0142,amplitude=.43,phase=-1.8),
        intention(waveform="sin",frequency=.133,amplitude=.24,phase=.15),
        intention(waveform="cos",frequency=.8,amplitude=.31,phase=1.2),
    ]
    transcript=[]
    source_requests_before=source.source_requests
    for spec in holdouts:
        before=perf_counter_ns()
        proposed=ground.observe(spec,sparse,expected_head=ground.head)
        proposed_end=perf_counter_ns()
        # The reference is consulted only AFTER the candidate new answer
        # and its state position have been fixed.
        observed_start=perf_counter_ns()
        expected=ground.native_reference(spec,sparse)
        observed_end=perf_counter_ns()
        err=max(abs(x-y) for x,y in zip(proposed["values"],expected))
        if err>3e-9:raise ValueError("Held-out trained source disproved generative Ground")
        transcript.append({
            "intent_digest":spec["intent_digest"],
            "unseen_witness":spec["intent_digest"] not in
                 ground.witness["witness_intentions"],
            "position":proposed["position"]["coordinate"],
            "requested_indices":list(sparse),
            "output_sha256":proposed["event"]["result_sha256"],
            "generated_without_original_native_math":True,
            "prediction_ns":proposed_end-before,
            "verified_native_reference_ns":observed_end-observed_start,
            "max_abs_heldout_error":err,
            "source_requests_for_new_output":proposed["event"]["source_range_requests"],
            "new_trained_eval_during_prediction":
                proposed["event"]["original_native_operator_evaluations"],
        })
    checkpoint=ground.checkpoint()
    old_source=source.source_requests
    before=perf_counter_ns()
    restored=GenerativeTrainedOperatorGround.restore(
        index,checkpoint,expected_head=checkpoint["checkpoint_digest"])
    restored_ns=perf_counter_ns()-before
    never_requested=intention(waveform="sin",frequency=.0917,
                              amplitude=.28,phase=1.1)
    predicted=restored.observe(never_requested,sparse,
                              expected_head=restored.head)
    independent=ground.native_reference(never_requested,sparse)
    novel_err=max(abs(a-b) for a,b in zip(predicted["values"],independent))
    if novel_err>3e-9:raise ValueError("Restored generative Ground failed unseen intention")
    if source.source_requests!=old_source:
        raise ValueError("Original source was recontacted for heldout generative state")
    negative_spec=intention(waveform="cos",frequency=.0193,
                            amplitude=.56,phase=2.3)
    begin=perf_counter_ns()
    full=restored.observe(negative_spec,tuple(range(width)),
                          expected_head=restored.head)
    full_time=perf_counter_ns()-begin
    start_ref=perf_counter_ns()
    reference=ground.native_reference(negative_spec,tuple(range(width)))
    full_native_ns=perf_counter_ns()-start_ref
    full_error=max(abs(a-b) for a,b in zip(full["values"],reference))
    if full_error>3e-9:raise ValueError("Full output failed original trained reference")
    return {
        "schema":"vl-i.shape007-trained-generative-sufficient-statistic.v1",
        "source_model":source.model,
        "source_revision":source.revision,
        "source_index_sha256":source.index_sha256,
        "original_full_pretrained_checkpoint":ground.original_pretrained,
        "full_original_trained_norm_width":width,
        "source_root":ground.source_root,
        "learned_law":ground.chosen_law,
        "candidate_law_max_training_errors":
             ground.witness["candidate_max_abs_errors"],
        "source_qualification_native_witness_evaluations":
             ground.training_original_calls,
        "source_qualification_native_ns":ground.training_ns,
        "source_plus_ground_bootstrap_ns":t1-t0,
        "original_source_requests_after_bootstrap":source_requests_before,
        "original_bytes_transferred":source.source_bytes,
        "heldout_samples":transcript,
        "heldout_max_error":max(x["max_abs_heldout_error"] for x in transcript),
        "generated_heldout_coordinates":len(transcript)*len(sparse),
        "source_free_restore_verified":True,
        "source_free_restore_ns":restored_ns,
        "source_free_reconstructed_new_unseen_intention":{
            "position":predicted["position"]["coordinate"],
            "new_input_intention":never_requested["intent_digest"],
            "max_abs_original_reference_error":novel_err,
            "source_requests":predicted["event"]["source_range_requests"],
            "original_native_evaluations":
                predicted["event"]["original_native_operator_evaluations"],
        },
        "negative_full_output_control":{
            "coordinates_requested":width,
            "generated_ns":full_time,
            "original_full_native_reference_ns":full_native_ns,
            "max_abs_error":full_error,
            "generated_slower_than_native":full_time>full_native_ns,
        },
        "ECPU_coefficient_storage":{
            "manifest":ground.gain_manifest["manifest_digest"],
            "bytes":ground.width*8,
            "source_trained_weight_sha256":ground.original_weight_sha256,
        },
        "hard_limits":[
            "finite sin/cos analytic hidden-state input family only",
            "not trained token embeddings or arbitrary language-model state",
            "law selected from supplied symbolic candidates, not autonomously invented",
            "finitely witnessed; mathematical formula checked numerically on held-outs",
            "original pretrained normalization only; not complete Qwen attention or decoder",
            "full output still requires proportional output work; sparse observer can be cheaper",
            "original-source acquisition, law training and ECPU checkpoint have real costs",
            "original HTTPS host trust distinct from SHA-256 integrity pins",
        ]
    }


def loopback():
    idx,shard=_fixture()
    counters={"tcp_calls":0,"tcp_bytes":0}
    class Receiver(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            match=re.fullmatch(r"bytes=(\d+)-(\d+)",
                               self.headers.get("Range",""))
            if self.path!="/"+SHARD or not match:
                self.send_error(400);return
            a,b=map(int,match.groups())
            if a>b or b>=len(shard):
                self.send_error(416);return
            payload=shard[a:b+1]
            self.send_response(206)
            self.send_header("Content-Range",f"bytes {a}-{b}/{len(shard)}")
            self.send_header("Content-Length",str(len(payload)))
            self.end_headers();self.wfile.write(payload)
    server=ThreadingHTTPServer(("127.0.0.1",0),Receiver)
    thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        url=f"http://127.0.0.1:{server.server_port}/{SHARD}"
        def host(req):
            a=req["byte_begin"];b=req["byte_end_exclusive"]
            with urlopen(Request(
                url,headers={"Range":f"bytes={a}-{b-1}"},method="GET"),
                timeout=8) as response:
                if response.status!=206 or response.headers.get("Content-Range")!=(
                    f"bytes {a}-{b-1}/{len(shard)}"):
                    raise ValueError("TCP source returned wrong original range")
                payload=response.read(b-a+1)
            if len(payload)!=b-a:raise ValueError("Unqualified TCP payload")
            counters["tcp_calls"]+=1;counters["tcp_bytes"]+=len(payload)
            return {**req,"status":"QUALIFIED_EXACT_SHARD_RANGE",
                    "observer_host":"synthetic-untrusted-loopback-TCP",
                    "payload":payload,"sha256":sha256(payload).hexdigest()}
        result=_experiment(idx,host,{"synthetic-untrusted-loopback-TCP"},
                           fixture=True)
        result["mode"]="SYNTHETIC_ORIGINAL_SHAPE_OVER_REAL_TCP"
        result["tcp_source_calls"]=counters["tcp_calls"]
        result["tcp_source_bytes"]=counters["tcp_bytes"]
        return result
    finally:
        server.shutdown();server.server_close();thread.join(timeout=5)


def live():
    from experiments.model_fabric.scaled_model_lab import (
        _fetch_metadata,_url,pinned_hf_range_host)
    entry=SOURCES["qwen-27b"]
    idx=_fetch_metadata(_url(MODEL,REVISION,"model.safetensors.index.json"),
                        MAX_INDEX_BYTES)
    if sha256(idx).hexdigest()!=entry["index_sha256"]:
        raise ValueError("Pinned original Qwen source index changed")
    result=_experiment(idx,pinned_hf_range_host(MODEL,REVISION),
                       {"pinned-huggingface-https-range"},fixture=False)
    result["mode"]="LIVE_ORIGINAL_27B_UNSEEN_GENERATIVE_OPERATOR_GROUND"
    result["original_checkpoint_advertised_bytes"]=entry["advertised_bytes"]
    return result


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("loopback","live"),required=True)
    parser.add_argument("--out",required=True)
    args=parser.parse_args(argv)
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    try:result=loopback() if args.mode=="loopback" else live()
    except (ValueError,OSError,TimeoutError,RuntimeError) as e:
        result={
            "schema":"vl-i.shape007-trained-generative-sufficient-statistic.v1",
            "mode":args.mode,"status":"UNRESOLVED",
            "error_class":type(e).__name__,"reason":str(e)[:260]}
        path.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
        print("VL_I_SHAPE007 "+json.dumps(result,sort_keys=True))
        return 1
    result["status"]="MEASURED"
    path.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("VL_I_SHAPE007 "+json.dumps(result,sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
