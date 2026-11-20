"""SHAPE-008: infer and use a new original token-conditioned computational law.

Select the minimal polynomial degree from actual native VL-64 observations of
two original token embedding rows. Predict output for unseen mixtures BEFORE
consulting original trained normalization. On the full original checkpoint,
tokens 63204 and 628 are taken from the independently observed 13-token
Qwen paragraph transcript retained by VL-I INTERACT-004.

The two-token affine-mixture hidden inputs are NOT claimed to be the original
Qwen network's actual conversational prefill states or generated tokens.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import re
from threading import Thread
from time import perf_counter_ns
from urllib.request import Request,urlopen

from experiments.model_fabric.scaled_model_fabric import MAX_INDEX_BYTES
from vli.pretrained_range_memory import QualifiedPretrainedRangeMemory
from vli.token_conditioned_law import TokenConditionedLawGround
from vli.large_model_comparison import SOURCES
from examples.token_span_fixture import original_shape_fixture,SHARD

MODEL="Qwen/Qwen3.8-27B"
REVISION="37561816e66b5b9153b1b64c10968c18c1e6f4a0"


def experiment(index,host,approved,*,token_ids,synthetic):
    original=not synthetic
    t0=perf_counter_ns()
    source=QualifiedPretrainedRangeMemory(
        model=MODEL,revision=REVISION,index_raw=index,
        upstream_host=host,approved_upstream_hosts=set(approved),
        max_resident_bytes=4*1024*1024)
    ground=TokenConditionedLawGround(
        source,index,token_ids=token_ids,synthetic_fixture=synthetic)
    t1=perf_counter_ns()
    heldout=(.13,.37,.67,.91)
    width=ground.width
    sparse=(0,1,2,width//4,width//2,3*width//4,width-2,width-1)
    results=[]
    # Commit every prediction before invoking an original pretrained oracle.
    staged=[]
    for t in heldout:
        before=source.source_requests
        result=ground.generate(t,sparse,expected_head=ground.head)
        if source.source_requests!=before:
            raise ValueError("Unseen output unexpectedly contacted original source")
        staged.append((t,result))
    for t,result in staged:
        start=perf_counter_ns()
        real=ground.native_reference(t,sparse)
        reference_ns=perf_counter_ns()-start
        err=max(abs(a-b) for a,b in zip(real,result["values"]))
        if err>2e-9:
            raise ValueError("New token-conditioned Ground failed independent heldout")
        results.append({
            "mixture":t,"original_token_ids":list(token_ids),
            "position":result["coordinate"],
            "result_sha256":result["event"]["output_sha256"],
            "generated_sparse_coordinates":len(sparse),
            "generated_ns":result["duration_ns"],
            "native_original_reference_ns":reference_ns,
            "native_reference_invoked_only_after_prediction":True,
            "max_abs_error":err,
            "native_calls_during_prediction":result["event"][
                "original_native_operator_calls"],
            "original_network_requests_during_prediction":result["event"][
                "original_host_range_calls"],
        })
    checkpoint=ground.checkpoint()
    known_source_calls=source.source_requests
    before=perf_counter_ns()
    restored=TokenConditionedLawGround.restore(
        index,checkpoint,expected_head=checkpoint["checkpoint_digest"])
    restored_ns=perf_counter_ns()-before
    new_t=.83
    before=perf_counter_ns()
    new_result=restored.generate(new_t,sparse,expected_head=restored.head)
    unseen_ns=perf_counter_ns()-before
    native=ground.native_reference(new_t,sparse)
    err=max(abs(a-b) for a,b in zip(new_result["values"],native))
    if err>2e-9 or source.source_requests!=known_source_calls:
        raise ValueError("Source-free unseen-token-mixture generation not qualified")
    full_t=.29
    before=perf_counter_ns()
    full=restored.generate(full_t,tuple(range(width)),
                           expected_head=restored.head)
    full_ns=perf_counter_ns()-before
    before=perf_counter_ns()
    reference=ground.native_reference(full_t,tuple(range(width)))
    reference_ns=perf_counter_ns()-before
    full_err=max(abs(a-b) for a,b in zip(full["values"],reference))
    if full_err>2e-9:raise ValueError("Complete original normalization mismatch")
    return {
        "schema":"vl-i.shape008-real-token-conditioned-inferred-law.v1",
        "source_model":MODEL,"pinned_revision":REVISION,
        "real_original_pretrained_checkpoint":ground.original_pretrained,
        "real_original_token_id_origins":(
            "INTERACT004_pinned_Qwen_paragraph_tokenizer" if original else
            "synthetic_fake_token_embedding_fixture"),
        "original_source_index_sha256":source.index_sha256,
        "source_bytes_transferred":source.source_bytes,
        "source_network_requests":source.source_requests,
        "width":width,"tokens":list(token_ids),
        "original_embedding_witnesses":ground.embedding_witnesses,
        "native_norm_weight_sha256":ground.original_norm_weight_sha256,
        "original_source_ground":ground.source_root,
        "learned_polynomial_degree":ground.proof["admitted_degree"],
        "inferred_coefficients":list(ground.coefficients),
        "candidate_counterexample_errors":ground.proof[
            "candidate_challenge_errors"],
        "training_native_operator_calls":ground.training_calls,
        "training_native_operator_ns":ground.training_ns,
        "source_and_inference_bootstrap_ns":t1-t0,
        "heldout_predictions_precede_native_checks":True,
        "heldout":results,
        "max_abs_heldout_error":max(r["max_abs_error"] for r in results),
        "source_free_recovery":{
            "success":True,"recovery_ns":restored_ns,
            "checkpoint_digest":checkpoint["checkpoint_digest"],
            "checkpoint_serialized_bytes":len(json.dumps(checkpoint)),
            "new_unseen_mixture":new_t,"new_generated_ns":unseen_ns,
            "new_prediction_max_abs_error":err,
            "new_native_calls_during_prediction":new_result["event"][
                "original_native_operator_calls"],
            "new_original_source_requests_during_prediction":new_result["event"][
                "original_host_range_calls"],
        },
        "full_output_negative_control":{
            "width":width,"generated_full_ns":full_ns,
            "native_original_reference_ns":reference_ns,
            "max_abs_error":full_err,
            "generated_slower_than_native":full_ns>reference_ns,
        },
        "ECPU_original_embedding_and_gain_objects":{
            "objects":{k:v["manifest_digest"] for k,v in ground.manifests.items()},
            "complete_bytes":3*width*8,
        },
        "scope_limits":[
            "two original model token rows only, affine mixtures inside their span",
            "not a real paragraph prefill, not a next-token decoder",
            "quadratic degree family selected from bounded candidate set",
            "coefficients learned from actual original operator output probes",
            "independent algebra supports degree-2 only on this affine span",
            "new inputs unseen to training probes, reference checked afterward",
            "original checkpoint not downscaled for live test",
            "source bytes, probe computation, ECPU page checking costed",
        ],
    }


def loopback():
    idx,raw_shard=original_shape_fixture()
    data={"calls":0,"bytes":0}
    class Receiver(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            m=re.fullmatch(r"bytes=(\d+)-(\d+)",
                            self.headers.get("Range",""))
            if self.path!="/"+SHARD or not m:
                self.send_error(400);return
            a,b=map(int,m.groups())
            if b<a or b>=len(raw_shard):
                self.send_error(416);return
            payload=raw_shard[a:b+1]
            self.send_response(206)
            self.send_header("Content-Range",f"bytes {a}-{b}/{len(raw_shard)}")
            self.send_header("Content-Length",str(len(payload)))
            self.end_headers();self.wfile.write(payload)
    server=ThreadingHTTPServer(("127.0.0.1",0),Receiver)
    th=Thread(target=server.serve_forever,daemon=True);th.start()
    try:
        url=f"http://127.0.0.1:{server.server_port}/{SHARD}"
        def source(req):
            a=req["byte_begin"];b=req["byte_end_exclusive"]
            with urlopen(Request(url,headers={
                "Range":f"bytes={a}-{b-1}"},method="GET"),timeout=6) as res:
                if res.status!=206 or res.headers.get("Content-Range")!=(
                    f"bytes {a}-{b-1}/{len(raw_shard)}"):
                    raise ValueError("Local TCP exact range is not source-qualified")
                payload=res.read(b-a+1)
            if len(payload)!=b-a:
                raise ValueError("Short synthetic TCP source byte observation")
            data["calls"]+=1;data["bytes"]+=len(payload)
            return {**req,"status":"QUALIFIED_EXACT_SHARD_RANGE",
                    "observer_host":"synthetic-unauthenticated-loopback-TCP",
                    "payload":payload,"sha256":sha256(payload).hexdigest()}
        result=experiment(idx,source,{"synthetic-unauthenticated-loopback-TCP"},
                          token_ids=(100,101),synthetic=True)
        result["mode"]="SYNTHETIC_BF16_OVER_REAL_LOOPBACK_TCP"
        result["loopback_network_metrics"]=data
        return result
    finally:
        server.shutdown();server.server_close();th.join(timeout=5)


def live():
    from experiments.model_fabric.scaled_model_lab import (
        _fetch_metadata,_url,pinned_hf_range_host)
    cfg=SOURCES["qwen-27b"]
    idx=_fetch_metadata(_url(MODEL,REVISION,"model.safetensors.index.json"),
                        MAX_INDEX_BYTES)
    if sha256(idx).hexdigest()!=cfg["index_sha256"]:
        raise ValueError("Actual original pretrained model index changed")
    result=experiment(
        idx,pinned_hf_range_host(MODEL,REVISION),
        {"pinned-huggingface-https-range"},
        token_ids=(63204,628),synthetic=False)
    result["mode"]="LIVE_ORIGINAL_QWEN27B_TOKEN_SPAN_LAW"
    result["original_checkpoint_advertised_bytes"]=cfg["advertised_bytes"]
    return result


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("loopback","live"),required=True)
    parser.add_argument("--out",required=True)
    args=parser.parse_args(argv)
    destination=Path(args.out)
    destination.parent.mkdir(parents=True,exist_ok=True)
    try:result=loopback() if args.mode=="loopback" else live()
    except (ValueError,OSError,RuntimeError,TimeoutError) as e:
        result={"schema":"vl-i.shape008-real-token-conditioned-inferred-law.v1",
                "mode":args.mode,"status":"UNRESOLVED",
                "error_class":type(e).__name__,"reason":str(e)[:280]}
        destination.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
        print("VL_I_SHAPE008 "+json.dumps(result,sort_keys=True))
        return 1
    result["status"]="MEASURED"
    destination.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("VL_I_SHAPE008 "+json.dumps(result,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
