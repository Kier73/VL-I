"""SHAPE-009: discover a Ground from ORIGINAL Qwen inter-layer causal states.

Actual layer-0 output vectors are carried into the native layer-1 executor
during 4 successive original-token causal transitions; their witnessed source
positions become a three-position mathematical Ground. This does NOT claim
a complete 64-layer decoder or PyTorch-equivalent original trained execution.

Original model mode reuses QUALIFIED-006 host, tokenizer, layer-chain and
original native VL-64 modules. Fixture mode uses those same native layer
functions with a small synthetic weight fixture; it is never original Qwen
evidence.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter_ns

from vli.causal_hidden_manifold import (
    CausalHiddenManifoldGround, TOL)
from vli.large_model_comparison import SOURCES
from vli.qwen_layer_chain import QwenCausalGDNChain


def _verify_observer(ground,fourth):
    width=ground.width
    sparse=(0,1,2,width//2,width-1)
    sparse=tuple(dict.fromkeys(sparse))
    heldout=((.13,.27),(.44,.19),(.08,.73))
    staged=[]
    # Form all novel outputs before the independent original native
    # normalization reference is ever called on a held-out mixture.
    for s,t in heldout:
        predicted=ground.generate(s,t,sparse,expected_head=ground.head)
        staged.append((s,t,predicted))
    observations=[]
    for s,t,predicted in staged:
        native_start=perf_counter_ns()
        native=ground.native_reference(s,t,sparse)
        native_ns=perf_counter_ns()-native_start
        err=max(abs(x-y) for x,y in zip(predicted["values"],native))
        if err>TOL:
            raise ValueError("Original native hidden-state counterexample rejected learned law")
        observations.append({
            "barycentric_coordinates":[s,t],
            "position":predicted["position"],
            "requested_original_output_indices":list(sparse),
            "generated_ns":predicted["duration_ns"],
            "native_reference_ns":native_ns,
            "max_abs_heldout_error":err,
            "generated_before_native_reference":True,
            "original_native_operator_calls_during_generation":0,
        })
    extra=ground.probe_new_original_causal_position(fourth)
    if extra["inside_declared_convex_hull"]:
        raise ValueError("Expected genuine fourth causal token to test outside learned convex span")
    checkpoint=ground.checkpoint()
    t0=perf_counter_ns()
    recovered=CausalHiddenManifoldGround.restore(
        checkpoint,expected_head=checkpoint["checkpoint_digest"])
    recovery_ns=perf_counter_ns()-t0
    new_s,new_t=.24,.41
    before=perf_counter_ns()
    new=recovered.generate(
        new_s,new_t,sparse,expected_head=recovered.head)
    new_ns=perf_counter_ns()-before
    oracle=ground.native_reference(new_s,new_t,sparse)
    err=max(abs(a-b) for a,b in zip(new["values"],oracle))
    if err>TOL:
        raise ValueError("Source-free reconstructed causal Ground did not generate new state")
    full_intention=(.21,.34)
    before=perf_counter_ns()
    all_coords=recovered.generate(
        *full_intention,tuple(range(width)),expected_head=recovered.head)
    full_ns=perf_counter_ns()-before
    before=perf_counter_ns()
    ref=ground.native_reference(*full_intention,tuple(range(width)))
    native_full_ns=perf_counter_ns()-before
    full_err=max(abs(a-b) for a,b in zip(all_coords["values"],ref))
    if full_err>TOL:
        raise ValueError("Full hidden-state native reference failed")
    return {
        "schema":"vl-i.shape009-original-causal-interlayer-manifold.v1",
        "real_original_pretrained":ground.original_pretrained,
        "model":ground.model,"revision":ground.revision,
        "index_sha256":ground.index_sha256,
        "source_root":ground.source_root,
        "input_width":width,
        "actual_original_source_layer_positions":[0,1,2,3],
        "native_original_causal_chain_heads":[
            r["chain_head"] for r in ground.recorded_causal_state_witnesses],
        "original_source_token_ids":[
            r["original_token_id"] for r in ground.recorded_causal_state_witnesses]+[
            extra["original_token_id"]],
        "original_interlayer_handoff_digests":[
            r["layer1_input_f64_digest"]
            for r in ground.recorded_causal_state_witnesses],
        "original_native_layer1_norm_sha256":ground.native_norm_sha256,
        "third_true_causal_state_not_on_two_state_line":{
            "line_projection":ground.prior_line_parameter,
            "off_line_relative_residual":ground.off_line_residual},
        "candidate_max_challenge_errors":ground.proof["candidate_max_error"],
        "admitted_law":ground.proof["admitted_family"],
        "native_training_calls":ground.native_training_calls,
        "native_training_ns":ground.native_training_ns,
        "observation_based_fit_positions":ground.proof["training_probes"],
        "falsification_positions":ground.proof["counterexamples"],
        "unseen_convex_span_intentions":observations,
        "heldout_max_abs_error":max(x["max_abs_heldout_error"] for x in observations),
        "fourth_real_causal_hidden_state_negative_probe":extra,
        "no_unproven_future_causal_state_admitted":True,
        "ECPU_reusable_hidden_state_ground":{
            "objects":{k:v["manifest_digest"] for k,v in ground.objects.items()},
            "source_causal_hidden_bytes":3*width*8,
            "native_layer1_trained_weight_bytes":len(
                bytes.fromhex(ground.native_norm_sha256)) if False else 2*width,
            "source_free_restoration_verified":True,
            "checkpoint_digest":checkpoint["checkpoint_digest"],
            "checkpoint_serialized_bytes":len(json.dumps(checkpoint)),
            "recovery_ns":recovery_ns},
        "recovered_new_unseen_mixture":{
            "mixture":[new_s,new_t],
            "position":new["position"],
            "generated_ns":new_ns,
            "max_abs_native_reference_error":err,
            "native_layer_evaluations_during_prediction":0},
        "complete_output_negative_control":{
            "width":width,
            "generated_full_ns":full_ns,
            "original_native_full_norm_ns":native_full_ns,
            "generated_slower":full_ns>native_full_ns,
            "max_abs_error":full_err},
        "independent_real_original_weight_PyTorch_whole_layer_oracle":False,
        "complete_original_model_decoder_executed":False,
        "scope":[
            "genuine native original first two complete Qwen GDN layer states for four tokens in live mode",
            "learned norm law only on convex interpolation of three real causal layer0 output hidden states",
            "fourth real causal position is a counterexample to automatic extrapolation, not a generated output",
            "native original layer1 normalization reference queried only after heldout predictions",
            "source-free ECPU structural memory preserves exact original causal outputs and trained norm coefficients",
            "independent PyTorch/Transformers whole-layer parity remains unresolved",
            "no generated next token and no full 64-layer decoder",
        ],
    }


def fixture():
    from test_qwen_layer_chain import two_layer_fixture,executor
    from vli.qwen_all_heads import QwenVectorizedCompleteLayer
    nav=two_layer_fixture()
    chain=QwenCausalGDNChain(
        executor(nav,QwenVectorizedCompleteLayer))
    reports=[]
    for token,hidden in (
        (63204,[.5,-.2,.3]),
        (628,[-.1,.2,.8]),
        (1761,[.7,.6,-.3]),
        (33303,[.3,-.4,.65]),
    ):
        _,receipt=chain.step(hidden,token_id=token,
                             capture_intermediates=True)
        reports.append(receipt)
    raw,desc=nav.get_raw("model.language_model.layers.1.input_layernorm.weight")
    ground=CausalHiddenManifoldGround(
        reports=reports[:3],native_norm_raw=raw,
        native_norm_dtype=desc["dtype"],
        model=nav.index["model"],revision=nav.index["revision"],
        index_sha256=nav.index["index_sha256"],
        index_ground_digest=nav.index["ground_digest"])
    report=_verify_observer(ground,reports[3])
    report["mode"]="SYNTHETIC_NATIVE_QWEN_TWO_LAYER_CAUSAL_FIXTURE"
    report["not_original_pretrained"]=True
    return report


def live():
    from experiments.model_fabric.scaled_model_fabric import (
        QualifiedShardNavigator,MAX_INDEX_BYTES)
    from experiments.model_fabric.scaled_model_lab import _url,_fetch_metadata
    from vli.operative_model import ground as model_ground
    from vli.pinned_text_ground import (
        admit,text_intention,tokenize,validate_observation,
        MAX_ASSET_BYTES,TOKENIZER_SOURCES)
    from vli.token_embedding_ground import token_intention,observe_token_row
    from vli.qualified_projection_host import HOST_NAME,BoundedModelRangeHost
    from vli.qwen_all_heads import QwenVectorizedCompleteLayer
    from examples.qwen_layer_chain_frontier import DEFAULT_TEXT

    cfg=SOURCES["qwen-27b"]
    model,revision=cfg["model"],cfg["revision"]
    source_start=perf_counter_ns()
    index=_fetch_metadata(
        _url(model,revision,"model.safetensors.index.json"),MAX_INDEX_BYTES)
    if sha256(index).hexdigest()!=cfg["index_sha256"]:
        raise ValueError("Original pretrained checkpoint index pin changed")
    host=BoundedModelRangeHost(
        model=model,revision=revision,
        block_bytes=8*1024*1024,budget_bytes=128*1024*1024)
    def nav():
        return QualifiedShardNavigator(
            model=model,revision=revision,index_raw=index,
            range_host=host,approved_hosts={HOST_NAME})
    n=nav()
    g=model_ground("qwen-27b",n.index)
    tok=TOKENIZER_SOURCES["qwen-27b"]
    if tok["repository"]!=model or tok["revision"]!=revision:
        raise ValueError("Pinned original tokenizer revision does not match source")
    tokenizer_url=(f"https://huggingface.co/{model}/resolve/"
                   f"{revision}/tokenizer.json")
    raw_tokenizer=_fetch_metadata(tokenizer_url,MAX_ASSET_BYTES)
    tground,tokenizer=admit(g,raw_tokenizer)
    intended=text_intention(tground,DEFAULT_TEXT)
    observed=tokenize(tground,tokenizer,intended)
    validate_observation(g,tground,intended,observed)
    original_ids=tuple(observed["token_ids"][:4])
    if original_ids!=(63204,628,1761,33303):
        raise ValueError("Pinned original paragraph token IDs differ from prior evidence")
    runners=[
        QwenVectorizedCompleteLayer(
            n,layer=i,key_heads=16,value_heads=48,
            key_dim=128,value_dim=128,budget_bytes=2_000_000_000)
        for i in (0,1)]
    chain=QwenCausalGDNChain(runners)
    real=[]
    for token_id in original_ids:
        embedding,source_witness=observe_token_row(
            n,g,token_intention(g,token_id))
        _,receipt=chain.step(
            embedding,token_id=token_id,capture_intermediates=True)
        if receipt["original_token_id"]!=token_id:
            raise ValueError("Observed original token did not bind to causal layer")
        real.append(receipt)
    before=perf_counter_ns()
    norm_raw,norm_type=n.get_raw(
        "model.language_model.layers.1.input_layernorm.weight")
    trained_source_end=perf_counter_ns()
    ground=CausalHiddenManifoldGround(
        reports=real[:3],
        native_norm_raw=norm_raw,native_norm_dtype=norm_type["dtype"],
        model=n.index["model"],revision=n.index["revision"],
        index_sha256=n.index["index_sha256"],
        index_ground_digest=n.index["ground_digest"])
    formed=_verify_observer(ground,real[3])
    formed.update({
        "mode":"ORIGINAL_QWEN27B_TWO_COMPLETE_GDN_LAYERS_FOUR_REAL_CAUSAL_POSITIONS",
        "actual_original_pretrained_source":True,
        "tokenizer_sha256":tground["asset_sha256"],
        "original_token_ids":list(original_ids),
        "original_token_sequence_sha256":observed["observation_digest"],
        "original_two_layer_chain_final_head":chain.head,
        "all_four_original_causal_positions_complete":chain.position==4,
        "original_model_host_bytes_and_ranges":host.stats(),
        "original_source_and_four_layer_chain_ns":
            trained_source_end-source_start,
        "original_norm_acquisition_ns":trained_source_end-before,
        "first_native_original_layer0_to_layer1_verification_grade":
            "VL64_NATIVE_SAME_IMPLEMENTATION_NOT_INDEPENDENT_PYTORCH_REFERENCE",
    })
    return formed


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("fixture","live"),required=True)
    parser.add_argument("--out",required=True)
    args=parser.parse_args(argv)
    path=Path(args.out)
    path.parent.mkdir(parents=True,exist_ok=True)
    try:
        report=fixture() if args.mode=="fixture" else live()
    except (OSError,TimeoutError,RuntimeError,ValueError,MemoryError,
            ImportError,OverflowError) as exc:
        report={
            "schema":"vl-i.shape009-original-causal-interlayer-manifold.v1",
            "status":"UNRESOLVED","mode":args.mode,
            "error_type":type(exc).__name__,"reason":str(exc)[:350],
            "no_claim_of_original_pretrained_causal_hidden_law":True}
        path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
        print("VL_I_SHAPE009 "+json.dumps(report,sort_keys=True))
        return 1
    report["status"]="MEASURED"
    path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("VL_I_SHAPE009 "+json.dumps(report,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
