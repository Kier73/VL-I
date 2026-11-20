"""QUALIFIED-008: indexed source topology and multi-hybrid-block causal parity."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import struct

import pytest

from experiments.model_fabric.complete_native_layers import QwenQualifiedLayer, LocalStatePages
from vli.qwen_architecture_navigator import (
    admit_indexed_qwen_topology, QwenArchitectureCausalPrefix,
)
from vli.qwen_hybrid_block import QwenQualifiedHybridBlock
from vli.qwen_layer_chain import _f64_transfer_digest
from test_architecture_probe import nav_for
from test_complete_native_layers import qwen_fixture
from test_qwen_hybrid_block import (
    attention_specs, runners, reference_attention,
)


def nav_eight_layers():
    """Two original-form hybrid blocks; every layer has all its own source tensors."""
    _,matrices,vectors,conv=qwen_fixture()
    tensors={}
    for layer in range(8):
        prefix=f"model.language_model.layers.{layer}."
        if layer%4==3:
            for name,entries in attention_specs().items():
                if isinstance(entries[0],list):
                    vals=[z for row in entries for z in row]
                    shape=[len(entries),len(entries[0])]
                else:
                    vals=entries
                    shape=[len(vals)]
                tensors[prefix+name]=("F32",shape,
                    struct.pack("<"+str(len(vals))+"f",*vals))
        else:
            for name,m in matrices.items():
                vals=[z for row in m for z in row]
                tensors[prefix+name]=("F32",[len(m),len(m[0])],
                    struct.pack("<"+str(len(vals))+"f",*vals))
            for name,v in vectors.items():
                tensors[prefix+name]=("F32",[len(v)],
                    struct.pack("<"+str(len(v))+"f",*v))
            vals=[z for row in conv for z in row]
            tensors[prefix+"linear_attn.conv1d.weight"]=(
                "F32",[len(conv),1,len(conv[0])],
                struct.pack("<"+str(len(vals))+"f",*vals))
    return nav_for(tensors)


def config_eight():
    return {
      "architectures":["Qwen3_5ForConditionalGeneration"],
      "model_type":"qwen3_5",
      "text_config":{
        "model_type":"qwen3_5_text",
        "num_hidden_layers":8,
        "layer_types":["linear_attention","linear_attention",
                       "linear_attention","full_attention"]*2,
        "full_attention_interval":4,
        "hidden_size":3,"intermediate_size":4,
        "linear_num_key_heads":1,"linear_num_value_heads":2,
        "linear_key_head_dim":2,"linear_value_head_dim":2,
        "num_attention_heads":2,"num_key_value_heads":1,
        "head_dim":2,"max_position_embeddings":256,
        "hidden_act":"silu","attention_bias":False,
        "rms_norm_eps":1e-6,
        "rope_parameters":{
            "rope_type":"default","rope_theta":10_000_000,
            "partial_rotary_factor":1.0,"mrope_section":[1,0,0],
        }
      }
    }


def encoded_config(data=None):
    return json.dumps(config_eight() if data is None else data,
                      sort_keys=True,separators=(",",":")).encode()


def plan_for(nav):
    return admit_indexed_qwen_topology(nav,encoded_config())


def auto_eight(nav=None):
    nav=nav or nav_eight_layers()
    return QwenArchitectureCausalPrefix(
        nav,plan_for(nav),stop_layer=8,max_positions=64)


def test_model_itself_selects_eight_mixed_operators_from_qualified_index():
    nav=nav_eight_layers()
    raw=encoded_config()
    cfg_hash=hashlib.sha256(raw).hexdigest()
    plan=admit_indexed_qwen_topology(nav,raw,expected_config_sha256=cfg_hash)
    assert plan["layer_count"]==8
    assert [x["kind"] for x in plan["layers"]]==(
        ["GDN","GDN","GDN","GATED_GQA"]*2)
    assert plan["index_ground_digest"]==nav.index["ground_digest"]
    assert plan["config_sha256"]==cfg_hash
    assert plan["external_config_sha256_pin_supplied"]
    chain=QwenArchitectureCausalPrefix(
        nav,plan,stop_layer=8,max_positions=64)
    assert [r.layer for r in chain.runners]==list(range(8))
    assert [type(r).__name__ for r in chain.runners]==[
        "QwenVectorizedCompleteLayer","QwenVectorizedCompleteLayer",
        "QwenVectorizedCompleteLayer","QwenQualifiedGatedAttention"]*2
    assert chain.position==0 and chain.width==3


def test_new_architecture_navigator_matches_independent_scalar_oracles():
    chain=auto_eight()
    independently=[QwenQualifiedLayer(nav_eight_layers(),layer=i,
        key_heads=1,value_heads=2,key_dim=2,value_dim=2)
        for i in (0,1,2,4,5,6)]
    histories={3:[],7:[]}
    first_block=QwenQualifiedHybridBlock(runners(nav_eight_layers()))
    for pos,hidden in enumerate(([.5,-.2,.3],[-.1,.2,.8],[.7,.6,-.3])):
        got,receipt=chain.step(list(hidden),token_id=pos)
        prefix_out,first_receipt=first_block.step(list(hidden),token_id=pos)
        scalar_out=list(hidden)
        gdn_idx=0
        for layer in range(8):
            if layer%4!=3:
                scalar_out,_=independently[gdn_idx].next(scalar_out)
                gdn_idx+=1
            else:
                scalar_out,histories[layer]=reference_attention(
                    scalar_out,histories[layer],pos)
            if layer==3:
                assert prefix_out==pytest.approx(scalar_out,abs=2e-10)
        assert got==pytest.approx(scalar_out,abs=3e-10)
        assert len(receipt["layer_witnesses"])==8
        assert [x["layer"] for x in receipt["layer_witnesses"]]==list(range(8))
        assert receipt["layer_witnesses"][3]["output_f64_digest"]==(
            first_receipt["layer_witnesses"][-1]["output_f64_digest"])
        assert receipt["layer_witnesses"][0]["input_f64_digest"]==(
            _f64_transfer_digest(hidden))
        for prev,next_ in zip(receipt["layer_witnesses"],receipt["layer_witnesses"][1:]):
            assert prev["output_f64_digest"]==next_["input_f64_digest"]
        assert [x["kind"] for x in receipt["layer_witnesses"]]==(
            ["GDN","GDN","GDN","GATED_GQA"]*2)
        assert receipt["layer_witnesses"][3]["kv_positions"]==pos+1
        assert receipt["layer_witnesses"][7]["kv_positions"]==pos+1
        assert receipt["complete_prefix_layers"]==8
        assert not receipt["vocabulary_logits_computed"]


def test_eight_states_checkpoint_restore_then_exact_continuation(tmp_path):
    original=auto_eight()
    original.step([.5,-.2,.3],token_id=63204)
    store=LocalStatePages(tmp_path)
    manifest=original.checkpoint(store,input_sequence_digest="a"*64)
    assert len(manifest["layer_checkpoints"])==8
    assert [x["layer"] for x in manifest["layer_checkpoints"]]==list(range(8))
    assert len([x for x in manifest["layer_checkpoints"] if "kv_pages" in x])==2
    assert len([x for x in manifest["layer_checkpoints"] if "causal_pages" in x])==6
    nav=nav_eight_layers()
    restored=QwenArchitectureCausalPrefix.restore(
        nav,plan_for(nav),store,manifest,
        expected_digest=manifest["checkpoint_digest"],
        expected_plan_digest=original.plan["plan_digest"],
        expected_input_sequence_digest="a"*64,max_positions=64)
    assert restored.head==original.head
    assert restored.position==original.position==1
    for tok,hidden in ((628,[-.1,.2,.8]),(123,[.7,.6,-.3])):
        a,r1=original.step(hidden,token_id=tok)
        b,r2=restored.step(hidden,token_id=tok)
        assert a==b
        assert r1["chain_head"]==r2["chain_head"]
        assert original._snapshot()==restored._snapshot()
    assert restored.position==3


def test_config_and_model_index_must_agree_before_instantiating():
    nav=nav_eight_layers()
    changed=deepcopy(config_eight())
    changed["text_config"]["layer_types"][3]="linear_attention"
    with pytest.raises(ValueError,match="interval|Missing|conflict"):
        admit_indexed_qwen_topology(nav,encoded_config(changed))
    changed=deepcopy(config_eight())
    changed["text_config"]["full_attention_interval"]=None
    changed["text_config"]["layer_types"][2]="full_attention"
    with pytest.raises(ValueError,match="Missing"):
        admit_indexed_qwen_topology(nav,encoded_config(changed))
    changed=deepcopy(config_eight())
    changed["text_config"]["num_hidden_layers"]=9
    with pytest.raises(ValueError,match="layer"):
        admit_indexed_qwen_topology(nav,encoded_config(changed))
    changed=deepcopy(config_eight())
    changed["text_config"]["num_attention_heads"]=3
    with pytest.raises(ValueError,match="topology|operator|geometry|shape"):
        admit_indexed_qwen_topology(nav,encoded_config(changed))
    with pytest.raises(ValueError,match="external pin"):
        admit_indexed_qwen_topology(nav,encoded_config(),
                                   expected_config_sha256="b"*64)
    with pytest.raises(ValueError,match="Duplicate"):
        admit_indexed_qwen_topology(nav,b'{"model_type":"qwen3_5","model_type":"qwen3_5"}')
    with pytest.raises(ValueError,match="Nonfinite"):
        admit_indexed_qwen_topology(nav,b'{"model_type":NaN}')


def test_missing_indexed_tensor_even_though_config_claims_it():
    nav=nav_eight_layers()
    del nav.weight_map["model.language_model.layers.7.self_attn.v_proj.weight"]
    with pytest.raises(ValueError,match="Missing trained|Shard header|index tensor"):
        admit_indexed_qwen_topology(nav,encoded_config())


def test_checkpoint_enforces_external_plan_and_exact_layer_lineage(tmp_path):
    original=auto_eight()
    original.step([.5,-.2,.3])
    store=LocalStatePages(tmp_path)
    manifest=original.checkpoint(store,input_sequence_digest="c"*64)
    def restore(plan,document=manifest,pin=None):
        nav=nav_eight_layers()
        return QwenArchitectureCausalPrefix.restore(
            nav,plan,store,document,
            expected_digest=manifest["checkpoint_digest"],
            expected_plan_digest=pin or original.plan["plan_digest"],
            expected_input_sequence_digest="c"*64,max_positions=64)
    with pytest.raises(ValueError,match="Unpinned"):
        restore(original.plan,{**manifest,"stop_layer":7})
    with pytest.raises(ValueError,match="Different"):
        restore(original.plan,pin="d"*64)
    config=deepcopy(config_eight())
    config["text_config"]["rope_parameters"]["rope_theta"]=12345
    altered=admit_indexed_qwen_topology(nav_eight_layers(),encoded_config(config))
    with pytest.raises(ValueError,match="Different"):
        restore(altered)
    target=manifest["layer_checkpoints"][7]["kv_pages"][0]
    (tmp_path/target["sha256"]).write_bytes(b"tampered")
    with pytest.raises(ValueError,match="Corrupted"):
        restore(original.plan)


def test_failure_of_second_attention_rolls_back_all_eight_states(monkeypatch):
    chain=auto_eight()
    chain.step([.5,-.2,.3])
    snapshot=deepcopy(chain._snapshot())
    head=chain.head
    last_digest=chain.last_output_digest
    old=chain.runners[7]._projection
    def interrupt(name,values):
        if name=="mlp.down_proj.weight":
            raise ConnectionError("full-attention worker departed")
        return old(name,values)
    monkeypatch.setattr(chain.runners[7],"_projection",interrupt)
    with pytest.raises(ConnectionError,match="departed"):
        chain.step([-.1,.2,.8])
    assert chain._snapshot()==snapshot
    assert chain.position==1
    assert chain.head==head
    assert chain.last_output_digest==last_digest


def test_plan_digests_and_operator_types_are_mutation_resistant():
    nav=nav_eight_layers()
    plan=plan_for(nav)
    mutated=deepcopy(plan)
    mutated["layers"][7]["kind"]="GDN"
    with pytest.raises(ValueError,match="Changed"):
        QwenArchitectureCausalPrefix(nav,mutated,stop_layer=8)
    with pytest.raises(ValueError,match="extent"):
        QwenArchitectureCausalPrefix(nav,plan,stop_layer=9)
    with pytest.raises(ValueError,match="extent"):
        QwenArchitectureCausalPrefix(nav,plan,stop_layer=0)
