"""QUALIFIED-008: independently restarted worker continues two hybrid blocks.

The original-topology fixture state is written by a worker process that is
forcibly terminated. Another Python interpreter must reconstruct the
architecture plan, re-open all eight state authorities from externally pinned
pages, and produce the identical next trained result and semantic head.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from vli.qwen_architecture_navigator import QwenArchitectureCausalPrefix
from test_qwen_architecture_navigator import auto_eight


CHILD=r'''
import json,sys,time
from pathlib import Path
from experiments.model_fabric.complete_native_layers import LocalStatePages
from vli.qwen_architecture_navigator import QwenArchitectureCausalPrefix
from test_qwen_architecture_navigator import nav_eight_layers,plan_for

root=Path(sys.argv[2])
store=LocalStatePages(root)
if sys.argv[1]=="checkpoint":
    nav=nav_eight_layers()
    chain=QwenArchitectureCausalPrefix(nav,plan_for(nav),
                                       stop_layer=8,max_positions=64)
    _,receipt=chain.step([.5,-.2,.3],token_id=63204)
    checkpoint=chain.checkpoint(store,input_sequence_digest="a"*64)
    (root/"checkpoint.json").write_text(json.dumps(checkpoint),encoding="utf8")
    (root/"plan-digest.txt").write_text(chain.plan["plan_digest"],encoding="ascii")
    print("EIGHT_LAYER_CHECKPOINT_COMMITTED",flush=True)
    while True: time.sleep(1)
if sys.argv[1]=="recover":
    nav=nav_eight_layers()
    plan=plan_for(nav)
    checkpoint=json.loads((root/"checkpoint.json").read_text("utf8"))
    chain=QwenArchitectureCausalPrefix.restore(
        nav,plan,store,checkpoint,
        expected_digest=sys.argv[3],
        expected_plan_digest=sys.argv[4],
        expected_input_sequence_digest="a"*64,
        max_positions=64)
    out,receipt=chain.step([-.1,.2,.8],token_id=628)
    print(json.dumps({"position":chain.position,
        "head":receipt["chain_head"],"result":out,
        "layer_types":[w["kind"] for w in receipt["layer_witnesses"]],
        "attention_history":[r.position for r in chain.runners
                             if r.__class__.__name__=="QwenQualifiedGatedAttention"]}),
        flush=True)
'''


def test_two_hybrid_blocks_survive_actual_worker_termination(tmp_path):
    env=os.environ.copy()
    env["PYTHONPATH"]=str(Path(__file__).resolve().parent)+os.pathsep+env.get("PYTHONPATH","")
    command=[sys.executable,"-u","-c",CHILD]
    worker=subprocess.Popen(
        command+["checkpoint",str(tmp_path)],stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,text=True,env=env)
    try:
        marker=worker.stdout.readline().strip()
        assert marker=="EIGHT_LAYER_CHECKPOINT_COMMITTED",(
            marker,worker.poll(),
            worker.stderr.read() if worker.poll() is not None else "")
    finally:
        worker.kill()
        worker.wait(timeout=10)
        worker.stdout.close()
        worker.stderr.close()
    assert worker.returncode!=0
    external=json.loads((tmp_path/"checkpoint.json").read_text())
    plan_head=(tmp_path/"plan-digest.txt").read_text()
    assert len(external["layer_checkpoints"])==8
    assert len(plan_head)==64
    resumed=subprocess.run(
        command+["recover",str(tmp_path),external["checkpoint_digest"],plan_head],
        capture_output=True,text=True,check=True,timeout=30,env=env)
    evidence=json.loads(resumed.stdout.strip())
    uninterrupted=auto_eight()
    uninterrupted.step([.5,-.2,.3],token_id=63204)
    output,receipt=uninterrupted.step([-.1,.2,.8],token_id=628)
    assert evidence["position"]==2
    assert evidence["result"]==output
    assert evidence["head"]==receipt["chain_head"]
    assert evidence["layer_types"]==["GDN","GDN","GDN","GATED_GQA"]*2
    assert evidence["attention_history"]==[2,2]
