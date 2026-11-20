"""QUALIFIED-007: real process-death recovery of hybrid causal mathematical state.

One operating-system worker checkpoints and is forcibly killed. A distinct
Python interpreter independently instantiates qualified navigators and
continues from pages, without reading the dead process's memory.
Only fixture-trained weights: original 27B process parity remains separate.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from vli.qwen_hybrid_block import QwenQualifiedHybridBlock
from test_qwen_hybrid_block import nav_hybrid,runners


CHILD=r'''
import json
from pathlib import Path
import sys,time
from experiments.model_fabric.complete_native_layers import LocalStatePages
from vli.qwen_hybrid_block import QwenQualifiedHybridBlock
from test_qwen_hybrid_block import nav_hybrid,runners

root=Path(sys.argv[2])
pages=LocalStatePages(root)
if sys.argv[1]=="checkpoint":
    chain=QwenQualifiedHybridBlock(runners(nav_hybrid()))
    out,receipt=chain.step([.5,-.2,.3],token_id=63204)
    checkpoint=chain.checkpoint(pages,input_sequence_digest="a"*64)
    (root/"pinned-manifest.json").write_text(json.dumps(checkpoint),encoding="utf8")
    print("CHECKPOINT_COMMITTED",flush=True)
    while True:
        time.sleep(1)
if sys.argv[1]=="resume":
    checkpoint=json.loads((root/"pinned-manifest.json").read_text("utf8"))
    chain=QwenQualifiedHybridBlock.restore(
        runners(nav_hybrid()),pages,checkpoint,
        expected_digest=sys.argv[3],
        expected_input_sequence_digest="a"*64)
    out,receipt=chain.step([-.1,.2,.8],token_id=628)
    print(json.dumps({"chain_head":receipt["chain_head"],
        "output":out,"position":chain.position,
        "attention_kv_positions":chain.runners[-1].position}),flush=True)
'''


def test_actual_killed_worker_process_resumes_four_causal_layers(tmp_path):
    env=os.environ.copy()
    tests=str(Path(__file__).resolve().parent)
    env["PYTHONPATH"]=tests+os.pathsep+env.get("PYTHONPATH","")
    cmd=[sys.executable,"-u","-c",CHILD]
    process=subprocess.Popen(
        cmd+["checkpoint",str(tmp_path)],stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,text=True,env=env)
    try:
        first=process.stdout.readline().strip()
        assert first=="CHECKPOINT_COMMITTED",(
            first,process.poll(),process.stderr.read() if process.poll() is not None else "")
    finally:
        process.kill()
        process.wait(timeout=10)
        process.stdout.close()
        process.stderr.close()
    assert process.returncode!=0
    pinned=json.loads((tmp_path/"pinned-manifest.json").read_text("utf8"))
    assert pinned["position"]==1
    external_head=pinned["checkpoint_digest"]
    recovered=subprocess.run(
        cmd+["resume",str(tmp_path),external_head],env=env,
        capture_output=True,text=True,check=True,timeout=35)
    result=json.loads(recovered.stdout.strip())
    assert result["position"]==2
    assert result["attention_kv_positions"]==2
    continuous=QwenQualifiedHybridBlock(runners(nav_hybrid()))
    continuous.step([.5,-.2,.3],token_id=63204)
    expected,receipt=continuous.step([-.1,.2,.8],token_id=628)
    assert result["output"]==expected
    assert result["chain_head"]==receipt["chain_head"]
