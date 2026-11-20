"""Adversarial tests for bounded host coalescing and fast typed projections.

The worker may fetch 8 MiB HTTP blocks, but the VL64 navigator must see
<=64 KiB logical slices; no source block may be relabelled a remote HTTP
observation on its second use.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from unittest.mock import patch

import numpy as np
import pytest

from experiments.model_fabric.scaled_model_fabric import QualifiedShardNavigator
from experiments.model_fabric.qualified_subgraphs import observed_rows
from vli.large_model_comparison import SOURCES
from vli.qualified_projection_host import (
    HOST_NAME, BoundedModelRangeHost, fast_rows, MAX_LOGICAL_RANGE,
)


def bf16(v):
    words=[]
    for z in v:
        bits=struct.unpack('<I',struct.pack('<f',float(z)))[0]
        words.append(bits>>16)
    return struct.pack('<'+str(len(words))+'H',*words)


def source(*,padding=False):
    tensors={
      'layer.linear.weight':('BF16',[18,7],
       bf16([(r-9)*.03125+(c-3)*.125 for r in range(18) for c in range(7)])),
      'layer.f32.weight':('F32',[5,7],
       struct.pack('<35f',*[(r-2)*.03125+(c-3)*.125
                            for r in range(5) for c in range(7)])),
    }
    if padding:
        tensors['padding']=('U8',[132000],bytes([0x44])*132000)
    header={};cursor=0
    for name,(dtype,shape,raw) in tensors.items():
        header[name]={'dtype':dtype,'shape':shape,
                      'data_offsets':[cursor,cursor+len(raw)]}
        cursor+=len(raw)
    encoded=json.dumps(header,separators=(',',':')).encode()
    file=struct.pack('<Q',len(encoded))+encoded+b''.join(v[2] for v in tensors.values())
    repo=SOURCES['qwen-27b']
    index=json.dumps({'metadata':{'total_size':cursor},
        'weight_map':{k:'model-00001-of-00018.safetensors'
                      for k in tensors}}).encode()
    calls=[]
    def fetch(model,revision,shard,start,end):
        calls.append((model,revision,shard,start,end))
        if start>=len(file):raise ValueError('Beyond file')
        end=min(end,len(file))
        return {'status':206,
                'content_range':f'bytes {start}-{end-1}/{len(file)}',
                'payload':file[start:end],
                'source_host':'pinned-huggingface-https-range'}
    host=BoundedModelRangeHost(model=repo['model'],revision=repo['revision'],
          fetch=fetch,block_bytes=65536,budget_bytes=65536)
    nav=QualifiedShardNavigator(model=repo['model'],revision=repo['revision'],
         index_raw=index,range_host=host,approved_hosts={HOST_NAME})
    return nav,host,calls,file,fetch


def test_block_coalescing_means_fewer_real_http_requests():
    nav,host,calls,file,_=source()
    a=nav.get_raw('layer.linear.weight')[0]
    b=nav.get_raw('layer.f32.weight')[0]
    assert len(a)==18*7*2 and len(b)==5*7*4
    assert len(calls)==1
    assert host.stats()['remote_https_206_blocks']==1
    assert host.stats()['logical_64k_ranges']>1
    assert host.stats()['source_blocks_reused_by_logical_reads']>0
    assert host.stats()['model_weights_stored_in_VL_ground_bytes']==0
    assert host.stats()['resident_model_host_bytes']==len(file)


def test_fast_bf16_matches_independent_original_scalar():
    nav,host,_,_,_=source()
    x=[.125,-.25,.75,-.1,.6,.8,-.4]
    actual=fast_rows(nav,'layer.linear.weight',x,batch_rows=3)
    reference=observed_rows(nav,'layer.linear.weight',x)
    assert actual==pytest.approx(reference,abs=1e-12)
    assert len(actual)==18 and nav.weight_bytes>0
    assert host.stats()['remote_https_206_blocks']==1


def test_fast_f32_admits_partial_rows():
    nav,*_=source()
    x=[.25]*7
    values=fast_rows(nav,'layer.f32.weight',x,first=1,count=3,
                     batch_rows=2)
    reference=observed_rows(nav,'layer.f32.weight',x,first=1,count=3)
    assert values==pytest.approx(reference,abs=1e-12)


def test_host_wrong_revision_and_oversized_logical_request_refused():
    nav,host,*_=source()
    before=host.stats()['remote_https_206_blocks']
    request={
      'model':nav.index['model'],'revision':nav.index['revision'],
      'index_ground_digest':nav.index['ground_digest'],
      'shard':'model-00001-of-00018.safetensors',
      'byte_begin':0,'byte_end_exclusive':10,'purpose':'header'}
    with pytest.raises(ValueError,match='Unqualified'):
        host({**request,'revision':'f'*40})
    with pytest.raises(ValueError,match='Unqualified'):
        host({**request,'byte_end_exclusive':65537})
    assert host.stats()['remote_https_206_blocks']==before


@pytest.mark.parametrize('broken',[
    {'status':200}, {'source_host':'unknown'},
    {'content_range':'bytes 2-3/300'},
    {'content_range':'bytes 0-20/7'},
    {'payload':b''},
])
def test_wrong_HTTP_range_is_refused_before_VL_publication(broken):
    nav,host,calls,file,fetch=source()
    def bad(*args):
        original=fetch(*args)
        original.update(broken)
        return original
    host.fetch=bad
    with pytest.raises(ValueError):
        nav.get_raw('layer.linear.weight')
    assert host.stats()['remote_https_206_blocks']==0
    assert nav.weight_bytes==0


def test_evict_host_side_without_mutating_VL_history():
    nav,host,calls,file,_=source(padding=True)
    start=nav.range_count
    nav.get_raw('padding',flat_start=0,count=110000)
    assert host.stats()['evicted_blocks']>0
    assert host.stats()['resident_model_host_bytes']<=65536
    nav.get_raw('layer.linear.weight')
    assert host.stats()['remote_https_206_blocks']>2
    assert nav.range_count>start
    assert len(nav.chain)==64


def test_fast_projection_does_not_accept_wrong_width_or_nonfinite_data():
    nav,host,*_=source()
    with pytest.raises(ValueError,match='Unqualified'):
        fast_rows(nav,'layer.linear.weight',[0]*6)
    with pytest.raises(ValueError,match='Unqualified'):
        fast_rows(nav,'layer.linear.weight',[float('nan')]*7)
    with pytest.raises(ValueError,match='Unqualified'):
        fast_rows(nav,'layer.linear.weight',[0]*7,batch_rows=300)
    assert host.stats()['remote_https_206_blocks']==0


def test_worker_source_observation_chain_changes_only_after_real_HTTP():
    nav,host,calls,*_=source()
    nav.get_raw('layer.linear.weight')
    first=host.stats()['network_observation_head']
    nav.get_raw('layer.linear.weight')
    assert host.stats()['network_observation_head']==first
    assert host.stats()['remote_https_206_blocks']==1
    assert host.stats()['source_blocks_reused_by_logical_reads']>1
