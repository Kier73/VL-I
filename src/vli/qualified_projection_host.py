"""Host-side bounded HTTP prefetch and vectorized qualified matrix projection.

This module is NOT the 64 KiB interpreter. It belongs to a weight-bearing
model host. The VL64 QualifiedShardNavigator still requests and receives
<=65536-byte exact slices, and validates each slice's SHA256. Original source
HTTP 206 observations are separately hashed and accounted.

A block from a pinned HTTP source can serve multiple logical VL range glances,
but these are *not* counted as repeated remote observations. No block is
persisted inside VL-I/VL-64's Ground; eviction changes performance, not math.

This does not make trained arithmetic happen in the network. It is an
experiment to separate work at the model host from lightweight VL controls.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import math
import re
import urllib.parse
import urllib.request

from .ground import _digest, _require
from .large_model_comparison import SOURCES

OBSERVATION = "vl-i.pinned-coalesced-host-block.v1"
HOST_NAME = "qualified-coalesced-model-host"
MAX_LOGICAL_RANGE = 65536
MAX_BLOCK = 8 * 1024 * 1024
_ALLOWED = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.safetensors\Z")
_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)\Z")


def _host_allowed(url):
    parsed=urllib.parse.urlparse(url)
    name=parsed.hostname or ""
    return (parsed.scheme=="https" and
            (name in ("huggingface.co","hf.co") or
             name.endswith(".huggingface.co") or name.endswith(".hf.co")))


def hf_block_fetch(model, revision, shard, begin, end):
    """Read one immutable-model HTTPS range; may return fewer EOF bytes."""
    if (model not in (v["model"] for v in SOURCES.values())
            or revision not in (v["revision"] for v in SOURCES.values())
            or not _ALLOWED.fullmatch(shard)
            or type(begin) is not int or type(end) is not int
            or not 0<=begin<end<=begin+MAX_BLOCK):
        raise ValueError("Unqualified full-scale HTTP observation")
    # Do not let the caller combine a different revision with a source.
    if not any(model==v["model"] and revision==v["revision"]
               for v in SOURCES.values()):
        raise ValueError("Original model and revision are not the same Ground")
    url=f"https://huggingface.co/{model}/resolve/{revision}/{shard}"
    request=urllib.request.Request(url,method="GET",headers={
        "Range":f"bytes={begin}-{end-1}",
        "Accept-Encoding":"identity",
        "User-Agent":"VL-I-qualified-projection-host/1.0"})
    with urllib.request.urlopen(request,timeout=75) as response:
        if response.status!=206 or not _host_allowed(response.geturl()):
            raise ValueError("Unqualified large model host or byte-range response")
        content_range=response.headers.get("Content-Range","")
        raw=response.read(MAX_BLOCK+1)
    if len(raw)>MAX_BLOCK:
        raise ValueError("Model host returned an oversized block")
    return {"status":206,"content_range":content_range,
            "payload":raw,"source_host":"pinned-huggingface-https-range"}


class BoundedModelRangeHost:
    """Adapter from evidence-verified host HTTP blocks to VL64 range glances."""

    def __init__(self, *, model, revision, fetch=hf_block_fetch,
                 block_bytes=2*1024*1024, budget_bytes=64*1024*1024):
        if (not any(model==s["model"] and revision==s["revision"]
                    for s in SOURCES.values()) or not callable(fetch)
                or type(block_bytes) is not int or not
                    MAX_LOGICAL_RANGE<=block_bytes<=MAX_BLOCK
                or type(budget_bytes) is not int or not
                    block_bytes<=budget_bytes<=1<<32):
            raise ValueError("Unqualified bounded source block host configuration")
        self.model,self.revision,self.fetch=model,revision,fetch
        self.block_bytes,self.budget_bytes=block_bytes,budget_bytes
        self._cache=OrderedDict()
        self.used_bytes=0
        self.source_reads=0
        self.source_bytes=0
        self.logical_reads=0
        self.logical_bytes=0
        self.reused_logical_reads=0
        self.evicted_blocks=0
        self.head="0"*64

    def _block(self, shard, start):
        key=(shard,start)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key],True
        reply=self.fetch(self.model,self.revision,shard,start,start+self.block_bytes)
        if (not isinstance(reply,dict) or reply.get("status")!=206 or
            reply.get("source_host")!="pinned-huggingface-https-range"
            or type(reply.get("payload")) is not bytes or not reply["payload"]):
            raise ValueError("Unqualified original HTTPS source block")
        content_range=reply.get("content_range")
        match=_CONTENT_RANGE.fullmatch(content_range or "")
        if not match:
            raise ValueError("Missing complete original source Content-Range")
        lo,hi,total=(int(x) for x in match.groups())
        raw=reply["payload"]
        if not (lo==start and hi==start+len(raw)-1 and
                len(raw)<=self.block_bytes and hi<total and total>0):
            raise ValueError("Source block is incomplete or not exact")
        sha=hashlib.sha256(raw).hexdigest()
        observed={
            "source_model":self.model,"revision":self.revision,
            "shard":shard,"byte_start":lo,"byte_end_exclusive":hi+1,
            "source_file_total_bytes":total,
            "source_sha256":sha,
            "source_host":"pinned-huggingface-https-range",
        }
        next_head=_digest(OBSERVATION,{"previous":self.head,"observation":observed})
        while self._cache and self.used_bytes+len(raw)>self.budget_bytes:
            _old,removed=self._cache.popitem(last=False)
            self.used_bytes-=len(removed["raw"])
            self.evicted_blocks+=1
        block={"raw":raw,"sha256":sha,"total":total,"witness":next_head}
        self._cache[key]=block
        self.used_bytes+=len(raw)
        self.source_reads+=1
        self.source_bytes+=len(raw)
        self.head=next_head
        return block,False

    def __call__(self, request):
        if not isinstance(request,dict):
            raise ValueError("Unqualified logical shard request")
        keys={"model","revision","index_ground_digest","shard",
              "byte_begin","byte_end_exclusive","purpose"}
        if (set(request)!=keys or request["model"]!=self.model or
            request["revision"]!=self.revision or
            request["purpose"] not in ("header","tensor") or
            not _ALLOWED.fullmatch(request["shard"]) or
            type(request["byte_begin"]) is not int or
            type(request["byte_end_exclusive"]) is not int or
            not 0<=request["byte_begin"]<request["byte_end_exclusive"]
                <=request["byte_begin"]+MAX_LOGICAL_RANGE):
            raise ValueError("Unqualified VL logical byte-range operation")
        lo,hi=request["byte_begin"],request["byte_end_exclusive"]
        parts=[];cache_hit=True
        pos=lo
        while pos<hi:
            first=(pos//self.block_bytes)*self.block_bytes
            block,reused=self._block(request["shard"],first)
            off=pos-first
            stop=min(hi,first+len(block["raw"]))
            if off>=len(block["raw"]) or stop<=pos:
                raise ValueError("Weight slice exceeds qualified original shard")
            parts.append(block["raw"][off:off+(stop-pos)])
            pos=stop
            cache_hit &=reused
        payload=b"".join(parts)
        if len(payload)!=hi-lo:
            raise ValueError("Source observation did not cover full requested slice")
        self.logical_reads+=1
        self.logical_bytes+=len(payload)
        if cache_hit:self.reused_logical_reads+=1
        return {**request,"status":"QUALIFIED_EXACT_SHARD_RANGE",
                "observer_host":HOST_NAME,"sha256":hashlib.sha256(payload).hexdigest(),
                "payload":payload,
                "cache_disposition":("RETAINED_ORIGINAL_OBSERVATION"
                        if cache_hit else "FRESH_HOST_OBSERVATION")}

    def stats(self):
        return {"schema":OBSERVATION,
                "remote_https_206_blocks":self.source_reads,
                "remote_http_bytes":self.source_bytes,
                "logical_64k_ranges":self.logical_reads,
                "logical_range_bytes":self.logical_bytes,
                "source_blocks_reused_by_logical_reads":self.reused_logical_reads,
                "evicted_blocks":self.evicted_blocks,
                "resident_model_host_bytes":self.used_bytes,
                "resident_model_host_budget_bytes":self.budget_bytes,
                "network_observation_head":self.head,
                "model_weights_stored_in_VL_ground_bytes":0,
                "host_compute_happens_in_worker_not_network":True}


def fast_rows(nav, name, x, *, first=0, count=None, batch_rows=96):
    """Bounded model-worker NumPy matvec from original qualified VL ranges.

    The independent existing scalar observed_rows remains the reference.
    Every acquired input byte still passes nav.get_raw() and its 64KiB
    range receipts. The host-side matrix buffer is discarded per block.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ValueError("Host-side NumPy numeric backend not installed") from exc
    from experiments.model_fabric.scaled_model_fabric import QualifiedShardNavigator
    if not isinstance(nav,QualifiedShardNavigator):
        raise ValueError("Qualified VL original source required")
    d=nav.locate(name);shape=d["shape"]
    if (len(shape)!=2 or type(first) is not int or first<0 or
        type(batch_rows) is not int or not 1<=batch_rows<=256 or
        not isinstance(x,(list,tuple)) or len(x)!=shape[1] or not x or
        any(type(v) not in (int,float) or not math.isfinite(v) for v in x)
        or d["dtype"] not in ("BF16","F16","F32")):
        raise ValueError("Unqualified original numeric matrix/vector/extent")
    if count is None:count=shape[0]-first
    if type(count) is not int or not 0<count<=shape[0]-first:
        raise ValueError("Unqualified native matvec row selection")
    width=shape[1]
    vector=np.array(x,dtype=np.float64)
    out=[]
    per_batch=max(1,min(batch_rows,MAX_BLOCK//(width*8)))
    for start in range(first,first+count,per_batch):
        rows=min(per_batch,first+count-start)
        raw,desc=nav.get_raw(name,flat_start=start*width,count=rows*width)
        if desc!=d:
            raise ValueError("Tensor identity changed between qualified observations")
        if d["dtype"]=="BF16":
            u=np.frombuffer(raw,dtype="<u2").astype("<u4")
            u<<=16
            values=u.view("<f4")
        elif d["dtype"]=="F16":
            values=np.frombuffer(raw,dtype="<f2")
        else:
            values=np.frombuffer(raw,dtype="<f4")
        if values.size!=rows*width or not np.isfinite(values).all():
            raise ValueError("Incomplete or nonfinite trained matrix rows")
        work=values.reshape(rows,width).astype(np.float64)
        projected=work @ vector
        if projected.size!=rows or not np.isfinite(projected).all():
            raise ValueError("Nonfinite trained projection")
        out.extend(float(v) for v in projected)
    return out
