"""Synthetic token embedding fixture for SHAPE-008 controls.

None of these are original trained checkpoint bytes. Only the live HTTPS
experiment observes authentic original Qwen token embedding weights.
"""
from __future__ import annotations
import json
import struct

SHARD="model-00001-of-00020.safetensors"
EMBED="model.language_model.embed_tokens.weight"
NORM="model.language_model.layers.0.input_layernorm.weight"


def _bf16(number):
    return struct.pack("<H",struct.unpack("<I",struct.pack("<f",number))[0]>>16)


def original_shape_fixture():
    width=64;vocab=128
    left=[((j%11)-5)/32 for j in range(width)]
    right=[((j%7)-3)/16 for j in range(width)]
    embeddings=[]
    for k in range(vocab):
        values=(left if k==100 else right if k==101 else
                [((j+k)%5-2)/128 for j in range(width)])
        embeddings.extend(values)
    emb=b"".join(_bf16(v) for v in embeddings)
    norm=b"".join(_bf16((j%9)/256) for j in range(width))
    h={
        EMBED:{"dtype":"BF16","shape":[vocab,width],
               "data_offsets":[0,len(emb)]},
        NORM:{"dtype":"BF16","shape":[width],
              "data_offsets":[len(emb),len(emb)+len(norm)]},
    }
    hb=json.dumps(h,sort_keys=True,separators=(",",":")).encode()
    shard=struct.pack("<Q",len(hb))+hb+emb+norm
    index=json.dumps({
        "metadata":{"total_size":len(emb)+len(norm)},
        "weight_map":{EMBED:SHARD,NORM:SHARD},
    },sort_keys=True,separators=(",",":")).encode()
    return index,shard
