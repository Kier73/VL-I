"""INTERACT-004: a pinned original-vocabulary tokenizer Ground.

Converts real observer text to qualified model token IDs before trained-weight
acquisition. The Rust tokenizer computes from pinned source assets, not a
catalogue of expected responses. This is literal text, not model chat-template
formatting and not yet complete pretrained language inference.

Qwen assets live at its *same checkpoint revision*. DeepSeek-V4.1-Flash-1017
does not ship a root tokenizer.json in its pinned repo; the related Flash
tokenizer source is separately pinned and explicitly carries weaker provenance.
"""
from __future__ import annotations

import hashlib
import json

from .ground import _digest, _hex64, _require
from .large_model_comparison import SOURCES
from .operative_model import _check_ground

SCHEMA="vl-i.pinned-text-tokenizer-ground.v1"
INTENTION="vl-i.literal-text-intention.v1"
OBSERVATION="vl-i.literal-text-tokens.v1"
MAX_ASSET_BYTES=20*1024*1024
MAX_TEXT_BYTES=2048
MAX_TOKENS=1024

TOKENIZER_SOURCES={
  "qwen-27b":{
    "repository":"Qwen/Qwen3.8-27B",
    "revision":"37561816e66b5b9153b1b64c10968c18c1e6f4a0",
    "filename":"tokenizer.json",
    "source_grade":"SAME_PINNED_CHECKPOINT_TOKENIZER",
  },
  "deepseek-v41-flash":{
    "repository":"deepseek-ai/DeepSeek-V4.1-Flash",
    "revision":"dba1be0a40aa45a94ad051997016db3960a90277",
    "filename":"tokenizer.json",
    "source_grade":"RELATED_FLASH_RELEASE_NOT_PROVEN_IDENTICAL_TO_1017",
  },
}


def _native_tokenizer(raw):
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise ValueError("Pinned Rust tokenizers dependency missing") from exc
    try:
        return Tokenizer.from_str(raw.decode("utf-8"))
    except (ValueError,TypeError,UnicodeError) as exc:
        raise ValueError("Unqualified native tokenizer serialization") from exc


def admit(g, raw: bytes):
    """Admit one immutable tokenizer source; retain digest, not its contents."""
    _check_ground(g)
    route=g["route"]; cfg=TOKENIZER_SOURCES[route]
    _require(type(raw) is bytes and 10<=len(raw)<=MAX_ASSET_BYTES,
             "Missing/oversized tokenizer asset")
    tokenizer=_native_tokenizer(raw)
    vocab=tokenizer.get_vocab_size(with_added_tokens=True)
    _require(type(vocab) is int and 128<=vocab<=1_000_000,
             "Unqualified original model vocabulary")
    asset_hash=hashlib.sha256(raw).hexdigest()
    body={"schema":SCHEMA,"model":g["model"],"model_revision":g["revision"],
          "model_ground_digest":g["ground_digest"],
          "asset_repository":cfg["repository"],
          "asset_revision":cfg["revision"],"asset_file":cfg["filename"],
          "asset_sha256":asset_hash,"asset_size_bytes":len(raw),
          "tokenizer_vocab_size":vocab,"source_grade":cfg["source_grade"],
          "input_scope":"literal_text_no_special_tokens_no_chat_template",
          "trained_weight_bytes_stored":0}
    return {**body,"ground_digest":_digest(SCHEMA,body)},tokenizer


def text_intention(text_ground, text: str):
    _require(type(text_ground) is dict and text_ground.get("schema")==SCHEMA
             and _hex64(text_ground.get("ground_digest"))
             and text_ground["ground_digest"]==_digest(SCHEMA,{
                 k:v for k,v in text_ground.items() if k!="ground_digest"}),
             "Tokenizer Ground identity altered")
    _require(type(text) is str and text and
             len(text.encode("utf-8"))<=MAX_TEXT_BYTES
             and "\0" not in text,"Unqualified literal text intention")
    body={"schema":INTENTION,
          "tokenizer_ground_digest":text_ground["ground_digest"],
          "text":text,
          "mode":"literal_raw_no_chat_template",
          "model":text_ground["model"],
          "model_revision":text_ground["model_revision"]}
    return {**body,"intent_digest":_digest(INTENTION,body)}


def tokenize(text_ground,tokenizer,intent):
    """Execute actual BPE or other tokenizer law from admitted source bytes."""
    _require(type(intent) is dict and intent.get("schema")==INTENTION
             and _hex64(intent.get("intent_digest"))
             and intent["intent_digest"]==_digest(INTENTION,{
                  k:v for k,v in intent.items() if k!="intent_digest"})
             and intent==text_intention(text_ground,intent.get("text")),
             "Text intention changed before tokenization")
    from tokenizers import Tokenizer
    _require(isinstance(tokenizer,Tokenizer)
             and tokenizer.get_vocab_size(with_added_tokens=True)
                 ==text_ground["tokenizer_vocab_size"],
             "Tokenizer runtime differs from admitted source")
    encoded=tokenizer.encode(intent["text"],add_special_tokens=False)
    ids=encoded.ids
    _require(1<=len(ids)<=MAX_TOKENS
             and all(type(i) is int and 0<=i<text_ground["tokenizer_vocab_size"]
                     for i in ids),
             "Native tokenizer produced invalid/excessive token IDs")
    recovered=tokenizer.decode(ids,skip_special_tokens=False)
    _require(recovered==intent["text"],
             "Literal round-trip tokenizer normalization not admitted")
    # Offsets are character positions in the original text. They qualify
    # segmentation, not the model's chat-template semantics.
    offsets=[[int(a),int(b)] for a,b in encoded.offsets]
    _require(len(offsets)==len(ids)
             and all(0<=a<=b<=len(intent["text"]) for a,b in offsets),
             "Unqualified original tokenizer offsets")
    body={"schema":OBSERVATION,
          "model":text_ground["model"],
          "model_revision":text_ground["model_revision"],
          "model_ground_digest":text_ground["model_ground_digest"],
          "tokenizer_ground_digest":text_ground["ground_digest"],
          "intent_digest":intent["intent_digest"],
          "text_sha256":hashlib.sha256(intent["text"].encode()).hexdigest(),
          "token_ids":list(ids),"token_offsets":offsets,
          "token_count":len(ids),"decoded_roundtrip_exact":True,
          "source_grade":text_ground["source_grade"],
          "scope":"native_literal_tokenization_not_chat_prefill_or_full_inference"}
    return {**body,"observation_digest":_digest(OBSERVATION,body)}


def validate_observation(g,tg,intent,observation):
    """Admit a token transcript without accepting a caller-supplied re-label."""
    _check_ground(g)
    _require(tg.get("model_ground_digest")==g["ground_digest"]
             and tg.get("model")==g["model"]
             and tg.get("model_revision")==g["revision"],
             "Tokenizer/model source mismatch")
    _require(type(observation) is dict
             and observation.get("schema")==OBSERVATION
             and _hex64(observation.get("observation_digest"))
             and observation["observation_digest"]==_digest(OBSERVATION,{
                   k:v for k,v in observation.items() if k!="observation_digest"})
             and observation.get("model_ground_digest")==g["ground_digest"]
             and observation.get("tokenizer_ground_digest")==tg["ground_digest"]
             and observation.get("intent_digest")==intent["intent_digest"]
             and observation.get("source_grade")==tg["source_grade"]
             and observation.get("token_count")==len(observation.get("token_ids",[]))
             and observation.get("decoded_roundtrip_exact") is True
             and observation.get("text_sha256")==hashlib.sha256(
                   intent["text"].encode("utf-8")).hexdigest(),
             "Unqualified tokenizer/intent receipt")
    return observation
