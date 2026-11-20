"""Binary historical-attention Ground: one navigable coordinate per head.

For exactly two causal KV entries, normalized attention has one degree of
freedom. The *difference* between two QK scores is the natural coordinate:
    delta = (q dot k_prior - q dot k_current) / sqrt(head_dim)
    p_prior = sigmoid(delta)
    context = v_current + p_prior*(v_prior - v_current)

A trained *source operator* still needs to realize delta and the V
endpoints. This quotient only describes and reconstructs that conditional
relation; it is not proof of eliminating any matrix arithmetic, and does
not generalize to longer histories without a different Ground.

This is a VL-I research adapter, not a modification to VL-64.
"""
from __future__ import annotations

from hashlib import sha256
import math

from .ground import _digest,_hex64

SCHEMA="vl-i.binary-causal-attention-coordinate.v1"
SOURCE="vl-i.attested-two-position-softmax.v1"


def _sigmoid(value):
    if value>=0:
        return 1/(1+math.exp(-value))
    exp=math.exp(value)
    return exp/(1+exp)


def binary_softmax_source_digest(probability_pairs):
    """Identify the exact measured attention probability evidence."""
    if (not isinstance(probability_pairs,(list,tuple)) or
        not 1<=len(probability_pairs)<=1024 or
        any(not isinstance(pair,(list,tuple)) or len(pair)!=2 or
            any(type(x) not in (int,float) or not math.isfinite(x) or
                not 0<x<1 for x in pair) or
            not math.isclose(math.fsum(pair),1.0,rel_tol=0,abs_tol=2e-12)
            for pair in probability_pairs)):
        raise ValueError("Incomplete or unqualified two-position attention simplex")
    return _digest(SOURCE,[[float(x) for x in pair] for pair in probability_pairs])


def admit_binary_attention_quotient(probability_pairs, *,
                                    expected_source_digest):
    """Collapse complete two-position softmax to one score-gap coordinate.

    Caller must pin the independently observed source probability
    record. The quotient is algebraically reconstructible, but source
    authenticity remains a separate upstream admission responsibility.
    """
    actual=binary_softmax_source_digest(probability_pairs)
    if not _hex64(expected_source_digest) or expected_source_digest!=actual:
        raise ValueError("Missing external source observation pin")
    coordinates=[]
    errors=[]
    for p_previous,p_current in probability_pairs:
        p0,p1=float(p_previous),float(p_current)
        delta=math.log(p0)-math.log(p1)
        a=_sigmoid(delta)
        b=1-a
        if not math.isfinite(delta):
            raise ValueError("Nonfinite binary attention coordinate")
        errors.append(max(abs(a-p0),abs(b-p1)))
        coordinates.append(delta)
    error=max(errors)
    if error>2e-12:
        raise ValueError("Binary relation does not reconstruct observed softmax")
    body={
        "schema":SCHEMA,
        "source_probability_digest":actual,
        "causal_history_positions":2,
        "head_count":len(coordinates),
        "head_score_difference_coordinates":coordinates,
        "max_probability_reconstruction_error":error,
        "original_probability_scalar_count":2*len(coordinates),
        "quotient_probability_coordinate_count":len(coordinates),
        "not_a_claim_of_eliminated_weight_projections":True,
        "not_a_claim_of_full_attention_state_compression":True,
        "scope":"exact two-position softmax simplex quotient conditional on admitted trained qk scores and value endpoints",
    }
    return {**body,"quotient_digest":_digest(SCHEMA,body)}


def reconstruct_binary_attention(quotient, *, head):
    """Reconstruct exact two-key softmax probabilities from one coordinate."""
    if (not isinstance(quotient,dict) or
        not _hex64(quotient.get("quotient_digest")) or
        _digest(SCHEMA,{k:v for k,v in quotient.items()
               if k!="quotient_digest"})!=quotient["quotient_digest"] or
        quotient.get("schema")!=SCHEMA or
        quotient.get("causal_history_positions")!=2 or
        not isinstance(quotient.get("head_score_difference_coordinates"),list) or
        type(head) is not int or
        not 0<=head<len(quotient["head_score_difference_coordinates"])):
        raise ValueError("Unadmitted binary historical-attention Ground")
    delta=quotient["head_score_difference_coordinates"][head]
    if type(delta) not in (int,float) or not math.isfinite(delta):
        raise ValueError("Nonfinite binary attention coordinate")
    prior=_sigmoid(delta)
    return [prior,1-prior]


def realize_binary_context(quotient,*,head,prior_values,current_values):
    """Realize V-current + p-prior times the signed historical V difference."""
    probabilities=reconstruct_binary_attention(quotient,head=head)
    if (not isinstance(prior_values,(list,tuple)) or
        not isinstance(current_values,(list,tuple)) or
        not 1<=len(prior_values)==len(current_values)<=32768 or
        any(type(x) not in (int,float) or not math.isfinite(x)
            for x in list(prior_values)+list(current_values))):
        raise ValueError("Missing valid causal historical value endpoints")
    p=probabilities[0]
    output=[float(now)+p*(float(earlier)-float(now))
            for earlier,now in zip(prior_values,current_values)]
    if not all(math.isfinite(x) for x in output):
        raise ValueError("Nonfinite binary attention realization")
    return output
