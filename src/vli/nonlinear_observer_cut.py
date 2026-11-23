"""NOC-001: observer-first nonlinear feed-forward cut (NumPy, opt-in).

Given a readout a, residual u, normalized MLP input x, and a gated MLP
v=SiLU(Gx)*(Px), the requested real-arithmetic score is
    a @ (u + D v) = a @ u + (D.T @ a) @ v.

A source-independent Cauchy upper bound encloses omitted channels. The
caller can choose an additional error guard for numerical semantics. An
identified sign is a real-arithmetic statement about the compiled Ground;
agreement with a specific FP32 model runtime is checked by a separate
reference, not inferred from the bound. No neural model server is called.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


def _np():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Install the optional VL-I numerical extra for NOC-001") from exc
    return np


def _vector(x, *, name, n=None):
    np = _np()
    v = np.asarray(x, dtype=np.float64)
    if v.ndim != 1 or (n is not None and len(v) != n) or not np.isfinite(v).all():
        raise ValueError(f"Invalid finite {name} vector")
    return v


def _matrix(x, *, name, shape):
    np = _np()
    v = np.asarray(x, dtype=np.float64)
    if v.ndim != 2 or v.shape != shape or not np.isfinite(v).all():
        raise ValueError(f"Invalid finite {name} matrix shape")
    return v


def _silu(t):
    # Scalar stable sigmoid, retaining negative-saturation without exp overflow.
    if t >= 0:
        return t/(1.0 + math.exp(-t))
    e = math.exp(t)
    return t*e/(1.0+e)


@dataclass(frozen=True)
class NonlinearObserverCut:
    """Single requested score, source-derived readout, and MLP row norms."""
    a: object
    b: object
    gate: object
    up: object
    g_l2: object
    p_l2: object
    g_l1: object
    p_l1: object

    @property
    def width(self):
        return len(self.a)

    @property
    def channels(self):
        return len(self.b)


def compile_nonlinear_cut(a, down, gate, up):
    """Compile a=D-derived score pullback and source row-norm information."""
    np = _np()
    a = _vector(a,name="readout")
    if not len(a):
        raise ValueError("Readout must be nonempty")
    down = np.asarray(down, dtype=np.float64)
    if down.ndim != 2 or down.shape[0] != len(a) or down.shape[1] < 1:
        raise ValueError("Invalid down-projection geometry")
    width, channels = down.shape
    down = _matrix(down, name="down", shape=(width,channels))
    gate = _matrix(gate, name="gate", shape=(channels,width))
    up = _matrix(up, name="up", shape=(channels,width))
    b = down.T @ a
    g_l2 = np.linalg.norm(gate,axis=1)
    p_l2 = np.linalg.norm(up,axis=1)
    g_l1 = np.sum(np.abs(gate),axis=1)
    p_l1 = np.sum(np.abs(up),axis=1)
    if not all(np.isfinite(z).all() for z in (b,g_l2,p_l2,g_l1,p_l1)):
        raise ValueError("Nonfinite compiled Ground coefficients")
    return NonlinearObserverCut(a,b,gate,up,g_l2,p_l2,g_l1,p_l1)


def evaluate_cut(compiled, x, u, *, numerical_guard=0.0, order=None):
    """Attempt to settle sign without evaluating all nonlinear MLP channels.

    Bounds use min(L2*||x||2, L1*||x||inf) per source row and
    |SiLU(t)|<=|t|. A 1e-10 multiplicative pad protects typical F64
    reductions but is not a formal correctly-rounded interval library.
    status=DECIDED_REAL_ARITHMETIC reports the stated bound for the
    compiled coefficients, NOT exact equivalence to a CPU FP32 oracle.

    Count only evaluated gate/up rows. For unresolved ties, return status
    UNRESOLVED and leave downstream execution to the caller's fallback.
    """
    np = _np()
    if not isinstance(compiled, NonlinearObserverCut):
        raise ValueError("A compiled nonlinear observer is required")
    x = _vector(x,name="MLP input",n=compiled.width)
    u = _vector(u,name="attention residual",n=compiled.width)
    if type(numerical_guard) not in (float,int) or not math.isfinite(numerical_guard) or numerical_guard < 0:
        raise ValueError("Invalid declared numerical guard")
    x2 = float(np.linalg.norm(x))
    xinf = float(np.max(np.abs(x)))
    gbound = np.minimum(compiled.g_l2*x2,compiled.g_l1*xinf)
    pbound = np.minimum(compiled.p_l2*x2,compiled.p_l1*xinf)
    bound = np.abs(compiled.b) * gbound * pbound * (1.0+1e-10)
    if not np.isfinite(bound).all():
        raise ValueError("Unbounded nonlinear observer influence")
    if order is None:
        choices = sorted(range(compiled.channels),key=lambda k:(-float(bound[k]),k))
    else:
        choices = list(order)
        if (len(choices) != compiled.channels or
                any(type(k) is not int or not 0<=k<compiled.channels for k in choices) or
                len(set(choices)) != compiled.channels):
            raise ValueError("Channel order is not a permutation")
    base = float(compiled.a @ u)
    terms = []
    work = 0
    remaining = math.fsum(float(q) for q in bound)
    # Use conservative allowance for FP64 dot/sum evaluation; caller supplies
    # a stronger guard if its numerical contract requires it.
    for place in range(compiled.channels+1):
        partial = math.fsum((base,*terms))
        # Absolute error allowance for finite-precision measurement of score.
        numerical_envelope = (float(numerical_guard) +
            128*2.220446049250313e-16*(abs(base)+math.fsum(map(abs,terms))+1.0))
        if abs(partial) > remaining + numerical_envelope:
            return {"status":"DECIDED_REAL_ARITHMETIC", "sign":1 if partial>0 else -1,
                    "channels_evaluated":work, "channels_total":compiled.channels,
                    "partial_score":partial, "remaining_bound":remaining,
                    "numeric_guard":numerical_envelope,
                    "gate_up_dot_products_avoided":2*(compiled.channels-work),
                    "source_execution_calls":0}
        if place == compiled.channels:
            break
        k = choices[place]
        g = float(compiled.gate[k] @ x)
        p = float(compiled.up[k] @ x)
        terms.append(float(compiled.b[k])*_silu(g)*p)
        remaining = max(0.0, math.fsum(float(bound[j]) for j in choices[place+1:]))
        work += 1
    return {"status":"UNRESOLVED_NUMERICAL_MARGIN", "sign":None,
            "channels_evaluated":work,"channels_total":compiled.channels,
            "partial_score":math.fsum((base,*terms)), "remaining_bound":remaining,
            "numeric_guard":numerical_envelope,
            "gate_up_dot_products_avoided":0, "source_execution_calls":0}
