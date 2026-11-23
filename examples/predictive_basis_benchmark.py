"""PBG-001: no-network exact-rational token-generation benchmark.

Run: PYTHONPATH=src python -m examples.predictive_basis_benchmark
"""
from fractions import Fraction as F
from itertools import product
import json
import time

from vli.predictive_basis_ground import (
    StochasticSource, compile_predictive_ground, PredictiveSession,
)


def example_source(n=8):
    a,b=[],[]
    for i in range(n):
        q=F(3,4) if i%2==0 else F(1,4)
        z=[F(0)]*n; o=[F(0)]*n
        z[(i+2)%n]=q
        o[(i+1)%n]=1-q
        a.append(tuple(z));b.append(tuple(o))
    return StochasticSource(("0","1"),(tuple(a),tuple(b)),
                            (F(1),)+(F(0),)*(n-1))


def run():
    source=example_source()
    before=time.perf_counter_ns()
    ground=compile_predictive_ground(source)
    compilation_ns=time.perf_counter_ns()-before
    checked=0
    for length in range(10):
        for word in product(source.actions,repeat=length):
            session=PredictiveSession(ground)
            p=F(1)
            for action in word:
                if session.distribution()!=source.next_distribution(session.history):
                    raise AssertionError("Missing source-equivalent future probability")
                p*=session.observe(action)
            if p!=source.word_probability(word) or session.distribution()!=source.next_distribution(word):
                raise AssertionError("Candidate predictive Ground differs from source")
            checked+=1
    # Timing deliberately measures the same length of *incremental* source
    # state updates and compiled Ground updates. Sparse originals may win.
    stream="010011"*16
    start=time.perf_counter_ns()
    row=source.initial
    from vli.predictive_basis_ground import _rowmat
    for action in stream:
        row=_rowmat(row,source.kernels[source.actions.index(action)])
        mass=sum(row,F())
        row=tuple(v/mass for v in row)
    source_ns=time.perf_counter_ns()-start
    session=PredictiveSession(ground)
    start=time.perf_counter_ns()
    for action in stream:
        session.observe(action)
    ground_ns=time.perf_counter_ns()-start
    if source.next_distribution(stream)!=session.distribution():
        raise AssertionError("Long unobserved continuation fails source parity")
    return {"experiment":"PBG-001","source_state_dimension":len(source.initial),
            "compiled_predictive_dimension":ground.rank,
            "original_transition_matrices":len(source.kernels),
            "sampled_action_words_verified_exact":checked,
            "maximum_exhaustive_word_length":9,
            "heldout_stream_length":len(stream),
            "ground_digest":ground.ground_digest,
            "compile_ns":compilation_ns,
            "source_incremental_transition_ns":source_ns,
            "virtual_incremental_transition_ns":ground_ns,
            "scope":"exact rational finite stochastic fixture; sparse reference; single-run timings not performance claims"}


if __name__=="__main__":
    print(json.dumps(run(),indent=2))
