"""PBG-001: independent exact future-output and persistence checks."""
from dataclasses import replace
from fractions import Fraction as F
from itertools import product
import pytest

from vli.predictive_basis_ground import (
    StochasticSource, PredictiveSession, compile_predictive_ground,
    verify_predictive_ground, _hash,
)


def redundant_source(n=8):
    assert n % 2 == 0
    zero, one = [], []
    for i in range(n):
        p = F(3,4) if i % 2 == 0 else F(1,4)
        r0, r1 = [F(0)]*n, [F(0)]*n
        r0[(i+2)%n] = p
        r1[(i+1)%n] = 1-p
        zero.append(r0); one.append(r1)
    return StochasticSource(("0","1"), (tuple(map(tuple,zero)),tuple(map(tuple,one))),
                            (F(1),)+(F(0),)*(n-1))


def test_exact_rank_reduction_and_unseen_long_continuations():
    source = redundant_source(8)
    ground = compile_predictive_ground(source)
    assert ground.rank == 2 < len(source.initial)
    assert verify_predictive_ground(source, ground)
    for length in range(9):
        for word in product(source.actions, repeat=length):
            session = PredictiveSession(ground)
            prefix_p = F(1)
            for token in word:
                assert session.distribution() == source.next_distribution(session.history)
                prefix_p *= session.observe(token)
            assert prefix_p == source.word_probability(word)
            assert session.distribution() == source.next_distribution(word)
    assert ground.rank == 2


def test_snapshot_rehome_and_source_free_continuation():
    ground = compile_predictive_ground(redundant_source())
    one = PredictiveSession(ground)
    for a in '101101001001101001':
        one.observe(a)
    snap = one.snapshot()
    two = PredictiveSession.restore(ground, snap, expected_digest=snap['snapshot_digest'])
    assert two.state == one.state
    for a in '001111001':
        assert two.distribution() == one.distribution()
        assert two.observe(a) == one.observe(a)
    assert one.state == two.state


def test_changed_source_changes_ground_and_bad_certificates_refuse():
    source = redundant_source()
    ground = compile_predictive_ground(source)
    kernels = [[list(row) for row in mat] for mat in source.kernels]
    # Split two formerly equivalent hidden states with an altered emission.
    kernels[0][2][4] -= F(1,8)
    kernels[1][2][3] += F(1,8)
    changed = StochasticSource(source.actions, tuple(tuple(tuple(row) for row in mat)
                                        for mat in kernels), source.initial)
    with pytest.raises(ValueError, match='source|Source|Wrong'):
        verify_predictive_ground(changed, ground)
    changed_ground = compile_predictive_ground(changed)
    assert changed_ground.rank > ground.rank
    bad = replace(ground, transforms=tuple(tuple(tuple(v+1 if i==j==0 else v
                                for j,v in enumerate(row)) for i,row in enumerate(mat))
                                for mat in ground.transforms))
    with pytest.raises(ValueError, match='identity'):
        verify_predictive_ground(source, bad)
    # Rehashing a forged transition still fails the independent source check.
    bad = replace(bad, ground_digest=_hash(bad.material()))
    with pytest.raises(ValueError, match='commute|closure'):
        verify_predictive_ground(source, bad)


def test_zero_probability_refusal_and_snapshot_mutation():
    source = StochasticSource(('0','1'), (((F(1),),), ((F(0),),)), (F(1),))
    ground = compile_predictive_ground(source)
    session = PredictiveSession(ground)
    assert session.distribution() == {'0':F(1),'1':F(0)}
    with pytest.raises(ValueError, match='zero-probability'):
        session.observe('1')
    assert session.state == ground.initial
    session.observe('0')
    snap=session.snapshot()
    forged={**snap, 'state':['2/1']}
    forged['snapshot_digest']=_hash({k:v for k,v in forged.items() if k!='snapshot_digest'})
    with pytest.raises(ValueError, match='replay'):
        PredictiveSession.restore(ground,forged, expected_digest=forged['snapshot_digest'])
    with pytest.raises(ValueError, match='Unpinned'):
        PredictiveSession.restore(ground,snap, expected_digest='0'*64)
    with pytest.raises(ValueError, match='Unqualified'):
        session.observe('unexpected')


def test_inputs_reject_float_or_invalid_normalization():
    with pytest.raises(ValueError, match='stochastic source kernels'):
        StochasticSource(('x',), (((0.9,),),), (1,))
    with pytest.raises(ValueError, match='normalized'):
        StochasticSource(('x',), (((F(2,3),),),), (1,))
