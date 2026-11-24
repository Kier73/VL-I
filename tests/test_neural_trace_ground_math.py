"""PBG-006 source-trace mathematical tests using explicitly synthetic tapes.

These unit tests monkeypatch the native-only verifier. The native Qwen
observation and source-specific receipt checks live in the integration tests.
"""
from fractions import Fraction as Q
from itertools import product
import pytest

from vli.neural_trace_ground import (
    NeuralObserver, CausalTape, WitnessedNeuralTraceEnsemble, ALPHABET,
)
from vli.source_hankel_ground import (
    HankelLearner, certify_source_equivalence, source_trace_probability,
)


def fixture(monkeypatch, *, altered=False):
    monkeypatch.setattr(CausalTape,"verify", lambda self: True)
    obs=NeuralObserver(0,0.)
    samples=(("0","0","1"),("0","1","0"),("1","1","0"))
    if altered:
        samples=(("1","0","1"),*samples[1:])
    tapes=tuple(CausalTape(
        'synthetic-fixture','s1','a'*64,'b'*64,obs,
        ((100,(.1,)),)*len(words),tuple({"fixture":1} for _ in words),
        words,chr(100+i)*64) for i,words in enumerate(samples))
    return WitnessedNeuralTraceEnsemble(tapes)


def close_ground(exp, budget=4000):
    atlas=exp.atlas(budget)
    learn=HankelLearner(atlas)
    candidate=learn.build()
    initial=candidate.rank
    witnesses=[]
    for _ in range(16):
        proof=exp.certify(candidate)
        if proof['status']=='EQUIVALENT_ALL_WORDS':
            return initial,candidate,atlas,witnesses
        word=proof['word']
        witnesses.append(word)
        candidate=learn.counterexample(word)
    raise AssertionError('Finite observed source did not converge')


def test_native_trace_uniform_mixture_is_exact_rational_source(monkeypatch):
    exp=fixture(monkeypatch)
    assert exp.probability(())==1
    assert exp.probability(('sense:readout:0',))==Q(2,3)
    assert exp.probability(('sense:readout:1',))==Q(1,3)
    assert exp.probability(('act:step',)*3+('sense:readout:2',))==1
    assert exp.probability(('act:step','sense:readout:1'))==Q(2,3)
    for depth in range(5):
        for word in product(ALPHABET, repeat=depth):
            assert exp.probability(word)==source_trace_probability(exp.source,word)


def test_hankel_derives_new_relations_from_qualified_trace_questions(monkeypatch):
    exp=fixture(monkeypatch)
    initial,learned,atlas,witnesses=close_ground(exp)
    assert len(witnesses)>0
    assert learned.rank>initial
    assert exp.certify(learned)['status']=='EQUIVALENT_ALL_WORDS'
    assert learned.source_digest==exp.source.digest
    assert len(atlas.cache)>0 and atlas.calls>=2
    for depth in range(5):
        for word in product(ALPHABET, repeat=depth):
            assert learned.probability(word)==exp.probability(word)


def test_trace_source_revision_changes_native_evidence_identity(monkeypatch):
    a=fixture(monkeypatch)
    b=fixture(monkeypatch,altered=True)
    assert a.native_witness_digest==b.native_witness_digest  # deliberately same fake receipt list
    assert a.source.digest!=b.source.digest
    rec=a.read((('sense:readout:0',),))[0]
    assert rec.source_digest!=b.source.digest
    with pytest.raises(ValueError):
        b.atlas().reader((('unknown:measurement',),))


def test_inadmissible_projection_and_hypothesis(monkeypatch):
    with pytest.raises(ValueError):NeuralObserver(-1)
    with pytest.raises(ValueError):NeuralObserver(0,float('inf'))
    exp=fixture(monkeypatch)
    with pytest.raises(ValueError):exp.probability(('act:unknown',))
    with pytest.raises(ValueError):exp.atlas(1).observe_many([(),('act:step',)])
