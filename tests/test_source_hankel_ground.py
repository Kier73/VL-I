"""PBG-005: exact source-query learned law, structural counterexample + proof."""
from fractions import Fraction as Q
from dataclasses import replace
from itertools import product
import pytest

from vli.source_hankel_ground import (
    SourceObservation, QueryAtlas, HankelLearner, LearnedGround, digest,
    source_trace_probability, verify_source_closure,
    find_counterexample, certify_source_equivalence,
)
from vli.intention_predictive_ground import ControlledSource


def delayed_source(delay=2):
    n=delay+1
    initial=(Q(1),)+tuple(Q(0) for _ in range(delay))
    goal=tuple(Q(int(i==delay)) for i in range(n))
    advance=[]
    for i in range(n):
        row=[Q() for _ in range(n)];row[min(n-1,i+1)]=Q(1)
        advance.append(tuple(row))
    sensors=[]
    for label in ('0','1'):
        rows=[]
        for i in range(n):
            row=[Q() for _ in range(n)]
            if (i==delay)==(label=='1'):
                row[i]=Q(1)
            rows.append(tuple(row))
        sensors.append((label,tuple(rows)))
    return ControlledSource(initial,goal,(('step',tuple(advance),Q(1,100)),),
                            (('goal',tuple(sensors),Q(1,100)),))


def alphabet():
    return ('act:step','sense:goal:0','sense:goal:1')


def probe(source,*,wrong_word=None,bad_hash=False):
    """Injected source reader, not access to source matrices inside learner."""
    def query(words):
        return tuple(SourceObservation(w,
                  (Q(0) if w==wrong_word else source_trace_probability(source,w)),
                  ('0'*64 if bad_hash else source.digest),
                  digest({'question':list(w),'source':source.digest})) for w in words)
    return query


def make_learner(source):
    atlas=QueryAtlas(alphabet(),source.digest,probe(source),max_queries=1300)
    learner=HankelLearner(atlas)
    return learner,atlas


def test_delayed_observation_discovers_missing_predictive_direction():
    source=delayed_source(2)
    learner,atlas=make_learner(source)
    first=learner.build()
    assert first.rank==1
    assert first.probability(('act:step',)*2+('sense:goal:1',))==0
    proof=certify_source_equivalence(source,first)
    assert proof['status']=='COUNTEREXAMPLE'
    assert proof['word']==('act:step','act:step','sense:goal:0')
    assert proof['actual']==0 and proof['predicted']==1
    second=learner.counterexample(proof['word'])
    assert second.rank==3
    assert second.ground_digest!=first.ground_digest
    assert verify_source_closure(source,second)
    assert certify_source_equivalence(source,second)['status']=='EQUIVALENT_ALL_WORDS'
    assert second.probability(proof['word'])==0
    assert first.probability(proof['word'])==1 # parent remains immutable
    assert atlas.calls>0 and len(atlas.cache)>1


def test_independent_all_word_oracle_certifies_with_linear_span_not_exhaustive_enumeration():
    source=delayed_source(2)
    learner,atlas=make_learner(source)
    learner.build()
    witness=find_counterexample(source,learner.ground)
    grown=learner.counterexample(witness[0])
    proof=certify_source_equivalence(source,grown)
    assert proof['status']=='EQUIVALENT_ALL_WORDS'
    assert proof['reachable_rank']<=len(source.initial)+grown.rank
    assert proof['witness_count']<=len(source.initial)+grown.rank
    for length in range(7):
        for word in product(alphabet(),repeat=length):
            assert source_trace_probability(source,word)==grown.probability(word)


def test_longer_delay_requires_another_rank_and_produces_new_source_identity():
    s3=delayed_source(3)
    learner,atlas=make_learner(s3)
    base=learner.build()
    proof=certify_source_equivalence(s3,base)
    assert proof['status']=='COUNTEREXAMPLE'
    assert proof['word']==('act:step',)*3+('sense:goal:0',)
    grown=learner.counterexample(proof['word'])
    assert grown.rank==4
    assert certify_source_equivalence(s3,grown)['status']=='EQUIVALENT_ALL_WORDS'
    assert verify_source_closure(s3,grown)
    assert s3.digest!=delayed_source(2).digest


def test_false_or_unqualified_receipt_cannot_form_source_ground():
    s=delayed_source()
    with pytest.raises(ValueError,match='bind exact'):
        HankelLearner(QueryAtlas(alphabet(),s.digest,probe(s,bad_hash=True))).build()
    atlas=QueryAtlas(alphabet(),s.digest,probe(s))
    ground=HankelLearner(atlas).build()
    assert ground.rank==1
    with pytest.raises(ValueError,match='identity'):
        verify_source_closure(delayed_source(3),ground)
    with pytest.raises(ValueError,match='closure'):
        verify_source_closure(s,ground)


def test_lookalike_and_stale_oracle_is_refused_by_source_independent_verifier():
    s=delayed_source()
    wrong=('act:step','act:step','sense:goal:1')
    atlas=QueryAtlas(alphabet(),s.digest,probe(s,wrong_word=wrong))
    l=HankelLearner(atlas)
    first=l.build()
    with pytest.raises(ValueError,match='Not a source-distinguishing'):
        l.counterexample(('sense:goal:1',))
    # A falsely reporting query is never silently promoted by independent source audit.
    with pytest.raises(ValueError,match='counterexample'):
        l.counterexample(wrong)
    assert find_counterexample(s,first)[0]==('act:step','act:step','sense:goal:0')


def test_lossless_source_observation_and_cache_no_duplicate_queries():
    s=delayed_source()
    calls=[]
    def count(words):
        calls.append(tuple(words));return probe(s)(words)
    atlas=QueryAtlas(alphabet(),s.digest,count,max_queries=10)
    a=atlas.observe_many(((),('act:step',),(),('act:step',)))
    assert a==(1,1)
    assert sum(map(len,calls))==2
    assert atlas.observe_many(((),('act:step',)))==(1,1)
    assert len(calls)==1
    with pytest.raises(ValueError,match='budget'):
        atlas.observe_many(tuple(('act:step',)*i for i in range(11)))
    with pytest.raises(ValueError):
        SourceObservation((),Q(5,4),s.digest,digest({}))


def test_evidence_tamper_and_alternative_source_not_admitted():
    s=delayed_source()
    learner,atlas=make_learner(s)
    ground=learner.build()
    assert ground.probability(())==1
    forged=replace(ground,initial=(Q(2),))
    with pytest.raises(ValueError,match='identity'):
        forged.probability(())
    with pytest.raises(ValueError,match='identity'):
        certify_source_equivalence(delayed_source(3),ground)


def test_counterexample_refinement_does_not_reexecute_any_material_action():
    s=delayed_source(2)
    native_read_count=[]
    def reader(words):
        native_read_count.append(len(words))
        return probe(s)(words)
    atlas=QueryAtlas(alphabet(),s.digest,reader)
    learner=HankelLearner(atlas)
    learner.build()
    result=learner.counterexample(('act:step','act:step','sense:goal:1'))
    assert result.rank==3
    assert sum(native_read_count)==len(atlas.cache)
    assert all(isinstance(x,SourceObservation) for x in probe(s)(((),)))
    # These are *source trace probability queries*, not physical actuation.
