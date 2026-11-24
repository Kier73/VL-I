"""PBG-007: source-free candidate discrimination and active-source admission.

Synthetic finite-source tests, not genuine Qwen numerical evidence.
The separate native suite exercises QwenCausalGDNChain, E-CPU and VGI.
"""
from dataclasses import replace
from fractions import Fraction as Q
import pytest

from vli.source_hankel_ground import HankelLearner, QueryAtlas, certify_source_equivalence
from vli.active_neural_acquisition import (
    Candidate, ActiveTraceObserver, distinguish, select_native_run,
    select_observer_attention, apply_observer_attention,
)
from test_source_hankel_ground import delayed_source, alphabet, probe
from test_neural_trace_ground_math import fixture


def learn_full(source):
    atlas=QueryAtlas(alphabet(),source.digest,probe(source),max_queries=1500)
    learner=HankelLearner(atlas)
    law=learner.build()
    for _ in range(12):
        proof=certify_source_equivalence(source,law)
        if proof['status']=='EQUIVALENT_ALL_WORDS':return law
        law=learner.counterexample(proof['word'])
    raise AssertionError('Exact-source finite learner did not converge')


def fresh_observer():
    true=delayed_source(2)
    alternative=delayed_source(3)
    expected=learn_full(true)
    other=learn_full(alternative)
    atlas=QueryAtlas(alphabet(),true.digest,probe(true),max_queries=1500)
    learner=HankelLearner(atlas)
    base=learner.build()
    assert base.rank==1
    controller=ActiveTraceObserver(atlas,learner,
        (Candidate('two-step',expected,Q(1)),Candidate('three-step',other,Q(1))))
    return true,controller,expected,other


def test_paired_exact_linear_span_finds_shortest_uncached_distinguishing_question():
    true,controller,expected,other=fresh_observer()
    diff=distinguish(expected,other)
    assert diff['status']=='CANDIDATE_DISTINCTION'
    assert diff['word']==('act:step','act:step','sense:goal:0')
    assert diff['prediction_a']==0 and diff['prediction_b']==1
    assert diff['reachable_rank']<=expected.rank+other.rank
    plan=controller.propose(nonce='1'*64)
    assert plan.word==diff['word']
    assert plan.word not in controller.atlas.cache
    result=controller.execute(plan)
    assert result['status']=='QUALIFIED_SOURCE_PROBE'
    assert result['matching_forecasts']==('two-step',)
    assert result['observed']==Q(0)
    assert result['source_receipt_digest']==controller.atlas.receipts[plan.word]
    assert result['native_trained_calls_during_prediction']==0
    assert result['native_trained_calls_during_source_trace_query']==0
    assert controller.learner.ground.rank==3
    assert certify_source_equivalence(true,controller.learner.ground)['status']=='EQUIVALENT_ALL_WORDS'
    assert len(controller.seen_nonces)==1
    with pytest.raises(ValueError,match='nonce'):
        controller.execute(plan)


def test_all_word_candidate_agreement_does_not_claim_neural_source_truth():
    _,_,first,_=fresh_observer()
    assert distinguish(first,first)['status']=='CANDIDATES_EQUIVALENT_ALL_WORDS'
    assert distinguish(first,first)['source_queries']==0


def test_stale_plan_is_fenced_before_new_source_execution():
    _,controller,_,_=fresh_observer()
    plan=controller.propose(nonce='2'*64)
    new_word=('act:step',)*4+('sense:goal:0',)
    assert new_word not in controller.atlas.cache
    controller.atlas.probability(new_word)
    with pytest.raises(ValueError,match='Stale|substituted'):
        controller.execute(plan)
    assert plan.nonce not in controller.seen_nonces


def test_hypothesis_tamper_and_cost_contract_are_refused():
    _,controller,one,two=fresh_observer()
    forged=replace(one,initial=tuple(Q(2)*v for v in one.initial))
    with pytest.raises(ValueError,match='identity'):
        Candidate('altered',forged,Q(1))
    with pytest.raises(ValueError,match='cost'):
        ActiveTraceObserver(controller.atlas,controller.learner,
                            (Candidate('a',one,Q(1)),Candidate('b',two,Q(1))),
                            costs={a:Q(-1) for a in alphabet()})
    with pytest.raises(ValueError,match='Candidates'):
        ActiveTraceObserver(controller.atlas,controller.learner,
                            (Candidate('same',one,Q(1)),Candidate('same',two,Q(1))))


def test_native_input_novelty_selector_does_not_execute_source(monkeypatch):
    exp=fixture(monkeypatch)
    candidate_a=((200,(.11,)),(201,(.10,)))
    candidate_b=((202,(.95,)),(203,(.45,)))
    plan=select_native_run(exp,(candidate_a,candidate_b),nonce='a'*64)
    assert plan.selected==candidate_b
    assert plan.cost==2 and plan.novelty>0
    assert plan.parent_digest==exp.native_witness_digest
    assert plan.proposal_digest!=plan.selected_digest
    with pytest.raises(ValueError,match='Duplicate'):
        select_native_run(exp,(candidate_a,candidate_a),nonce='b'*64)
    with pytest.raises(ValueError,match='vector'):
        select_native_run(exp,(((1,(float('nan'),)),),),nonce='c'*64)
    with pytest.raises(ValueError,match='already observed'):
        select_native_run(exp,(exp.tapes[0].inputs,),nonce='d'*64)
    assert exp.tapes[0].reports[0]=={'fixture':1}


def test_native_source_revision_requires_new_observer_source_identity(monkeypatch):
    exp=fixture(monkeypatch)
    new=fixture(monkeypatch,altered=True)
    assert new.source_digest!=exp.source_digest
    plan=select_native_run(exp,(((40,(.3,)),),),nonce='e'*64)
    assert plan.parent_digest==exp.native_witness_digest
    with pytest.raises(ValueError,match='Duplicate'):
        select_native_run(exp,(plan.selected,plan.selected),nonce='f'*64)


def test_source_witnessed_observer_attention_selects_target_distinction(monkeypatch):
    from vli.neural_trace_ground import CausalTape, WitnessedNeuralTraceEnsemble, NeuralObserver
    monkeypatch.setattr(CausalTape,'verify',lambda self: True)
    obs=NeuralObserver(0,0.)
    values=((.2,.1),(.4,-.3),(.7,-.2))
    tapes=tuple(CausalTape('fixture','r','a'*64,'b'*64,obs,
       ((i,(.1,.2)),),({'receiver_local_intermediates':[None,vec]},),
       ('1',),'c'*64) for i,vec in enumerate(values))
    source=WitnessedNeuralTraceEnsemble(tapes)
    plan=select_observer_attention(source,(2,))
    assert plan.distinction==Q(1)
    assert plan.position==0 and plan.coordinate==0
    assert plan.threshold==.7
    assert plan.original_witness_digest==source.native_witness_digest
    # The mathematical fixture proves selection only. Native witness capture,
    # recapture and reframe are verified in the native integration suite.
    with pytest.raises(ValueError,match='cost'):
        select_observer_attention(source,(2,),coordinate_costs={0:Q(-1),1:Q(1)})
    with pytest.raises(ValueError,match='nonempty'):
        select_observer_attention(source,())
    with pytest.raises(ValueError,match='nonempty'):
        select_observer_attention(source,(0,1,2))
