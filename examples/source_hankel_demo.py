"""PBG-005: source-query-only discovery of a delayed predictive law.

python examples/source_hankel_demo.py
Only synthetic exact-rational model queries are used; no physical action.
"""
from fractions import Fraction as Q
import json

from vli.intention_predictive_ground import ControlledSource
from vli.source_hankel_ground import (
    QueryAtlas, HankelLearner, SourceObservation, HankelObserverSession,
    source_trace_probability, certify_source_equivalence, verify_source_closure,
    digest,
)


def delayed_source():
    initial=(Q(1),Q(0),Q(0))
    goal=(Q(0),Q(0),Q(1))
    transition=((Q(0),Q(1),Q(0)),(Q(0),Q(0),Q(1)),(Q(0),Q(0),Q(1)))
    sensing=(('0',((Q(1),Q(0),Q(0)),(Q(0),Q(1),Q(0)),(Q(0),Q(0),Q(0)))),
             ('1',((Q(0),Q(0),Q(0)),(Q(0),Q(0),Q(0)),(Q(0),Q(0),Q(1)))))
    return ControlledSource(initial,goal,(('step',transition,Q(1,100)),),
                            (('goal',sensing,Q(1,100)),))


def run():
    source=delayed_source()
    alphabet=('act:step','sense:goal:0','sense:goal:1')
    # Source matrices live ONLY in the source-owned closure. The learner is
    # given a qualified membership-query interface with identical semantics.
    def reader(words):
        return tuple(SourceObservation(w,source_trace_probability(source,w),
                source.digest,digest({'synthetic-probe':list(w),'source':source.digest}))
                for w in words)
    atlas=QueryAtlas(alphabet,source.digest,reader,max_queries=1500)
    learner=HankelLearner(atlas)
    base=learner.build()
    first=certify_source_equivalence(source,base)
    assert first['status']=='COUNTEREXAMPLE' and base.rank==1
    grown=learner.counterexample(first['word'])
    verified=certify_source_equivalence(source,grown)
    assert verified['status']=='EQUIVALENT_ALL_WORDS'
    assert verify_source_closure(source,grown)
    session=HankelObserverSession(grown,source)
    before=session.mass('sense:goal:1')
    session.apply('step');session.apply('step')
    after=session.mass('sense:goal:1')
    return {'case':'pbg005-exact-rational-synthetic-source',
            'source_states':len(source.initial),
            'initial_candidate_rank':base.rank,'expanded_rank':grown.rank,
            'counterexample_word':list(first['word']),
            'observed_weight':str(first['actual']),
            'initial_prediction':str(first['predicted']),
            'initial_goal_probability':str(before),
            'goal_probability_after_two_predicted_steps':str(after),
            'exact_source_query_count':len(atlas.cache),
            'source_query_batches':atlas.calls,
            'all_future_equivalence':verified['status'],
            'paired_reachable_rank':verified['reachable_rank'],
            'source_digest':source.digest,'ground_digest':grown.ground_digest,
            'scope':'exact finite source with independent native matrix audit; not a pretrained model or external material actuation'}


if __name__=='__main__':
    print(json.dumps(run(),sort_keys=True,indent=2))
