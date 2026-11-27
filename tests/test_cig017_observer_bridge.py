"""Execute real In-Formation-Space CIG-015/017 mathematics on PBG-010's CNF.

The CIG proof is independent of the PBG-010 finite-pair checker. No Python
mocks of CIG's source/compiler/verifier are used in this integration test.
"""
from collections import Counter
from fractions import Fraction as Q
import pytest

from vli.observer_sufficiency import (
    SourceBit, FiniteEvidence, compile_language, verify_language,
    export_cig017_source, _digest,
)
from global_sufficiency import synthesize
from verify_global_sufficiency import verify
from language_extension import Language


def corpus(targets,cols):
    return FiniteEvidence('a'*64,'b'*64,
        tuple(_digest({'tape':i}) for i in range(len(targets))),
        SourceBit(2,0,0.),tuple(targets),
        tuple(SourceBit(0,i,0.) for i in range(len(cols))),
        tuple(tuple(c) for c in cols),tuple(Q(1) for c in cols))


def test_cig017_independently_proves_sufficiency_of_admitted_finite_source():
    table=corpus([0,1],[[0,1]])
    local=compile_language(table)
    assert local['status']=='SUFFICIENT_ON_WITNESSED_ENSEMBLE'
    assert verify_language(table,local)
    n,source,q,f=export_cig017_source(table,local['selected'])
    independent=synthesize(n,source,q,f,width=12,cells=200000)
    assert independent['status']=='SUFFICIENT',independent
    assert verify(n,source,q,f,independent,width=12,cells=200000)


def test_cig017_exposes_pair_when_observation_is_not_sufficient():
    table=corpus([0,1],[[0,0]])
    local=compile_language(table)
    assert local['status']=='MISSING_OBSERVER_DISTINCTION'
    n,source,q,f=export_cig017_source(table,())
    independent=synthesize(n,source,q,f,width=12,cells=200000)
    assert independent['status']=='INSUFFICIENT',independent
    assert verify(n,source,q,f,independent,width=12,cells=200000)


def test_cig015_refines_source_histogram_by_goal_without_new_neural_execution():
    table=corpus([0,1,1],[[0,1,1]])
    n,source,q,f=export_cig017_source(table,(0,))
    # CIG-015 recomputes a certified source-fiber split with a completely
    # different algorithm and independent checker.
    lang=Language(n,list(range(1,n+1)),list(source),[q[0][0]])
    before=lang.snapshot()
    lang.ensure(list(f),Counter())
    after=lang.snapshot()
    assert len(after['descriptor']['predicates'])==1
    assert sum(int(x[1]) for x in before['bins'])==len(table.targets)
    assert sum(int(x[1]) for x in after['bins'])==len(table.targets)
