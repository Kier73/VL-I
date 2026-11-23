"""NOC-001: independently calculated dense nonlinear oracles and safety checks."""
import numpy as np
import pytest
from vli.nonlinear_observer_cut import compile_nonlinear_cut, evaluate_cut


def baseline(a, down, gate, up, u, x):
    # Independent dense first-order mathematical path: reconstruct h=u+Dv.
    g = gate@x
    sigmoid = np.where(g>=0, 1/(1+np.exp(-g)), np.exp(g)/(1+np.exp(g)))
    v = (g*sigmoid)*(up@x)
    return float(a @ (u+down@v))


def fixture(seed=20, *, m=24, d=9):
    rng=np.random.default_rng(seed)
    a=rng.normal(size=d)
    down=rng.normal(size=(d,m))/10
    gate=rng.normal(size=(m,d))/10
    up=rng.normal(size=(m,d))/10
    x=rng.normal(size=d)
    u=rng.normal(size=d)
    return a,down,gate,up,u,x


def test_zero_channel_decision_matches_full_nonlinear_oracle():
    a,d,g,p,u,x=fixture()
    u = 1000*a
    compiled=compile_nonlinear_cut(a,d,g,p)
    result=evaluate_cut(compiled,x,u)
    assert result['status']=='DECIDED_REAL_ARITHMETIC'
    assert result['channels_evaluated']==0
    assert result['gate_up_dot_products_avoided']==2*compiled.channels
    assert result['sign']==np.sign(baseline(a,d,g,p,u,x))


def test_adaptive_channels_match_independent_dense_oracle_and_bounds():
    for seed in range(45):
        a,d,g,p,u,x=fixture(seed)
        ground=compile_nonlinear_cut(a,d,g,p)
        actual=baseline(a,d,g,p,u,x)
        result=evaluate_cut(ground,x,u, numerical_guard=1e-10)
        assert result['status']=='DECIDED_REAL_ARITHMETIC'
        assert result['sign']==int(np.sign(actual))
        assert 0<=result['channels_evaluated']<=ground.channels
        assert result['source_execution_calls']==0
        assert result['gate_up_dot_products_avoided']==2*(ground.channels-result['channels_evaluated'])


def test_channel_order_and_exact_cancellation_refuse_unjustified_decision():
    a=np.array([1.0]); d=np.array([[1.0,1.0]])
    g=np.array([[1.0],[1.0]]); p=np.array([[1.0],[-1.0]])
    cut=compile_nonlinear_cut(a,d,g,p)
    result=evaluate_cut(cut,[1.0],[0.0],order=[1,0])
    assert result['status']=='UNRESOLVED_NUMERICAL_MARGIN'
    assert result['sign'] is None
    assert result['channels_evaluated']==2
    assert abs(baseline(a,d,g,p,np.zeros(1),np.ones(1)))<1e-15


def test_large_guard_requires_fallback_not_unsafe_sign():
    a,d,g,p,u,x=fixture()
    cut=compile_nonlinear_cut(a,d,g,p)
    result=evaluate_cut(cut,x,u,numerical_guard=1e6)
    assert result['status']=='UNRESOLVED_NUMERICAL_MARGIN'


def test_nonfinite_shapes_and_invalid_orders_refuse():
    a,d,g,p,u,x=fixture()
    cut=compile_nonlinear_cut(a,d,g,p)
    with pytest.raises(ValueError,match='geometry'):
        compile_nonlinear_cut(a,d[:3],g,p)
    with pytest.raises(ValueError,match='matrix shape'):
        compile_nonlinear_cut(a,d,g.T,p)
    with pytest.raises(ValueError,match='finite'):
        evaluate_cut(cut,np.full(len(x),np.nan),u)
    with pytest.raises(ValueError,match='permutation'):
        evaluate_cut(cut,x,u,order=list(range(cut.channels-1))+[0])
    with pytest.raises(ValueError,match='guard'):
        evaluate_cut(cut,x,u,numerical_guard=float('nan'))
