import numpy as np
import pytest

from trade_research.tail_formula_endpoint_robust import huber_leaf, positive_reference_target, weighted_quantile


def test_weighted_quantile_uses_lower_value_at_exact_boundary():
    assert weighted_quantile([1.,5.,9.],[1.,2.,1.],.25)==1.
    assert weighted_quantile([1.,5.,9.],[1.,2.,1.],.5)==5.


def test_huber_leaf_limits_outlier_and_respects_price_unit():
    values=np.array([0.,0.,100.]);weights=np.ones(3)
    np.testing.assert_allclose(huber_leaf(values,weights,1.),1/3,rtol=0,atol=1e-12)
    np.testing.assert_allclose(huber_leaf(3*values,weights,3.),3*huber_leaf(values,weights,1.),rtol=0,atol=1e-12)


def test_clipped_residual_matches_negative_gradient_of_huber_loss():
    residual=np.array([-2.,-.4,0.,.4,2.]);delta=.75;eps=1e-6
    loss=lambda r:np.where(np.abs(r)<=delta,r*r/2,delta*(np.abs(r)-delta/2))
    derivative=(loss(residual+eps)-loss(residual-eps))/(2*eps)
    np.testing.assert_allclose(derivative,np.clip(residual,-delta,delta),rtol=0,atol=1e-9)


def test_positive_reference_event_excludes_zero_and_rejects_missing():
    np.testing.assert_array_equal(positive_reference_target([-1e-12,0.,1e-12]),[0.,0.,1.])
    with pytest.raises(AssertionError,match='Missing reference'):
        positive_reference_target([.01,np.nan])
