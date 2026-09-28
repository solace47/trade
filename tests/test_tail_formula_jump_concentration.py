import numpy as np

from trade_research.tail_formula_jump_concentration import concentration, native_value


def test_flat_uniform_single_and_two_jumps():
    changes = np.zeros((4,29))
    changes[1] = 1
    changes[2,14] = 1
    changes[3,[1,27]] = [1,-1]
    value,total = concentration(changes)
    np.testing.assert_allclose(value,[0,0,100,100*(29/2-1)/28],atol=2e-13)
    np.testing.assert_array_equal(total,[0,29,1,2])
    np.testing.assert_allclose(value,concentration(-changes*17)[0],atol=2e-13)


def test_generated_native_first_last_and_outside_boundaries():
    for offset in [0,14,28]:
        changes = np.zeros(29)
        changes[offset] = .01
        price = np.r_[10.,10*np.exp(np.cumsum(changes))]
        np.testing.assert_allclose(native_value(price),100,rtol=0,atol=2e-12)
        assert native_value(price) == native_value(price,.01)
    assert native_value(np.ones(30)*10) == 0
