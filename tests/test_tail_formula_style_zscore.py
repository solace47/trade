import numpy as np
import pandas as pd

from trade_research.tail_formula_style_zscore import FIELDS, transform


def sample():
    return pd.DataFrame(dict(date=['2025-01-02']*4,code=['a','b','c','bad'],
        formula_input_valid=[True,True,True,False], **{n:[1.,2.,3.,999.] for n in FIELDS}))


def test_input_pool_excludes_invalid_and_positive_affine_scale_cancels():
    f,s = transform(sample()); assert s.style_members.tolist() == [3]
    np.testing.assert_allclose(f.set_index('code').loc[['a','b','c'],'ZV01'],[-np.sqrt(1.5),0,np.sqrt(1.5)])
    assert f.set_index('code').loc['bad',['Z'+n for n in FIELDS]].isna().all()
    d=sample();d[FIELDS]=7*d[FIELDS]+11;other,_=transform(d)
    np.testing.assert_allclose(f[['Z'+n for n in FIELDS]],other[['Z'+n for n in FIELDS]],rtol=0,atol=2e-12,equal_nan=True)


def test_constant_or_single_member_keeps_keys_but_invalidates_entire_date():
    d=sample();d['S01']=2.;f,_=transform(d)
    assert len(f)==4 and not f.formula_input_valid.any()
    d=sample();d['formula_input_valid']=[True,False,False,False];f,_=transform(d)
    assert len(f)==4 and not f.formula_input_valid.any()
