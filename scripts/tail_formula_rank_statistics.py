"""Verified rank pairs for binary labels and three ordered categories.

Ordinal categories use -3, 0, 1; their magnitudes do not enter pair counts.
"""
import numpy as np
import pandas as pd
from trade_research import tail_formula_additive as numeric

def binary_daily(frame, field):
    p = frame[['date', 'score', field]].copy()
    assert p[field].isin([0, 1]).all() and np.isfinite(p.score).all()
    p['rank'] = p.groupby('date').score.rank(method='average')
    p['positive_rank'] = p['rank'].where(p[field].eq(1), 0.)
    d = p.groupby('date').agg(rows=(field, 'size'), positives=(field, 'sum'), positive_ranks=('positive_rank', 'sum')).reset_index()
    d['pairs'] = d.positives * (d.rows - d.positives)
    d['numerator2'] = (2*d.positive_ranks - d.positives*(d.positives+1)).astype('int64')
    d['auc'] = d.numerator2 / (2*d.pairs.replace(0, np.nan))
    return d[['date', 'pairs', 'numerator2', 'auc']]


def daily_metrics(frame):
    """Higher score should precede the higher label; score ties count one half."""
    d = frame.groupby('date').size().rename('rows').reset_index()
    for field in ['positive_before', 'space_before']:
        part = binary_daily(frame, field).rename(columns={n:field+'_'+n for n in ['pairs','numerator2','auc']})
        d = d.merge(part, on='date', validate='one_to_one')
    d['utility_pairs'] = 0
    d['utility_numerator2'] = 0
    for high, low in [(0, -3), (1, -3), (1, 0)]:
        part = frame.loc[frame.utility.isin([high, low]), ['date','score','utility']].copy()
        part['higher'] = part.utility.eq(high).astype('int64')
        b = binary_daily(part, 'higher').set_index('date')
        for field in ['pairs', 'numerator2']:
            d['utility_'+field] += d.date.map(b[field]).fillna(0).astype('int64')
    d['utility_auc'] = d.utility_numerator2 / (2*d.utility_pairs.replace(0, np.nan))
    return d


def verify_metrics(frame, actual):
    c = numeric.conn(); c.register('diagnostic_rows', frame)
    expected = c.sql('SELECT date,count(*) AS rows FROM diagnostic_rows GROUP BY date ORDER BY date').df()
    def one(where, label):
        return c.sql(f'''WITH ranked AS (SELECT date,{label} AS positive,
            rank() OVER(PARTITION BY date ORDER BY score) +
            (count(*) OVER(PARTITION BY date,score)-1)/2. AS average_rank
            FROM diagnostic_rows {where}), counts AS (
            SELECT date,count(*) AS n,sum(positive::INT) AS m,
            sum(CASE WHEN positive THEN average_rank ELSE 0. END) AS total
            FROM ranked GROUP BY date)
            SELECT date,(m*(n-m))::BIGINT AS pairs,(2*total-m*(m+1))::BIGINT AS numerator2 FROM counts ORDER BY date''').df()
    for field in ['positive_before','space_before']:
        b=one('', field+'=1').rename(columns={n:field+'_'+n for n in ['pairs','numerator2']})
        expected=expected.merge(b,on='date',validate='one_to_one')
    expected['utility_pairs']=0; expected['utility_numerator2']=0
    for high,low in [(0,-3),(1,-3),(1,0)]:
        b=one(f'WHERE utility IN ({high},{low})',f'utility={high}').set_index('date')
        for field in ['pairs','numerator2']:
            expected['utility_'+field]+=expected.date.map(b[field]).fillna(0).astype('int64')
    for field in ['positive_before','space_before','utility']:
        expected[field+'_auc']=expected[field+'_numerator2']/(2*expected[field+'_pairs'].replace(0,np.nan))
    c.close()
    pd.testing.assert_frame_equal(actual[expected.columns],expected,check_dtype=False,rtol=0,atol=2e-12)

