"""Fixed characterization of completed selectors; no selection changes or fits."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from trade_research.research_io import check_runtime, check_sources, save_json, sha
from trade_research.tail_formula_additive import conn
from tail_formula_reports import checked_selection
from trade_research.tail_formula_boundary_evaluation import period, number

ROOT = Path('data/research/tail_formula_order_target/reflection')
PROTOCOL = Path('config/tail_formula_order_target_reflection.json')
FEATURES = ['V01', 'visible_return', 'A12', 'R01']


def review():
    check_runtime()
    assert not (ROOT/'report.json').exists()
    p = json.loads(PROTOCOL.read_text())
    assert subprocess.check_output(['git','show',f'HEAD:{PROTOCOL}']) == PROTOCOL.read_bytes()
    assert sha(PROTOCOL) in subprocess.check_output(['git','show','HEAD:docs/selection-formula.md']).decode()
    check_sources(p['source_hashes'])
    f = pd.read_parquet(p['features'], columns=['date','code','half','formula_input_valid','V01','NA01','A12','R01'])
    f = f.loc[f.date.ge('2024-01-01')].copy()
    assert len(f) == 1258085 and int(f.formula_input_valid.sum()) == 1117397
    f['visible_return'] = f.NA01 * f.V01
    daily = []; summaries = []
    c = conn()
    for group, path in p['selections'].items():
        frame = checked_selection(Path(path))
        q = f.merge(frame[['date','code','selected']], on=['date','code'], validate='one_to_one')
        assert len(q) == len(f) and q.loc[q.selected,'formula_input_valid'].all()
        chosen = q.loc[q.selected]; dates = chosen.date.unique()
        for arm, part in [('formula',chosen),('valid_pool_same_dates',q.loc[q.formula_input_valid & q.date.isin(dates)])]:
            d = part.groupby(['date','half']).agg(rows=('code','size'), **{n:(n,'mean') for n in FEATURES}).reset_index()
            c.register('characteristics',part)
            ex = c.sql('SELECT date,half,count(*) AS rows,'+','.join(f'avg({n}) AS {n}' for n in FEATURES)+' FROM characteristics GROUP BY date,half ORDER BY date,half').df()
            pd.testing.assert_frame_equal(d,ex,check_dtype=False,rtol=0,atol=2e-12)
            year = group[-4:]
            for window in [year+'H1',year+'H2',year]:
                z = period(d,window)
                summaries.append(dict(group=group,arm=arm,period=window,days=len(z),rows=int(z.rows.sum()), **{n:number(z[n].mean()) for n in FEATURES}))
            d['group']=group;d['arm']=arm;daily.append(d)
    labels = pd.read_parquet(p['targets'])
    assert len(labels)==1815129 and labels.date.lt('2026-01-01').all()
    known = labels.loc[labels.known15].copy()
    se=known.first_space_end15;be=known.first_bad315
    known['space_first'] = se.ge(0) & (be.lt(0) | se.lt(be))
    known['bad_first'] = be.ge(0) & (se.lt(0) | be.le(se))
    known['no_event'] = se.lt(0) & be.lt(0)
    assert (known[['space_first','bad_first','no_event']].sum(axis=1)==1).all()
    np.testing.assert_array_equal(known.space_first,known.ordered_utility.eq(1))
    training = []
    for fold in p['folds']:
        part=known.loc[known.date.ge(fold['training_start']) & known.next_date.lt(fold['training_end'])]
        d=part.groupby('date')[['space_first','bad_first','no_event']].mean()
        c.register('mature_labels',part)
        ex=c.sql('SELECT date,avg((first_space_end15>=0 AND (first_bad315<0 OR first_space_end15<first_bad315))::INT) AS space_first,avg((first_bad315>=0 AND (first_space_end15<0 OR first_bad315<=first_space_end15))::INT) AS bad_first,avg((first_space_end15<0 AND first_bad315<0)::INT) AS no_event FROM mature_labels GROUP BY date ORDER BY date').df().set_index('date')
        pd.testing.assert_frame_equal(d,ex,check_dtype=False,rtol=0,atol=2e-12)
        training.append(dict(fold=fold['id'],mature_known_rows=len(part),days=len(d), **{n:float(d[n].mean()) for n in d.columns}))
    c.close()
    pd.concat(daily,ignore_index=True).to_parquet(ROOT/'daily.parquet',index=False,compression='zstd')
    save_json(ROOT/'report.json',dict(passed=True,protocol_sha256=sha(PROTOCOL),source_hashes=p['source_hashes'],daily_sha256=sha(ROOT/'daily.parquet'),summaries=summaries,training_categories=training,all_characteristics_and_training_categories_SQL_verified=True,selected_lists_unchanged=True,new_fits=0,new_features_used_to_select=0,no_new_raw_prices=True,new_2026_prices_read=False,no_exit_rules=True))
    print(json.dumps(dict(report_sha256=sha(ROOT/'report.json'),summaries=summaries,training_categories=training),ensure_ascii=False),flush=True)


if __name__=='__main__':
    review()
