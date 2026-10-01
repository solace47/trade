"""Review all eight training periods of fixed rules; never change a rule."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from trade_research import tail_formula_baseline as original
from trade_research import tail_formula_additive as numeric
from trade_research import tail_formula_rule_search as rules
from trade_research.research_io import check_sources, save_json, sha
import run_tail_formula_rule_search as fit


def review():
    root = fit.ROOT
    protocol = root / 'training_reference_diagnosis_protocol.json'
    output = root / 'training_reference_diagnosis.json'
    assert not output.exists()
    p = json.loads(protocol.read_text())
    text = subprocess.check_output(['git', 'show', 'HEAD:docs/selection-formula.md']).decode()
    assert sha(protocol) in text
    check_sources(p['source_hashes'])
    f = original.original()
    t = pd.read_parquet(fit.TARGETS)
    pd.testing.assert_frame_equal(f[['date', 'code']], t[['date', 'code']], check_exact=True)
    records = []
    for name in ['2024h1', '2024h2', '2025h1', '2025h2']:
        model = json.loads((root/name/'model_report.json').read_text())
        fold = model['fold']
        scope = f.formula_input_valid & f.date.ge(fold['training_start']) & f.date.lt(fold['training_end'])
        visible = f.loc[scope].reset_index(drop=True)
        labels = t.loc[scope].reset_index(drop=True)
        chosen = model['chosen_rule']
        flags = rules.masks(numeric.encode(visible), chosen['conditions']) if chosen else np.zeros(len(visible), dtype=bool)
        mature = labels.next_date.lt(fold['training_end'])
        known = mature & labels.known15
        reference_known = known & labels.target_valid & np.isfinite(labels.net)
        r = visible[['date', 'code']].copy()
        r['known'] = known
        r['success'] = known & labels.opportunity15.eq(1)
        r['reference_known'] = reference_known
        r['missing_reference_known'] = known & ~reference_known
        r['reference'] = np.expm1(labels.net/100).where(reference_known)
        r['bad3'] = labels.adverse_return15.le(-.03).astype(float).where(known & np.isfinite(labels.adverse_return15))
        d = r.loc[flags].groupby('date').agg(rows=('code','size'), known=('known','sum'),
            success=('success','sum'), reference_rows=('reference_known','sum'),
            missing_reference_known=('missing_reference_known','sum'), reference=('reference','mean'), bad3=('bad3','mean')).reset_index()
        d['lower'] = d.success/d.rows
        d['rate'] = d.success/d.known.replace(0,np.nan)
        con = numeric.conn()
        con.register('visible', visible[['date','code', *model['feature_names']]])
        terms = [f"floor(least(greatest(100*{model['feature_names'][j]}+10000+.000001,0),999999))::INT {'<=' if op==0 else '>'} {cut}" for j,op,cut in chosen['conditions']] if chosen else ['false']
        k = f"coalesce(t.next_date<'{fold['training_end']}' AND t.known15,false)"
        rk = f'({k}) AND t.target_valid AND isfinite(t.net)'
        q = con.sql(f'''WITH selected AS(SELECT v.date,v.code,t.* EXCLUDE(date,code),{k} AS known,
            {rk} AS reference_known FROM visible v LEFT JOIN read_parquet('{fit.TARGETS}') t USING(date,code)
            WHERE {' AND '.join(terms)}) SELECT date,count(*) AS rows,
            count(*) FILTER(WHERE known) AS known,count(*) FILTER(WHERE known AND opportunity15=1) AS success,
            count(*) FILTER(WHERE reference_known) AS reference_rows,
            count(*) FILTER(WHERE known AND NOT reference_known) AS missing_reference_known,
            avg(exp(net/100)-1) FILTER(WHERE reference_known) AS reference,
            avg((adverse_return15<=-.03)::INT) FILTER(WHERE known AND isfinite(adverse_return15)) AS bad3,
            count(*) FILTER(WHERE known AND opportunity15=1)/count(*) AS lower,
            count(*) FILTER(WHERE known AND opportunity15=1)/nullif(count(*) FILTER(WHERE known),0) AS rate
            FROM selected GROUP BY date ORDER BY date''').df()
        con.close()
        pd.testing.assert_frame_equal(d[q.columns],q,check_dtype=False,atol=2e-12,rtol=0)
        for half in [0,1]:
            a = d.loc[d.date.lt(fold['split']) if half==0 else d.date.ge(fold['split'])]
            b = q.loc[q.date.lt(fold['split']) if half==0 else q.date.ge(fold['split'])]
            result = dict(fold=name,training_half=half,days=len(a),rows=int(a.rows.sum()),
                known=int(a.known.sum()),reference_rows=int(a.reference_rows.sum()),
                missing_reference_known=int(a.missing_reference_known.sum()))
            for metric in ['lower','rate','reference','bad3']:
                value = a[metric].mean()
                other = b[metric].mean()
                assert (pd.isna(value) and pd.isna(other)) or abs(value-other)<=2e-12
                result[metric] = float(value) if pd.notna(value) else None
            records.append(result)
    assert len(records)==p['groups']==8
    save_json(output,dict(passed=True,protocol_sha256=sha(protocol),implementation_sha256=sha(Path(__file__)),
        source_hashes=p['source_hashes'],summaries=records,all_eight_daily_and_group_statistics_SQL_rebuilt=True,
        original_rules_unchanged=True,new_candidates=0,new_fits=0,new_2026_prices_read=False))
    return dict(sha256=sha(output),summaries=records)


if __name__=='__main__':
    print(json.dumps(review(),ensure_ascii=False))
