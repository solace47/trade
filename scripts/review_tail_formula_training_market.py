"""Review all fixed training groups against their same-date visible base pool."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from trade_research import tail_formula_baseline as original
from trade_research import tail_formula_additive as numeric
from trade_research import tail_formula_rule_search as rules
from trade_research.research_io import check_sources, save_json, sha
from run_tail_formula_rule_search import condition_key


ROOT = Path('data/research/tail_formula_profit_rule_search')
PROTOCOL = ROOT/'training_market_decomposition_protocol.json'
EXECUTION = ROOT/'training_market_decomposition_execution.json'
OUTPUT = ROOT/'training_market_decomposition.json'
PROGRAM = Path('scripts/review_tail_formula_training_market.py')


def number(value):
    return float(value) if pd.notna(value) else None


def review():
    assert not OUTPUT.exists(), 'Never repeat a completed diagnosis'
    p, e = json.loads(PROTOCOL.read_text()), json.loads(EXECUTION.read_text())
    document = subprocess.check_output(['git','show','HEAD:docs/selection-formula.md']).decode()
    assert sha(PROTOCOL) in document and sha(EXECUTION) in document
    assert subprocess.check_output(['git','show',f'HEAD:{PROGRAM}']) == PROGRAM.read_bytes()
    check_sources(p['source_hashes'])
    check_sources(e['source_hashes'])
    f = original.original()
    targets = pd.read_parquet('data/research/tail_formula_phase_split/targets.parquet')
    pd.testing.assert_frame_equal(f[['date','code']],targets[['date','code']],check_exact=True)
    feature_names = list(original.CONTROL)
    records, receipts = [], {}
    for fold_id in ['2024h1','2024h2','2025h1','2025h2']:
        report = json.loads((Path('data/research')/p['studies'][0]/fold_id/'model_report.json').read_text())
        fold = report['fold']
        scope = f.formula_input_valid & f.date.ge(fold['training_start']) & f.date.lt(fold['training_end'])
        train, label = f.loc[scope].reset_index(drop=True), targets.loc[scope].reset_index(drop=True)
        x = numeric.encode(train)
        mature = label.next_date.lt(fold['training_end'])
        known = mature & label.known15
        price_known = known & label.target_valid & np.isfinite(label.net)
        rows = train[['date','code']].copy()
        rows['known'] = known
        rows['success'] = known & label.opportunity15.eq(1)
        rows['no_trade'] = mature & label.known_no_trade
        rows['unknown'] = ~known & ~rows.no_trade
        rows['reference'] = np.expm1(label.net/100).where(price_known)
        base = rows.groupby('date').agg(base_reference=('reference','mean'),base_reference_rows=('reference','count')).reset_index()
        con = numeric.conn()
        con.register('visible',train[['date','code',*feature_names]])
        enc = ','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in feature_names)
        con.execute(f'''CREATE TABLE observations AS SELECT v.date,v.code,{enc},
            coalesce(t.next_date<'{fold['training_end']}' AND t.known15,false) AS known,
            coalesce(t.next_date<'{fold['training_end']}' AND t.known15 AND t.opportunity15=1,false) AS success,
            coalesce(t.next_date<'{fold['training_end']}' AND t.known_no_trade,false) AS no_trade,
            NOT known AND NOT no_trade AS unknown,
            CASE WHEN known AND t.target_valid AND isfinite(t.net) THEN exp(t.net/100)-1 END AS reference
            FROM visible v JOIN read_parquet('data/research/tail_formula_phase_split/targets.parquet') t USING(date,code)''')
        con.execute('''CREATE TABLE base_daily AS SELECT date,avg(reference) AS base_reference,
            count(reference) AS base_reference_rows FROM observations GROUP BY date''')
        sql_base = con.sql('SELECT * FROM base_daily ORDER BY date').df()
        pd.testing.assert_frame_equal(base,sql_base,check_dtype=False,check_exact=False,atol=2e-12,rtol=0)
        memo = {}
        for study in p['studies']:
            path = Path('data/research')/study/fold_id/'model_report.json'
            chosen = json.loads(path.read_text())['chosen_rule']
            key = condition_key(chosen['conditions']) if chosen else ()
            if key not in memo:
                selected = rules.masks(x,key) if key else np.zeros(len(x),dtype=bool)
                d = rows.loc[selected].groupby('date').agg(rows=('code','size'),known=('known','sum'),
                    success=('success','sum'),no_trade=('no_trade','sum'),unknown=('unknown','sum'),
                    reference_rows=('reference','count'),reference=('reference','mean')).reset_index()
                d['lower'] = d.success/d.rows
                d = d.merge(base,on='date',how='left',validate='one_to_one')
                d['excess'] = d.reference-d.base_reference
                terms = [f"{feature_names[j]} {'<=' if op==0 else '>'} {cut}" for j,op,cut in key] or ['false']
                con.execute(f'''CREATE OR REPLACE TABLE selected_daily AS SELECT date,count(*) AS rows,
                    sum(known::INT) AS known,sum(success::INT) AS success,sum(no_trade::INT) AS no_trade,
                    sum(unknown::INT) AS unknown,count(reference) AS reference_rows,avg(reference) AS reference,
                    avg(success::INT) AS lower FROM observations WHERE {' AND '.join(terms)} GROUP BY date''')
                sql_daily = con.sql('''SELECT s.*,b.base_reference,b.base_reference_rows,
                    s.reference-b.base_reference AS excess FROM selected_daily s JOIN base_daily b USING(date) ORDER BY date''').df()
                pd.testing.assert_frame_equal(d,sql_daily,check_dtype=False,check_exact=False,atol=2e-12,rtol=0)
                for name in ['date','rows','known','success','no_trade','unknown','reference_rows','base_reference_rows']:
                    pd.testing.assert_series_equal(d[name],sql_daily[name],check_dtype=False,check_exact=True)
                destination = ROOT/('training_market_daily_'+fold_id+'_'+str(len(memo))+'.parquet')
                assert not destination.exists()
                d.to_parquet(destination,index=False,compression='zstd')
                receipts[str(destination)] = sha(destination)
                memo[key] = (d,sql_daily,destination)
            d,independent,destination = memo[key]
            con.register('diagnostic_daily',independent)
            for half in [0,1]:
                q = d.loc[d.date.ge(fold['split']) if half else d.date.lt(fold['split'])]
                paired = q.loc[q.reference.notna() & q.base_reference.notna()]
                result = dict(study=study,fold=fold_id,half=half,days=len(q),
                    rows=int(q.rows.sum()),known=int(q.known.sum()),no_trade=int(q.no_trade.sum()),
                    unknown=int(q.unknown.sum()),reference_rows=int(q.reference_rows.sum()),
                    matched_reference_days=len(paired),lower=number(q.lower.mean()),
                    rule_reference=number(paired.reference.mean()),
                    same_dates_base_reference=number(paired.base_reference.mean()),excess=number(paired.excess.mean()),
                    model_report_sha256=sha(path),daily_source=str(destination),
                    empty_rule=chosen is None)
                operator = '>=' if half else '<'
                sql_summary = con.sql(f'''SELECT count(*) AS days,coalesce(sum(rows),0) AS rows,
                    coalesce(sum(known),0) AS known,coalesce(sum(no_trade),0) AS no_trade,
                    coalesce(sum(unknown),0) AS unknown,coalesce(sum(reference_rows),0) AS reference_rows,
                    count(excess) AS matched_reference_days,avg(lower) AS lower,
                    avg(reference) FILTER(WHERE excess IS NOT NULL) AS rule_reference,
                    avg(base_reference) FILTER(WHERE excess IS NOT NULL) AS same_dates_base_reference,
                    avg(excess) AS excess FROM diagnostic_daily WHERE CAST(date AS VARCHAR) {operator} '{fold['split']}' ''').df().iloc[0]
                for name in ['days','rows','known','no_trade','unknown','reference_rows','matched_reference_days']:
                    assert result[name] == int(sql_summary[name])
                for name in ['lower','rule_reference','same_dates_base_reference','excess']:
                    value = number(sql_summary[name])
                    assert (result[name] is None) == (value is None)
                    if value is not None:
                        assert abs(result[name]-value)<=2e-12
                records.append(result)
        con.close()
        print(json.dumps(dict(fold_complete=fold_id,groups=len(records),unique_rules=len(memo))),flush=True)
    assert len(records) == p['groups'] == 24
    check_sources(receipts)
    save_json(OUTPUT,dict(passed=True,protocol_sha256=sha(PROTOCOL),execution_sha256=sha(EXECUTION),
        source_hashes=receipts,groups=records,every_daily_and_group_SQL_verified=True,
        empty_rules_and_unknowns_retained=True,new_fits=0,changed_rules=0,new_selections=0,
        new_2026_prices_read=False,no_exit_rules=True,posthoc_diagnosis_not_causal_identification=True))
    return dict(output_sha256=sha(OUTPUT),groups=len(records),unique_daily_files=len(receipts))


if __name__ == '__main__':
    print(json.dumps(review(),ensure_ascii=False))
