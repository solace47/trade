"""Reuse a frozen external screen and every original group for the morning target."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_boundary_evaluation as evaluation
from .corporate_cash import save_json, sha

STEM = 'tail_formula_external_morning'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
PRIOR = Path('data/research/external_tail_combo')
KEYS = Path('data/research/tail_formula_1000/observation_keys.parquet')
COLUMNS = ['date', 'code', 'half', 'board', 'decision_shares']
GROUPS = {
    'eligible': 'TRUE',
    'base': 'base_combo',
    'recent_only': 'base_combo AND recent_activity',
    'vwap_only': 'base_combo AND vwap_support',
    'neither': "grp='0:0'",
    'vwap_without_activity': "grp='0:1'",
    'activity_without_vwap': "grp='1:0'",
    'full': "grp='1:1'",
    'detail_unknown': "grp='detail_unknown'",
    'base_unknown': "grp='base_unknown'",
    'outside_base': "grp='outside_base'",
}


def checked():
    p = json.loads(PROTOCOL.read_text())
    assert p['groups'] == list(GROUPS) and p['primary_group'] == 'full'
    assert p['window_end'] == '09:59' and not p['new_2026_prices_allowed']
    for path, digest in p['source_hashes'].items():
        assert sha(Path(path)) == digest
    r = json.loads((PRIOR / 'input_report.json').read_text())
    v = json.loads((PRIOR / 'input_verification.json').read_text())
    assert v['passed'] and v['input_report_sha256'] == sha(PRIOR / 'input_report.json')
    for name in ['pool', 'history', 'paths']:
        assert r['outputs_sha256'][name] == sha(PRIOR / (name + '.parquet'))
    return p


def group_masks(pool):
    b = pool.base_combo.fillna(False)
    return dict(eligible=pd.Series(True, index=pool.index), base=b,
        recent_only=b & pool.recent_activity.fillna(False),
        vwap_only=b & pool.vwap_support.fillna(False),
        **{name: pool['group'].eq({'neither': '0:0', 'vwap_without_activity': '0:1',
            'activity_without_vwap': '1:0', 'full': '1:1'}.get(name, name))
           for name in list(GROUPS)[4:]})


def freeze():
    checked()
    assert not (ROOT / 'selection_manifest.json').exists()
    pool = pd.read_parquet(PRIOR / 'pool.parquet')
    eligible = pool.loc[pool.necessary_tradeable].copy()
    assert len(eligible) == 388448 and eligible.code.str.startswith(('sz.002', 'sz.003')).all()
    keys = pd.read_parquet(KEYS)
    assert keys.date.between('2024-01-01', '2025-12-31').all()
    q = eligible[['date', 'code', 'half', 'decision_shares']].merge(keys,
        on=['date', 'code'], how='left', validate='one_to_one', suffixes=('', '_keys'), indicator=True)
    assert q._merge.eq('both').all(), 'A missing historical hard-filter key must not be silently removed'
    assert q.half.eq(q.half_keys).all() and q.decision_shares.eq(q.decision_shares_keys).all()
    keys['board'] = 'main'
    keys = keys[COLUMNS].sort_values(['date', 'code']).reset_index(drop=True)
    definitions = group_masks(eligible)
    assert definitions['full'].sum() == 220 and definitions['base'].sum() == 1708
    assert np.array_equal(definitions['full'], eligible.primary)
    partition = sum(definitions[name].astype(int) for name in checked()['partition_groups'])
    np.testing.assert_array_equal(partition, definitions['base'].astype(int))
    records = []
    for name, mask in definitions.items():
        out = ROOT / name; out.mkdir(parents=True, exist_ok=True)
        selected = eligible.loc[mask, ['date', 'code']].assign(selected=True)
        f = keys.merge(selected, on=['date', 'code'], how='left', validate='one_to_one')
        f['selected'] = f.selected.eq(True)
        assert int(f.selected.sum()) == int(mask.sum())
        f.to_parquet(out / 'selection.parquet', index=False, compression='zstd')
        counts = f.loc[f.selected].groupby('half').agg(selected=('code','size'), days=('date','nunique')).reset_index()
        r = dict(protocol_sha256=sha(PROTOCOL), source_input_report_sha256=sha(PRIOR / 'input_report.json'),
            selection_sha256=sha(out / 'selection.parquet'), selected=int(f.selected.sum()),
            days=int(f.loc[f.selected,'date'].nunique()), group=name, counts=counts.to_dict('records'),
            outcome_based_selection=False, new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
        save_json(out / 'selection_report.json', r)
        records.append(dict(group=name, root=str(out), selection_report_sha256=sha(out / 'selection_report.json')))
    result = dict(protocol_sha256=sha(PROTOCOL), records=records, all_original_groups_retained=True,
        all_388448_original_eligible_keys_and_shares_mapped=True, group_unknowns_are_diagnostic_not_signals=True,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'selection_manifest.json', result)
    return result


def verify():
    checked()
    manifest = json.loads((ROOT / 'selection_manifest.json').read_text())
    assert manifest['protocol_sha256'] == sha(PROTOCOL) and len(manifest['records']) == len(GROUPS)
    c = base.conn()
    c.execute(f"CREATE VIEW original AS SELECT * FROM read_parquet('{PRIOR}/pool.parquet')")
    # Rebuild the old nullable condition groups rather than accepting their text labels.
    c.execute(f"""CREATE VIEW rebuilt AS WITH atoms AS (
        SELECT o.date,o.code,o.half,o.decision_shares,o.necessary_tradeable,
        round(o.price_1449*100)*100 BETWEEN round(o.preclose*100)*103 AND round(o.preclose*100)*105 AS gain,
        CASE WHEN h.volume5_rows=5 AND h.volume5_sum>0 THEN o.volume_1449*5>h.volume5_sum END AS vol,
        o.turnover_1449_proxy BETWEEN 5 AND 10 AS turnover,
        o.float_cap_proxy BETWEEN 5e9 AND 2e10 AS size, o.return_1449>o.market_return AS market,
        CASE WHEN h.history_rows=20 AND h.history_dates=20 AND h.valid_history_rows=20
            THEN h.big_up_days>0 END AS recent_activity,
        CASE WHEN p.path_source_valid THEN p.above_samples>=34 AND p.late_above_samples=15
            AND p.current_above END AS vwap_support
        FROM original o JOIN read_parquet('{PRIOR}/history.parquet') h USING(date,code)
        LEFT JOIN read_parquet('{PRIOR}/paths.parquet') p USING(date,code)), gates AS (
        SELECT *,gain AND vol AND turnover AND size AND market AS base_combo FROM atoms)
        SELECT *,CASE WHEN base_combo IS NULL THEN 'base_unknown' WHEN NOT base_combo THEN 'outside_base'
            WHEN recent_activity IS NULL OR vwap_support IS NULL THEN 'detail_unknown'
            ELSE recent_activity::INTEGER::VARCHAR||':'||vwap_support::INTEGER::VARCHAR END AS grp FROM gates""")
    differences = c.sql("""SELECT count(*) FROM original o JOIN rebuilt r USING(date,code)
        WHERE o.base_combo IS DISTINCT FROM r.base_combo OR o.recent_activity IS DISTINCT FROM r.recent_activity
        OR o.vwap_support IS DISTINCT FROM r.vwap_support OR o."group" IS DISTINCT FROM r.grp""").fetchone()[0]
    assert differences == 0
    proofs = []
    for item in manifest['records']:
        name = item['group']; out = Path(item['root'])
        assert item['selection_report_sha256'] == sha(out / 'selection_report.json')
        r = json.loads((out / 'selection_report.json').read_text())
        assert r['selection_sha256'] == sha(out / 'selection.parquet')
        expected = c.sql(f"""SELECT k.date,k.code,k.half,'main' AS board,k.decision_shares,
            coalesce(r.necessary_tradeable AND ({GROUPS[name]}),false) AS selected
            FROM read_parquet('{KEYS}') k LEFT JOIN rebuilt r USING(date,code) ORDER BY k.date,k.code""").df()
        actual = pd.read_parquet(out / 'selection.parquet')
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        counts = expected.loc[expected.selected].groupby('half').agg(selected=('code','size'),days=('date','nunique')).reset_index()
        assert counts.to_dict('records') == r['counts'] and int(expected.selected.sum()) == r['selected']
        proof = dict(passed=True, selection_report_sha256=sha(out / 'selection_report.json'),
            all_selection_flags_and_original_nullable_groups_independently_rebuilt=True,
            original_input_rows=448257, hard_filtered_rows=len(actual), selected=r['selected'],
            new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
        save_json(out / 'selection_verification.json', proof)
        proofs.append(dict(**item, selection_verification_sha256=sha(out / 'selection_verification.json'), counts=r['counts']))
    c.close()
    result = dict(passed=True, protocol_sha256=sha(PROTOCOL), selection_manifest_sha256=sha(ROOT / 'selection_manifest.json'),
        records=proofs, all_eleven_groups_frozen_together=True, new_group_outcomes_read=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'joint_selection_freeze.json', result)
    return result


def analyze():
    checked()
    path = ROOT / 'joint_selection_freeze.json'
    joint = json.loads(path.read_text())
    assert joint['passed'] and joint['protocol_sha256'] == sha(PROTOCOL)
    committed = subprocess.run(['git','show','HEAD:docs/selection-formula.md'],capture_output=True,text=True,check=True).stdout
    assert sha(path) in committed, 'All lists must be committed before group outcomes'
    precheck = json.loads((ROOT / 'full_selection_precheck.json').read_text())
    assert precheck['passed'] and precheck['joint_sha256'] == sha(path)
    assert all(not row['equal'] for row in precheck['comparisons']), 'Reuse identical lists before evaluation'
    # No half-year directories: each group report already includes all four halves.
    for item in joint['records']:
        out = Path(item['root'])
        assert item['selection_report_sha256'] == sha(out / 'selection_report.json')
        assert item['selection_verification_sha256'] == sha(out / 'selection_verification.json')
        if not (out / 'analysis_report.json').exists():
            with (out / 'analysis_command.log').open('w') as log:
                result = evaluation.analyze(out, PROTOCOL)
                log.write(json.dumps(result,ensure_ascii=False))
        for tool in ['verify_tail_formula_before1000.py', 'audit_tail_formula_reference_coverage.py']:
            command = ['.venv/bin/python','scripts/'+tool]
            if tool.startswith('verify'):
                command += ['analysis']
            else:
                if (out / 'reference_coverage_verification.json').exists():
                    proof = json.loads((out / 'reference_coverage_verification.json').read_text())
                    assert proof['passed'] and proof['analysis_report_sha256'] == sha(out / 'analysis_report.json')
                    assert proof['scope_years'] == [2024, 2025]
                    continue
                command += ['--years', '2024', '2025']
            command += ['--root',str(out)]
            with (out / (tool + '.log')).open('w') as log:
                subprocess.run(command,stdout=log,check=True)
        r = json.loads((out / 'analysis_report.json').read_text())
        print(json.dumps(dict(group=item['group'],primary=[s for s in r['summaries'] if s['arm']=='formula'
            and s['bps']==15 and not s['sensitive'] and s['period']=='2025']),ensure_ascii=False),flush=True)
    return dict(passed=True, all_eleven_group_analyses_and_reference_coverage_verified=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['freeze','verify','analyze'])
    a=p.parse_args()
    print(json.dumps(globals()[a.stage](),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
