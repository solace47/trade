"""Audit matured expanded-pool executable-event targets, without model fits."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from trade_research import tail_formula_additive as base
from trade_research import tail_formula_pool_scope_inputs as inputs
from trade_research import tail_formula_pool_scope_labels as labels
from trade_research.corporate_cash import save_json, sha

ROOT = Path('data/research/tail_formula_pool_executable')
PROTOCOL = Path('config/tail_formula_pool_executable_audit_protocol.json')
INTENT = Path('config/tail_formula_pool_executable_intent.json')
COLS = ['date', 'code', 'next_date', 'known15', 'opportunity15', 'known_no_trade',
        'entry_source_valid', 'period_entry_bad_day', 'period_bad_symbol', 'entry_source_unknown',
        'entry_recorded', 'entry_queue_unknown', 'entry_fill_status']


def checked():
    p = json.loads(PROTOCOL.read_text()); labels.checked('training')
    assert p['intent_sha256'] == sha(INTENT) and p['expressions'] == inputs.EXPRESSIONS
    assert p['input_only'] and not p['new_2026_prices_allowed']
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    root = labels.directory('training')
    r = json.loads((root/'full_label_report.json').read_text())
    v = json.loads((root/'full_label_verification.json').read_text())
    assert v['passed'] and v['label_report_sha256'] == sha(root/'full_label_report.json')
    assert r['labels_sha256'] == sha(root/'full_labels.parquet')
    return p


def audit():
    p = checked(); assert not (ROOT/'training_audit_verification.json').exists()
    ROOT.mkdir(parents=True, exist_ok=True)
    f = pd.read_parquet(inputs.OUT/'features.parquet', columns=[*inputs.META, *inputs.EXPRESSIONS])
    records = []; artifacts = {}
    for fold, spec in p['folds'].items():
        c = base.conn()
        l = c.execute('SELECT '+','.join(COLS)+' FROM read_parquet(?) WHERE date>=? AND next_date<? ORDER BY date,code',
            [str(labels.directory('training')/'full_labels.parquet'), spec['training_start'], spec['training_end']]).df()
        assert l.next_date.max() < spec['evaluation_start'] and not (l.known15 & l.known_no_trade).any()
        source_unknown = ~l.entry_source_valid | l.period_entry_bad_day | l.period_bad_symbol
        np.testing.assert_array_equal(l.entry_source_unknown, source_unknown)
        np.testing.assert_array_equal(l.known_no_trade, ~source_unknown & ~l.entry_recorded)
        assert not (l.known_no_trade & l.entry_queue_unknown).any()
        assert not (l.known_no_trade & l.entry_fill_status.eq('not_submitted')).any()
        assert np.isfinite(l.loc[l.known15, 'opportunity15']).all()
        assert l.loc[l.known15, 'opportunity15'].isin([0, 1]).all()
        l['utility'] = np.where(l.known_no_trade, 0., np.where(l.known15, l.opportunity15, np.nan))
        available = l.loc[l.known15 | l.known_no_trade].copy()
        available['target'] = available.utility-available.groupby('date').utility.transform('mean')
        t = f.loc[f.formula_input_valid].merge(available, on=['date', 'code'], validate='one_to_one')
        t = t.sort_values(['date', 'code']).reset_index(drop=True)
        t['w'] = 1/t.groupby('date').code.transform('size')
        unknown = ~l.known15 & ~l.known_no_trade
        assert np.isnan(l.loc[unknown, 'utility']).all()
        assert t.loc[t.known_no_trade, 'utility'].eq(0).all()
        c.register('features', f[['date', 'code', 'formula_input_valid', *inputs.EXPRESSIONS]])
        c.register('source', l[COLS])
        encoded = ','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in inputs.EXPRESSIONS)
        d = c.sql('''WITH identified AS(SELECT date,code,next_date,known15,known_no_trade,
            CASE WHEN known_no_trade THEN 0.0 ELSE opportunity15 END AS utility
            FROM source WHERE known15 OR known_no_trade), centered AS(SELECT *,
            utility-avg(utility) OVER(PARTITION BY date) AS target FROM identified)
            SELECT date,code,next_date,known15,known_no_trade,utility,target,
            1./count(*) OVER(PARTITION BY date) AS w,'''+encoded+'''
            FROM features JOIN centered USING(date,code) WHERE formula_input_valid ORDER BY date,code''').df(); c.close()
        pd.testing.assert_frame_equal(t[['date', 'code', 'next_date', 'known15', 'known_no_trade']],
            d[['date', 'code', 'next_date', 'known15', 'known_no_trade']], check_exact=True)
        np.testing.assert_allclose(t[['utility', 'target', 'w']], d[['utility', 'target', 'w']], rtol=0, atol=2e-12)
        old_known = f.loc[f.formula_input_valid, ['date', 'code']].merge(l.loc[l.known15, ['date', 'code']],
            on=['date', 'code'], validate='one_to_one').sort_values(['date', 'code']).reset_index(drop=True)
        pd.testing.assert_frame_equal(t.loc[t.known15, ['date', 'code']].reset_index(drop=True), old_known, check_exact=True)
        encoded_python = np.floor(np.clip(100*t[list(inputs.EXPRESSIONS)].to_numpy(float)+10000+.000001, 0, 999999)).astype('int32')
        np.testing.assert_array_equal(encoded_python, d[list(inputs.EXPRESSIONS)].to_numpy('int32'))
        root = ROOT/fold; root.mkdir(exist_ok=True)
        cols = ['date', 'code', 'next_date', 'known15', 'known_no_trade', 'utility', 'target', 'w']
        t[cols].to_parquet(root/'training_keys.parquet', index=False, compression='zstd')
        candidate = dict(**spec, expected_training_rows=len(t), expected_training_days=int(t.date.nunique()),
            feature_names=list(inputs.EXPRESSIONS), parameters=p['parameters'])
        save_json(root/'candidate_lookup_protocol.json', candidate)
        result = subprocess.run(['.venv/bin/python', 'scripts/find_existing_tail_formula_models.py',
            '--protocol', str(root/'candidate_lookup_protocol.json'), '--variant', 'relative_executable_opportunity'],
            text=True, capture_output=True, check=True)
        lookup = json.loads(result.stdout); save_json(root/'candidate_lookup_verification.json', lookup)
        r = dict(fold=fold, rows=len(t), days=int(t.date.nunique()), original_known_training_rows=len(old_known),
            added_no_trade_rows=int(t.known_no_trade.sum()), base_known_rows=int(l.known15.sum()),
            base_no_trade_rows=int(l.known_no_trade.sum()), base_unknown_rows=int(unknown.sum()),
            added_fill_statuses=t.loc[t.known_no_trade].groupby('entry_fill_status').size().to_dict(),
            last_observation=t.next_date.max(), existing_model_candidates=len(lookup['matches']),
            raw_training_keys_and_targets_independently_rebuilt=True)
        records.append(r)
        for file in ['training_keys.parquet', 'candidate_lookup_protocol.json', 'candidate_lookup_verification.json']:
            artifacts[str(root/file)] = sha(root/file)
    out = dict(passed=True, protocol_sha256=sha(PROTOCOL), input_artifacts=artifacts, folds=records,
        known_no_trade_is_zero_opportunity_not_realized_return=True, all_unknowns_excluded_only_from_training=True,
        all_original_known_training_rows_and_input_quality_retained=True,
        all_complete_pool_targets_weights_and_encodings_independently_rebuilt=True,
        no_new_model_fit_or_group_evaluation=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT/'training_audit_verification.json', out); return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=['audit'])
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
