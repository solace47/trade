"""Reuse bounded Q1 observations and extract only newly required candidate dates."""
import argparse
import json
from pathlib import Path

import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_q1_candidates as study
from . import tail_formula_q1_candidate_inputs as inputs
from .corporate_cash import MINUTES, save_json, sha
from .market_study import _quality_symbols
from .turnover_reference import CALENDAR

ROOT = study.ROOT
LABELS = ROOT/'labels'
CATALOG = LABELS/'catalog'
OLD = study.OLD/'labels'


def prepare():
    p, gate = study.checked_models(); assert not (LABELS/'input_manifest.json').exists()
    report = json.loads((ROOT/'selection_report.json').read_text())
    proof = json.loads((ROOT/'selection_verification.json').read_text())
    assert proof['passed'] and proof['selection_report_sha256'] == sha(ROOT/'selection_report.json')
    dates = set()
    for variant, digest in report['selection_report_sha256'].items():
        folder = ROOT/variant; r = json.loads((folder/'selection_report.json').read_text())
        assert sha(folder/'selection_report.json') == digest and r['selection_sha256'] == sha(folder/'selection.parquet')
        assert proof['selection_verification_sha256'][variant] == sha(folder/'selection_verification.json')
        selected = pd.read_parquet(folder/'selection.parquet', columns=['date', 'selected'])
        dates.update(selected.loc[selected.selected, 'date'])
    assert sorted(dates) == report['union_selected_dates']
    b = json.loads((inputs.OLD/'base_report.json').read_text())
    assert b['universe_sha256'] == sha(inputs.OLD/'universe.parquet')
    universe = pd.read_parquet(inputs.OLD/'universe.parquet')
    keys = universe.loc[universe.date.isin(dates)].copy().sort_values(['date', 'code']).reset_index(drop=True)
    cal = pd.read_parquet(CALENDAR)
    days = sorted(cal.loc[cal.is_trading_day.eq('1') & cal.calendar_date.between(p['signal_first'], p['observation_last']), 'calendar_date'])
    keys['next_date'] = keys.date.map(dict(zip(days[:-1], days[1:])))
    assert keys.next_date.gt(keys.date).all() and keys.next_date.le(p['observation_last']).all()
    assert keys.date.between(p['signal_first'], p['signal_last']).all()
    LABELS.mkdir(parents=True, exist_ok=True)
    keys.to_parquet(LABELS/'keys.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(study.PROTOCOL), selection_report_sha256=sha(ROOT/'selection_report.json'),
        selection_verification_sha256=sha(ROOT/'selection_verification.json'), models_gate_sha256=sha(ROOT/'models_gate.json'),
        source_manifest_sha256=sha(inputs.OLD/'source_manifest.json'), keys_sha256=sha(LABELS/'keys.parquet'),
        calendar_sha256=sha(CALENDAR), rows=len(keys), dates=len(dates), codes=keys.code.nunique(),
        first_signal=keys.date.min(), last_signal=keys.date.max(), last_observation=keys.next_date.max(),
        all_candidate_dates_union=True, full_base_on_selected_dates=True, q1_new_group_outcomes_read=False,
        prior_q1_outcomes_exposed=True, strict_blind=False, no_q2_signal_prices_read=True, no_exit_rules=True)
    save_json(LABELS/'input_manifest.json', r); return r


def checked_keys():
    p, gate = study.checked_models(); m = json.loads((LABELS/'input_manifest.json').read_text())
    assert m['protocol_sha256'] == sha(study.PROTOCOL) and m['selection_report_sha256'] == sha(ROOT/'selection_report.json')
    assert m['selection_verification_sha256'] == sha(ROOT/'selection_verification.json')
    assert m['keys_sha256'] == sha(LABELS/'keys.parquet') and m['calendar_sha256'] == sha(CALENDAR)
    assert m['source_manifest_sha256'] == sha(inputs.OLD/'source_manifest.json')
    return p, m, pd.read_parquet(LABELS/'keys.parquet')


def reuse_evidence():
    p, m, keys = checked_keys(); assert not (LABELS/'evidence_reuse.json').exists()
    # The old coverage is code/year and quality is the whole fixed quarter;
    # neither is limited to the old formula's 39 selected dates.
    old_proof = json.loads((OLD/'full_label_verification.json').read_text())
    old_report = json.loads((OLD/'full_label_report.json').read_text())
    assert old_proof['passed'] and old_proof['label_report_sha256'] == sha(OLD/'full_label_report.json')
    cr = json.loads((OLD/'catalog/date_report.json').read_text())
    cp = json.loads((OLD/'catalog/date_verification.json').read_text())
    assert cp['passed'] and cp['date_report_sha256'] == sha(OLD/'catalog/date_report.json')
    assert old_report['catalog_verification_sha256'] == sha(OLD/'catalog/date_verification.json')
    assert cr['complete'] and cr['events_sha256'] == sha(OLD/'catalog/date_events.parquet')
    assert cr['coverage_sha256'] == sha(OLD/'catalog/date_coverage.parquet')
    coverage = pd.read_parquet(OLD/'catalog/date_coverage.parquet')
    assert set(keys.code) <= set(coverage.loc[coverage.year.eq('2026'), 'code']), 'Missing catalogue codes need explicit extension'
    quality = json.loads((OLD/'quality/report.json').read_text())
    assert old_report['quality_report_sha256'] == sha(OLD/'quality/report.json')
    assert (quality['symbol_audit_first'], quality['symbol_audit_last'], quality['day_issue_keys_last']) == (p['signal_first'], p['signal_last'], p['observation_last'])
    assert quality['bad_days_sha256'] == sha(OLD/'quality/bad_days.parquet')
    needed = set(_quality_symbols(Path('data/research/market_issues_ci')).code) & set(keys.code)
    assert needed <= {x['code'] for x in quality['checks']}, 'New suspicious codes require source audit'
    CATALOG.mkdir(exist_ok=True)
    for name in ['date_report.json', 'date_verification.json', 'date_events.parquet', 'date_coverage.parquet']:
        (CATALOG/name).symlink_to((OLD/'catalog'/name).resolve())
    folder = LABELS/'quality'; folder.mkdir(exist_ok=True)
    (folder/'bad_days.parquet').symlink_to((OLD/'quality/bad_days.parquet').resolve())
    quality['reused_quality_report_sha256'] = sha(OLD/'quality/report.json')
    quality['reused_quality_full_label_proof_sha256'] = sha(OLD/'full_label_verification.json')
    quality['input_manifest_sha256'] = sha(LABELS/'input_manifest.json')
    save_json(folder/'report.json', quality)
    r = dict(passed=True, input_manifest_sha256=sha(LABELS/'input_manifest.json'),
        reused_catalog_report_sha256=sha(OLD/'catalog/date_report.json'),
        reused_quality_report_sha256=sha(OLD/'quality/report.json'), all_codes_covered=True,
        quarter_wide_evidence_not_restricted_to_old_selection_dates=True, no_new_queries_or_prices_required=True)
    save_json(LABELS/'evidence_reuse.json', r); return r


def raw():
    p, m, keys = checked_keys(); assert not (LABELS/'raw_report.json').exists()
    sources = json.loads((inputs.OLD/'source_manifest.json').read_text())
    old_manifest = json.loads((OLD/'input_manifest.json').read_text())
    old_raw = json.loads((OLD/'raw_report.json').read_text())
    old_windows = json.loads((OLD/'window_report.json').read_text())
    old_proof = json.loads((OLD/'window_verification.json').read_text())
    assert old_manifest['keys_sha256'] == sha(OLD/'keys.parquet')
    assert old_raw['input_manifest_sha256'] == sha(OLD/'input_manifest.json')
    assert old_windows['raw_report_sha256'] == sha(OLD/'raw_report.json')
    assert old_proof['passed'] and old_proof['window_report_sha256'] == sha(OLD/'window_report.json')
    assert old_manifest['source_manifest_sha256'] == sha(inputs.OLD/'source_manifest.json')
    old_keys = pd.read_parquet(OLD/'keys.parquet', columns=['date', 'code', 'next_date'])
    identity = keys[['date', 'code', 'next_date']].merge(old_keys.assign(reused=True), on=['date', 'code'], how='left', validate='one_to_one', suffixes=('', '_old'))
    known = identity.reused.eq(True)
    assert identity.loc[known, 'next_date'].eq(identity.loc[known, 'next_date_old']).all()
    missing = keys.loc[~known, ['date', 'code', 'next_date']]
    folder = LABELS/'raw_parts'; folder.mkdir(exist_ok=True)
    parts = {}; count = 0; reused_rows = 0
    for index, (file, digest) in enumerate(old_raw['parts_sha256'].items()):
        assert sha(Path(file)) == digest
        f = pd.read_parquet(file).merge(keys[['date', 'code']], on=['date', 'code'], validate='many_to_one')
        path = folder/f'reused_{index:03d}.parquet'
        f.to_parquet(path, index=False, compression='zstd'); parts[str(path)] = sha(path)
        count += len(f); reused_rows += len(f)
    codes = sorted(missing.code.unique())
    for offset in range(0, len(codes), 64):
        subset = codes[offset:offset+64]; q = missing.loc[missing.code.isin(subset)]
        windows = pd.concat([q[['date', 'code']].assign(source_date=q.date, kind='entry'),
            q[['date', 'code']].assign(source_date=q.next_date, kind='morning')], ignore_index=True)
        files = [MINUTES/code[:2].upper()/(code[3:]+'.parquet') for code in subset]
        for file in files:
            assert sha(file) == sources['minute_sha256'][str(file)]
        c = base.conn(); c.register('windows', windows); c.read_parquet([str(x) for x in files]).create_view('source')
        f = c.execute('''WITH s AS(SELECT lower(exchange)||'.'||symbol AS code,
            strftime(timestamp,'%Y-%m-%d') AS source_date,strftime(timestamp,'%H:%M') AS clock,
            timestamp,open::DOUBLE AS open,high::DOUBLE AS high,low::DOUBLE AS low,close::DOUBLE AS close,
            volume::DOUBLE AS volume,turnover::DOUBLE AS amount FROM source
            WHERE timestamp>=CAST(? AS TIMESTAMP) AND timestamp<CAST(? AS TIMESTAMP)
            AND(strftime(timestamp,'%H:%M') BETWEEN '14:52' AND '14:55'
                OR strftime(timestamp,'%H:%M') BETWEEN '09:31' AND '10:00'))
            SELECT w.date,w.code,w.kind,s.* EXCLUDE(code) FROM s JOIN windows w USING(source_date,code)
            WHERE(w.kind='entry' AND clock BETWEEN '14:52' AND '14:55')
                OR(w.kind='morning' AND clock BETWEEN '09:31' AND '10:00')
            ORDER BY w.date,w.code,w.kind,timestamp''', [p['signal_first'], p['observation_last']+' 10:01']).df(); c.close()
        path = folder/f'added_{offset//64:03d}.parquet'; f.to_parquet(path, index=False, compression='zstd')
        parts[str(path)] = sha(path); count += len(f)
        print(json.dumps(dict(additional_codes=min(offset+64,len(codes)), total_codes=len(codes), raw_rows=count)), flush=True)
    r = dict(protocol_sha256=sha(study.PROTOCOL), input_manifest_sha256=sha(LABELS/'input_manifest.json'),
        old_raw_report_sha256=sha(OLD/'raw_report.json'), old_window_verification_sha256=sha(OLD/'window_verification.json'),
        parts_sha256=parts, raw_rows=count, reused_rows=reused_rows, reused_stock_days=int(known.sum()), added_stock_days=len(missing),
        first_signal=p['signal_first'], last_signal=p['signal_last'], last_observation=p['observation_last'],
        retain_30th_bar_for_unchanged_source_validity_only=True, evaluation_window='09:31-09:59',
        new_2026_prices_read=True, only_april_morning_prices_read=True, no_q2_signal_prices_read=True, no_exit_rules=True)
    save_json(LABELS/'raw_report.json', r); return {k:v for k,v in r.items() if k != 'parts_sha256'}


def setup_legacy():
    from . import tail_formula_forward_raw as raw_module
    from . import tail_formula_forward_labels as labels_module
    for module in [raw_module, labels_module]:
        module.PROTOCOL = study.PROTOCOL; module.OUT = inputs.OLD; module.LABELS = LABELS
        module.checked_keys = checked_keys
    labels_module.CATALOG = CATALOG
    return raw_module, labels_module


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'reuse_evidence', 'raw', 'windows', 'labels30']); a = parser.parse_args()
    if a.stage in ['windows', 'labels30']:
        raw_module, labels_module = setup_legacy()
        result = raw_module.windows() if a.stage == 'windows' else labels_module.labels()
    else:
        result = globals()[a.stage]()
    print(json.dumps(result, ensure_ascii=False, indent=2))
