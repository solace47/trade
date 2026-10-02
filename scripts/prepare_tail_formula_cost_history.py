"""Prepare five-stock-day cost quotations without forecasting or outcome reads."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

import probe_tail_formula_cost_history as proxy
from trade_research import tail_formula_additive as numeric
from trade_research import tail_formula_baseline as baseline
from trade_research.research_io import check_runtime, check_sources, save_json, sha

ROOT = Path('data/research/tail_formula_cost_history')
PROTOCOL = Path('config/tail_formula_cost_history_input_protocol.json')


def checked():
    check_runtime()
    p = json.loads(PROTOCOL.read_text())
    assert subprocess.check_output(['git','show',f'HEAD:{PROTOCOL}']) == PROTOCOL.read_bytes()
    assert sha(PROTOCOL) in subprocess.check_output(['git','show','HEAD:docs/selection-formula.md']).decode()
    check_sources(p['source_hashes'])
    source = json.loads(Path(p['source_manifest']).read_text())
    assert len(source['codes']) == 3223 and p['history_days'] == 5 and p['new_model_fits_allowed'] == 0
    return p, source


def calendar():
    p, source = checked()
    assert not (ROOT/'calendar_report.json').exists()
    frames = []
    for code in source['codes']:
        item = source['daily_sources'][code]; path = Path(item['path'])
        assert sha(path) == item['sha256']
        d = pd.read_parquet(path,columns=['date','code','tradestatus'],
            filters=[('date','>=',p['history_start']),('date','<=',p['signal_last'])])
        assert d.code.eq(code).all() and not d.date.duplicated().any()
        frames.append(d.loc[d.tradestatus.eq(1),['date','code']])
    f = pd.concat(frames,ignore_index=True).sort_values(['code','date']).reset_index(drop=True)
    assert f.date.le('2025-12-30').all() and not f.duplicated(['date','code']).any()
    ROOT.mkdir(exist_ok=True)
    f.to_parquet(ROOT/'stock_days.parquet',index=False,compression='zstd')
    save_json(ROOT/'calendar_report.json',dict(passed=True,protocol_sha256=sha(PROTOCOL),
        calendar_sha256=sha(ROOT/'stock_days.parquet'),rows=len(f),codes=f.code.nunique(),
        all_calendar_fields_from_fixed_sources=True,no_price_columns_read=True,new_2026_prices_read=False))
    return dict(calendar_sha256=sha(ROOT/'stock_days.parquet'),rows=len(f))


def windows():
    p, source = checked()
    assert not (ROOT/'window_report.json').exists()
    folder = ROOT/'window_parts'; folder.mkdir(exist_ok=True)
    previous = Path(p['pilot_root'])
    proof = json.loads((previous/'pilot_verified.json').read_text())
    assert proof['passed'] and proof['all_raw_windows_independently_SQL_and_Pandas_rebuilt']
    check_sources(proof['outputs'])
    pilot = pd.read_parquet(previous/'pilot_windows.parquet')
    codes = sorted(set(source['codes'])-set(proof['pilot_codes']))
    receipts, raw_rows = {}, 0
    for offset in range(0,len(codes),64):
        subset = codes[offset:offset+64]
        path = folder/f'part_{offset//64:03d}.parquet'; record = path.with_suffix('.json')
        digests = {source['minute_sources'][code]['path']:source['minute_sources'][code]['sha256'] for code in subset}
        identity = {source['minute_sources'][code]['path']:code for code in subset}
        if record.exists():
            meta = json.loads(record.read_text())
            assert meta['protocol_sha256'] == sha(PROTOCOL) and meta['producer_sha256'] == sha(Path(__file__))
            assert meta['codes'] == subset and meta['source_hashes'] == digests and meta['window_sha256'] == sha(path)
        else:
            for name, digest in digests.items():
                assert sha(Path(name)) == digest
            c = numeric.conn(); c.read_parquet(list(digests),filename=True).create_view('source')
            raw = c.execute('''SELECT filename,lower(exchange)||'.'||symbol AS code,
                strftime(timestamp,'%Y-%m-%d') AS date,timestamp,
                (extract(hour FROM timestamp)*100+extract(minute FROM timestamp))::INT AS clock,
                open::DOUBLE AS open,high::DOUBLE AS high,low::DOUBLE AS low,close::DOUBLE AS close,
                volume::DOUBLE AS volume,turnover::DOUBLE AS amount FROM source
                WHERE timestamp>=?::DATE AND timestamp<?::DATE+INTERVAL 1 DAY
                AND (strftime(timestamp,'%H%M') BETWEEN '0931' AND '0959'
                  OR strftime(timestamp,'%H%M') IN ('1449','1452','1453','1454','1455'))
                ORDER BY code,date,timestamp''',[p['history_start'],p['signal_last']]).df(); c.close()
            assert raw.code.eq(raw.filename.map(identity)).all() and raw.date.le('2025-12-30').all()
            assert raw.timestamp.eq(raw.timestamp.dt.floor('min')).all()
            f = proxy.aggregate(raw.drop(columns='filename'))
            f.to_parquet(path,index=False,compression='zstd')
            meta = dict(protocol_sha256=sha(PROTOCOL),producer_sha256=sha(Path(__file__)),
                codes=subset,source_hashes=digests,window_sha256=sha(path),raw_minutes=len(raw),
                all_raw_windows_SQL_and_Pandas_verified=True,all_file_identities_verified=True)
            save_json(record,meta)
        receipts[str(path)] = meta['window_sha256']; receipts[str(record)] = sha(record)
        raw_rows += meta['raw_minutes']
        print(json.dumps(dict(codes=offset+len(subset),total=len(codes),raw_minutes=raw_rows)),flush=True)
    frames = [pilot]+[pd.read_parquet(Path(n)) for n in receipts if n.endswith('.parquet')]
    full = pd.concat(frames,ignore_index=True).sort_values(['code','date']).reset_index(drop=True)
    assert not full.duplicated(['date','code']).any()
    full.to_parquet(ROOT/'windows.parquet',index=False,compression='zstd')
    save_json(ROOT/'window_report.json',dict(passed=True,protocol_sha256=sha(PROTOCOL),
        parts_sha256=receipts,windows_sha256=sha(ROOT/'windows.parquet'),rows=len(full),
        pilot_windows_reused_exactly=True,pilot_window_sha256=sha(previous/'pilot_windows.parquet'),
        new_raw_minutes=raw_rows,new_2026_prices_read=False,new_model_fits=0))
    return dict(window_report_sha256=sha(ROOT/'window_report.json'),rows=len(full))


def inputs():
    p, _ = checked()
    destination = ROOT/'feature_verification.json'; assert not destination.exists()
    cr = json.loads((ROOT/'calendar_report.json').read_text()); wr = json.loads((ROOT/'window_report.json').read_text())
    assert cr['passed'] and cr['calendar_sha256'] == sha(ROOT/'stock_days.parquet')
    assert wr['passed'] and wr['windows_sha256'] == sha(ROOT/'windows.parquet')
    check_sources(wr['parts_sha256'])
    days = pd.read_parquet(ROOT/'stock_days.parquet'); w = pd.read_parquet(ROOT/'windows.parquet')
    atoms = proxy.history(w,days)[['date','code','reference_date','proxy_mark','proxy_positive','atom_valid']]
    rolling = atoms.groupby('code',sort=False)[['proxy_mark','proxy_positive']].rolling(5,min_periods=5).mean().reset_index(level=0,drop=True)
    atoms['mean5'] = rolling.proxy_mark; atoms['frequency5'] = 100*rolling.proxy_positive
    atoms['history_valid'] = atoms.mean5.notna() & atoms.frequency5.notna()
    c = numeric.conn(); c.register('atoms',atoms[['date','code','proxy_mark','proxy_positive','atom_valid']])
    expected = c.sql('''SELECT date,code,CASE WHEN count(proxy_mark) OVER w=5 THEN avg(proxy_mark) OVER w END AS mean5,
        CASE WHEN count(proxy_positive) OVER w=5 THEN 100*avg(proxy_positive) OVER w END AS frequency5
        FROM atoms WINDOW w AS(PARTITION BY code ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)
        ORDER BY code,date''').df(); c.close()
    pd.testing.assert_frame_equal(atoms[['date','code','mean5','frequency5']],expected,check_dtype=False,atol=2e-10,rtol=0)
    assert atoms.loc[atoms.atom_valid,'reference_date'].lt(atoms.loc[atoms.atom_valid,'date']).all()
    # The pilot proves atom chronology; full-pool scalar replay checks the
    # changed five-observation window rather than silently reusing 20-day math.
    original = baseline.original()
    f = original.merge(atoms[['date','code','mean5','frequency5','history_valid']],on=['date','code'],how='left',validate='one_to_one')
    pd.testing.assert_frame_equal(f[original.columns],original,check_exact=True)
    f['CM01'] = f.mean5/f.V01; f['CM02'] = f.frequency5
    valid = original.formula_input_valid & f.history_valid.eq(True) & np.isfinite(f[['CM01','CM02']]).all(axis=1)
    f['original_formula_input_valid'] = f.formula_input_valid
    f['formula_input_valid'] = valid
    f = f.drop(columns=['mean5','frequency5','history_valid'])
    atoms.to_parquet(ROOT/'history_atoms.parquet',index=False,compression='zstd')
    f.to_parquet(ROOT/'features.parquet',index=False,compression='zstd')
    save_json(ROOT/'feature_report.json',dict(protocol_sha256=sha(PROTOCOL),features_sha256=sha(ROOT/'features.parquet'),
        history_atoms_sha256=sha(ROOT/'history_atoms.parquet'),rows=len(f),codes=f.code.nunique(),
        previous_valid=int(original.formula_input_valid.sum()),valid=int(valid.sum()),
        newly_invalid=int((original.formula_input_valid & ~valid).sum()),
        by_half=f.groupby('half').agg(rows=('code','size'),old_valid=('original_formula_input_valid','sum'),valid=('formula_input_valid','sum')).reset_index().to_dict('records'),
        all_original_metadata_and_50_values_unchanged=True,new_model_fits=0,new_strategy_economic_results_read=False,new_2026_prices_read=False,
        historical_quote_proxy_not_actual_profit_or_queue_fill=True,native_client_parity_verified=False))
    save_json(destination,dict(passed=True,protocol_sha256=sha(PROTOCOL),feature_report_sha256=sha(ROOT/'feature_report.json'),
        source_windows_SQL_and_Pandas_verified=True,all_five_stock_day_windows_SQL_rebuilt=True,
        no_bad_window_skipped_or_unknown_filled=True,all_original_metadata_and_50_values_unchanged=True,
        pilot_atom_time_mutation_receipt_sha256=sha(Path(p['pilot_root'])/'pilot_verified.json'),
        full_client_parity_verified=False,new_model_fits=0,new_2026_prices_read=False))
    return dict(feature_report_sha256=sha(ROOT/'feature_report.json'),verification_sha256=sha(destination),valid=int(valid.sum()),rows=len(f))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['calendar','windows','inputs'])
    print(json.dumps(globals()[parser.parse_args().stage](),ensure_ascii=False))
