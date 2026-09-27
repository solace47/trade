"""Four immediately preceding stock-day inputs for next-morning selection."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_context_2024 as linkage
from . import tail_formula_float as previous
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from .corporate_cash import save_json, sha
from .turnover_reference import CALENDAR

STEM = 'tail_formula_prior_day'
ROOT = Path('data/research')/STEM
PROTOCOL = Path('config')/(STEM+'_protocol.json')
COMBINED_PROTOCOL = Path('config')/(STEM+'_combined_protocol.json')
CONTROL = Path('data/research')/(STEM+'_control')
OLD_SELECTION = Path('data/research/tail_formula_float_2025')
NEW_EXPRESSIONS = {
    'Y01': '100*(DCP1/DCP2-1)/V01',
    'Y02': '100*(DCP1-YDL)/MAX(YDH-YDL,0.01)',
    'Y03': '100*(YDH-YDL)/DCP2/V01',
    'Y04': '20*YDV01/('+ '+'.join(f'YDV{i:02d}' for i in range(1,21))+')'}
EXPRESSIONS = {**previous.EXPRESSIONS, **NEW_EXPRESSIONS}
HEADER = previous.HEADER+'YDH:=REF(HHV(H,B0),B0);\nYDL:=REF(LLV(L,B0),B0);\n'
HEADER += ''.join(f'YDV{i:02d}:=REF(SUM(V,B0),B{i-1});\n' for i in range(1,21))


def source_files():
    p = json.loads(PROTOCOL.read_text())
    assert p['previous_feature_report_sha256'] == sha(previous.ROOT/'feature_report.json')
    assert p['daily_feature_report_sha256'] == sha(base.SOURCE/'feature_report.json')
    r = json.loads((previous.ROOT/'feature_report.json').read_text())
    v = json.loads((previous.ROOT/'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(previous.ROOT/'feature_report.json')
    assert r['features_sha256'] == sha(previous.ROOT/'features.parquet')
    assert p['history_stock_days'] == 20 and p['history_first'] == '2023-06-01'
    return previous.context.market.stock.source_files()


def features():
    if (ROOT/'feature_report.json').exists():
        raise ValueError('Do not replace the frozen preceding-day inputs')
    files = source_files(); c = base.conn(); c.read_parquet(files).create_view('daily')
    lagged = ','.join(f'lag({field}) OVER w AS yd_{field}' for field in ['open','high','low','close','volume','preclose','adjustflag'])
    hist = c.sql(f'''WITH active AS(SELECT date,code,open::DOUBLE AS open,high::DOUBLE AS high,
        low::DOUBLE AS low,close::DOUBLE AS close,volume::DOUBLE AS volume,
        preclose::DOUBLE AS preclose,adjustflag::DOUBLE AS adjustflag FROM daily
        WHERE date BETWEEN '2023-06-01' AND '2025-12-30' AND tradestatus=1),
        a AS(SELECT *,coalesce(isfinite(volume) AND volume>0 AND volume=floor(volume) AND adjustflag=3,false) AS good FROM active),
        h AS(SELECT date,code,{lagged},lag(date) OVER w AS yd_source_date,lag(date,2) OVER w AS yd_reference_date,
            lag(close,2) OVER w AS yd_previous_close,lag(adjustflag,2) OVER w AS yd_reference_adjustflag,
            avg(volume) OVER hist AS yd_volume_mean,count(*) OVER hist AS yd_volume_rows,
            sum(good::INT) OVER hist AS yd_volume_good,min(date) OVER hist AS yd_volume_first_date
            FROM a WINDOW w AS(PARTITION BY code ORDER BY date),
            hist AS(PARTITION BY code ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING))
        SELECT * FROM h WHERE date>='2024-01-01' ORDER BY date,code''').df(); c.close()
    old = pd.read_parquet(previous.ROOT/'features.parquet')
    f = old.merge(hist, on=['date','code'], how='left', validate='one_to_one').sort_values(['date','code']).reset_index(drop=True)
    prices = f[['yd_open','yd_high','yd_low','yd_close','yd_previous_close']]
    valid = (np.isfinite(prices).all(axis=1) & prices.gt(0).all(axis=1)
        & (prices-prices.round(2)).abs().le(.0001).all(axis=1)
        & f.yd_high.add(.0001).ge(f[['yd_open','yd_low','yd_close']].max(axis=1))
        & f.yd_low.sub(.0001).le(f[['yd_open','yd_close']].min(axis=1))
        & f.yd_adjustflag.eq(3) & f.yd_reference_adjustflag.eq(3)
        & f.yd_volume_rows.eq(20) & f.yd_volume_good.eq(20) & f.yd_volume_mean.gt(0)
        & f.yd_source_date.lt(f.date) & f.yd_reference_date.lt(f.yd_source_date) & f.V01.gt(0))
    f['prior_day_valid'] = valid
    f['Y01'] = (100*(f.yd_close/f.yd_previous_close-1)/f.V01).where(valid)
    f['Y02'] = (100*(f.yd_close-f.yd_low)/np.maximum(f.yd_high-f.yd_low,.01)).where(valid)
    f['Y03'] = (100*(f.yd_high-f.yd_low)/f.yd_previous_close/f.V01).where(valid)
    f['Y04'] = (f.yd_volume/f.yd_volume_mean).where(valid)
    cal = pd.read_parquet(CALENDAR)
    days = sorted(cal.loc[cal.is_trading_day.eq('1') & cal.calendar_date.between('2023-06-01','2025-12-30'),'calendar_date'])
    ranks = {d:i for i,d in enumerate(days)}
    f['yd_source_gap'] = f.date.map(ranks)-f.yd_source_date.map(ranks)
    f['yd_history_span'] = f.yd_source_date.map(ranks)-f.yd_volume_first_date.map(ranks)+1
    f['yd_reference_break'] = (f.yd_preclose-f.yd_previous_close).abs().gt(.005)
    f['prior_formula_input_valid'] = f.formula_input_valid
    f['formula_input_valid'] &= valid & np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    ROOT.mkdir(parents=True,exist_ok=True); f.to_parquet(ROOT/'features.parquet',index=False,compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), previous_feature_report_sha256=sha(previous.ROOT/'feature_report.json'),
        daily_feature_report_sha256=sha(base.SOURCE/'feature_report.json'), calendar_sha256=sha(CALENDAR),
        features_sha256=sha(ROOT/'features.parquet'), rows=len(f), valid=int(f.formula_input_valid.sum()),
        previous_valid=int(f.prior_formula_input_valid.sum()), newly_invalid=int((f.prior_formula_input_valid&~f.formula_input_valid).sum()),
        flat_prior_days=int((f.formula_input_valid & f.yd_high.eq(f.yd_low)).sum()),
        prior_day_reference_breaks=int((f.formula_input_valid&f.yd_reference_break).sum()),
        prior_stock_day_gaps=int((f.formula_input_valid&f.yd_source_gap.gt(1)).sum()),
        volume_history_gaps=int((f.formula_input_valid&f.yd_history_span.gt(20)).sum()),
        first_history_date=f.yd_volume_first_date.min(), last_history_date=f.yd_source_date.max(),
        expressions=EXPRESSIONS, native_header=HEADER, all_previous_48_inputs_retained=True,
        native_source_parity_verified=False, software_compilation_verified=False,
        outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT/'feature_report.json',r)
    return {k:v for k,v in r.items() if k not in ['expressions','native_header']}


def control(stage):
    r = json.loads((ROOT/'feature_report.json').read_text()); v = json.loads((ROOT/'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(ROOT/'feature_report.json')
    assert r['features_sha256'] == sha(ROOT/'features.parquet')
    assert sha(OLD_SELECTION/'selection_report.json') == '07de34caa25a339fa8e7b8d3665c7d04c044792d01fa170d0fd48f6b77b8f5c5'
    oldr = json.loads((OLD_SELECTION/'selection_report.json').read_text()); oldv = json.loads((OLD_SELECTION/'selection_verification.json').read_text())
    assert oldv['passed'] and oldv['selection_report_sha256'] == sha(OLD_SELECTION/'selection_report.json')
    assert oldr['selection_sha256'] == sha(OLD_SELECTION/'selection.parquet')
    old = pd.read_parquet(OLD_SELECTION/'selection.parquet')
    if stage == 'freeze_control':
        if (CONTROL/'selection_report.json').exists():
            raise ValueError('Do not replace the same-quality original control')
        assert not any((Path('data/research')/(STEM+'_'+fold)/'analysis_report.json').exists() for fold in ['2024','recent','2025'])
        f = pd.read_parquet(ROOT/'features.parquet',columns=['date','code','formula_input_valid'])
        pd.testing.assert_frame_equal(old[['date','code']],f[['date','code']],check_exact=True)
        out = old.copy(); out['selected'] &= f.formula_input_valid
        CONTROL.mkdir(exist_ok=True); out.to_parquet(CONTROL/'selection.parquet',index=False,compression='zstd')
        result = dict(protocol_sha256=sha(COMBINED_PROTOCOL),feature_report_sha256=sha(ROOT/'feature_report.json'),
            original_selection_report_sha256=sha(OLD_SELECTION/'selection_report.json'),selection_sha256=sha(CONTROL/'selection.parquet'),
            selected=int(out.selected.sum()),unchanged_original_selection=out.equals(old),model_refitted=False,
            new_group_outcomes_read=False,year_2025_is_exploratory=True,new_2026_prices_read=False,no_exit_rules=True)
        save_json(CONTROL/'selection_report.json',result); return result
    r = json.loads((CONTROL/'selection_report.json').read_text())
    for key,path in [('protocol_sha256',COMBINED_PROTOCOL),('feature_report_sha256',ROOT/'feature_report.json'),
                     ('original_selection_report_sha256',OLD_SELECTION/'selection_report.json'),('selection_sha256',CONTROL/'selection.parquet')]:
        assert r[key] == sha(path)
    c = base.conn()
    expected = c.sql(f'''SELECT s.date,s.code,s.half,s.board,s.decision_shares,s.selected AND f.formula_input_valid AS selected
        FROM read_parquet('{OLD_SELECTION}/selection.parquet') s JOIN read_parquet('{ROOT}/features.parquet') f USING(date,code)
        ORDER BY date,code''').df(); c.close()
    pd.testing.assert_frame_equal(pd.read_parquet(CONTROL/'selection.parquet'),expected,check_exact=True)
    assert r['unchanged_original_selection'] == expected.equals(old)
    proof = dict(passed=True,selection_report_sha256=sha(CONTROL/'selection_report.json'),rows=len(expected),
        all_control_selection_flags_rebuilt=True,outcomes_read=False,new_2026_prices_read=False)
    save_json(CONTROL/'selection_verification.json',proof); return proof


def setup(fold):
    if fold == 'combined':
        p = json.loads(COMBINED_PROTOCOL.read_text())
        paths = [Path('config')/(STEM+'_'+f+'_protocol.json') for f in ['2024','recent']]
        assert p['fold_protocols'] == [str(x) for x in paths]
        for path, name in zip(paths,['2024','recent']):
            for file in ['model_report.json','selection_report.json']:
                assert json.loads((Path('data/research')/(STEM+'_'+name)/file).read_text())['protocol_sha256'] == sha(path)
        linkage.ROOT = Path('data/research')/(STEM+'_2024'); linkage.H2 = Path('data/research')/(STEM+'_recent')
        linkage.COMBINED = Path('data/research')/(STEM+'_2025'); linkage.PROTOCOL = COMBINED_PROTOCOL
        linkage.H2_SELECTION_SHA = None; linkage.H2_MODEL_SHA = None; linkage.H2_OUTCOMES_PREVIOUSLY_SEEN = False
    else:
        previous.setup(fold)
        root = Path('data/research')/(STEM+'_'+fold); protocol = Path('config')/(STEM+'_'+fold+'_protocol.json')
        base.ROOT = root; base.PROTOCOL = protocol; relative.PROTOCOL = protocol; study.ROOT = root; study.PROTOCOL = protocol
        base.FEATURES = ROOT; base.EXPRESSIONS = EXPRESSIONS; base.HEADER = HEADER


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['features','model','verify_model','scores','freeze','verify','analyze','freeze_control','verify_control'])
    p.add_argument('--fold',choices=['2024','recent','combined'],default='2024'); a = p.parse_args()
    if a.stage == 'features':
        result = features()
    elif a.stage.endswith('_control'):
        result = control(a.stage)
    else:
        setup(a.fold)
        if a.fold == 'combined':
            assert a.stage in ['freeze','verify','analyze']
            result = (linkage.common_analysis(linkage.COMBINED,linkage.PROTOCOL) if a.stage == 'analyze'
                      else getattr(linkage,'combine' if a.stage == 'freeze' else 'verify_combined')())
        elif a.stage in ['model','verify_model']:
            result = getattr(relative,a.stage)('relative')
        elif a.stage in ['freeze','verify']:
            result = getattr(study,a.stage)()
        else:
            result = getattr(base,a.stage)()
    print(json.dumps(result,ensure_ascii=False,indent=2))
