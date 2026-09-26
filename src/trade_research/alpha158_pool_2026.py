"""Frozen 2026 temporal validation inputs; never fit on validation prices."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .absolute_ridge import choose, match_controls
from .absolute_ridge_1449 import feature_frame
from .alpha158_asof import DEFINITION, features_for_symbol
from .alpha158_inputs import DAILY, DAILY_COLUMNS, PREFIX_COLUMNS
from .alpha158_models import transform
from .corporate_cash import save_json, sha
from .minute_prefix_1449 import build_year
from .positive_pool_1449 import choose_pool, SALT
from .quote_precision import quote_cents, fixed_quote_shares
from .reference_gain_eval import MINUTES
from .shallow_tree_1449 import decision_buyable
from .turnover_reference import CALENDAR

ROOT = Path('data/research/alpha158_pool_2026')
PROTOCOL = Path('config/alpha158_pool_2026_protocol.json')
RULE_COMMIT = '47073f3'


def policy() -> dict:
    frozen = json.loads(PROTOCOL.read_text())
    if (sha(Path(frozen['model_path'])) != frozen['model_sha256']
            or frozen['hash_salt'] != SALT or frozen['refit'] is not False):
        raise ValueError('Frozen validation weights or selection rule changed')
    return frozen


def unchanged(path: Path, content: dict) -> None:
    if path.exists() and json.loads(path.read_text()) != content:
        raise ValueError(f'Frozen validation source changed: {path}')
    save_json(path, content)


def prepare(output: Path = ROOT) -> dict:
    p = policy()
    if (output / 'input_report.json').exists():
        raise ValueError('Cannot replace validation inputs after selections are frozen')
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted([*(MINUTES/'SH').glob('60*.parquet'), *(MINUTES/'SZ').glob('00*.parquet')])
    if not paths:
        raise ValueError('No historical main-board raw-minute files')
    source = {'rule_commit': RULE_COMMIT, 'protocol_sha256': sha(PROTOCOL),
        'minute_sha256': {str(f): sha(f) for f in paths},
        'snapshot_sha256': {str(f): sha(f) for f in sorted(Path('data/research/market_snapshots_ci').glob('*.parquet'))},
        'calendar_sha256': sha(CALENDAR), 'definition_sha256': sha(DEFINITION),
        'code_sha256': {name: sha(Path(__file__).with_name(name)) for name in
            ('minute_prefix_1449.py', 'absolute_ridge_1449.py', 'alpha158_asof.py', 'alpha158_pool_2026.py')},
        'first': p['signal_first'], 'last': p['signal_last'], 'new_holding_results_read': False}
    unchanged(output/'prefix_manifest.json', source)
    for start in range(0, len(paths), 256):
        part = output/'prefix'/f'part_{start//256:03d}.parquet'
        if not part.exists():
            build_year([str(f) for f in paths[start:start+256]], 2026, part,
                       threads=4, validation_last=p['signal_last'])
        print(f'2026 prefixes {min(start+256,len(paths))}/{len(paths)}', flush=True)
    c = duckdb.connect(); c.execute('SET threads=4')
    base = feature_frame(c, prefix_pattern=str(output/'prefix/*.parquet'),
        date_ranges=((p['signal_first'], p['signal_last']),), validation_last=p['signal_last'])
    c.close()
    base['decision_buyable'] = [decision_buyable(r.code, r.price_1449, r.preclose) for r in base.itertuples()]
    base = base.loc[base.decision_buyable & base.price_1449.ge(5)].sort_values(['date','code']).reset_index(drop=True)
    base.to_parquet(output/'base_features.parquet', index=False, compression='zstd')
    report = {'rows':len(base), 'symbols':base.code.nunique(), 'dates':base.date.nunique(),
        'first':base.date.min(), 'last':base.date.max(), 'base_sha256':sha(output/'base_features.parquet'),
        'manifest_sha256':sha(output/'prefix_manifest.json'),
        'prefix_sha256':{str(f):sha(f) for f in sorted((output/'prefix').glob('*.parquet'))},
        'holdout_inputs_read':True, 'new_holding_results_read':False}
    save_json(output/'base_report.json', report)
    return report


def feature_part(item) -> dict:
    code, prefix, output, last_date, adapter_hash = item
    root = Path(output)/'parts'; root.mkdir(parents=True, exist_ok=True)
    path = root/f'{code}.parquet'; report_path = path.with_suffix('.json')
    daily_path = DAILY/(code.replace('.','_')+'.parquet')
    expected = {'code':code, 'daily_sha256':sha(daily_path), 'last_date':last_date,
        'prefix_sha256':hashlib.sha256(pd.util.hash_pandas_object(prefix,index=False).values.tobytes()).hexdigest(),
        'adapter_sha256':adapter_hash}
    if path.exists() and report_path.exists():
        cached = json.loads(report_path.read_text())
        if all(cached.get(k)==v for k,v in expected.items()) and sha(path)==cached['output_sha256']:
            return cached
        raise ValueError('Changed frozen validation feature part')
    daily = pd.read_parquet(daily_path, columns=DAILY_COLUMNS,
        filters=[('date','>=','2019-01-01'),('date','<=',last_date)])
    values = features_for_symbol(daily,prefix); values.insert(1,'code',code)
    pd.testing.assert_series_equal(values.date,prefix.date.reset_index(drop=True))
    values.to_parquet(path,index=False,compression='zstd')
    result = {**expected, 'path':str(path), 'rows':len(values), 'output_sha256':sha(path)}
    save_json(report_path,result)
    return result


def features(output: Path = ROOT) -> dict:
    p = policy(); base_report = json.loads((output/'base_report.json').read_text())
    if sha(output/'base_features.parquet') != base_report['base_sha256'] or (output/'input_report.json').exists():
        raise ValueError('Base changed or selections already frozen')
    base = pd.read_parquet(output/'base_features.parquet', columns=PREFIX_COLUMNS).sort_values(['code','date'])
    adapter = sha(Path(__file__).with_name('alpha158_asof.py'))
    jobs = [(code,g.reset_index(drop=True),str(output),p['signal_last'],adapter) for code,g in base.groupby('code',sort=True)]
    parts = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i, result in enumerate(pool.map(feature_part,jobs),1):
            parts.append(result)
            if i%100==0 or i==len(jobs):
                print({'2026_features':i,'total':len(jobs)},flush=True)
    values = pd.concat([pd.read_parquet(x['path']) for x in parts],ignore_index=True).sort_values(['date','code']).reset_index(drop=True)
    expected = base[['date','code']].sort_values(['date','code']).reset_index(drop=True)
    pd.testing.assert_frame_equal(values[['date','code']],expected)
    values.to_parquet(output/'features.parquet',index=False,compression='zstd')
    save_json(output/'feature_parts.json',parts)
    result = {'rows':len(values),'symbols':len(parts),'features_sha256':sha(output/'features.parquet'),
        'base_sha256':base_report['base_sha256'],'adapter_sha256':adapter,
        'missing_by_feature':values.drop(columns=['date','code']).isna().sum().to_dict(),
        'holdout_inputs_read':True,'new_holding_results_read':False}
    save_json(output/'feature_report.json',result)
    return result


def freeze(output: Path = ROOT) -> dict:
    p = policy()
    if (output/'input_report.json').exists():
        raise ValueError('Do not overwrite the validation selection')
    report = json.loads((output/'feature_report.json').read_text())
    for name,key in [('base_features','base_sha256'),('features','features_sha256')]:
        if sha(output/(name+'.parquet')) != report[key]:
            raise ValueError('Frozen validation feature file changed')
    checks = json.loads((output/'independent_feature_checks.json').read_text())
    if checks['factor_values_checked'] < 158*24 or checks['maximum_difference']>5e-8:
        raise ValueError('Independent validation features have not passed')
    base, values = pd.read_parquet(output/'base_features.parquet'),pd.read_parquet(output/'features.parquet')
    pd.testing.assert_frame_equal(base[['date','code']],values[['date','code']])
    model = json.loads(Path(p['model_path']).read_text())
    with threadpool_limits(limits=4):
        score = transform(values,model)@np.array(model['coefficients'])+model['intercept']
    ranked = base[['date','code']].assign(score=score)
    if not np.isfinite(score).all():
        raise ValueError('A validation score is missing or infinite')
    ranked.to_parquet(output/'all_scores.parquet',index=False,compression='zstd')
    base['price_1449'] = base.price_1449.map(lambda q:quote_cents(q)/100)
    base['price_signal'] = base.price_1449
    table = pd.read_parquet(CALENDAR)
    calendar = sorted(table.loc[table.is_trading_day.eq('1') & table.calendar_date.between(p['signal_first'],p['execution_last']),'calendar_date'])
    models = {}
    for name,selector in [('positive_pool',choose_pool),('highest_score',choose)]:
        high = selector(ranked,calendar); low = match_controls(high,base)
        high['arm'],high['pair_id'] = 'high',high.code;low['arm']='low'
        members = pd.concat([high,low],ignore_index=True);members['pair_id']=members.date+':'+members.pair_id
        signals = members.merge(base,on=['date','code'],validate='one_to_one')
        signals['isST'],signals['reference_gap'],signals['listing_age_sessions'] = 0,False,20
        signals['half'] = '2026'+np.where(signals.date.str[5:7].le('06'),'H1','H2')
        signals['decision_shares'] = [fixed_quote_shares(r.code,r.price_1449,20000) for r in signals.itertuples()]
        signals = signals.sort_values(['date','arm','daily_rank','code']).reset_index(drop=True)
        if len(signals)!=len(members) or signals.duplicated(['date','code']).any():
            raise ValueError('Validation identities changed')
        folder = output/name;folder.mkdir(exist_ok=True)
        signals.to_parquet(folder/'signals.parquet',index=False,compression='zstd')
        part = {'candidates':len(high),'controls':len(low),'signals_sha256':sha(folder/'signals.parquet'),
            'dates':high.date.nunique(),'by_period':[{'period':label,'arm':arm,'rows':len(g),'dates':g.date.nunique()}
                for label,first,last in p['descriptive_periods'] for arm,g in signals.loc[signals.date.between(first,last)].groupby('arm')]}
        save_json(folder/'input_report.json',part);models[name]=part
    result = {'rule_commit':RULE_COMMIT,'protocol_sha256':sha(PROTOCOL),'model_sha256':p['model_sha256'],
        'all_scores_sha256':sha(output/'all_scores.parquet'),'feature_report_sha256':sha(output/'feature_report.json'),
        'models':models,'positive_pool_rows':int((score>0).sum()),'holdout_inputs_read':True,
        'new_holding_results_read':False,'strict_blind':False}
    save_json(output/'input_report.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','features','freeze'])
    args=parser.parse_args();result=globals()[args.stage]()
    print(json.dumps({k:v for k,v in result.items() if k not in ('prefix_sha256','missing_by_feature')},ensure_ascii=False,indent=2))
