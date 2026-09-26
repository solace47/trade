"""Independently verify the period-quality overlay without deleting any case."""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json,sha

source=Path('data/research/economic_winner');root=source/'period_quality'
report=json.loads((root/'label_report.json').read_text())
assert report['labels_sha256']==sha(root/'labels.parquet')
for name in ['label_report','label_verification','analysis_report']:
    assert report['original_'+name+'_sha256']==sha(source/(name+'.json'))
assert report['protocol_sha256']==sha(Path('config/economic_winner_quality.json'))
assert report['period_report_sha256']==sha(Path('data/research/quality_period_2024_2025.json'))
for path,digest in report['issue_files_sha256'].items():assert sha(Path(path))==digest
c=duckdb.connect();c.execute('SET threads=4')
c.read_parquet(str(source/'labels.parquet')).create_view('old')
c.read_csv(list(report['issue_files_sha256']),all_varchar=True).create_view('issues')
period=json.loads(Path('data/research/quality_period_2024_2025.json').read_text())
c.register('symbols',pd.DataFrame({'code':pd.Series(period['period_bad_symbols'],dtype='string')}))
flags=c.sql('''WITH bad AS(SELECT DISTINCT date,code FROM issues WHERE date BETWEEN '2024-01-01' AND '2025-12-31'
  AND kind IN('ohlc_disagreement','partial_minute_day','unexpected_minute_on_suspended_day','active_no_trade','missing_active_minute')),
  q AS(SELECT o.*,
    EXISTS(SELECT 1 FROM bad b WHERE b.code=o.code AND b.date=o.date) AS entry_bad,
    EXISTS(SELECT 1 FROM bad b WHERE b.code=o.code AND b.date=o.next_date) AS exit_bad,
    EXISTS(SELECT 1 FROM symbols b WHERE b.code=o.code) AS symbol_bad FROM old o)
  SELECT date,code,entry_bad,exit_bad,symbol_bad,
    necessary_tradeable AND(entry_bad OR symbol_bad OR(exit_bad AND label15<>'no_trade')) AS source_unknown
  FROM q ORDER BY date,code''').df()
keys=['date','code'];old=pd.read_parquet(source/'labels.parquet').set_index(keys).sort_index()
new=pd.read_parquet(root/'labels.parquet').set_index(keys).sort_index();flags=flags.set_index(keys).sort_index()
assert len(old)==len(new)==len(flags)==2404280
pd.testing.assert_index_equal(old.index,new.index);pd.testing.assert_index_equal(old.index,flags.index)
for target,field in [('period_entry_bad_day','entry_bad'),('period_exit_bad_day','exit_bad'),('period_bad_symbol','symbol_bad'),('period_source_unknown','source_unknown')]:
    np.testing.assert_array_equal(new[target],flags[field])
changed={'base_status'}
for bps in [5,15]:
    expected=old[f'label{bps}'].where(~flags.source_unknown,'unknown')
    pd.testing.assert_series_equal(new[f'label{bps}'],expected)
    np.testing.assert_array_equal(new[f'known_profit{bps}'],old[f'known_profit{bps}']&~flags.source_unknown)
    pd.testing.assert_series_equal(new[f'original_label{bps}'],old[f'label{bps}'],check_names=False)
    for field in ['buy_cash','sell_cash','net_return']:
        name=f'{field}{bps}';pd.testing.assert_series_equal(new[name],old[name].where(~flags.source_unknown))
    changed.update(f'{name}{bps}' for name in ['buy_cash','sell_cash','net_return','known_profit','label'])
    assert report['changed_counts'][str(bps)]==int(expected.ne(old[f'label{bps}']).sum())
    assert report['label_counts'][str(bps)]==new[f'label{bps}'].value_counts().to_dict()
untouched=[name for name in old.columns if name not in changed]
pd.testing.assert_frame_equal(new[untouched],old[untouched])
pd.testing.assert_series_equal(new.original_base_status,old.base_status,check_names=False)
expected_status=old.base_status.mask(flags.source_unknown&old.base_status.isin(['ordinary_t1','not_bought']),'period_source_unknown')
pd.testing.assert_series_equal(new.base_status,expected_status)
result={'passed':True,'label_report_sha256':sha(root/'label_report.json'),
    'rows':len(new),'unchanged_original_columns':len(untouched),'flags_checked':len(flags)*4,
    'changed_counts':report['changed_counts'],'new_2026_prices_read':False}
save_json(root/'label_verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
