"""Audit existing historical classifications using metadata columns only."""
import json
from pathlib import Path

import duckdb
import pandas as pd

from trade_research.corporate_cash import save_json, sha

ROOT=Path('data/research/tail_formula_industry_baseline')
ROOT.mkdir(parents=True,exist_ok=True)
OUTPUT=ROOT/'classification_metadata_lookup_verification.json'
assert not OUTPUT.exists()
old=Path('data/research/industry_intervals.parquet')
assert sha(old)==json.loads(Path('data/research/sector_strength_short/source_manifest.json').read_text())['source_sha256'][str(old)]
original=pd.read_parquet(old)
asofs=set(original['asof'].unique())
files=sorted(file for file in Path('data/research/industry_history').glob('*.parquet') if file.stem in asofs)
assert {file.stem for file in files}==asofs and len(files)==8
frames=[];hashes={str(old):sha(old),str(Path(__file__).relative_to(Path.cwd())):sha(Path(__file__))}
for file in files:
    frame=pd.read_parquet(file);asof=file.stem
    assert not frame.code.duplicated().any() and frame.updateDate.le(asof).all()
    assert frame.industryClassification.eq('证监会行业分类').all()
    frame['snapshot_date']=asof;frames.append(frame);hashes[str(file)]=sha(file)
history=pd.concat(frames,ignore_index=True)
c=duckdb.connect();c.execute('SET threads=2');c.register('snapshots',history)
rebuilt=c.sql('''WITH r AS(SELECT code,industry,snapshot_date AS "asof",updateDate,
    strftime(greatest(CAST(snapshot_date AS DATE),CAST(updateDate AS DATE)+INTERVAL 1 DAY),'%Y-%m-%d') AS effective_date
    FROM snapshots WHERE industry<>'') SELECT *,lead(effective_date) OVER(PARTITION BY code ORDER BY effective_date,"asof")
    AS next_effective_date FROM r ORDER BY code,effective_date,"asof"''').df()
pd.testing.assert_frame_equal(original,rebuilt[original.columns],check_exact=True,check_dtype=False)
assert not original.duplicated(['code','effective_date']).any()
labels=Path('data/research/tail_formula_morning_range/inputs/full_labels.parquet')
features=Path('data/research/tail_formula_morning_range/inputs/features.parquet')
hashes[str(labels)]=sha(labels);hashes[str(features)]=sha(features)
c.read_parquet(str(old)).create_view('industries');counts=[]
for label,path in [('all_label_keys',labels),('original_50_keys',features)]:
    keys=c.execute("SELECT date,code FROM read_parquet(?) WHERE date>='2024-01-01' AND date<'2026-01-01'",[str(path)]).df()
    c.register('keys',keys)
    member=c.sql('''SELECT k.date,k.code,h.industry,h."asof",h.updateDate,h.effective_date,h.next_effective_date
        FROM keys k LEFT JOIN industries h ON h.code=k.code AND k.date>=h.effective_date
        AND(h.next_effective_date IS NULL OR k.date<h.next_effective_date)
        AND h."asof"<k.date AND h.updateDate<k.date AND date_diff('day',CAST(h."asof" AS DATE),CAST(k.date AS DATE))<=370
        ORDER BY k.date,k.code''').df()
    assert len(member)==len(keys) and not member.duplicated(['date','code']).any()
    pd.testing.assert_frame_equal(member[['date','code']],keys.sort_values(['date','code']).reset_index(drop=True),check_exact=True)
    ok=member.industry.notna()
    assert member.loc[ok,'asof'].lt(member.loc[ok,'date']).all() and member.loc[ok,'updateDate'].lt(member.loc[ok,'date']).all()
    counts.append(dict(pool=label,rows=len(member),visible_classifications=int(ok.sum()),unknown_classifications=int((~ok).sum()),industries=member.industry.nunique()))
    if label=='all_label_keys':
        full=member;full.to_parquet(ROOT/'historical_memberships.parquet',index=False,compression='zstd')
c.close()
r=dict(passed=True,source_hashes=hashes,snapshots=len(files),intervals=len(original),
    full_memberships_sha256=sha(ROOT/'historical_memberships.parquet'),full_label_keys=len(full),counts=counts,
    historical_industry_intervals_independently_rebuilt=True,strictly_earlier_query_and_update_dates=True,
    all_label_and_original_50_key_joins_unique=True,unknown_memberships=int(full.industry.isna().sum()),
    no_price_score_or_outcome_columns_read=True,new_2026_prices_read=False,no_exit_rules=True)
save_json(OUTPUT,r)
print(json.dumps(dict(metadata_sha256=sha(OUTPUT),counts=counts),ensure_ascii=False))
