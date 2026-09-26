"""Fixed CSI 1000 deletion cohort from contemporaneous official attachments."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re

import duckdb
import numpy as np
import pandas as pd

from .absolute_ridge import match_controls
from .corporate_cash import save_json, sha
from .quote_precision import fixed_quote_shares, quote_cents

ROOT = Path('data/research/index_rebalance')
RULE_COMMIT = 'cf4a922'
BATCHES = {'15267': '2024-06-14', '15471': '2024-12-13',
           '15690': '2025-06-13', '3006000': '2025-12-12'}


def parse_attachment(pages: list[str]) -> pd.DataFrame:
    """Track actual-index headings across pages, never treating reserves as adds."""
    rows, index = [], None
    for page_number, page in enumerate(pages, 1):
        for line in page.splitlines():
            compact = re.sub(r'\s+', '', line)
            header = re.fullmatch(r'((?:沪深|中证)[A0-9]+)指数样本调整名单[：:]', compact)
            if header:
                index = header[1]
                continue
            if '指数备选名单' in compact:
                index = None
                continue
            if index is None or not re.match(r'^\s*\d{6}\s', line):
                continue
            record = re.fullmatch(r'\s*(\d{6})\s+(.+?)\s+(\d{6})\s+(.+?)\s*', line)
            if not record:
                raise ValueError(f'Unparsed actual adjustment row on page {page_number}')
            for side, code, name in [('delete', record[1], record[2]), ('add', record[3], record[4])]:
                rows.append({'index': index, 'side': side, 'symbol': code,
                             'name': re.sub(r'\s+', '', name), 'page': page_number})
    result = pd.DataFrame(rows)
    if result.empty or result.duplicated(['index', 'side', 'symbol']).any():
        raise ValueError('Missing or duplicate adjustment identities')
    if result.duplicated(['index', 'symbol']).any():
        raise ValueError('A security cannot be both added and removed from one index')
    return result


def source_cohort(folder: Path) -> tuple[pd.DataFrame, list[dict]]:
    frames, sources = [], []
    for notice_id, date in BATCHES.items():
        stem = folder / f'notice_{notice_id}'
        notice = json.loads(stem.with_suffix('.json').read_text())
        expected = {m[1]: int(m[2]) for m in re.finditer(
            r'((?:沪深|中证)[A0-9]+)指数更换(\d+)只样本', notice['article'])}
        effective = re.search(r'于(\d{4})年(\d+)月(\d+)日收市后生效', notice['article'])
        if not effective or '-'.join([effective[1], effective[2].zfill(2), effective[3].zfill(2)]) != date:
            raise ValueError('Effective date does not match the frozen protocol')
        if not '2024-01-01' <= notice['published'] < date <= '2025-12-31':
            raise ValueError('The historical announcement must precede the decision')
        primary = parse_attachment(json.loads(stem.with_suffix('.pages.json').read_text()))
        independent = parse_attachment(json.loads(stem.with_suffix('.independent_text.json').read_text()))
        keys = ['index', 'side', 'symbol', 'name', 'page']
        pd.testing.assert_frame_equal(primary[keys], independent[keys])
        for idx, group in primary.groupby('index'):
            if idx not in expected or group.side.value_counts().to_dict() != {'delete': expected[idx], 'add': expected[idx]}:
                raise ValueError('PDF row counts do not match the original notice')
        if set(primary['index']) != set(expected) or expected.get('中证1000') != 100:
            raise ValueError('The complete historical index set is required')
        # Grid extraction is independent of the heading-aware text row parser.
        tables = json.loads(stem.with_suffix('.tables.json').read_text())
        grid_pairs = [tuple((str(r[0]), str(r[2]))) for page in tables for table in page for r in table
                      if len(r) == 4 and re.fullmatch(r'\d{6}', str(r[0]))
                      and re.fullmatch(r'\d{6}', str(r[2]))]
        text_pairs = list(zip(primary.loc[primary.side.eq('delete'), 'symbol'],
                              primary.loc[primary.side.eq('add'), 'symbol']))
        if Counter(grid_pairs) != Counter(text_pairs):
            raise ValueError('Independent PDF grids disagree with the parsed changes')
        primary['date'], primary['published'], primary['notice_id'] = date, notice['published'], notice_id
        primary['code'] = np.where(primary.symbol.str.startswith('6'), 'sh.', 'sz.') + primary.symbol
        if not primary.symbol.str.startswith(('0', '3', '6')).all():
            raise ValueError('An unhandled stock exchange occurs in the source')
        frames.append(primary)
        suffixes = ['.json', '.pdf', '.pages.json', '.independent_text.json', '.tables.json']
        if stem.with_suffix('.download.json').exists():
            suffixes.append('.download.json')
        sources.append({'notice_id': notice_id, 'date': date, 'published': notice['published'],
            'url': notice['url'], 'attachment_url': notice['links'][0]['url'], 'index_counts': expected,
            'sha256': {str(stem.with_suffix(s)): sha(stem.with_suffix(s)) for s in suffixes},
            'independent_text_and_grid_agree': True})
    return pd.concat(frames, ignore_index=True), sources


def valid_cent(price: float) -> bool:
    try:
        quote_cents(price)
        return True
    except ValueError:
        return False


def visible_pool() -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    prefixes = sorted(Path('data/research/minute_prefix_1449').glob('202[45]/*.parquet'))
    snapshots = sorted(Path('data/research/market_snapshots_ci').glob('*.parquet'))
    c = duckdb.connect()
    c.execute('SET threads=4')
    c.read_parquet([str(p) for p in prefixes]).create_view('prefix')
    c.read_parquet([str(p) for p in snapshots]).create_view('snapshots')
    c.register('decision_dates', pd.DataFrame({'date': list(BATCHES.values())}))
    joined = c.execute('''SELECT p.date,p.code,p.price_1449,p.amount_1449,p.volume_1449,
            p.high_1449,p.low_1449,p.quote_outside_traded_range,
            s.preclose,s.isST,s.listing_age_sessions,s.reference_gap,s.return20_prior_adjusted
        FROM prefix p JOIN decision_dates d USING(date)
        LEFT JOIN snapshots s USING(date,code) ORDER BY p.date,p.code''').df()
    c.close()
    if joined.duplicated(['date', 'code']).any():
        raise ValueError('Duplicate visible stock-day')
    joined['main_board'] = joined.code.str.startswith(('sh.60', 'sz.00'))
    joined['valid_cent'] = joined.price_1449.map(valid_cent)
    joined['eligible'] = (joined.main_board & joined.isST.eq(0) & joined.listing_age_sessions.ge(60)
        & joined.reference_gap.eq(False) & joined.quote_outside_traded_range.eq(False)
        & joined.price_1449.ge(5) & joined.volume_1449.gt(0) & joined.amount_1449.gt(0)
        & joined.preclose.gt(0) & joined.valid_cent)
    frame = joined.loc[joined.eligible].copy()
    frame['raw_price_1449'] = frame.price_1449
    frame['price_1449'] = frame.price_1449.map(lambda p: quote_cents(p)/100)
    frame['return_1450'] = frame.price_1449/frame.preclose-1  # Legacy matching name; uses 14:49.
    frame['board'], frame['price_signal'], frame['amount_signal'] = 'main', frame.price_1449, frame.amount_1449
    return frame, joined, prefixes+snapshots


def freeze(output: Path = ROOT) -> dict:
    if (output/'repriced.parquet').exists():
        raise ValueError('Cannot replace event inputs after reading their outcomes')
    changes, sources = source_cohort(output/'source_docs')
    base, joined, input_files = visible_pool()
    deleted = changes.loc[changes['index'].eq('中证1000') & changes.side.eq('delete')].copy()
    promotions = set(zip(changes.loc[changes.side.eq('add'), 'date'], changes.loc[changes.side.eq('add'), 'code']))
    deleted['also_added_elsewhere'] = [(d,c) in promotions for d,c in zip(deleted.date,deleted.code)]
    audit = deleted.merge(joined, on=['date','code'], how='left', validate='one_to_one', indicator=True)
    candidate_keys = audit.loc[~audit.also_added_elsewhere & audit.eligible.eq(True), ['date','code']]
    chosen = candidate_keys.merge(base, on=['date','code'], validate='one_to_one')
    chosen = chosen.sort_values(['date','amount_1449','code'], ascending=[True,False,True])
    chosen['daily_rank'] = chosen.groupby('date').cumcount()+1
    changed_keys = set(zip(changes.date, changes.code))
    chosen_keys = set(zip(chosen.date, chosen.code))
    controls_pool = base.loc[[(d,c) not in changed_keys or (d,c) in chosen_keys for d,c in zip(base.date,base.code)]]
    controls = match_controls(chosen[['date','code','daily_rank']], controls_pool)
    controls = controls.merge(base, on=['date','code'], validate='one_to_one')
    chosen['arm'], chosen['pair_id'] = 'high', chosen.code
    controls['arm'] = 'low'
    signals = pd.concat([chosen,controls], ignore_index=True).sort_values(['date','arm','daily_rank','code'])
    signals['pair_id'] = signals.date+':'+signals.pair_id
    signals['primary_top5'] = signals.daily_rank.le(5)
    signals['half'] = signals.date.str[:4]+np.where(signals.date.str[5:7].astype(int).le(6),'H1','H2')
    signals['decision_shares'] = [fixed_quote_shares(r.code,r.price_1449,20000) for r in signals.itertuples()]
    if signals.duplicated(['date','code']).any() or len(deleted) != 400:
        raise ValueError('Invalid complete event identities')
    for name, frame in [('changes',changes),('deletion_audit',audit),('base',base),('signals',signals)]:
        frame.to_parquet(output/(name+'.parquet'),index=False,compression='zstd')
    save_json(output/'source_docs/notice_sources.json',sources)
    cells=[]
    for date in BATCHES.values():
        a=audit.loc[audit.date.eq(date)]
        h=chosen.loc[chosen.date.eq(date)]
        l=controls.loc[controls.date.eq(date)]
        cells.append({'date':date,'deletions':len(a),'also_added_elsewhere':int(a.also_added_elsewhere.sum()),
            'missing_visible_join':int(a._merge.ne('both').sum()),'qualified':len(h),'controls':len(l),
            'primary':int(h.daily_rank.le(5).sum()),'primary_controls':int(l.daily_rank.le(5).sum()),
            'unmatched':len(h)-len(l)})
    result={'rule_commit':RULE_COMMIT,'by_batch':cells,'signals_sha256':sha(output/'signals.parquet'),
        'input_sha256':{str(p):sha(p) for p in input_files},'sources':sources,
        'candidates':len(chosen),'controls':len(controls),'notional':20000,'horizons':[1,5],
        'holdout_read':False,'new_selected_returns_read':False,'number_of_independent_batches':4}
    save_json(output/'input_report.json',result)
    return result


if __name__ == '__main__':
    report=freeze()
    print(json.dumps({k:report[k] for k in ['by_batch','signals_sha256','candidates','controls']},ensure_ascii=False,indent=2))
