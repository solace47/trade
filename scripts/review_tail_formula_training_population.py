"""Zero-fit audit of the benchmark used by an existing relative target."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from trade_research.tail_formula_additive import conn
from trade_research.research_io import check_runtime, check_sources, sha, save_json

PROTOCOL = Path('config/tail_formula_training_population.json')


def main():
    check_runtime()
    p = json.loads(PROTOCOL.read_text())
    assert subprocess.check_output(['git', 'show', f'HEAD:{PROTOCOL}']) == PROTOCOL.read_bytes()
    assert sha(PROTOCOL) in subprocess.check_output(['git', 'show', 'HEAD:docs/selection-formula.md']).decode()
    check_sources(p['source_hashes'])
    gate = json.loads(Path(p['completed_receipt']).read_text())
    assert gate['passed']; check_sources(gate['source_hashes'])
    root = Path(p['output_root']); root.mkdir(exist_ok=False)
    c = conn()
    f = pd.read_parquet(p['features'], columns=['date', 'code', 'formula_input_valid'])
    assert len(f) == 1815129 and f.date.lt('2026-01-01').all()
    c.register('features', f)
    daily = []; summaries = []; sources = dict(p['source_hashes'])
    for spec in p['folds']:
        for arm in ['original', 'primary']:
            path = Path(spec[arm + '_labels']); m = json.loads(Path(spec[arm + '_model']).read_text())
            assert m['last_observation'] < spec['training_end'] and m['variant'] == 'relative'
            labels = pd.read_parquet(path, columns=['date', 'code', 'next_date', 'known15', 'opportunity15'])
            assert labels.date.lt('2026-01-01').all() and not labels.duplicated(['date', 'code']).any()
            c.register('labels', labels)
            d = c.execute('''WITH scoped AS(SELECT date,code,opportunity15,formula_input_valid
                FROM labels JOIN features USING(date,code)
                WHERE known15 AND date>=? AND next_date<?)
                SELECT date,count(*)::BIGINT AS source_rows,
                sum(formula_input_valid::INT)::BIGINT AS fitted_rows,
                avg(opportunity15) AS source_mean,
                avg(opportunity15) FILTER(WHERE formula_input_valid) AS fitted_mean,
                avg(opportunity15) FILTER(WHERE NOT formula_input_valid) AS invalid_mean,
                var_pop(opportunity15) FILTER(WHERE formula_input_valid) AS within_date_variance
                FROM scoped GROUP BY date ORDER BY date''', [spec['training_start'], spec['training_end']]).df()
            mature = labels.loc[labels.known15 & labels.date.ge(spec['training_start']) & labels.next_date.lt(spec['training_end'])]
            q = mature.merge(f, on=['date', 'code'], how='left', validate='one_to_one', indicator=True)
            assert q._merge.eq('both').all() and int(d.source_rows.sum()) == len(mature)
            ex = q.groupby('date').opportunity15.agg(source_rows='size', source_mean='mean')
            valid = q.loc[q.formula_input_valid]
            part = valid.groupby('date').opportunity15.agg(fitted_rows='size', fitted_mean='mean')
            part['within_date_variance'] = valid.groupby('date').opportunity15.var(ddof=0)
            ex = ex.join(part)
            ex['fitted_rows'] = ex.fitted_rows.fillna(0).astype('int64')
            ex['invalid_mean'] = q.loc[~q.formula_input_valid].groupby('date').opportunity15.mean()
            ex = ex.reset_index()
            pd.testing.assert_frame_equal(d[ex.columns], ex, check_dtype=False, rtol=0, atol=2e-12)
            assert int(d.fitted_rows.sum()) == m['rows'] and int(d.fitted_rows.gt(0).sum()) == m['days']
            d['trained_date_mean'] = d.fitted_mean - d.source_mean
            defined = d.fitted_rows.gt(0)
            v = d.loc[defined]; between = float(np.mean(v.trained_date_mean ** 2))
            within = float(v.within_date_variance.mean())
            residual = valid.merge(d[['date', 'source_mean', 'fitted_mean']], on='date', validate='many_to_one')
            mse = ((residual.opportunity15 - residual.source_mean) ** 2).groupby(residual.date).mean().mean()
            np.testing.assert_allclose(mse, within + between, rtol=0, atol=2e-12)
            summaries.append(dict(fold=spec['id'], arm=arm, source_days=len(d), fitted_days=int(defined.sum()),
                source_rows=int(d.source_rows.sum()), fitted_rows=int(d.fitted_rows.sum()),
                mean_trained_date_mean=float(v.trained_date_mean.mean()),
                maximum_absolute_trained_date_mean=float(v.trained_date_mean.abs().max()),
                median_absolute_trained_date_mean=float(v.trained_date_mean.abs().median()),
                between_date_mean_square=between, within_date_variance=within,
                benchmark_gap_fraction_of_centered_target_mean_square=between / (within + between)))
            d['fold'] = spec['id']; d['arm'] = arm; daily.append(d)
            print(json.dumps(dict(population_verified=spec['id'], arm=arm)), flush=True)
    c.close()
    path = root / 'population_daily.parquet'
    pd.concat(daily, ignore_index=True).to_parquet(path, index=False, compression='zstd')
    sources[str(path)] = sha(path)
    save_json(root / 'report.json', dict(passed=True, protocol_sha256=sha(PROTOCOL), source_hashes=sources,
        summaries=summaries, all_daily_populations_and_means_independently_rebuilt=True,
        all_fitted_rows_and_dates_match_existing_models=True,
        date_equal_mean_square_decomposition_verified=True,
        zero_valid_dates_preserved=True, new_fits=0, new_selection_lists=0,
        new_economic_groups=0, new_2026_prices_read=False, no_exit_rules=True,
        target_mean_square_is_not_score_or_profit_explanation=True))
    print(json.dumps(dict(report_sha256=sha(root / 'report.json'))), flush=True)


if __name__ == '__main__':
    main()
