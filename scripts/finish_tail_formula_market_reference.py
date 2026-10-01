"""Use the existing comparator's supported per-year guard without changing the study."""
import json
from pathlib import Path

import pandas as pd

from trade_research.corporate_cash import save_json, sha
import run_tail_formula_market_reference as study
import compare_tail_formula_shared_unknowns as comparator

REPAIR = Path('config/tail_formula_market_reference_finish_repair.json')


def per_year_compare(spec, receipt):
    year = Path(spec['left']).name[-4:]
    assert year in ['2024', '2025'] and Path(spec['right']).name.endswith(year)
    adapted = dict(receipt, signal_range=['2024-01-01', '2024-12-31'] if year == '2024' else None)
    return comparator.compare(spec, adapted)


def run():
    repair = json.loads(REPAIR.read_text())
    assert repair['unchanged_protocol_sha256'] == sha(study.PROTOCOL)
    assert repair['unchanged_joint_sha256'] == sha(study.ROOT / 'joint_selection_freeze.json')
    for file, digest in repair['source_hashes'].items(): assert sha(Path(file)) == digest, file
    for year, guard in [('2024', ['2024-01-01', '2024-12-31']), ('2025', None)]:
        comparator.check_signal_dates(pd.Series([year + '-02-01']), guard)
    study.shared_compare = per_year_compare
    result = study.finish()
    forecast = json.loads((study.ROOT / 'forecast_diagnostic.json').read_text())
    coverage = []
    for record in forecast['records']:
        fold = record['fold']; _, model, root = study.checked_model(fold)
        inputs = pd.read_parquet(root / 'market_scores.parquet', columns=['date', 'exchange'])
        inputs = inputs.loc[inputs.date.ge(model['evaluation_start']) & inputs.date.lt(model['evaluation_end'])]
        known = pd.read_parquet(root / 'forecast_daily.parquet', columns=['date'])
        assert known.date.is_unique and set(known.date) <= set(inputs.date)
        missing = sorted(set(inputs.date) - set(known.date))
        coverage.append(dict(fold=fold, evaluation_calendar_days=inputs.date.nunique(),
            known_reference_target_days=len(known), undefined_reference_target_dates=missing,
            existing_mse_and_ci_are_known_reference_conditional=True,
            full_calendar_mse_delta_ci=None if missing else record['mse_delta_ci']))
    save_json(study.ROOT / 'forecast_calendar_coverage_verification.json', dict(passed=True,
        forecast_sha256=sha(study.ROOT / 'forecast_diagnostic.json'), records=coverage,
        undefined_calendar_dates_preserved=True, no_new_price_or_target_reaggregation=True,
        new_2026_prices_read=False))
    receipts = {str(study.ROOT / f): sha(study.ROOT / f) for f in
        ['complete_results_manifest.json', 'forecast_calendar_coverage_verification.json', 'joint_selection_freeze.json']}
    receipts[str(REPAIR)] = sha(REPAIR)
    save_json(study.ROOT / 'finish_repair_verification.json', dict(passed=True, source_hashes=receipts,
        existing_models_scores_lists_and_statistics_unchanged=True,
        only_pair_year_guard_adapted=True, existing_partial_comparison_reused=True,
        zero_new_fits_or_economic_group_aggregations=True, new_2026_prices_read=False))
    return dict(**result, repair_sha256=sha(study.ROOT / 'finish_repair_verification.json'), forecast_coverage=coverage)


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False, indent=2))
