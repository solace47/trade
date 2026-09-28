"""Run the existing independent checks against the new bounded common Q1 scope."""
import argparse
import json
import pandas as pd

from trade_research import tail_formula_q1_candidate_observations as study
from trade_research import tail_formula_q1_candidate_inputs as inputs
from trade_research.corporate_cash import save_json, sha


def main(stage):
    if stage == 'windows':
        import verify_tail_formula_forward_windows as verifier
    else:
        import verify_tail_formula_forward_labels as verifier
    verifier.LABELS = study.LABELS; verifier.CATALOG = study.CATALOG
    verifier.OUT = inputs.OLD; verifier.checked_keys = study.checked_keys
    verifier.main()
    if stage == 'labels30':
        old = pd.read_parquet(study.OLD/'full_labels.parquet')
        new = pd.read_parquet(study.LABELS/'full_labels.parquet')
        keys = old[['date','code']].merge(new[['date','code']], validate='one_to_one')
        a = keys.merge(old, on=['date','code'], validate='one_to_one').sort_values(['date','code']).reset_index(drop=True)
        b = keys.merge(new, on=['date','code'], validate='one_to_one').sort_values(['date','code']).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b[a.columns], check_dtype=False, rtol=0, atol=2e-10)
        status = [x for x in a if a[x].dtype == bool or 'status' in x]
        pd.testing.assert_frame_equal(a[status], b[status], check_exact=True)
        r = dict(passed=True, old_label_report_sha256=sha(study.OLD/'full_label_report.json'),
            label_report_sha256=sha(study.LABELS/'full_label_report.json'), overlap_rows=len(a),
            all_old_columns_reproduced=True, all_old_unknown_and_fill_states_exact=True)
        save_json(study.LABELS/'old_overlap_verification.json', r)
        print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=['windows','labels30'])
    main(p.parse_args().stage)
