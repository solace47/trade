"""Three fixed selectors on common 2024 inputs; six chronological fits at most."""
import json
from pathlib import Path

import pandas as pd

from . import tail_formula_stock_2024_inputs as inputs
from . import tail_formula_additive as base
from .corporate_cash import save_json, sha

STEM = inputs.STEM
ROOT = inputs.ROOT
INPUTS = inputs.INPUTS
PROTOCOL = inputs.PROTOCOL
INTENT = inputs.INTENT
prior = inputs.prior
META = inputs.META
HEADER = inputs.prior.HEADER
EXPRESSIONS = inputs.EXPRESSIONS
CORE_GATE = 'AMREADY AND RTREADY'
ARMS = {arm: EXPRESSIONS for arm in ['full', 'group0', 'group1']}
MODEL_PROTOCOL = Path('config') / (STEM + '_model_protocol.json')


def model_root(fold, arm):
    assert fold in ['2024h1', '2024h2'] and arm in ARMS
    return ROOT / 'models' / fold / arm


def checked():
    inputs.checked()
    p = json.loads(MODEL_PROTOCOL.read_text())
    assert p['input_protocol_sha256'] == sha(PROTOCOL) and p['arms'] == ARMS
    assert p['folds'] == json.loads(INTENT.read_text())['folds']
    assert p['threshold'] == .995 and p['new_model_fits_allowed'] == 6
    assert p['groups'] == ['morning2024', 'agreement2024', 'mean2024']
    assert not p['new_2026_prices_allowed']
    for file, digest in p['source_hashes'].items(): assert sha(Path(file)) == digest, file
    fr = json.loads((INPUTS/'feature_report.json').read_text())
    assert fr['features_sha256'] == sha(INPUTS/'features.parquet') and fr['rows'] == 1144320
    for name in ['feature', 'native_input']:
        proof = json.loads((INPUTS/(name+'_verification.json')).read_text())
        assert proof['passed'] and proof['feature_report_sha256'] == sha(INPUTS/'feature_report.json')
    lr = json.loads((INPUTS/'full_label_report.json').read_text())
    lv = json.loads((INPUTS/'full_label_verification.json').read_text())
    assert lv['passed'] and lv['label_report_sha256'] == sha(INPUTS/'full_label_report.json')
    assert lr['labels_sha256'] == sha(INPUTS/'full_labels.parquet')
    assert fr['valid'] == 1006916 and lr['rows'] == 1144320
    return p


def project_labels(fold, arm):
    p = checked(); dates = p['folds'][fold]
    root = model_root(fold, arm) / 'labels'; root.mkdir(parents=True, exist_ok=True)
    assert not (root/'full_label_report.json').exists()
    group = None if arm == 'full' else int(arm[-1])
    condition = '' if group is None else f' AND (cast(substr(code,9,1) AS INT)>=5)::INT={group}'
    c=base.conn(); fields=','.join(inputs.LABEL_FIELDS)
    actual=c.execute('SELECT '+fields+' FROM read_parquet(?) WHERE date>=? AND next_date<?'+condition+' ORDER BY date,code',
                     [str(INPUTS/'full_labels.parquet'),dates['training_start'],dates['training_end']]).df()
    parent=pd.read_parquet(INPUTS/'full_labels.parquet')
    valid=parent.date.ge(dates['training_start']) & parent.next_date.lt(dates['training_end'])
    if group is not None:
        assert parent.code.str.fullmatch(r'(sh|sz)\.\d{6}').all()
        valid &= parent.code.str[-1].astype(int).ge(5).eq(bool(group))
    expected=parent.loc[valid,inputs.LABEL_FIELDS].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual,expected,check_exact=True)
    assert actual.next_date.gt(actual.date).all() and actual.next_date.lt(dates['evaluation_start']).all()
    actual.to_parquet(root/'full_labels.parquet',index=False,compression='zstd')
    c.register('projected',actual)
    counts=c.execute('''SELECT count(*) AS rows,count(DISTINCT date) AS days,max(next_date) AS last_observation
        FROM projected l JOIN read_parquet(?) f USING(date,code) WHERE l.known15 AND f.formula_input_valid''',
                     [str(INPUTS/'features.parquet')]).df().iloc[0];c.close()
    expected_counts=p['expected_training'][fold][arm]
    assert counts.rows==expected_counts['rows'] and counts.days==expected_counts['days']
    assert counts.last_observation==expected_counts['last_observation']
    r=dict(protocol_sha256=sha(MODEL_PROTOCOL),fold=fold,arm=arm,training_group=group,
        labels_sha256=sha(root/'full_labels.parquet'),original_labels_sha256=sha(INPUTS/'full_labels.parquet'),
        original_label_report_sha256=sha(INPUTS/'full_label_report.json'),
        projected_rows=len(actual),training_rows=int(counts.rows),training_days=int(counts.days),
        training_labels_only=True,no_new_group_evaluation=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(root/'full_label_report.json',r)
    save_json(root/'full_label_verification.json',dict(passed=True,label_report_sha256=sha(root/'full_label_report.json'),
        all_original_seven_values_and_two_independent_scope_predicates_equal=True,
        full_known_daily_base_centering_before_input_intersection_required=True,
        new_2026_prices_read=False,no_exit_rules=True))
    return dict(fold=fold,arm=arm,projected_rows=len(actual),training_rows=int(counts.rows))
