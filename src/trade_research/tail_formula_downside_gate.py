"""Keep the frozen opportunity score and add an absolute downside-score gate."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_context_2024 as linkage
from . import tail_formula_float as original
from . import tail_formula_relative as relative
from .corporate_cash import save_json, sha


def setup(fold):
    if fold == 'combined':
        linkage.ROOT = Path('data/research/tail_formula_downside_gate_2024')
        linkage.H2 = Path('data/research/tail_formula_downside_gate_recent')
        linkage.COMBINED = Path('data/research/tail_formula_downside_gate_2025')
        linkage.PROTOCOL = Path('config/tail_formula_downside_gate_combined_protocol.json')
        linkage.H2_SELECTION_SHA = None
        linkage.H2_MODEL_SHA = None
        linkage.H2_OUTCOMES_PREVIOUSLY_SEEN = False
    else:
        original.setup(fold)
        base.ROOT = Path('data/research/tail_formula_downside_gate_' + fold)
        base.PROTOCOL = Path('config/tail_formula_downside_gate_' + fold + '_protocol.json')
        relative.PROTOCOL = base.PROTOCOL


def checked_inputs():
    root = base.ROOT
    config = json.loads(base.PROTOCOL.read_text())
    old = Path(config['opportunity_root'])
    assert sha(old/'selection_report.json') == config['opportunity_selection_report_sha256']
    assert sha(base.FEATURES/'feature_report.json') == config['feature_report_sha256']
    assert sha(base.SOURCE/'full_label_report.json') == config['label_report_sha256']
    for path in [root, old]:
        for kind in ['model', 'score']:
            report = path/f'{kind}_report.json'
            proof = json.loads((path/f'{kind}_verification.json').read_text())
            assert proof['passed'] and proof[f'{kind}_report_sha256'] == sha(report)
        score = json.loads((path/'score_report.json').read_text())
        assert score['scores_sha256'] == sha(path/'scores.parquet')
        assert score['model_report_sha256'] == sha(path/'model_report.json')
    selected = json.loads((old/'selection_report.json').read_text())
    proof = json.loads((old/'selection_verification.json').read_text())
    assert proof['passed'] and proof['selection_report_sha256'] == sha(old/'selection_report.json')
    assert selected['selection_sha256'] == sha(old/'selection.parquet')
    assert selected['core_sha256'] == sha(old/'frozen_numeric_core.tdx')
    assert selected['model_report_sha256'] == sha(old/'model_report.json')
    risk = json.loads((root/'model_report.json').read_text())
    opportunity = json.loads((old/'model_report.json').read_text())
    assert risk['protocol_sha256'] == sha(base.PROTOCOL)
    assert risk['variant'] == 'downside' and opportunity['variant'] == 'relative'
    for key in ['training_start', 'training_end']:
        assert risk[key] == opportunity[key] == config[key]
    assert risk['feature_names'] == opportunity['feature_names'] == list(base.EXPRESSIONS)
    assert risk['feature_report_sha256'] == opportunity['feature_report_sha256'] == sha(base.FEATURES/'feature_report.json')
    assert risk['last_observation'] < config['evaluation_start']
    assert opportunity['last_observation'] < config['evaluation_start']
    assert config['risk_cut_rule'] == 'date_equal_training_risk_frequency'
    assert config['risk_adverse_threshold'] == -.03
    assert 0 < risk['bias'] < 1
    return config, old, selected, risk, opportunity


def compose_core(opportunity_source, risk_source, opportunity_cut, risk_cut):
    prefix, opportunity_body = opportunity_source.split('T01:=', 1)
    other, risk_body = risk_source.split('T01:=', 1)
    assert prefix == other
    opportunity_body = 'T01:=' + opportunity_body.split('\nCORE:', 1)[0] + '\n'
    risk_body = 'T01:=' + risk_body.split('\nCORE:', 1)[0] + '\n'
    risk_body = re.sub(r'\bT(\d{2})\b', r'U\1', risk_body)
    risk_body = re.sub(r'\bSC\b', 'RC', risk_body)
    core = prefix + opportunity_body + risk_body + f'CORE:(SC>{opportunity_cut:.17g}) AND (RC<={risk_cut:.17g});\n'
    names = re.findall(r'\b([A-Z][A-Z0-9]*):=', core)
    assert len(names) == len(set(names))
    return core


def freeze():
    root = base.ROOT
    if (root/'selection_report.json').exists():
        raise ValueError('Do not replace the frozen opportunity-plus-downside selection')
    config, old, selected, risk, opportunity = checked_inputs()
    prior = pd.read_parquet(old/'selection.parquet')
    scores = pd.read_parquet(root/'scores.parquet')
    pd.testing.assert_frame_equal(prior.drop(columns='selected'), scores[prior.drop(columns='selected').columns], check_exact=True)
    out = prior.copy()
    out['selected'] &= scores.formula_input_valid & scores.score.le(risk['bias'])
    out.to_parquet(root/'selection.parquet', index=False, compression='zstd')
    # This component is an archival numerical-score expression. The combined
    # selector below explicitly uses <=, not this helper's default > clause.
    risk_core = base.native_core(risk, risk['bias'], base.EXPRESSIONS, base.HEADER)
    (root/'risk_numeric_core.tdx').write_text(risk_core)
    core = compose_core((old/'frozen_numeric_core.tdx').read_text(), risk_core,
                        selected['chosen_threshold']['threshold'], risk['bias'])
    (root/'frozen_numeric_core.tdx').write_text(core)
    r = dict(protocol_sha256=sha(base.PROTOCOL), model_report_sha256=sha(root/'model_report.json'),
        score_report_sha256=sha(root/'score_report.json'), opportunity_root=str(old),
        opportunity_selection_report_sha256=sha(old/'selection_report.json'),
        opportunity_model_report_sha256=sha(old/'model_report.json'),
        selection_sha256=sha(root/'selection.parquet'), core_sha256=sha(root/'frozen_numeric_core.tdx'),
        risk_core_sha256=sha(root/'risk_numeric_core.tdx'), chosen_threshold=selected['chosen_threshold'],
        chosen_threshold_applies_to='unchanged_opportunity_model_only', risk_score_cut=risk['bias'],
        risk_cut_rule=config['risk_cut_rule'], original_selected=int(prior.selected.sum()), selected=int(out.selected.sum()),
        by_half=out.groupby('half').selected.agg(['size','sum']).reset_index().to_dict('records'),
        evaluation_start=config['evaluation_start'], evaluation_end=config['evaluation_end'],
        two_frozen_scores_required=True, year_2025_is_exploratory=True, new_group_outcomes_read=False,
        opportunity_source_outcomes_previously_seen=True, risk_score_not_calibrated_probability=True,
        no_exit_rules=True, new_2026_prices_read=False, software_compilation_verified=False)
    save_json(root/'selection_report.json', r)
    return r


def verify():
    root = base.ROOT
    config, old, selected, risk, opportunity = checked_inputs()
    r = json.loads((root/'selection_report.json').read_text())
    for key, path in [('protocol_sha256',base.PROTOCOL),('model_report_sha256',root/'model_report.json'),
        ('score_report_sha256',root/'score_report.json'),('opportunity_selection_report_sha256',old/'selection_report.json'),
        ('opportunity_model_report_sha256',old/'model_report.json'),('selection_sha256',root/'selection.parquet'),
        ('core_sha256',root/'frozen_numeric_core.tdx'),('risk_core_sha256',root/'risk_numeric_core.tdx')]:
        assert r[key] == sha(path)
    cut = opportunity['thresholds'][3]
    assert r['chosen_threshold'] == cut and cut['training_quantile'] == .995
    assert r['risk_score_cut'] == risk['bias'] and r['risk_cut_rule'] == config['risk_cut_rule']
    c = base.conn()
    c.read_parquet(str(root/'scores.parquet')).create_view('risk')
    c.read_parquet(str(old/'scores.parquet')).create_view('opportunity')
    c.read_parquet(str(base.FEATURES/'features.parquet')).create_view('features')
    # Independently rebuild the date-equal frequency using the exact valid
    # training intersection, not the old opportunity label or a pooled mean.
    historical_frequency = c.execute('''SELECT avg(day_rate) FROM (
        SELECT l.date,avg(CAST(l.adverse_return15<=-.03 AS INTEGER)) AS day_rate
        FROM read_parquet(?) l JOIN features f USING(date,code)
        WHERE l.known15 AND f.formula_input_valid AND l.date>=? AND l.next_date<? GROUP BY l.date)''',
        [str(base.SOURCE/'full_labels.parquet'),config['training_start'],config['training_end']]).fetchone()[0]
    np.testing.assert_allclose(historical_frequency, r['risk_score_cut'], rtol=0, atol=2e-12)
    start, end = config['evaluation_start'], config['evaluation_end']
    expected = c.sql(f'''SELECT a.date,a.code,a.half,a.board,a.decision_shares,
        a.date>='{start}' AND a.date<'{end}' AND a.formula_input_valid AND b.formula_input_valid
        AND b.score>{cut['threshold']:.17e} AND a.score<={risk['bias']:.17e} AS selected
        FROM risk a JOIN opportunity b USING(date,code) ORDER BY date,code''').df()
    c.close()
    pd.testing.assert_frame_equal(pd.read_parquet(root/'selection.parquet'), expected, check_exact=True)
    assert r['selected'] == int(expected.selected.sum())
    assert r['selected'] <= r['original_selected'] == selected['selected']
    assert expected.loc[expected.selected, 'date'].ge(risk['training_end']).all()
    core = (root/'frozen_numeric_core.tdx').read_text()
    prefix, body = core.split('T01:=', 1)
    opportunity_body, risk_body = body.split('U01:=', 1)
    clause = f"CORE:(SC>{cut['threshold']:.17g}) AND (RC<={risk['bias']:.17g});\n"
    assert risk_body.endswith(clause)
    restored = prefix + 'T01:=' + opportunity_body + f"CORE:SC>{cut['threshold']:.17g};\n"
    assert restored == (old/'frozen_numeric_core.tdx').read_text()
    restored_risk = 'U01:=' + risk_body[:-len(clause)]
    restored_risk = re.sub(r'\bU(\d{2})\b', r'T\1', restored_risk)
    restored_risk = re.sub(r'\bRC\b', 'SC', restored_risk)
    restored_risk = prefix + restored_risk + f"CORE:SC>{risk['bias']:.17g};\n"
    assert restored_risk == (root/'risk_numeric_core.tdx').read_text()
    assert restored_risk == base.native_core(risk,risk['bias'],base.EXPRESSIONS,base.HEADER)
    proof = dict(passed=True,selection_report_sha256=sha(root/'selection_report.json'),rows=len(expected),
        all_selection_flags_rebuilt_from_both_raw_scores=True,
        date_equal_training_risk_frequency_rebuilt=True,both_core_components_and_sum_order_preserved=True,
        no_training_period_selection=True,no_new_group_outcomes_read=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(root/'selection_verification.json',proof)
    return proof


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('fold', choices=['2024','recent','combined'])
    p.add_argument('stage', choices=['model','verify_model','scores','freeze','verify','analyze'])
    a = p.parse_args(); setup(a.fold)
    if a.fold == 'combined':
        assert a.stage in ['freeze','verify','analyze']
        r = (linkage.common_analysis(linkage.COMBINED,linkage.PROTOCOL) if a.stage == 'analyze'
             else getattr(linkage,'combine' if a.stage == 'freeze' else 'verify_combined')())
    elif a.stage in ['model','verify_model']:
        r = getattr(relative,a.stage)('downside')
    elif a.stage in ['freeze','verify']:
        r = globals()[a.stage]()
    else:
        r = getattr(base,a.stage)()
    print(json.dumps(r,ensure_ascii=False,indent=2))
