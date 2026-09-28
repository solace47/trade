"""Report all three frozen Q1 variants, without selecting a winner from Q1."""
import argparse
import json

from trade_research import tail_formula_boundary_evaluation as evaluation
from trade_research import tail_formula_q1_candidates as study
from trade_research import tail_formula_q1_candidate_boundary as boundary
from trade_research.corporate_cash import save_json, sha
from verify_tail_formula_before1000 import analysis as verify_analysis
from reuse_tail_formula_selected_analysis import reuse
from compare_tail_formula_same_dates import compare
from compare_tail_formula_shared_unknowns import compare as coupled


def setup():
    p, gate = study.checked_models()
    evaluation.source.ROOT = boundary.ROOT
    evaluation.PERIODS = p['periods']
    return p, gate


def analyze(variant):
    p, gate = setup(); assert variant in gate['active_variants']
    return evaluation.analyze(study.ROOT/variant, study.PROTOCOL)


def verify(variant):
    p, gate = setup(); assert variant in gate['active_variants']
    return verify_analysis(study.ROOT/variant)


def comparisons():
    p, gate = setup(); output = study.ROOT/'comparisons'; output.mkdir(exist_ok=True)
    reports = {}; specs = []
    for variant in gate['active_variants']:
        if variant == 'control':
            continue
        quality = study.ROOT/('control_'+variant)
        equal = json.loads((quality/'selection_report.json').read_text())['identical_to_full_control']
        if equal:
            if not (quality/'analysis_reuse.json').exists():
                reuse(quality, study.ROOT/'control')
            control = study.ROOT/'control'
        else:
            if not (quality/'analysis_report.json').exists():
                evaluation.analyze(quality, study.PROTOCOL)
                verify_analysis(quality)
            control = quality
        left = study.ROOT/variant
        for suffix, right in [('full_control', study.ROOT/'control'), ('quality_control', control)]:
            if suffix == 'quality_control' and equal:
                continue
            path = output/f'{variant}_vs_{suffix}.json'
            if not path.exists():
                compare(left, right, path, p['periods'], intersection_only=True)
            reports[str(path)] = sha(path)
            specs.append(dict(left=str(left), right=str(right), output=str(output/f'{variant}_vs_{suffix}_coupled.json'),
                left_analysis_sha256=sha(left/'analysis_report.json'), right_analysis_sha256=sha(right/'analysis_report.json')))
    coupled_policy = dict(protocol_sha256=sha(study.PROTOCOL), signal_range=[p['signal_first'],p['signal_last']],
        periods=p['periods'], labels_sha256=sha(boundary.ROOT/'full_labels.parquet'), comparisons=specs,
        diagnostic_only=True, does_not_change_selected_lists=True)
    policy_path = output/'coupled_protocol.json'
    if not policy_path.exists():
        save_json(policy_path, coupled_policy)
    else:
        assert json.loads(policy_path.read_text()) == coupled_policy
    from pathlib import Path
    for spec in specs:
        if not Path(spec['output']).exists():
            coupled(spec, coupled_policy)
        reports[spec['output']] = sha(Path(spec['output']))
    result = dict(protocol_sha256=sha(study.PROTOCOL), reports_sha256=reports,
        all_predeclared_variants_reported=True, original_selections_unchanged=True,
        q1_previously_exposed=True, strict_blind=False, no_q2_signal_prices_read=True, no_exit_rules=True)
    save_json(output/'report.json', result); return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=['analyze','verify','comparisons'])
    p.add_argument('--variant'); a=p.parse_args()
    print(json.dumps(comparisons() if a.stage=='comparisons' else globals()[a.stage](a.variant), ensure_ascii=False, indent=2))
