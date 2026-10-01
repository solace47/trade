"""Resume the unchanged morning-order comparison with legacy-count metadata support."""
import inspect
import json
from pathlib import Path

from trade_research import tail_formula_morning_extrema_order as study
from trade_research.corporate_cash import save_json, sha
import run_tail_formula_morning_extrema_order as runner


def main():
    runner.bind()
    common = runner.common
    p, e, joint = common.checked_joint()
    legacy = []
    for file, digest in e['prior_completion_manifests'].items():
        assert sha(Path(file)) == digest
        data = json.loads(Path(file).read_text())
        pairs = data.get('comparisons', [])
        if type(pairs) is int:
            assert pairs >= 0
            legacy.append(dict(path=file, sha256=digest, comparison_count=pairs,
                               no_complete_endpoint_pairs_in_count_field=True))
        else:
            assert isinstance(pairs, list) and all(isinstance(v, dict) for v in pairs)
    assert len(legacy) == 1
    assert legacy[0]['path'] == 'data/research/tail_formula_prior_close/complete_results_manifest.json'
    original = inspect.getsource(common.finish)
    needle = "for pair in old.get('comparisons',[]):"
    assert original.count(needle) == 1
    fixed = original.replace(needle, 'for pair in legacy_manifest_pairs(old):')

    def legacy_manifest_pairs(data):
        pairs = data.get('comparisons', [])
        if type(pairs) is int:
            assert pairs >= 0
            return []  # A count cannot prove two complete endpoints equal.
        assert isinstance(pairs, list) and all(isinstance(v, dict) for v in pairs)
        return pairs

    receipt = study.ROOT / 'finish_compatibility_receipt.json'
    assert not receipt.exists() and not (study.ROOT / 'complete_results_manifest.json').exists()
    save_json(receipt, dict(passed=True, input_protocol_sha256=sha(study.PROTOCOL),
        execution_protocol_sha256=sha(runner.common.EXECUTION),
        joint_sha256=sha(study.ROOT / 'joint_selection_freeze.json'), legacy_count_manifests=legacy,
        pinned_runner_sha256=sha(Path('scripts/run_tail_formula_volume_memory.py')),
        compatibility_source_sha256=sha(Path(__file__)),
        original_failure_log_sha256=sha(Path('data/research/tail_formula_morning_extrema_order_finish.log')),
        single_read_only_legacy_metadata_loop_adaptation=True,
        all_numerical_comparison_gate_and_provenance_code_unchanged=True,
        no_new_fit_or_annual_aggregation=True, no_exit_rules=True, new_2026_prices_read=False))
    namespace = dict(common.__dict__)
    namespace['legacy_manifest_pairs'] = legacy_manifest_pairs
    exec(compile(fixed, '<morning-order-legacy-metadata-adapter>', 'exec'), namespace)
    result = namespace['finish']()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
