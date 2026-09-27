"""Compose audited decimal-compatible component cores without redoing research."""
import argparse
import json
from pathlib import Path
import re

from export_tail_formula_native_compatible import compatible_text
from trade_research import tail_formula_size_agreement as study
from trade_research.corporate_cash import save_json, sha


def main(fold):
    spec, components = study.checked(fold); root = study.folder(fold)
    assert not (root / 'native_compatibility_report.json').exists()
    selection = json.loads((root / 'selection_report.json').read_text())
    proof = json.loads((root / 'selection_verification.json').read_text())
    assert proof['passed'] and proof['selection_report_sha256'] == sha(root / 'selection_report.json')
    assert selection['core_sha256'] == sha(root / 'frozen_numeric_core.tdx')
    sources = []; cuts = []; reports = []
    for component, _ in components:
        report = json.loads((component / 'native_compatibility_report.json').read_text())
        assert report['passed'] and report['evaluation_changed_flags'] == 0
        assert report['model_report_sha256'] == sha(component / 'model_report.json')
        assert report['original_core_sha256'] == sha(component / 'frozen_numeric_core.tdx')
        assert report['native_core_sha256'] == sha(component / 'native_compatible_core.tdx')
        source = (component / 'native_compatible_core.tdx').read_text()
        assert source == compatible_text((component / 'frozen_numeric_core.tdx').read_text())
        cut = re.search(r'^CORE:SC>([^;]+);$', source, re.M).group(1)
        assert float(cut) == report['new_threshold']
        sources.append(source); cuts.append(cut)
        reports.append(dict(root=str(component), compatibility_report_sha256=sha(component / 'native_compatibility_report.json')))
    prefixes, bodies = study.prefixes_and_bodies(sources)
    second = re.sub(r'\bT(\d{2})\b', r'U\1', bodies[1]); second = re.sub(r'\bSC\b', 'SSC', second)
    composed = prefixes[1] + bodies[0] + second + f'CORE:(SC>{cuts[0]}) AND (SSC>{cuts[1]});\n'
    converted = compatible_text((root / 'frozen_numeric_core.tdx').read_text())
    assert composed == converted
    assert composed.split('T01:=', 1)[0] == (root / 'frozen_numeric_core.tdx').read_text().split('T01:=', 1)[0]
    assert not re.search(r'\d[eE][-+]?\d', composed)
    assert max(map(len, re.findall(r'\b\d+(?:\.\d+)?\b', composed))) <= 16
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', composed); assert len(names) == len(set(names))
    out = root / 'native_compatible_core.tdx'; out.write_text(composed)
    result = dict(passed=True, selection_report_sha256=sha(root / 'selection_report.json'),
        original_core_sha256=sha(root / 'frozen_numeric_core.tdx'), native_core_sha256=sha(out), components=reports,
        both_independently_replayed_component_selections_unchanged=True, original_50_input_expressions_unchanged=True,
        both_64_tree_orders_and_and_clause_preserved=True, no_new_economic_evaluation=True,
        selected=selection['selected'], evaluation_start=spec['evaluation_start'], evaluation_end=spec['evaluation_end'],
        complete_selector_verified=False, software_compilation_verified=False, native_source_parity_verified=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(root / 'native_compatibility_report.json', result); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('fold', choices=['2024', 'recent'])
    print(json.dumps(main(parser.parse_args().fold), ensure_ascii=False, indent=2))
