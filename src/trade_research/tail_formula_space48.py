"""A fixed one-percent morning opportunity target on the original 48 inputs."""
import argparse
import json
from pathlib import Path

from . import tail_formula_additive as base
from . import tail_formula_context_2024 as linkage
from . import tail_formula_float as inputs
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from .corporate_cash import sha

STEM = 'tail_formula_space48'


def setup(fold):
    if fold == 'combined':
        protocol = Path('config')/(STEM+'_combined_protocol.json')
        p = json.loads(protocol.read_text())
        paths = [Path('config')/(STEM+'_'+f+'_protocol.json') for f in ['2024','recent']]
        assert p['fold_protocols'] == [str(x) for x in paths]
        for path,name in zip(paths,['2024','recent']):
            for file in ['model_report.json','selection_report.json']:
                assert json.loads((Path('data/research')/(STEM+'_'+name)/file).read_text())['protocol_sha256'] == sha(path)
        linkage.ROOT = Path('data/research')/(STEM+'_2024'); linkage.H2 = Path('data/research')/(STEM+'_recent')
        linkage.COMBINED = Path('data/research')/(STEM+'_2025'); linkage.PROTOCOL = protocol
        linkage.H2_SELECTION_SHA = None; linkage.H2_MODEL_SHA = None; linkage.H2_OUTCOMES_PREVIOUSLY_SEEN = False
    else:
        inputs.setup(fold)
        root = Path('data/research')/(STEM+'_'+fold); protocol = Path('config')/(STEM+'_'+fold+'_protocol.json')
        base.ROOT = root; base.PROTOCOL = protocol; relative.PROTOCOL = protocol; study.ROOT = root; study.PROTOCOL = protocol
        p = json.loads(protocol.read_text())
        assert p['feature_report_sha256'] == sha(base.FEATURES/'feature_report.json')
        assert p['label_report_sha256'] == sha(base.SOURCE/'full_label_report.json')
        assert p['space_threshold'] == .01 and p['variant'] == 'space' and p['model_max_depth'] == 3


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('fold',choices=['2024','recent','combined'])
    p.add_argument('stage',choices=['model','verify_model','scores','freeze','verify','analyze'])
    a = p.parse_args(); setup(a.fold)
    if a.fold == 'combined':
        assert a.stage in ['freeze','verify','analyze']
        result = (linkage.common_analysis(linkage.COMBINED,linkage.PROTOCOL) if a.stage == 'analyze'
                  else getattr(linkage,'combine' if a.stage == 'freeze' else 'verify_combined')())
    elif a.stage in ['model','verify_model']:
        result = getattr(relative,a.stage)('space')
    elif a.stage in ['freeze','verify']:
        result = getattr(study,a.stage)()
    else:
        result = getattr(base,a.stage)()
    print(json.dumps(result,ensure_ascii=False,indent=2))
