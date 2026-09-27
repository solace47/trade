"""Freeze a three-year training comparison without changing the original 48 inputs."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

from . import tail_formula_additive as base
from . import tail_formula_context_2024 as linkage
from . import tail_formula_float as original
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from .corporate_cash import save_json, sha
from .tail_formula_long48_inputs import ROOT, OUT, PROTOCOL, policy
from .tail_formula_long48_observations import LABELS

OLD_LABELS = Path('data/research/tail_formula_1000')
STEM = 'tail_formula_long48'
FEATURE_COLUMNS = ['date','code','half','board','decision_shares','formula_input_valid',*original.EXPRESSIONS]
LABEL_COLUMNS = ['date','code','next_date','known15','opportunity15','known_no_trade','adverse_return15']


def checked_inputs():
    policy(); proofs = {}
    tax = json.loads((ROOT/'tax_verification.json').read_text())
    assert tax['passed'] and tax['classification_source_sha256']==sha(Path(__file__).with_name('tail_formula_long48_labels.py'))
    proofs[str(ROOT/'tax_verification.json')] = sha(ROOT/'tax_verification.json')
    for folder,proof,report,key in [(OUT,'feature_verification.json','feature_report.json','feature_report_sha256'),
        (original.ROOT,'feature_verification.json','feature_report.json','feature_report_sha256'),
        (LABELS,'full_label_verification.json','full_label_report.json','label_report_sha256'),
        (OLD_LABELS,'full_label_verification.json','full_label_report.json','label_report_sha256')]:
        p = json.loads((folder/proof).read_text()); r = json.loads((folder/report).read_text())
        assert p['passed'] and p[key]==sha(folder/report)
        file = 'features.parquet' if key=='feature_report_sha256' else 'full_labels.parquet'
        digest = 'features_sha256' if file=='features.parquet' else 'labels_sha256'
        assert r[digest]==sha(folder/file)
        proofs[str(folder/proof)] = sha(folder/proof)
        proofs[str(folder/report)] = sha(folder/report)
    notice = json.loads((OUT/'notice_verification.json').read_text())
    assert notice['passed'] and notice['notice_report_sha256']==sha(OUT/'notice_report.json')
    proofs[str(OUT/'notice_verification.json')] = sha(OUT/'notice_verification.json')
    return proofs


def assemble():
    proofs = checked_inputs()
    if (ROOT/'feature_report.json').exists() or (ROOT/'full_label_report.json').exists():
        raise ValueError('Do not replace the assembled historical training inputs')
    c = base.conn()
    for kind,columns,first,second,file,report,hash_key in [
        ('feature',FEATURE_COLUMNS,OUT,original.ROOT,'features.parquet','feature_report.json','features_sha256'),
        ('label',LABEL_COLUMNS,LABELS,OLD_LABELS,'full_labels.parquet','full_label_report.json','labels_sha256')]:
        cols = ','.join(columns)
        c.execute(f"COPY (SELECT {cols} FROM read_parquet('{first}/{file}') UNION ALL SELECT {cols} FROM read_parquet('{second}/{file}') ORDER BY date,code) TO '{ROOT}/{file}' (FORMAT PARQUET,COMPRESSION ZSTD)")
        stats = c.sql(f"SELECT count(*) AS rows,count(DISTINCT (date,code)) AS unique_keys,min(date) AS first,max(date) AS last FROM read_parquet('{ROOT}/{file}')").df().iloc[0].to_dict()
        assert stats['rows']==stats['unique_keys'] and stats['first']>='2022-01-01' and stats['last']<'2026-01-01'
        r = dict(protocol_sha256=sha(PROTOCOL),source_proofs_sha256=proofs,columns=columns,**stats,
            sources=[str(first/file),str(second/file)],source_sha256={str(x/file):sha(x/file) for x in [first,second]},
            new_2026_prices_read=False,no_exit_rules=True)
        r[hash_key] = sha(ROOT/file)
        if kind=='feature':
            r.update(expressions=original.EXPRESSIONS,native_header=original.HEADER)
        else:
            r.update(training_projection_only=True,evaluation_uses_original_2025_full_labels=True)
        save_json(ROOT/report,r)
    c.close()
    return dict(feature_report_sha256=sha(ROOT/'feature_report.json'),label_report_sha256=sha(ROOT/'full_label_report.json'))


def verify_inputs():
    proofs = checked_inputs(); c = base.conn(); result = {}
    for file,report,key,verification,report_key in [
        ('features.parquet','feature_report.json','features_sha256','feature_verification.json','feature_report_sha256'),
        ('full_labels.parquet','full_label_report.json','labels_sha256','full_label_verification.json','label_report_sha256')]:
        r = json.loads((ROOT/report).read_text())
        assert r['protocol_sha256']==sha(PROTOCOL) and r['source_proofs_sha256']==proofs and r[key]==sha(ROOT/file)
        cols = ','.join(r['columns'])
        for source in r['sources']:
            assert sha(Path(source))==r['source_sha256'][source]
            where = "date<'2024-01-01'" if source==r['sources'][0] else "date>='2024-01-01'"
            for left,right in [(f"SELECT {cols} FROM read_parquet('{ROOT}/{file}') WHERE {where}",f"SELECT {cols} FROM read_parquet('{source}')"),
                               (f"SELECT {cols} FROM read_parquet('{source}')",f"SELECT {cols} FROM read_parquet('{ROOT}/{file}') WHERE {where}")]:
                assert c.sql(f'SELECT count(*) FROM (({left}) EXCEPT ALL ({right}))').fetchone()[0]==0
        v = dict(passed=True,rows=r['rows'],all_earlier_and_original_projected_values_unchanged=True,
            all_keys_unique=True,source_proofs_sha256=proofs,new_2026_prices_read=False,no_exit_rules=True)
        v[report_key] = sha(ROOT/report); save_json(ROOT/verification,v); result[verification] = sha(ROOT/verification)
    c.close(); return result


def setup(fold):
    if fold=='combined':
        protocol = Path('config')/(STEM+'_combined_protocol.json'); p = json.loads(protocol.read_text())
        paths = [Path('config')/(STEM+'_'+f+'_protocol.json') for f in ['2024','recent']]
        assert p['fold_protocols']==[str(x) for x in paths]
        for path,name in zip(paths,['2024','recent']):
            for file in ['model_report.json','selection_report.json']:
                assert json.loads((Path('data/research')/(STEM+'_'+name)/file).read_text())['protocol_sha256']==sha(path)
        linkage.ROOT = Path('data/research')/(STEM+'_2024'); linkage.H2 = Path('data/research')/(STEM+'_recent')
        linkage.COMBINED = Path('data/research')/(STEM+'_2025'); linkage.PROTOCOL = protocol
        linkage.H2_SELECTION_SHA = None; linkage.H2_MODEL_SHA = None; linkage.H2_OUTCOMES_PREVIOUSLY_SEEN = False
    else:
        original.setup(fold)
        root = Path('data/research')/(STEM+'_'+fold); protocol = Path('config')/(STEM+'_'+fold+'_protocol.json')
        base.ROOT = root; base.PROTOCOL = protocol; base.FEATURES = ROOT; base.SOURCE = ROOT
        relative.PROTOCOL = protocol; study.ROOT = root; study.PROTOCOL = protocol
        p = json.loads(protocol.read_text()); assert p['inputs_protocol_sha256']==sha(PROTOCOL)
        assert p['model_max_depth']==3 and p['variant']=='relative'


def verify_scores():
    spec = importlib.util.spec_from_file_location('independent_score_check',Path('scripts/verify_tail_formula_additive.py'))
    module = importlib.util.module_from_spec(spec)
    previous = sys.path.copy()
    try:
        sys.path.insert(0,str(Path('scripts').resolve()))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = previous
    module.ROOT = base.ROOT; module.PROTOCOL = base.PROTOCOL; module.FEATURES = ROOT; module.SOURCE = ROOT
    return module.scores()


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('fold',choices=['inputs','2024','recent','combined'])
    p.add_argument('stage',choices=['assemble','verify_inputs','model','verify_model','scores','verify_scores','freeze','verify','analyze'])
    a = p.parse_args()
    if a.fold=='inputs':
        assert a.stage in ['assemble','verify_inputs']; result = globals()[a.stage]()
    else:
        setup(a.fold)
        if a.fold=='combined':
            assert a.stage in ['freeze','verify','analyze']
            result = (linkage.common_analysis(linkage.COMBINED,linkage.PROTOCOL) if a.stage=='analyze'
                      else getattr(linkage,'combine' if a.stage=='freeze' else 'verify_combined')())
        elif a.stage in ['model','verify_model']:
            result = getattr(relative,a.stage)('relative')
        elif a.stage in ['freeze','verify']:
            result = getattr(study,a.stage)()
        elif a.stage=='verify_scores':
            result = verify_scores()
        else:
            assert a.stage in ['scores','analyze']; result = getattr(base,a.stage)()
    print(json.dumps(result,ensure_ascii=False,indent=2))
