"""Frozen relative-rank candidates with a separately fitted absolute opportunity gate."""
import argparse
import json
from pathlib import Path
import re

import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_context_2024 as linkage
from . import tail_formula_float as inputs
from . import tail_formula_relative as relative
from .corporate_cash import save_json,sha


def setup(fold):
    if fold=='combined':
        linkage.ROOT=Path('data/research/tail_formula_rank_gate_2024')
        linkage.H2=Path('data/research/tail_formula_rank_gate_recent')
        linkage.COMBINED=Path('data/research/tail_formula_rank_gate_2025')
        linkage.PROTOCOL=Path('config/tail_formula_rank_gate_combined_protocol.json')
        linkage.H2_SELECTION_SHA=None;linkage.H2_MODEL_SHA=None;linkage.H2_OUTCOMES_PREVIOUSLY_SEEN=False
    else:
        inputs.setup(fold)
        base.ROOT=Path('data/research/tail_formula_rank_gate_'+fold)
        base.PROTOCOL=Path('config/tail_formula_rank_gate_'+fold+'_protocol.json')
        relative.PROTOCOL=base.PROTOCOL


def checked_inputs():
    root=base.ROOT;config=json.loads(base.PROTOCOL.read_text());rank=Path(config['rank_root'])
    assert sha(rank/'selection_report.json')==config['rank_selection_report_sha256']
    for path in [root,rank]:
        for kind in ['model','score']:
            report=path/f'{kind}_report.json';proof=json.loads((path/f'{kind}_verification.json').read_text())
            assert proof['passed'] and proof[f'{kind}_report_sha256']==sha(report)
        score=json.loads((path/'score_report.json').read_text())
        assert score['scores_sha256']==sha(path/'scores.parquet')
        assert score['model_report_sha256']==sha(path/'model_report.json')
    selected=json.loads((rank/'selection_report.json').read_text())
    proof=json.loads((rank/'selection_verification.json').read_text())
    assert proof['passed'] and proof['selection_report_sha256']==sha(rank/'selection_report.json')
    assert selected['selection_sha256']==sha(rank/'selection.parquet')
    assert selected['core_sha256']==sha(rank/'frozen_numeric_core.tdx')
    assert selected['model_report_sha256']==sha(rank/'model_report.json')
    absolute=json.loads((root/'model_report.json').read_text())
    ranked=json.loads((rank/'model_report.json').read_text())
    assert absolute['variant']=='absolute' and ranked['variant']=='rank'
    for key in ['training_start','training_end']:
        assert absolute[key]==ranked[key]==config[key]
    assert absolute['feature_names']==ranked['feature_names']==list(base.EXPRESSIONS)
    assert absolute['feature_report_sha256']==ranked['feature_report_sha256']==sha(base.FEATURES/'feature_report.json')
    assert absolute['last_observation']<config['evaluation_start']
    assert ranked['last_observation']<config['evaluation_start']
    assert config['absolute_score_cut']==.60
    return config,rank,selected,absolute,ranked


def compose_core(rank_source,absolute_source,rank_cut,absolute_cut):
    """Share the indicator prefix and give the absolute model distinct local names."""
    prefix,rank_body=rank_source.split('T01:=',1)
    other,absolute_body=absolute_source.split('T01:=',1)
    assert prefix==other
    rank_body='T01:='+rank_body.split('\nCORE:',1)[0]+'\n'
    absolute_body='T01:='+absolute_body.split('\nCORE:',1)[0]+'\n'
    absolute_body=re.sub(r'\bT(\d{2})\b',r'U\1',absolute_body)
    absolute_body=re.sub(r'\bSC\b','AC',absolute_body)
    core=prefix+rank_body+absolute_body+f'CORE:(SC>{rank_cut:.17g}) AND (AC>{absolute_cut:.17g});\n'
    names=re.findall(r'\b([A-Z][A-Z0-9]*):=',core)
    assert len(names)==len(set(names)),'The two native model namespaces must not overwrite inputs or each other'
    return core


def freeze():
    root=base.ROOT
    if (root/'selection_report.json').exists():
        raise ValueError('Do not replace the frozen rank-plus-absolute selection')
    config,rank,selected,absolute,ranked=checked_inputs()
    old=pd.read_parquet(rank/'selection.parquet');scores=pd.read_parquet(root/'scores.parquet')
    pd.testing.assert_frame_equal(old.drop(columns='selected'),scores[old.drop(columns='selected').columns],check_exact=True)
    out=old.copy();out['selected'] &= scores.formula_input_valid&scores.score.gt(config['absolute_score_cut'])
    out.to_parquet(root/'selection.parquet',index=False,compression='zstd')
    absolute_core=base.native_core(absolute,config['absolute_score_cut'],base.EXPRESSIONS,base.HEADER)
    (root/'absolute_numeric_core.tdx').write_text(absolute_core)
    core=compose_core((rank/'frozen_numeric_core.tdx').read_text(),absolute_core,
        selected['chosen_threshold']['threshold'],config['absolute_score_cut'])
    (root/'frozen_numeric_core.tdx').write_text(core)
    r=dict(protocol_sha256=sha(base.PROTOCOL),model_report_sha256=sha(root/'model_report.json'),
        score_report_sha256=sha(root/'score_report.json'),rank_root=str(rank),
        rank_selection_report_sha256=sha(rank/'selection_report.json'),rank_model_report_sha256=sha(rank/'model_report.json'),
        selection_sha256=sha(root/'selection.parquet'),core_sha256=sha(root/'frozen_numeric_core.tdx'),
        absolute_core_sha256=sha(root/'absolute_numeric_core.tdx'),chosen_threshold=selected['chosen_threshold'],
        chosen_threshold_applies_to='rank_model_only',absolute_score_cut=config['absolute_score_cut'],
        original_rank_selected=int(old.selected.sum()),selected=int(out.selected.sum()),
        by_half=out.groupby('half').selected.agg(['size','sum']).reset_index().to_dict('records'),
        evaluation_start=config['evaluation_start'],evaluation_end=config.get('evaluation_end'),
        two_frozen_scores_required=True,year_2025_is_exploratory=True,new_group_outcomes_read=False,
        rank_source_outcomes_previously_seen=True,
        no_exit_rules=True,new_2026_prices_read=False,software_compilation_verified=False)
    save_json(root/'selection_report.json',r)
    return r


def verify():
    root=base.ROOT;config,rank,selected,absolute,ranked=checked_inputs()
    r=json.loads((root/'selection_report.json').read_text())
    for key,path in [('protocol_sha256',base.PROTOCOL),('model_report_sha256',root/'model_report.json'),
        ('score_report_sha256',root/'score_report.json'),('rank_selection_report_sha256',rank/'selection_report.json'),
        ('rank_model_report_sha256',rank/'model_report.json'),('selection_sha256',root/'selection.parquet'),
        ('core_sha256',root/'frozen_numeric_core.tdx'),('absolute_core_sha256',root/'absolute_numeric_core.tdx')]:
        assert r[key]==sha(path)
    cut=ranked['thresholds'][3];start=config['evaluation_start'];end=config.get('evaluation_end','2026-01-01')
    assert r['chosen_threshold']==cut and cut['training_quantile']==.995
    assert r['absolute_score_cut']==config['absolute_score_cut']==.60
    c=base.conn()
    c.read_parquet(str(root/'scores.parquet')).create_view('a')
    c.read_parquet(str(rank/'scores.parquet')).create_view('b')
    expected=c.sql(f'''SELECT a.date,a.code,a.half,a.board,a.decision_shares,
        a.date>='{start}' AND a.date<'{end}' AND a.formula_input_valid AND b.formula_input_valid
        AND b.score>{cut['threshold']:.17e} AND a.score>0.60 AS selected
        FROM a JOIN b USING(date,code) ORDER BY date,code''').df()
    pd.testing.assert_frame_equal(pd.read_parquet(root/'selection.parquet'),expected,check_exact=True)
    assert r['selected']==int(expected.selected.sum())
    assert r['selected']<=r['original_rank_selected']==selected['selected']
    assert expected.loc[expected.selected,'date'].ge(absolute['training_end']).all()
    # Recover each component from the text, including its exact summation order.
    core=(root/'frozen_numeric_core.tdx').read_text()
    prefix,body=core.split('T01:=',1);rank_body,absolute_body=body.split('U01:=',1)
    gate=f"CORE:(SC>{cut['threshold']:.17g}) AND (AC>0.59999999999999998);\n"
    assert absolute_body.endswith(gate)
    restored_rank=prefix+'T01:='+rank_body+f"CORE:SC>{cut['threshold']:.17g};\n"
    assert restored_rank==(rank/'frozen_numeric_core.tdx').read_text()
    restored_absolute='U01:='+absolute_body[:-len(gate)]
    restored_absolute=re.sub(r'\bU(\d{2})\b',r'T\1',restored_absolute)
    restored_absolute=re.sub(r'\bAC\b','SC',restored_absolute)
    restored_absolute=prefix+restored_absolute+'CORE:SC>0.59999999999999998;\n'
    assert restored_absolute==(root/'absolute_numeric_core.tdx').read_text()
    assert restored_absolute==base.native_core(absolute,.60,base.EXPRESSIONS,base.HEADER)
    proof=dict(passed=True,selection_report_sha256=sha(root/'selection_report.json'),rows=len(expected),
        all_selection_flags_rebuilt_from_both_raw_scores=True,both_numeric_core_components_and_order_preserved=True,
        no_training_period_selection=True,no_new_group_outcomes_read=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(root/'selection_verification.json',proof)
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('fold',choices=['2024','recent','combined'])
    p.add_argument('stage',choices=['model','verify_model','scores','freeze','verify','analyze'])
    a=p.parse_args();setup(a.fold)
    if a.fold=='combined':
        assert a.stage in ['freeze','verify','analyze']
        r=(linkage.common_analysis(linkage.COMBINED,linkage.PROTOCOL) if a.stage=='analyze'
            else getattr(linkage,'combine' if a.stage=='freeze' else 'verify_combined')())
    elif a.stage in ['model','verify_model']:
        r=getattr(relative,a.stage)('absolute')
    elif a.stage in ['freeze','verify']:
        r=globals()[a.stage]()
    else:
        r=getattr(base,a.stage)()
    print(json.dumps(r,ensure_ascii=False,indent=2))
