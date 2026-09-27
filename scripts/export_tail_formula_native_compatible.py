"""Export finite decimal literals and independently replay the native IF text."""
import argparse
from decimal import Decimal
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json,sha


def compatible_text(source):
    expanded=re.sub(r'\b\d+(?:\.\d+)?[eE][-+]?\d+\b',lambda m:format(Decimal(m.group()),'f'),source)
    return re.sub(r'\b\d+\.\d+\b',lambda m:format(float(m.group()),'.12f').rstrip('0').rstrip('.'),expanded)


def parse_tree(text):
    pattern=r'IF|X\d{2}|<=|-?\d+(?:\.\d+)?|[(),]'
    tokens=re.findall(pattern,text)
    assert ''.join(tokens)==text
    position=0

    def take(expected=None):
        nonlocal position
        token=tokens[position];position+=1
        assert expected is None or token==expected,(token,expected)
        return token

    def node():
        if tokens[position]!='IF':
            return float(take())
        take('IF');take('(')
        feature=int(take()[1:])-1;take('<=');cut=int(take());take(',')
        left=node();take(',');right=node();take(')')
        return feature,cut,left,right

    result=node();assert position==len(tokens)
    return result


def check_structure(parsed,model,node=0):
    left=model['children_left'][node]
    if left<0:
        assert isinstance(parsed,float)
        source=format(.05*model['value'][node],'.17g')
        assert parsed==float(compatible_text(source))
        return 1
    feature,cut,lower,upper=parsed
    assert feature==model['feature'][node] and cut==math.floor(model['threshold'][node])
    return check_structure(lower,model,left)+check_structure(upper,model,model['children_right'][node])


def evaluate(parsed,x):
    result=np.empty(len(x),dtype=float)

    def visit(node,mask):
        if isinstance(node,float):
            result[mask]=node
        else:
            feature,cut,left,right=node
            low=x[:,feature]<=cut
            visit(left,mask&low);visit(right,mask&~low)

    visit(parsed,np.ones(len(x),dtype=bool))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--features-root',type=Path,required=True)
    a=p.parse_args();root=a.root
    source=root/'frozen_numeric_core.tdx';output=root/'native_compatible_core.tdx'
    model=json.loads((root/'model_report.json').read_text())
    selection=json.loads((root/'selection_report.json').read_text())
    proof=json.loads((root/'score_verification.json').read_text())
    assert selection['core_sha256']==sha(source)
    assert proof['passed'] and proof['score_report_sha256']==sha(root/'score_report.json')
    score_report=json.loads((root/'score_report.json').read_text())
    assert score_report['scores_sha256']==sha(root/'scores.parquet')
    assert score_report['model_report_sha256']==sha(root/'model_report.json')
    fr=json.loads((a.features_root/'feature_report.json').read_text())
    assert model['feature_report_sha256']==sha(a.features_root/'feature_report.json')
    assert fr['features_sha256']==sha(a.features_root/'features.parquet')
    original=source.read_text();core=compatible_text(original)
    # Feature expressions and the encoder must remain byte-identical.
    assert core.split('T01:=')[0]==original.split('T01:=')[0]
    lines=dict(line.rstrip(';').split(':=',1) for line in core.splitlines() if ':=' in line)
    trees=[parse_tree(lines[f'T{i:02d}']) for i in range(1,65)]
    leaf_checks=sum(check_structure(t,m) for t,m in zip(trees,model['trees']))
    sums=lines['SC'].split('+')
    assert sums[1:]==[f'T{i:02d}' for i in range(1,65)]
    bias=float(sums[0]);threshold=float(re.search(r'^CORE:SC>([^;]+);$',core,re.M).group(1))
    f=pd.read_parquet(a.features_root/'features.parquet',columns=['date','code','half','formula_input_valid',*model['feature_names']])
    scores=pd.read_parquet(root/'scores.parquet')
    pd.testing.assert_frame_equal(f[['date','code']],scores[['date','code']],check_exact=True)
    good=f.formula_input_valid.to_numpy();f=f.loc[good].copy()
    raw=f[model['feature_names']].to_numpy(dtype=float)
    x=np.floor(np.clip(100*raw+10000+.000001,0,999999)).astype('int32')
    rebuilt=np.full(len(f),bias)
    for tree in trees:
        rebuilt+=evaluate(tree,x)
    old=scores.loc[good,'score'].to_numpy();oldcut=selection['chosen_threshold']['threshold']
    oldflags=old>oldcut;newflags=rebuilt>threshold
    evaluation=f.date.ge(selection['evaluation_start'])
    if selection.get('evaluation_end'):
        evaluation &= f.date.lt(selection['evaluation_end'])
    rows=[]
    for half in sorted(f.half.unique()):
        mask=f.half.eq(half).to_numpy()
        rows.append(dict(half=half,rows=int(mask.sum()),old_selected=int(oldflags[mask].sum()),
            new_selected=int(newflags[mask].sum()),changed_flags=int((oldflags[mask]!=newflags[mask]).sum())))
    changes=int(((oldflags!=newflags)&evaluation.to_numpy()).sum())
    assert changes==0,'Do not silently replace a formula with different evaluation selections'
    output.write_text(core)
    report=dict(passed=True,original_core_sha256=sha(source),native_core_sha256=sha(output),
        model_report_sha256=sha(root/'model_report.json'),feature_report_sha256=sha(a.features_root/'feature_report.json'),
        rows=len(f),leaf_literal_checks=leaf_checks,feature_expressions_and_encoder_unchanged=True,
        all_tree_paths_and_literal_values_rebuilt=True,maximum_score_difference=float(abs(rebuilt-old).max()),
        old_threshold=oldcut,new_threshold=threshold,evaluation_selected=int((newflags&evaluation.to_numpy()).sum()),
        evaluation_changed_flags=changes,by_half=rows,decimal_places=12,
        scope='仅复算导出文本在已编码输入上的分数，不代替客户端原始行情及指标计算一致性检验。',
        complete_selector_verified=False,outcomes_read=False,new_2026_prices_read=False)
    save_json(root/'native_compatibility_report.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
