"""Historical price co-movement adjusted relative inputs for the frozen next-morning folds."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_context_2024 as linkage
from . import tail_formula_float as previous
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from .corporate_cash import save_json,sha

ROOT=Path('data/research/tail_formula_beta')
PROTOCOL=Path('config/tail_formula_beta_protocol.json')
CONTROL=Path('data/research/tail_formula_beta_control')
COMBINED_PROTOCOL=Path('config/tail_formula_beta_combined_protocol.json')
OLD_SELECTION=Path('data/research/tail_formula_float_2025')
NEW_EXPRESSIONS={'KB01':'BTA', 'RB01':'(A01-KB01*J01)/V01',
    'RB02':'(A05-KB01*J02)/V01','RB03':'(A06-KB01*J03)/V01','RB04':'(A07-KB01*J04)/V01'}
EXPRESSIONS={**{k:v for k,v in previous.EXPRESSIONS.items() if k not in ['R01','R02','R03','R04']},
    **NEW_EXPRESSIONS}
HEADER=previous.HEADER+''.join(f'ICP{i}:=REF(INDEXC,B{i-1});\n' for i in range(1,22) if i not in [1,2,6,21])
HEADER+=''.join(f'BRS{i:02d}:=DCP{i}/DCP{i+1}-1;BRI{i:02d}:=ICP{i}/ICP{i+1}-1;\n' for i in range(1,21))
HEADER+='BRSM:=('+ '+'.join(f'BRS{i:02d}' for i in range(1,21))+')/20;\n'
HEADER+='BRIM:=('+ '+'.join(f'BRI{i:02d}' for i in range(1,21))+')/20;\n'
HEADER+='BRPM:=('+ '+'.join(f'BRS{i:02d}*BRI{i:02d}' for i in range(1,21))+')/20;\n'
HEADER+='BRQM:=('+ '+'.join(f'BRI{i:02d}*BRI{i:02d}' for i in range(1,21))+')/20;\n'
HEADER+='BRV:=BRQM-BRIM*BRIM;BTA:=(BRPM-BRSM*BRIM)/IF(BRV>0,BRV,1);\n'
INDEX=previous.context.market.ROOT


def source_files():
    config=json.loads(PROTOCOL.read_text())
    old=json.loads((previous.ROOT/'feature_report.json').read_text())
    proof=json.loads((previous.ROOT/'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256']==sha(previous.ROOT/'feature_report.json')
    for key,path in [('previous_feature_report_sha256',previous.ROOT/'feature_report.json'),
        ('daily_feature_report_sha256',base.SOURCE/'feature_report.json'),
        ('index_source_report_sha256',INDEX/'index_source_report.json'),
        ('index_feature_report_sha256',INDEX/'feature_report.json')]:
        assert config[key]==sha(path)
    assert old['features_sha256']==sha(previous.ROOT/'features.parquet')
    ip=json.loads((INDEX/'feature_verification.json').read_text())
    assert ip['passed'] and ip['feature_report_sha256']==sha(INDEX/'feature_report.json')
    assert sha(INDEX/'feature_verification.json')=='c2015b57f3f666b858d43e8235da74fe2b3801955b6b3dc3f59054f19279d244'
    ir=json.loads((INDEX/'index_source_report.json').read_text())
    assert ir['indices_sha256']==sha(INDEX/'indices.parquet')
    assert ir['indices_sha256']=='3e70480eee1f10d0d9ea6e615c5b74c95867ec4fe6986c2886ab6f99fae7b286'
    for path,digest in ir['raw_files_sha256'].items():
        assert sha(Path(path))==digest
    return previous.context.market.stock.source_files()


def features():
    if (ROOT/'feature_report.json').exists():
        raise ValueError('Do not replace frozen beta-adjusted inputs')
    files=source_files();c=base.conn();c.read_parquet(files).create_view('daily')
    c.read_parquet(str(INDEX/'indices.parquet')).create_view('indices')
    history=c.sql("""WITH active AS(
        SELECT s.date,s.code,s.close::DOUBLE AS cl,s.preclose::DOUBLE AS p,
            s.adjustflag::DOUBLE AS adj,i.close::DOUBLE AS ic FROM daily s LEFT JOIN indices i
            ON s.date=i.date AND i.code=CASE WHEN s.code LIKE 'sh.%' THEN 'sh.000001' ELSE 'sz.399001' END
        WHERE s.date BETWEEN '2023-06-01' AND '2025-12-30' AND s.tradestatus=1),
        lagged AS(SELECT *,lag(cl) OVER w AS pc,lag(ic) OVER w AS pic,
            lag(adj) OVER w AS padj,lag(date) OVER w AS pdate FROM active
            WINDOW w AS(PARTITION BY code ORDER BY date)),
        atoms AS(SELECT *,coalesce(cl>0 AND pc>0 AND ic>0 AND pic>0 AND adj=3 AND padj=3
            AND isfinite(cl) AND isfinite(pc) AND isfinite(ic) AND isfinite(pic),false) AS good,
            cl/pc-1 AS sr,ic/pic-1 AS ir,
            CASE WHEN abs(p-pc)>.005 THEN 1 ELSE 0 END AS reference_break FROM lagged),
        hist AS(SELECT date,code,count(*) OVER w AS beta_history_rows,
            sum(good::INTEGER) OVER w AS beta_history_good,
            min(date) OVER w AS beta_history_start,max(date) OVER w AS beta_history_end,
            min(pdate) OVER w AS beta_history_reference_start,
            avg(CASE WHEN good THEN sr END) OVER w AS beta_stock_mean,
            avg(CASE WHEN good THEN ir END) OVER w AS beta_index_mean,
            avg(CASE WHEN good THEN sr*ir END) OVER w AS beta_product_mean,
            avg(CASE WHEN good THEN ir*ir END) OVER w AS beta_index_square_mean,
            sum(reference_break) OVER w AS beta_history_reference_breaks FROM atoms
            WINDOW w AS(PARTITION BY code ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING))
        SELECT * FROM hist WHERE date>='2024-01-01' ORDER BY date,code""").df()
    c.close();old=pd.read_parquet(previous.ROOT/'features.parquet')
    f=old.merge(history,on=['date','code'],how='left',validate='one_to_one').sort_values(['date','code']).reset_index(drop=True)
    f['beta_index_variance']=f.beta_index_square_mean-f.beta_index_mean**2
    f['beta_covariance']=f.beta_product_mean-f.beta_stock_mean*f.beta_index_mean
    f['beta_source_valid']=(f.beta_history_rows.eq(20)&f.beta_history_good.eq(20)
        &f.beta_history_end.lt(f.date)&f.beta_history_reference_start.lt(f.beta_history_start)
        &f.beta_index_variance.gt(0)&f.V01.gt(0)
        &np.isfinite(f[['beta_index_variance','beta_covariance','V01']]).all(axis=1))
    f['KB01']=(f.beta_covariance/f.beta_index_variance).where(f.beta_source_valid)
    for name,stock,index in [('RB01','A01','J01'),('RB02','A05','J02'),('RB03','A06','J03'),('RB04','A07','J04')]:
        f[name]=((f[stock]-f.KB01*f[index])/f.V01).where(f.beta_source_valid)
    f['prior_formula_input_valid']=f.formula_input_valid
    f['formula_input_valid'] &= f.beta_source_valid&np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    ROOT.mkdir(parents=True,exist_ok=True);f.to_parquet(ROOT/'features.parquet',index=False,compression='zstd')
    r=dict(protocol_sha256=sha(PROTOCOL),previous_feature_report_sha256=sha(previous.ROOT/'feature_report.json'),
        daily_feature_report_sha256=sha(base.SOURCE/'feature_report.json'),
        index_source_report_sha256=sha(INDEX/'index_source_report.json'),
        index_feature_report_sha256=sha(INDEX/'feature_report.json'),
        index_verification_sha256=sha(INDEX/'feature_verification.json'),indices_sha256=sha(INDEX/'indices.parquet'),
        features_sha256=sha(ROOT/'features.parquet'),rows=len(f),valid=int(f.formula_input_valid.sum()),
        previous_valid=int(old.formula_input_valid.sum()),
        newly_invalid=int((f.prior_formula_input_valid&~f.formula_input_valid).sum()),
        valid_with_historical_reference_break=int((f.formula_input_valid&f.beta_history_reference_breaks.gt(0)).sum()),
        first_history_date=f.beta_history_reference_start.min(),last_history_date=f.beta_history_end.max(),
        native_expressions=EXPRESSIONS,native_header=HEADER,native_additional_validity='BRV>0 plus complete prior 21 prices',
        invalid_native_denominator_placeholder_is_not_a_valid_signal=True,
        outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True,native_source_parity_verified=False)
    save_json(ROOT/'feature_report.json',r)
    return {k:v for k,v in r.items() if k not in ['native_expressions','native_header']}


def control(stage):
    check=json.loads((ROOT/'feature_verification.json').read_text())
    assert check['passed'] and check['feature_report_sha256']==sha(ROOT/'feature_report.json')
    report=json.loads((ROOT/'feature_report.json').read_text())
    assert report['features_sha256']==sha(ROOT/'features.parquet')
    assert sha(OLD_SELECTION/'selection_report.json')=='07de34caa25a339fa8e7b8d3665c7d04c044792d01fa170d0fd48f6b77b8f5c5'
    old_report=json.loads((OLD_SELECTION/'selection_report.json').read_text())
    assert old_report['selection_sha256']==sha(OLD_SELECTION/'selection.parquet')
    old_proof=json.loads((OLD_SELECTION/'selection_verification.json').read_text())
    assert old_proof['passed'] and old_proof['selection_report_sha256']==sha(OLD_SELECTION/'selection_report.json')
    if stage=='freeze_control':
        if (CONTROL/'selection_report.json').exists():
            raise ValueError('Do not replace the same-input-quality control')
        assert not any((Path('data/research')/name/'analysis_report.json').exists()
            for name in ['tail_formula_beta_2024','tail_formula_beta_recent','tail_formula_beta_2025'])
        f=pd.read_parquet(ROOT/'features.parquet',columns=['date','code','formula_input_valid'])
        out=pd.read_parquet(OLD_SELECTION/'selection.parquet');pd.testing.assert_frame_equal(out[['date','code']],f[['date','code']])
        out['selected'] &= f.formula_input_valid
        CONTROL.mkdir(parents=True,exist_ok=True);out.to_parquet(CONTROL/'selection.parquet',index=False,compression='zstd')
        r=dict(protocol_sha256=sha(COMBINED_PROTOCOL),feature_report_sha256=sha(ROOT/'feature_report.json'),
            original_selection_report_sha256=sha(OLD_SELECTION/'selection_report.json'),
            selection_sha256=sha(CONTROL/'selection.parquet'),selected=int(out.selected.sum()),
            selection_is_original_48_and_new_input_valid=True,model_refitted=False,year_2025_is_exploratory=True,
            new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
        save_json(CONTROL/'selection_report.json',r);return r
    r=json.loads((CONTROL/'selection_report.json').read_text())
    assert r['protocol_sha256']==sha(COMBINED_PROTOCOL) and r['feature_report_sha256']==sha(ROOT/'feature_report.json')
    assert r['original_selection_report_sha256']==sha(OLD_SELECTION/'selection_report.json')
    assert r['selection_sha256']==sha(CONTROL/'selection.parquet')
    if stage=='verify_control':
        c=base.conn()
        expected=c.sql(f'''SELECT s.date,s.code,s.half,s.board,s.decision_shares,
            s.selected AND f.formula_input_valid AS selected
            FROM read_parquet('{OLD_SELECTION}/selection.parquet') s
            JOIN read_parquet('{ROOT}/features.parquet') f USING(date,code) ORDER BY date,code''').df()
        pd.testing.assert_frame_equal(pd.read_parquet(CONTROL/'selection.parquet'),expected,check_exact=True)
        assert int(expected.selected.sum())==r['selected']
        proof=dict(passed=True,selection_report_sha256=sha(CONTROL/'selection_report.json'),rows=len(expected),
            all_control_selection_flags_rebuilt=True,outcomes_read=False,new_2026_prices_read=False)
        save_json(CONTROL/'selection_verification.json',proof);return proof
    assert stage=='analyze_control'
    return linkage.common_analysis(CONTROL,COMBINED_PROTOCOL)


def setup(fold):
    if fold=='combined':
        linkage.ROOT=Path('data/research/tail_formula_beta_2024')
        linkage.H2=Path('data/research/tail_formula_beta_recent')
        linkage.COMBINED=Path('data/research/tail_formula_beta_2025')
        linkage.PROTOCOL=COMBINED_PROTOCOL
        linkage.H2_SELECTION_SHA=None;linkage.H2_MODEL_SHA=None;linkage.H2_OUTCOMES_PREVIOUSLY_SEEN=False
    else:
        previous.setup(fold)
        root=Path('data/research/tail_formula_beta_'+fold);protocol=Path('config/tail_formula_beta_'+fold+'_protocol.json')
        base.ROOT=root;base.PROTOCOL=protocol;relative.PROTOCOL=protocol;study.ROOT=root;study.PROTOCOL=protocol
        base.FEATURES=ROOT;base.EXPRESSIONS=EXPRESSIONS;base.HEADER=HEADER


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['features','model','verify_model','scores','freeze','verify','analyze',
        'freeze_control','verify_control','analyze_control'])
    p.add_argument('--fold',choices=['2024','recent','combined'],default='2024');a=p.parse_args()
    if a.stage=='features':
        r=features()
    elif a.stage.endswith('_control'):
        r=control(a.stage)
    else:
        setup(a.fold)
        if a.fold=='combined':
            assert a.stage in ['freeze','verify','analyze']
            r=(linkage.common_analysis(linkage.COMBINED,linkage.PROTOCOL) if a.stage=='analyze'
                else getattr(linkage,'combine' if a.stage=='freeze' else 'verify_combined')())
        elif a.stage in ['model','verify_model']:
            r=getattr(relative,a.stage)('relative')
        elif a.stage in ['freeze','verify']:
            r=getattr(study,a.stage)()
        else:
            r=getattr(base,a.stage)()
    print(json.dumps(r,ensure_ascii=False,indent=2))
