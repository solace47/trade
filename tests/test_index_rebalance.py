import pandas as pd
import pytest

from trade_research.index_rebalance import parse_attachment, valid_cent
from trade_research.absolute_ridge import match_controls


def test_reserves_ignored_and_later_actual_headings_reactivate_parser():
    pages=['''中证1000指数样本调整名单：
调出名单 调入名单
000001 平安银行 600000 浦发银行''', '''
000002 万科 A 600001 样本二
沪深300指数备选名单：
000099 不应读入 600099 不应读入
中证 A50 指数样本调整名单：
000003 样本三 600003 样本四''']
    rows=parse_attachment(pages)
    assert len(rows)==6
    assert set(rows['index'])=={'中证1000','中证A50'}
    assert rows.symbol.tolist()==['000001','600000','000002','600001','000003','600003']
    assert rows.loc[2,'name']=='万科A'
    assert rows.loc[2,'page']==2


@pytest.mark.parametrize('text',[
 '000001 样本 600000 样本\n000001 样本 600002 样本',
 '000001 样本 000001 样本',
 '000001 缺少另侧',
])
def test_bad_actual_rows_fail_closed(text):
    with pytest.raises(ValueError):
        parse_attachment(['中证1000指数样本调整名单：\n'+text])


@pytest.mark.parametrize('price,expected',[(5.00000001,True),(5.001,False),(0,False),(float('nan'),False)])
def test_only_storage_error_is_restored_to_cent(price,expected):
    assert valid_cent(price)==expected


def test_matching_uses_liquidity_order_and_never_reuses_control():
    features=pd.DataFrame({'date':['2024-06-14']*3,'code':['sh.600001','sh.600002','sh.600003'],
        'board':['main']*3,'price_signal':[10.]*3,'amount_signal':[1e8]*3,
        'return20_prior_adjusted':[0.]*3,'return_1450':[0.]*3})
    chosen=pd.DataFrame({'date':['2024-06-14']*2,'code':['sh.600002','sh.600001'],'daily_rank':[6,1]})
    result=match_controls(chosen,features)
    assert len(result)==1
    assert result.iloc[0].pair_id=='sh.600001'
    assert result.iloc[0].code=='sh.600003'
    assert result.iloc[0].daily_rank==1


def test_batch_summary_keeps_unknown_and_does_not_weight_by_stock_count():
    from trade_research.index_rebalance import BATCHES
    from trade_research.index_rebalance_eval import summarize
    rows=[]
    for n,date in enumerate(BATCHES.values(),1):
        for j in range(n):
            for arm in ('high','low'):
                for horizon in (1,5):
                    row={'date':date,'code':f'{arm}{j}','arm':arm,'pair_id':str(j),'horizon':horizon,
                         'primary_top5':True,'entry_status':'filled','exit_price':10.,
                         'known_return5':.01,'execution_source_valid':True,'catalog_action_applied':False}
                    for slip in (5,15):
                        for kind in ('return','lower','upper'):
                            row[f'catalog_scenario_{kind}{slip}']=n/100 if arm=='high' else 0.
                    rows.append(row)
    frame=pd.DataFrame(rows)
    _,summary=summarize(frame)
    assert all(abs(r['own_return']-.025)<1e-12 for r in summary)
    assert all(abs(r['edge_return']-.025)<1e-12 for r in summary)
    frame.loc[(frame.date.eq('2024-06-14') & frame.arm.eq('high')),'catalog_scenario_return15']=float('nan')
    cells,summary=summarize(frame)
    assert all(r['own_return'] is None and r['edge_return'] is None for r in summary if r['slip']==15)
    assert all(r['own_return'] is not None for r in summary if r['slip']==5)
