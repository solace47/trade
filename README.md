# A 股尾盘短线策略研究

目标是在交易日 14:50 利用当时已知的数据选股，模拟 14:52–14:55 买入，并检验 T+1 至 T+5 的退出结果。只有完成历史全市场、跨年份、样本外及交易约束验证后，才会给出通达信和同花顺公式。

## 当前数据结论

- [BaoStock](https://pypi.org/project/baostock/) 日线提供历史交易状态、ST 标识、昨收及开高低收。100 只先导样本的日线和 5 分钟线覆盖了 2025-09-01 至 2026-08-31：24,166 个正常交易日的 5 分钟记录均为完整 48 根，另有 22 个全零占位日与停牌状态一致。
- BaoStock 5 分钟线在 13,839 个正常交易日与其日线的开高低收不一致，**未通过价格一致性验收**。相关原始记录保留在本地供复核，不作为正式信号主数据。
- [公开 1 分钟档案](https://huggingface.co/datasets/neigezhu/china-a-share-1min-ohlcv) 固定到提交 `ba589a11534825044fe5a6b84838f50ba8d8d188`。当前先导验收区间为 2025-08-07 至 2026-08-06，每个正常交易日预期 241 根，包括 09:30 集合竞价记录。零成交量的报价占位记录不参与当日成交高低价计算；当日开盘价使用独立日线校验。
- 多年日线股票池从历史上市、退市日期构建，覆盖沪深两市并纳入退市股票。北交所尚未纳入这套 BaoStock 状态数据，因此现阶段不能称作全 A 股验证。
- 固定版本的 1 分钟档案列出 5,795 只股票（北交所 327、上交所 2,423、深交所 3,045）。多年 BaoStock 股票池中的 5,430 只沪深股票有 5,423 只在档案中；缺少 6 只档案截止前不久上市的新股和 1 只 2025 年退市股票，后续研究会明确剔除或补源。

公开 1 分钟档案的具体上游行情供应商和完整公司行动因子尚未独立核实。跨除权除息日的原始价格收益会单独标记，不能直接纳入绩效统计。

## 复现先导研究

需要 Python 3.12 和网络连接。原始行情和本地报告保存在 `data/`，不会被 Git 提交。

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
export PYTHONPATH=src

# 冻结历史时点股票池，下载 100 只样本的日线与 5 分钟线。
.venv/bin/python -m trade_research.ingest --workers 4
.venv/bin/python -m trade_research.audit

# 下载固定版本的 1 分钟数据，并生成数据审计、14:50 快照和未来成交结果。
.venv/bin/python -m trade_research.hf_download --workers 8
.venv/bin/python -m trade_research.hf_audit
.venv/bin/python -m trade_research.hf_snapshot
.venv/bin/python -m trade_research.hf_outcomes
.venv/bin/python -m trade_research.pilot_study

.venv/bin/pytest -q
```

扩大日线范围时运行：

```bash
.venv/bin/python -m trade_research.market_daily_ingest --workers 4
.venv/bin/python -m trade_research.market_daily_audit

# 冻结全量 1 分钟文件目录，再按批次继续下载同一版本的公开档案。
.venv/bin/python -m trade_research.hf_inventory
.venv/bin/python -m trade_research.hf_full_download --workers 8
```

该股票池覆盖 2020-01-01 至 2026-08-06，另从 2019-01-01 起保留指标预热数据。每只股票的下载结果会独立落盘，命令可重新执行以继续未完成部分。

## 研究口径

信号只读取 14:50 及之前的 1 分钟记录和交易前已知的日线信息。成交模拟使用 14:52–14:55 的成交均价，单笔目标金额 10 万元，参与量不超过该四分钟成交量的 10%。费用、滑点、涨跌停不能成交、停牌、T+1 和延迟卖出均在 `hf_outcomes.py` 中明确配置；这些成交仍是估算，无法代替逐笔订单簿。

`pilot_study.py` 的条件是预先冻结的研究起点，100 只股票按 2025-09-01 的历史股票池选取。其统计结果只用于检查研究流程，不据此宣称有可交易优势。印花税减半及交易规则分别参考[财政部、税务总局公告](https://shanxi.chinatax.gov.cn/web/detail/sx-11400-545-1780448)和[上交所交易规则](https://www.sse.com.cn/lawandrules/sselawsrules2025/stocks/exchange/c/c_20260424_10816482.shtml)；券商佣金及滑点为可调整的模拟参数。
