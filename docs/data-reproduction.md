# 数据复算

使用 Python 3.12。原始行情、版本锁、审计表和研究输出统一放在被 Git 忽略的 `data/`。

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

先导样本的复算顺序为 `trade_research.ingest`、`hf_download`、`hf_audit`、`hf_snapshot`、`hf_outcomes`、`pilot_study`；均使用 `PYTHONPATH=src .venv/bin/python -m` 运行。

## 沪深历史日线

```bash
PYTHONPATH=src .venv/bin/python -m trade_research.market_daily_ingest --workers 4
PYTHONPATH=src .venv/bin/python -m trade_research.market_daily_audit
```

## 全市场分钟研究

先在本地锁定公开分钟档案版本：

```bash
PYTHONPATH=src .venv/bin/python -c 'from trade_research.hf_download import locked_revision; locked_revision()'
```

如需在本地保留全市场分钟原始文件，可续传下载（约 43 GB）：

```bash
PYTHONPATH=src .venv/bin/python scripts/download_market_minute.py --workers 4
```

完成后，本地 `data/hf/pilot/market_download_manifest.json` 记录核验文件数、总字节数和源文件缺失路径。

然后在 GitHub Actions 的“历史行情分片验证”中，以 `first_shard=0`、`max_shards=4`、`limit_symbols=0` 启动首批。获得运行编号后执行：

```bash
PYTHONPATH=src .venv/bin/python scripts/orchestrate_market.py --initial-run-id RUN_ID
PYTHONPATH=src .venv/bin/python -m trade_research.market_quality
PYTHONPATH=src .venv/bin/python -m trade_research.market_integrity
PYTHONPATH=src .venv/bin/python -m trade_research.factor_scan
PYTHONPATH=src .venv/bin/python -m trade_research.strategy_scan
PYTHONPATH=src .venv/bin/python -m trade_research.strategy_select
```

将 `RUN_ID` 换成 Actions 运行编号。调度器会导入、核对并删除每个远程分片文件，再依次启动剩余批次。当前 2024/2025 年研究均为探索性，2025 年已被反复查看，不再称为盲测；只有跨期稳定且规则事前冻结后，才考虑用 `trade_research.holdout_eval` 验证 2026 年数据。2023 年及以前不计入策略收益。研究数据的具体版本保存在本地锁文件，不写入文档。

复算[策略结果](strategy-results.md)中的低成交额仓位敏感性：

```bash
PYTHONPATH=src .venv/bin/python -m trade_research.exploratory_signals --candidate low_amount_neutral --output data/research/size_signal_neutral.parquet
PYTHONPATH=src .venv/bin/python -m trade_research.size_sensitivity --signals data/research/size_signal_neutral.parquet --output data/research/size_sensitivity_neutral.parquet
```

第二步默认从本地公开分钟档案按需读取 14:52–14:55 分钟条，并分别以 2 万、5 万、10 万元单笔资金重新估算四个持有期。
按 14:50 市场广度复核这两组探索信号时运行 `PYTHONPATH=src .venv/bin/python scripts/regime_probe.py > data/research/regime_probe.jsonl`。

近期尾盘研究从原始分钟文件重建特征，再执行 2024 年开发扫描：

```bash
PYTHONPATH=src .venv/bin/python -m trade_research.intraday_features --threads 8
PYTHONPATH=src .venv/bin/python -m trade_research.intraday_scan --year 2024
PYTHONPATH=src .venv/bin/python scripts/intraday_factor_bins.py
PYTHONPATH=src .venv/bin/python scripts/late_ridge_probe.py
```

低成交额分层由 `trade_research.stratified_low_sample` 固定抽样，并用 `trade_research.size_sensitivity` 以每笔 2 万元重算；`trade_research.paired_controls` 按信号日同随机组配对。退出时点试验调用 `size_sensitivity --exit-windows close morning late_morning`；提前卖出试验见 `scripts/adaptive_exit_probe.py`。本地结果都写入 `data/research/`，不加入 README 或 Git。

市场相对收益与成交量试验运行 `PYTHONPATH=src .venv/bin/python -m trade_research.residual_liquidity`。它只读取 2024–2025 年信号，年末预留十个交易日，并把逐笔和汇总结果写入被忽略的 `data/research/`。

日内流动性冲击改写运行 `PYTHONPATH=src .venv/bin/python -m trade_research.liquidity_shock`；它使用 2023 年日线仅预热此前 60 个交易日的非流动性均值，策略收益仍从 2024 年开始。

前述试验生成本地特征表后，运行 `PYTHONPATH=src .venv/bin/python -m trade_research.liquidity_conditioned` 可复核动量与板块控制。

周度适配运行 `PYTHONPATH=src .venv/bin/python -m trade_research.weekly_liquidity`。当前周只读信号日前的已完成日线和当日 14:50 快照；前 12 个有效周用于基准，2023 年只用于预热。

市场相对下跌信号的次日上午退出重算运行 `PYTHONPATH=src .venv/bin/python scripts/residual_exit_probe.py`，以原始分钟文件重新估算每笔 2 万和 5 万元的成交。

固定风险排除规则运行 `PYTHONPATH=src .venv/bin/python -m trade_research.risk_filter`，同一确定性随机顺序对照过滤前后的选股。

随机对照和卖出时段的信号清单先分别运行 `trade_research.exploratory_signals` 的 `random_low_amount`、`random_liquid`、`low_amount_neutral`、`mid_amount_prior_loser`，输出到 `data/research/` 下的 `random_low_all.parquet`、`random_liquid_all.parquet`、`size_signal_neutral.parquet`、`mid_loser_all.parquet`；再用 `scripts/assemble_research_lists.py --kind random|exit --year 2024` 汇总。2025 年随机对照同样运行 `--year 2025`。卖出时段汇总见 `scripts/exit_window_report.py`。以上选股清单先排名，盘后质量审计只标记结果。

价格非同步度与融资兴趣组合依次运行 `trade_research.industry_history --start 2023-07-01 --intervals data/research/industry_intervals_warmup.parquet`、`trade_research.price_nonsynch`、`trade_research.nonsynch_margin_study`、`trade_research.nonsynch_margin_eval`。原始分钟复价用 `trade_research.size_sensitivity --signals data/research/nonsynch_margin_reprice_signals.parquet --output data/research/nonsynch_margin_repriced.parquet --notionals 20000 100000 --horizons 1 5`，然后运行 `trade_research.nonsynch_margin_eval --repriced data/research/nonsynch_margin_repriced.parquet`。2023 年数据只用于 120 日指标预热；专项方案和结果见[组合记录](nonsynch-margin-plan.md)。

开收盘非流动性组合依次运行 `trade_research.open_close_illiquidity`、`trade_research.oc_amihud_margin_study`、`trade_research.oc_amihud_margin_eval`；再用 `trade_research.size_sensitivity --signals data/research/oc_amihud_margin_reprice_signals.parquet --output data/research/oc_amihud_margin_repriced.parquet --notionals 20000 100000 --horizons 1 5` 从分钟档案复价，并运行 `trade_research.oc_amihud_margin_eval --repriced data/research/oc_amihud_margin_repriced.parquet` 校验与汇总。2023 年价格仅用于 60 日指标预热；结论见[专项记录](open-close-illiquidity-plan.md)。
