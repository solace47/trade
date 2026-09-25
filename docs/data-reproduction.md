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
平静上涨与市场广度的同日配对检验运行 `PYTHONPATH=src .venv/bin/python -m trade_research.quiet_breadth`，输出保存在 `data/research/quiet_breadth/`。
历史时段尾盘量能检验运行 `PYTHONPATH=src .venv/bin/python -m trade_research.tail_volume_surprise`；以其 `repricing_signals.parquet` 调用 `trade_research.size_sensitivity --notionals 100000 --horizons 1 5` 可逐笔核对原始分钟成交。
市场尾盘方向分层先运行 `PYTHONPATH=src .venv/bin/python -m trade_research.late_market_direction inputs`，输入门槛通过后运行同模块 `evaluate`，重用已核对的回落配对原始分钟复价；见[专项记录](late-market-direction-plan.md)。
14:20 前全市场方向先运行 `PYTHONPATH=src .venv/bin/python -m trade_research.market_pre_tail_state`，输入门槛通过后运行 `PYTHONPATH=src .venv/bin/python -m trade_research.market_pre_tail_eval`，重用同一冻结配对的真实分钟成交，并附全市场未严配诊断；见[专项记录](market-pre-tail-state-plan.md)。
同行业相对尾盘压力仅做输入审计：`PYTHONPATH=src .venv/bin/python -m trade_research.late_sector_pressure_inputs`；严格跨行业对照未过预设门槛，详见[专项记录](late-sector-pressure-plan.md)。
市场尾盘下跌日逆市上涨对平稳股的检验依次运行 `trade_research.down_market_rally_inputs`、`trade_research.size_sensitivity --signals data/research/down_market_rally/repricing_signals.parquet --output data/research/down_market_rally/repriced.parquet --notionals 20000 100000 --horizons 1 5 --exit-windows morning close`、`trade_research.down_market_rally_eval`；各命令以 `PYTHONPATH=src .venv/bin/python -m` 开头，详见[配对记录](down-market-rally-avoidance-plan.md)。
下跌尾盘的成交时点仅做输入审计：`PYTHONPATH=src .venv/bin/python -m trade_research.late_volume_timing`；严格同日配对未过冻结门槛，见[专项记录](late-volume-timing-plan.md)。
尾盘后半段路径仅做 14:50 输入审计：`PYTHONPATH=src .venv/bin/python -m trade_research.late_half_pressure`；配对覆盖未过预设门槛，程序不会输出可用于复价的名单。
尾盘回落与次晨反转依次运行 `trade_research.late_to_open_reversal`、`trade_research.size_sensitivity --signals data/research/late_to_open_reversal/repricing_signals.parquet --output data/research/late_to_open_reversal/repriced.parquet --notionals 20000 100000 --horizons 1 5 --exit-windows morning close`、`trade_research.late_to_open_reversal_eval`；命令均以 `PYTHONPATH=src .venv/bin/python -m` 开头，结果见[专项记录](late-to-open-reversal-plan.md)。
同项的前段轨迹匹配敏感性将首步改为 `trade_research.late_to_open_reversal --match-pre-tail`，复价输入、输出和末步评价的 `--output` 均使用 `data/research/late_pretrend_match/`；原方案默认名单不变。
收盘集合竞价入场先运行 `PYTHONPATH=src .venv/bin/python -m trade_research.closing_auction_entry` 审核 15:00 单条成交和容量；门槛通过后，以原 `late_to_open_reversal/repricing_signals.parquet` 运行 `trade_research.size_sensitivity --output data/research/closing_auction_entry/repriced.parquet --notionals 20000 100000 --horizons 1 --exit-windows morning close --entry-windows baseline auction`，最后运行 `trade_research.closing_auction_entry_eval`。第二、三步同样以 `PYTHONPATH=src .venv/bin/python -m` 开头；见[入场记录](closing-auction-entry-plan.md)。
次晨 09:34 条件退出沿用原 1,063 对和两种 T+1 原始分钟复价，先运行 `PYTHONPATH=src .venv/bin/python -m trade_research.next_morning_exit` 只审核输入；门槛通过后运行 `PYTHONPATH=src .venv/bin/python -m trade_research.next_morning_exit_eval` 比较条件与两种固定卖出时段，结果见[退出记录](next-morning-exit-plan.md)。
尾盘相对波动风险检验依次运行 `trade_research.late_variance --threads 8`、`trade_research.late_variance_risk`，再将 `data/research/late_variance_risk/repricing_signals.parquet` 交给 `trade_research.size_sensitivity --notionals 20000 100000 --horizons 1 5`，输出至同目录 `repriced.parquet`，最后运行 `trade_research.late_variance_risk_eval`。事后固定排序敏感性运行 `PYTHONPATH=src .venv/bin/python scripts/variance_seed_stability.py`，十组均写入同目录 `seed_stability.json`；口径与结果见[专项记录](late-variance-risk-plan.md)。
尾盘下跌半方差的输入审计运行 `trade_research.late_variance --threads 8 --output-dir data/research/late_semivariance`，随后运行 `trade_research.late_semivariance_input`；匹配上界未过事前门槛，程序不读收益，见[专项记录](late-semivariance-plan.md)。
尾盘延迟一分钟的同股复价先运行 `trade_research.entry_delay` 审核五分钟输入，再以 `trade_research.size_sensitivity --signals data/research/late_variance_risk/repricing_signals.parquet --output data/research/entry_delay/repriced.parquet --notionals 20000 100000 --horizons 1 5 --entry-windows baseline delay_one_minute` 复价，最后运行 `trade_research.entry_delay --evaluate`；见[执行记录](entry-delay-plan.md)。
市场尾盘波动状态的输入先运行 `trade_research.market_variance_state`，确认两种状态各半年的配对覆盖后运行 `trade_research.market_variance_state --evaluate`；它重用已核对的原始分钟成交结果，见[分层记录](market-variance-state-plan.md)。
尾盘剩余方差先运行 `trade_research.late_residual_variance --threads 8`，再运行同命令加 `--market-proxy clipped_mean`；分别运行 `trade_research.late_residual_variance_risk` 及加 `--market-proxy clipped_mean` 冻结两版名单。只有输入审计通过才对各目录的 `repricing_signals.parquet` 运行 `trade_research.size_sensitivity --notionals 20000 100000 --horizons 1 5`，结果写为同目录 `repriced.parquet`，再以 `trade_research.late_variance_risk_eval --output` 指向相应目录；见[剩余方差记录](late-residual-variance-plan.md)。

近期尾盘研究从原始分钟文件重建特征，再执行 2024 年开发扫描：

```bash
PYTHONPATH=src .venv/bin/python -m trade_research.intraday_features --threads 8
PYTHONPATH=src .venv/bin/python -m trade_research.intraday_scan --year 2024
PYTHONPATH=src .venv/bin/python -m trade_research.intraday_scan --year 2025
PYTHONPATH=src .venv/bin/python scripts/intraday_factor_bins.py
PYTHONPATH=src .venv/bin/python scripts/late_ridge_probe.py
PYTHONPATH=src .venv/bin/python scripts/tail_return_decomposition.py
PYTHONPATH=src .venv/bin/python -m trade_research.entry_timing --workers 4
PYTHONPATH=src .venv/bin/python -m trade_research.steady_path
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
