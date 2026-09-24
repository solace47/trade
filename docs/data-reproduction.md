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

将 `RUN_ID` 换成 Actions 运行编号。调度器会导入、核对并删除每个远程分片文件，再依次启动剩余批次。当前研究以 2024 年开发、2025 年验证；只有通过筛选后才把条件和卖出周期写入 `config/strategy-freeze.json`，通过 `trade_research.holdout_eval` 验证 2026 年数据。2023 年及以前不计入策略收益。研究数据的具体版本保存在本地锁文件，不写入文档。

复算[策略结果](strategy-results.md)中的低成交额仓位敏感性：

```bash
PYTHONPATH=src .venv/bin/python -m trade_research.exploratory_signals --candidate low_amount_neutral --output data/research/size_signal_neutral.parquet
PYTHONPATH=src .venv/bin/python -m trade_research.size_sensitivity --signals data/research/size_signal_neutral.parquet --output data/research/size_sensitivity_neutral.parquet
```

第二步默认从本地公开分钟档案按需读取 14:52–14:55 分钟条，并分别以 2 万、5 万、10 万元单笔资金重新估算四个持有期。
按 14:50 市场广度复核这两组探索信号时运行 `PYTHONPATH=src .venv/bin/python scripts/regime_probe.py > data/research/regime_probe.jsonl`。
