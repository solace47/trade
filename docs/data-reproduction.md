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

将 `RUN_ID` 换成 Actions 运行编号。调度器会导入、核对并删除每个远程分片文件，再依次启动剩余批次。只有 2022–2023 年筛选出合格条件后，才将条件与卖出周期写入 `config/strategy-freeze.json`，并通过 `trade_research.holdout_eval` 验证 2024 年以后数据。本次筛选结果为空，因此没有该配置文件。研究数据的具体版本保存在本地锁文件，不写入文档。
