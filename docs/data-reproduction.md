# 数据复算

运行环境和先导样本命令见仓库首页。原始行情、版本锁、审计表和研究输出统一放在被 Git 忽略的 `data/`。

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

然后在 GitHub Actions 的“历史行情分片验证”中，以 `first_shard=0`、`max_shards=4`、`limit_symbols=0` 启动首批。获得运行编号后执行：

```bash
PYTHONPATH=src .venv/bin/python scripts/orchestrate_market.py --initial-run-id RUN_ID
PYTHONPATH=src .venv/bin/python -m trade_research.market_quality
PYTHONPATH=src .venv/bin/python -m trade_research.strategy_scan
PYTHONPATH=src .venv/bin/python -m trade_research.strategy_select
```

将 `RUN_ID` 换成 Actions 运行编号。调度器会导入、核对并删除每个远程分片文件，再依次启动剩余批次。完整 2024 年及以后结果只能在选定条件与卖出周期写入 `config/strategy-freeze.json` 后，通过 `trade_research.holdout_eval` 读取。研究数据的具体版本保存在本地锁文件，不写入文档。
