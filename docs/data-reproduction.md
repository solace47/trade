# 数据复算

Python 3.12；原始行情、锁定版本、审计表和研究输出均放在被 Git 忽略的 `data/`。2023 年及以前只供指标预热；2024 年及以后才用于筛选和收益。2025 年非盲测，2026 年尽量留给事前冻结验证。

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

以下模块命令均在仓库根目录运行，前缀为 `PYTHONPATH=src .venv/bin/python -m`；各模块的 `--help` 列出输入、输出和可选参数。研究结果及门槛见[策略结果](strategy-results.md)和[输入停止记录](input-gates.md)，数据验收见[研究状态](research-status.md)。历史上已结束的逐项命令仍可在 Git 历史中查阅，不继续累加到本页。

## 原始数据与全市场快照

```bash
PYTHONPATH=src .venv/bin/python -m trade_research.market_daily_ingest --workers 4
PYTHONPATH=src .venv/bin/python -m trade_research.market_daily_audit
PYTHONPATH=src .venv/bin/python -c 'from trade_research.hf_download import locked_revision; locked_revision()'
PYTHONPATH=src .venv/bin/python scripts/download_market_minute.py --workers 4
```

分钟原档约 43 GB，可续传；本地 `data/hf/pilot/market_download_manifest.json` 记录校验文件数、总字节数及缺失路径。在 GitHub Actions 的“历史行情分片验证”中以 `first_shard=0`、`max_shards=4`、`limit_symbols=0` 启动首批，把返回的运行编号交给：

```bash
PYTHONPATH=src .venv/bin/python scripts/orchestrate_market.py --initial-run-id RUN_ID
PYTHONPATH=src .venv/bin/python -m trade_research.market_quality
PYTHONPATH=src .venv/bin/python -m trade_research.market_integrity
```

调度器会核对每个远程分片并导入本地。全市场 14:50 快照、T+1/2/3/5 存档和问题日分别在 `data/research/market_snapshots_ci/`、`market_outcomes_ci/`、`market_issues_ci/`；先检查完整性，再运行研究。

## 尾盘特征与输入门槛

```bash
PYTHONPATH=src .venv/bin/python -m trade_research.intraday_features --threads 4
PYTHONPATH=src .venv/bin/python -m trade_research.late_variance --threads 4
PYTHONPATH=src .venv/bin/python -m trade_research.minute_autocovariance --threads 4
PYTHONPATH=src .venv/bin/python -m trade_research.minute_prefix_1449 --threads 4
```

14:49 前缀程序按证券分批读取原档，输出到 `data/research/minute_prefix_1449/` 并审计完整标签、唯一键及与 14:50 快照的覆盖；新研究须先用该前缀，见[执行边界](research-status.md#执行边界)。分钟自相关程序先从原档提取 14:20–14:50，再用快照和方差表审计严格同日配对；四个半年段均未过预设输入门槛，所以没有生成复价名单，也不应读取本项后续成交或收益。尾盘容量基准依次运行 `trade_research.pretrade_tail_capacity --build-last5 2024` 和 `--build-last5 2025`、`trade_research.pretrade_tail_capacity --window 5`、`trade_research.pretrade_tail_profile --build-history`、`trade_research.pretrade_tail_profile`；其历史 20% 分位版本也停在输入门槛，口径见[容量记录](pretrade-tail-capacity-plan.md)。

其他检验由对应模块的默认命令重建输入，只有审计文件中的 `outcome_gate_passed` 为真且 `repricing_signals.parquet` 已生成，才能进入下一步。已完成的专项规则及结果留在各专项文档；本页只维护公共流程。

## 原始分钟复价与验证

`trade_research.size_sensitivity` 按固定名单逐股读取原始分钟，在 14:52–14:55 估算入场，并按指定时段和持有期重算成交、税费、滑点及现金感知结果。以下是**已过输入门槛的历史名单**的复算示例，不应用于上节失败的分钟自相关名单：

```bash
PYTHONPATH=src .venv/bin/python -m trade_research.size_sensitivity \
  --signals data/research/late_to_open_reversal/repricing_signals.parquet \
  --output data/research/late_to_open_reversal/repriced.parquet \
  --notionals 20000 100000 --horizons 1 5 --exit-windows morning close
```

`trade_research.late_to_open_reversal_eval` 汇总该历史名单的同日配对；其他专项由各自评价模块核对存档与原始分钟复价。任何未成交、延期退出和质量异常都须单列覆盖，不能只报告成交样本收益。当前没有可发布的选股公式。
