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

按研究年份复核整只股票源文件剔除名单：`PYTHONPATH=src .venv/bin/python -m trade_research.quality_period`。逐日问题仍由原审计排除；`scripts/tail_return_decomposition.py` 和近期两项毛空间程序支持 `--period-quality data/research/quality_period_2024_2025.json` 与单独输出目录，供分期清单敏感性复算，不覆盖历史主结果。

## 尾盘特征与输入门槛

```bash
PYTHONPATH=src .venv/bin/python -m trade_research.intraday_features --threads 4
PYTHONPATH=src .venv/bin/python -m trade_research.late_variance --threads 4
PYTHONPATH=src .venv/bin/python -m trade_research.minute_autocovariance --threads 4
PYTHONPATH=src .venv/bin/python -m trade_research.minute_prefix_1449 --threads 4
```

14:49 前缀程序按证券分批读取原档，输出到 `data/research/minute_prefix_1449/` 并审计完整标签、唯一键及与 14:50 快照的覆盖；新研究须先用该前缀，见[执行边界](research-status.md#执行边界)。分钟自相关程序先从原档提取 14:20–14:50，再用快照和方差表审计严格同日配对；四个半年段均未过预设输入门槛，所以没有生成复价名单，也不应读取本项后续成交或收益。尾盘容量的三种历史方案及停止原因已收敛到[输入门槛记录](input-gates.md)，复算程序分别为 `trade_research.pretrade_tail_capacity` 和 `trade_research.pretrade_tail_profile`。

分钟回跳**执行价**检验采用保守 14:49 截止。依次运行 `trade_research.minute_autocovariance --cutoff-label 1449 --output data/research/minute_autocovariance_1449`、`trade_research.minute_bounce_cost freeze` 和 `trade_research.minute_bounce_cost entry`。冻结阶段只处理全市场输入；执行价阶段只读归档的 14:52–14:55 买入状态与价格，不读卖出或持有收益。预设门槛及失败原因见[研究结果](strategy-results.md)与模块开头；2026 年不参与开发。

其他检验由对应模块的默认命令重建输入，只有审计文件中的 `outcome_gate_passed` 为真且 `repricing_signals.parquet` 已生成，才能进入下一步。已完成的专项规则及结果留在各专项文档；本页只维护公共流程。

## 固定交易的统计分辨率

旧执行价来源审计执行 `PYTHONPATH=src .venv/bin/python -m trade_research.fixed_execution_quality freeze`、同模块 `evaluate`。冻结现有 12,648 行会计记录及其 22,210 个已模拟买卖事件，只新增读取原买卖日期的四分钟 OHLC 并复核量额；原费用、收益和未知持仓字段不变。19 个原始窗口存在分钟价格矛盾，不能将旧账本的可复算性当作分钟源字段已获核准；具体边界见[研究状态](research-status.md#旧固定交易的执行价来源读取新增-ohlc-前冻结)。

再运行 `trade_research.fixed_return_ranges freeze`、`evaluate`，在两种预登记的条件价格情景下重新计算费用与已知经济收益范围。固定原日期、股数、公司行动和配对，不读新行情；包含未知终值的分组完整范围保持缺失。它只量化旧结论对执行价假设的敏感性，不产生新选股公式，也不改变原账本。

依次运行 `PYTHONPATH=src .venv/bin/python -m trade_research.statistical_resolution freeze`、同模块 `evaluate`。只使用已核算旧名单的 2 万元 T+1 结果，核对原同日配对后进行预定的逐日、周块和两周块抽样；存档每日差、180,000 次抽样误差和固定平移情景。它不读取新策略收益，不改变旧策略结论；适用边界及结果见[研究状态](research-status.md#旧固定交易的统计分辨率重抽样前冻结)。

## 整数价位输入检验

源字段诊断执行 `PYTHONPATH=src .venv/bin/python -m trade_research.minute_amount_consistency freeze`、同模块 `evaluate`。固定前次异常和哈希参考键，只读取这些日期的分钟与独立日线，并按已锁定下载校验值核对原文件；公开说明亦锁定至原归档提交。输出同日量额对账、逐分钟均价区间、固定偏移及四分钟总额边界，不计算持有收益、不更改源数据。诊断与限制见[研究状态](research-status.md#分钟量额与价格区间固定异常的来源诊断)。

已有入场价格范围分析执行 `trade_research.entry_price_ranges freeze`、`evaluate`。完全复用已读四分钟与原条件权重，对 284 个已标记窗口及全部窗口分别计算条件 OHLC 范围；上下界依权重符号求得，保留原日期分母。这里只分析入场价假设，不读取新行情或持有收益，不生成新策略名单。范围的附加假设及限制见[研究状态](research-status.md#已有入场诊断的价格范围计算边界前冻结)。

全样本条件关联诊断另行执行 `PYTHONPATH=src .venv/bin/python -m trade_research.round_number_geometry`。源为前版全部 `side_inputs.parquet`，不取其前五名或配对子集；输出所有日期的输入矩阵审计及带符号的条件对比权重，这些权重本身不是可交易组合。输入通过 391 个日期、114,657 条后，再运行 `trade_research.round_number_entry freeze`、`evaluate`，按固定名单提取四根原始入场分钟，保存原条、源文件信息、两档订单与逐分钟压力结果。入场质量门槛失败，因此没有持有收益入口；未知来源行和部分成交数量均保留，不记零或丢弃。

运行 `PYTHONPATH=src .venv/bin/python -m trade_research.round_number_1449`，复用只从 2024 年起计算的基础池及已有 14:49 原条审计分片。报价恢复为整数分后分类，冻结每日名单及同日对照；输出源指纹、分组覆盖和输入平衡。四期覆盖率及距整数距离平衡均未通过，因此没有成交或收益阶段入口。程序与固定规则见[输入检验](input-gates.md#整数价位两侧的尾盘价格位置读取输入分组前冻结)。

## 纯现金除息事件与输入目录

历史换手参考价的初始化诊断运行 `PYTHONPATH=src .venv/bin/python -m trade_research.turnover_reference`。它冻结基础池及日线文件 SHA，只投影各股 2024 年起、最后信号前一正常交易日以前的字段，输出严格滞后的参考状态及三种初始化敏感性，不调用成交或收益程序。数据位于本地忽略的 `data/research/turnover_reference/`；提供方换手定义、固定停牌原接口响应在 `source_docs/`，将停牌空量误作未知的首版在 `strict_volume_v1/`，不能覆盖该历史记录。输入结论见[初始化诊断](input-gates.md#历史换手参考价2024-起点的初始化诊断)。

在这些固定输入上运行 `PYTHONPATH=src .venv/bin/python -m trade_research.reference_gain_inputs`，重建初始化稳定且当日下跌的两组与严格同日配对，输出目录为 `data/research/reference_gain_inputs/`。四期输入门槛全部失败，不能连接到成交或收益程序；匹配器新增的历史换手条件是可选参数，原整数位置的 596 对已逐项验证不变。

依次执行 `trade_research.cash_dividend_catalog freeze`、`fetch`、`assemble`，再执行 `trade_research.cash_dividend_notices collect`、`reconcile` 与 `trade_research.cash_dividend_primary`。均使用 `PYTHONPATH=src .venv/bin/python -m` 前缀。前者冻结 2024–2025 历史主板证券年度键，逐项缓存完整或明确为空的响应；错误不冒充空结果。公告索引只取 2024–2025 披露记录，分页缺口会停止或拆分重取；发行人/日期的补充检索缓存在 `cash_dividend_catalog/notice_gaps/`。

[`cash_dividend_vendor_reviews.json`](../config/cash_dividend_vendor_reviews.json) 保存精确原始记录指纹及同日冲突核准；[`cash_dividend_primary_supplements.json`](../config/cash_dividend_primary_supplements.json) 保存遗漏、字段纠正、重复公告和跨期原件。原件 SHA、现金、日期与差异化参考金额均独立验证；补充结果写入 `events_augmented.parquet`，不覆盖原始接口或原目录。全目录仍不能一概视为原件条款全部核准。

随后执行 `trade_research.cash_ex_inputs build`、`potential`。只用 2024 年起的严格滞后日线指标和 14:49 前缀，输出基础池及候选数量上界。再依次执行 `trade_research.cash_ex_notice_sources freeze`、`fetch`，`trade_research.cash_ex_notices` 和 `trade_research.cash_ex_matching`；641 个分配状态原件按指纹缓存，保留原始逐页文本和去页码后的文本，字段歧义对照[核准表](../config/cash_ex_notice_reviews.json)。原件纠正重建的 371 个候选只配成 283 对，其中 2024 下半年 46 对不足门槛，因此不能接到分钟收益程序。冻结规则、源差异及结论见[输入检验](input-gates.md#纯现金除息后的价格压力事件数据阶段)。

## 原始分钟复价与验证

`trade_research.size_sensitivity` 按固定名单逐股读取原始分钟，在 14:52–14:55 估算入场，并按指定时段和持有期重算成交、税费、滑点及现金感知结果。以下是**已过输入门槛的历史名单**的复算示例，不应用于上节失败的分钟自相关名单：

```bash
PYTHONPATH=src .venv/bin/python -m trade_research.size_sensitivity \
  --signals data/research/late_to_open_reversal/repricing_signals.parquet \
  --output data/research/late_to_open_reversal/repriced.parquet \
  --notionals 20000 100000 --horizons 1 5 --exit-windows morning close
```

`trade_research.late_to_open_reversal_eval` 汇总该历史名单的同日配对；其他专项由各自评价模块核对存档与原始分钟复价。任何未成交、延期退出和质量异常都须单列覆盖，不能只报告成交样本收益。当前没有可发布的选股公式。

对最近一次 14:49 岭模型的固定名单做记零及资金审计，依次执行 `trade_research.fill_accounting freeze`、`evaluate`、`continue`。冻结会保存输入 SHA-256 并拒绝覆盖变化后的清单；后两步核对旧报告、计入合格延期的已知贡献，并从原始分钟追踪原来没有退出日期的买入至 2025 年末。公司行动或质量问题未核准的收益及回款保持为空，不输出整组完整收益，详见[会计审计](research-status.md#已买入记录的记零与资金审计读取分解前冻结)。

固定公司行动记录再依次执行 `trade_research.corporate_cash freeze`、`fetch`、`validate`、`evaluate`。`fetch` 只下载这 40 个分派事件的实施公告及独立分红字段，原件保存在本地忽略的 `data/research/corporate_cash/source/`；`validate` 对照[人工核准表](../config/corporate_cash_reviews.json)验证原件 SHA-256、日期、实际分红、送转股及除息参考价。核算只恢复符合冻结规则的纯现金记录，送转股及其他未知值仍留空，所有旧评分保留。

余下 9 条记录再执行 `trade_research.holding_exceptions freeze`、`evaluate`；它重读三个既有开盘单项差异日、按转增后的全部股数复验卖出，并逐日追踪固定未退出持仓。恢复 7 条经济核算后，海印股份的两档 T+5 仍无已核准终值；本金全损情景与实际结果分开，原股数、严格质量及旧评分字段不覆盖。

共用资金前的诊断依次执行 `trade_research.orderbook_funding freeze`、`quotes`、`evaluate`。原始四分钟切片按固定执行键重读并保存 SHA-256；金额、持有期、模型/对照各自成账，检查合并容量，再以同窗卖出款不预支的顺序统计实际支出与 14:49 候选预留的最低资金需求。逐日账簿和两种成本在本地忽略的 `data/research/orderbook_funding/`；这里不计算以事后最小本金优化的策略收益。

单根决策报价的精度审计依次执行 `trade_research.quote_precision freeze`、`audit`、`entries`。固定 6,324 个订单没有股数差异，`entries` 因而不读取原始成交或新收益；精确分位下单函数与旧浮点复算路径分开，禁止将非分位 VWAP 按此规则取整。

全市场执行价漂移的输入冻结、成交评估与固定样本原档核验依次运行 `trade_research.execution_drift freeze`、`evaluate`、`verify-raw`。看到主结果后的探索性基准敏感性另运行 `trade_research.minute_prefix_1449 --output-dir data/research/minute_prefix_1449_vwap`，再运行 `trade_research.execution_drift anchor-sensitivity` 与 `verify-raw`；结果和限制见[执行价检验](execution-drift.md)。

均价走势与末价偏离的收益分解依次运行 `trade_research.tail_vwap_decomposition freeze`、`evaluate`；固定样本再由 `trade_research.size_sensitivity` 以 2 万/10 万元、T+1/T+5 重算，最后运行 `trade_research.tail_vwap_decomposition verify-raw` 对账。该项没有通过发布门槛，细节合并在[同一研究记录](execution-drift.md)。
