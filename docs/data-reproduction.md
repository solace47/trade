# 数据复算

Python 3.12；原始行情、锁定版本、审计表和研究输出均放在被 Git 忽略的 `data/`。旧实验的 2023 年及以前仅供指标预热；最新固定研究增加 2022–2023 年训练标签，策略评价仍用 2024 年以后。2025 年非盲测，2026 年尽量留给事前冻结验证。

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

用户授权恢复历史预热后的新版本依次运行 `trade_research.turnover_reference_long`、`trade_research.reference_gain_pairs`、`trade_research.reference_gain_eval`、`trade_research.reference_gain_accounting`（同样使用 `PYTHONPATH=src .venv/bin/python -m`）。四步分别重建 2019 年起初始化、冻结先匹配后排序的名单、重算原始分钟执行并保留未知核算、生成目录条款和固定成交成立条件下的经济情景与周块区间。新目录为 `turnover_reference_long/` 与 `reference_gain_pairs/`，不覆盖旧版；`catalog_scenario` 不是已核准完整收益，`known_return` 的未知行不会被情景覆盖。本版经济结果仍不稳定，详见[记录](input-gates.md#先匹配后排序的收益检验)。

在这些固定输入上运行 `PYTHONPATH=src .venv/bin/python -m trade_research.reference_gain_inputs`，重建初始化稳定且当日下跌的两组与严格同日配对，输出目录为 `data/research/reference_gain_inputs/`。四期输入门槛全部失败，不能连接到成交或收益程序；匹配器新增的历史换手条件是可选参数，原整数位置的 596 对已逐项验证不变。

分析师旧研报的固定来源核验运行 `PYTHONPATH=src .venv/bin/python -m trade_research.analyst_source_probe`。它按 `config/analyst_source_probe_reviews.json` 锁定五份接口记录、PDF SHA、预测财年和人工审阅值，验证表格及内部日期矛盾；本地原响应、原件、页面、独立 Poppler 提取及核验报告在 `data/research/analyst_revision_source/`。PDF 制作时间和文档编号不当作首次公开证明，当前核验未通过可交易事件来源要求。默认首页的当前研报元数据单独留在 `source_docs/live_page_isolated.html`，不接入策略输入；复算程序只读固定 2024 年原件。

东吴 2024–2025 年目录运行 `PYTHONPATH=src .venv/bin/python -m trade_research.analyst_catalog`；首次下载需 `--fetch`，已有原响应按请求参数和 SHA 核对，不自动覆盖。目录 SHA 为 `e88551396b05a53aab5c5499fc17d54543d397fd4295d4b3d9e98a8af0c68477`，32 份样本 SHA 为 `d3ecbeea521add585bdaddf56bf1a9b12aa59f10538533ae488b5f9ae2800001`。随后运行 `trade_research.analyst_catalog_sources` 缓存固定原件、`trade_research.analyst_dongwu_audit` 按 `config/analyst_dongwu_source_reviews.json` 核准，均不读取行情文件。原响应、年度数量交叉检查、详情、PDF、页面和独立文本核验在 `analyst_revision_source/dongwu_catalog/`；实际公开时间与旧预测来源仍有单独的未解决标记。

同财年前件依次运行 `trade_research.analyst_predecessors`、`trade_research.analyst_predecessor_values`，仅首次获取缺少的详情或原件时加 `--fetch`。前者从 255 份详情的时间/身份投影冻结 22 个唯一前件，后者按冻结名单核对原件；10 个无前件样本保留。`predecessors/strict_code_v1/` 保存代码映射核准前的初版，`source_audit_before_printed_code_check.json` 保存最初未单列印刷代码核验的样本报告。新旧代码关系仅在两个固定文档中按 `config/analyst_source_identity_alias.json` 使用，交易证券代码和原始目录均不改；官方对照表快照在上级 `source_docs/`。全部输出仍带公开时点未核准标记，不连接收益程序。

依次执行 `trade_research.cash_dividend_catalog freeze`、`fetch`、`assemble`，再执行 `trade_research.cash_dividend_notices collect`、`reconcile` 与 `trade_research.cash_dividend_primary`。均使用 `PYTHONPATH=src .venv/bin/python -m` 前缀。前者冻结 2024–2025 历史主板证券年度键，逐项缓存完整或明确为空的响应；错误不冒充空结果。公告索引只取 2024–2025 披露记录，分页缺口会停止或拆分重取；发行人/日期的补充检索缓存在 `cash_dividend_catalog/notice_gaps/`。

[`cash_dividend_vendor_reviews.json`](../config/cash_dividend_vendor_reviews.json) 保存精确原始记录指纹及同日冲突核准；[`cash_dividend_primary_supplements.json`](../config/cash_dividend_primary_supplements.json) 保存遗漏、字段纠正、重复公告和跨期原件。原件 SHA、现金、日期与差异化参考金额均独立验证；补充结果写入 `events_augmented.parquet`，不覆盖原始接口或原目录。全目录仍不能一概视为原件条款全部核准。

随后执行 `trade_research.cash_ex_inputs build`、`potential`。只用 2024 年起的严格滞后日线指标和 14:49 前缀，输出基础池及候选数量上界。再依次执行 `trade_research.cash_ex_notice_sources freeze`、`fetch`，`trade_research.cash_ex_notices` 和 `trade_research.cash_ex_matching`；641 个分配状态原件按指纹缓存，保留原始逐页文本和去页码后的文本，字段歧义对照[核准表](../config/cash_ex_notice_reviews.json)。原件纠正重建的 371 个候选只配成 283 对，其中 2024 下半年 46 对不足门槛，旧版因此止于输入。用户随后授权调整这一数量条件，另行探索见下文，原失败记录保留。冻结规则、源差异及结论见[输入检验](input-gates.md#纯现金除息后的价格压力事件数据阶段)。

用户授权后的除息经济探索先运行 `trade_research.cash_ex_economics`，核准既有原件指纹、实际现金与参考价，冻结原 340 个候选及 283 个对照。然后分别调用 `reference_gain_eval.reprice(Path("data/research/cash_ex_economics") / str(n), expected_signal_sha="f49e816930f0a446edbad1b674576dcdbaac0309c49fa8fb2c1c9a2bcf1cd7b7", rule_commit="05998b5", notional=n)`，其中 `n` 为 20000、100000；使用 Python 入口时导入 `Path` 及对应模块。最后调用 `shallow_tree_eval.summarize(Path("data/research/cash_ex_economics"), model_names=("20000", "100000"), rule_commit="91a07ea", list_commit="05998b5")`。这里两个名称表示订单金额，不是机器学习模型。两档均无剩余未退出持仓，无需续查；当前除息不计给新买家，持有期后续分配单独核算。完整结果与限制见[经济探索](input-gates.md#已核准除息事件保留原名单的经济探索)。

固定买入名单的每日条件退出依次运行 `trade_research.tail_barrier_exit freeze`、`reprice`、`summarize`，均使用前述模块命令前缀。第一步只固定已有买入和未来各持仓日的 14:49 顺序决策，第二步逐股复现原 T+5 并按固定触发日重算卖出，原第十日以内与续查分别保存。公共核算文件按每笔实际计划退出日分组只是诊断，完整策略比较以 `tail_barrier_exit/policy_report.json` 为准，跨全部计划持有期保留同一分母。第三步报告目录经济情景、同笔增量、持有时长和逐笔亏损尾部；未知收益不补零，2026 年不用。

撤销风险警示研究的状态、目录和 70 份原件保存在 `risk_removal_1449/`：`state_transitions.parquet` 仅由历史实际交易行的 `isST` 前后变化生成；`notice_rows_raw.json` 是 2024／2025 年“撤销”“退市整理”四个完整分页查询，按代码和前 28 个自然日公告连接状态。`formal_source_manifest.json` 锁定原件与 pdfplumber 文本指纹，旁置 pypdf 独立文本；例外只按 `config/risk_removal_source_reviews.json` 的精确原件使用。两种提取需对生效日期、普通简称及 10% 条款给出一致结果。

名单生成入口为 `PYTHONPATH=src .venv/bin/python -m trade_research.risk_removal_1449`，已存在成交后拒绝覆盖。固定名单提交 `e8fb34a`、SHA `ba95f6756abc9b4f1c99d8fc613e51db9fffe43130cd26525cf1d133eb35a924`；按同指纹调用 `reference_gain_eval.reprice`，再调用 `shallow_tree_continuation.continue_model` 保存 `continued/`，以及 `reference_gain_accounting.evaluate` 计算固定比例成本。最后运行 `trade_research.risk_removal_eval`，逐笔按原始均价加减 `max(价格×基点,0.005)`，重算佣金最低值、过户费、卖出印花税和分配。`tick_report.json` 是本项主终点，原 `known_return*` 与目录账本全部保留。独立输入与经济核算脚本和报告也保存在本地目录；它们复现状态、选择、贪心配对、实际持有年度目录覆盖及原始窗口，不把未知当零。

同行坚挺下个股回落的接续入口为 `PYTHONPATH=src .venv/bin/python -m trade_research.sector_resilience_1449`，保留旧 `late_sector_pressure/`；新输出在 `sector_resilience_1449/`。完整同行池及候选条件在 `peers.parquet`／`features.parquet`，两个基准都扣除自身；历史行业和已公开退市公告按时点连接。名单提交 `2b83c29`、SHA `8b1b74f224ae1a88a47af7ea509cfc862107420d1fbecec28bd3a2f93a2f58c2`。随后按相同公共复价、续查、目录核算流程，并将新 `continued/` 路径传给 `risk_removal_eval.evaluate`，复用逐股成本下限及完整分母计算。该函数虽以最初事件命名，公式不依赖摘帽身份；候选定义仍由各自输入程序决定。`tick_report.json` 保存本项主结果，20 条分钟来源疑问不改变原实际未知字段。

## 原始分钟复价与验证

季度更新对照依次运行 `trade_research.rolling_ridge_1449 prepare` 和 `freeze`。2024 年训练标签直接核对复用，新增 2025 年标签只读至 09-30；六个季度分别检查最晚可能退出及实际退出早于拟合时点。`rolling_ridge_1449/static/` 复制已经固定的静态账本，只有 `rolling/` 新名单调用公共原始分钟复价，名单提交为 `44dde7d`。之后调用 `shallow_tree_eval.summarize(path, model_names=("static","rolling"), rule_commit="681ad90", list_commit="44dde7d")`；超期追踪同样另存 `continued/`，静态续查账本复用原档。核心方法和失败结论见[季度对照](input-gates.md#固定下行情景评分的季度更新对照)。

风险池与下行情景评分先运行 `trade_research.downside_ridge_inputs`，再运行 `trade_research.downside_ridge_1449`，固定原评分／新评分名单。输入步骤只对 2024 年训练持仓读原始四分钟，逐笔保留经济情景与未知标签；已有窗口缓存可在源指纹和完整键核准后，通过 `downside_ridge_inputs.build(reuse_window_cache=True)` 复用。输出在 `downside_ridge_1449/`，两个模型目录为 `old_score` 和 `downside`。按各自名单指纹调用公共 `reference_gain_eval.reprice` 后，使用 `shallow_tree_eval.summarize(path, model_names=("old_score","downside"), rule_commit="123fd1d", list_commit="2e1e400")` 汇总；需要持仓续查时对每版调用 `shallow_tree_continuation.continue_model(source, output)`，另存于 `continued/`，复制固定输入报告后用同一汇总函数评价。均使用 `PYTHONPATH=src .venv/bin/python`；[结论与条件](input-gates.md#事前风险池与下行情景训练评分)不因缓存重算而改变。

可买条件与浅层树对照先运行 `PYTHONPATH=src .venv/bin/python -m trade_research.shallow_tree_1449` 固定名单；已有复价结果时程序拒绝覆盖名单。随后分别对 `data/research/shallow_tree_1449/ridge` 和 `tree` 调用 `reference_gain_eval.reprice(path, expected_signal_sha=该目录输入报告中的指纹, rule_commit="4f8bb7c")`，再运行 `trade_research.shallow_tree_eval` 汇总原五日延迟规则的经济情景。继续运行 `trade_research.shallow_tree_continuation`，将尚未卖出的原持仓追踪至 2025 年末，结果另存 `continued/`；对该目录调用 `shallow_tree_eval.summarize(path)` 生成续查报告。剩余未知终值保留为空，本金全损情景单列，不能当成实际收益。上述模块均使用同一 Python 环境与 `PYTHONPATH=src` 前缀，固定规则与结论见[输入检验](input-gates.md#决策时可买条件与浅层树模型对照)。

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

## 中证1000历史调整事件

原公告与附件元数据在 `data/research/index_rebalance/source_docs/`，四个公告标识、日期与解析核对见 `index_rebalance.py`。需保留每份 PDF、pdfplumber 文本和表格 JSON、独立 pypdf 文本 JSON；两种文本的指数／方向／代码／名称／页码及网格代码配对必须一致。实际下载地址与 SHA256 保留，备选表不当作调入。

```bash
PYTHONPATH=src .venv/bin/python -m trade_research.index_rebalance
PYTHONPATH=src .venv/bin/python - <<'PYCODE'
import json
from pathlib import Path
from trade_research.reference_gain_eval import reprice
from trade_research.reference_gain_accounting import evaluate
p=Path('data/research/index_rebalance')
r=json.loads((p/'input_report.json').read_text())
reprice(p, expected_signal_sha=r['signals_sha256'], rule_commit='cf4a922', notional=20000)
evaluate(p, bootstrap=False)
PYCODE
PYTHONPATH=src .venv/bin/python -m trade_research.index_rebalance_eval
```

已有 `repriced.parquet` 时输入模块拒绝覆盖名单；复核原结果可直接运行汇总模块，不删除冻结产物来重选。`batch_report.json` 同时保存四批主名单、全体合格事件诊断及批次等权结果，未匹配者仍进自身分母。只有四个独立日期，关闭共享核算器的周块区间；不读取 2026 分钟或把目录情景覆盖到已知收益字段。

## 同日相对训练目标

```bash
PYTHONPATH=src .venv/bin/python -m trade_research.relative_ridge_1449
```

该模块先复现 `downside_ridge_1449` 的旧参数与全部旧选择，再固定 `relative_ridge_1449/relative/signals.parquet`；已有新执行时拒绝重写。使用 `reference_gain_eval.reprice` 对新名单指纹复价。把原绝对模型的完整输出复制为新目录的 `absolute/`（不覆盖原件），调用 `shallow_tree_eval.summarize`，参数 `model_names=("absolute","relative")`、`rule_commit="6cc751b"`、`list_commit="897dd9a"`。

初始结果与 `continued/` 结果分开保存；绝对对照的续查直接复用 `downside_ridge_1449/continued/downside/`，新模型使用 `shallow_tree_continuation.continue_model`。本次新模型无超期未退仓；续查目录保留相同名单，以便同口径比较。`comparison_report.json` 包含自身、同日配对、共同日期模型差和年度区间，实际未知不由训练评分替代。

## 增加早期训练历史

依次执行 `trade_research.long_history_inputs features`、`catalog_parallel`、`catalog_assemble`，再执行 `trade_research.long_history_labels raw`、`assemble`，最后运行 `trade_research.long_history_ridge_1449`，均使用前述模块命令前缀。只读取 2022–2023 新训练数据及已固定的 2024–2025 输入；分钟前缀按源文件分批缓存，训练成交按证券记录输入／代码指纹，完整旧产物按记录复用。分配查询区分完整空结果与错误，组装标签时按实际十日观察窗口补齐跨年目录。原 2024 标签、公共特征及旧模型选择必须全部复现。

新模型目录 `long_history_ridge_1449/long/` 的名单 SHA256 为 `c047deb1800c88748753be395264d21dedf42338a9ac70cca2ddbd517425e75d`。按此指纹调用 `reference_gain_eval.reprice`，规则提交 `802cc1e`、2 万元；旧模型完整文件复制为 `recent/`。随后调用 `shallow_tree_eval.summarize(path, model_names=("recent","long"), rule_commit="802cc1e", list_commit="104f868")`。旧对照的续查账本来自 `downside_ridge_1449/continued/downside/`；新模型没有超期未退出持仓，`continue_model` 直接复用完成账本，仍在 `continued/` 保存同口径比较。历史训练及测试收益分别有独立 SQL、原始窗口和冻结名单检查，见各 `independent_*_checks.json`。

在上述输入完成后，`trade_research.long_history_tree_1449` 复现同历史线性模型并训练固定浅树；名单、全部分数、两次拟合模型及指纹存于 `long_history_tree_1449/`。按 `tree/signals.parquet` 的指纹调用公共 `reprice`，规则提交 `7ef4059`；复制长历史线性账本为 `linear/`，使用 `summarize(path, model_names=("linear","tree"), rule_commit="7ef4059", list_commit="90f1e03")`。新树的 3 条超期持仓由 `continue_model` 延长核算，线性续查账本直接复用；`continued/` 内再按相同参数汇总。不能用原延期上限处的缺失收益替代完整持有损失，也不覆盖原始版本。

## 月末跨月持有

先运行 `trade_research.month_turn_1449`，固定 23 个月末及各自提前五交易日的日期和独立选择。2024 年末可跨到 2025 年；2025 年末因需要留出年价格而排除。按 `month_turn_1449/signals.parquet` 指纹调用公共 `reprice`，规则提交 `ef2911a`、名单提交 `6bba4a5`、名义 2 万元。然后执行 `trade_research.month_turn_eval`；它关闭通用按周区间，另按月份与固定五槽位核算，使用连续两个月的循环块区间。

如需统一超期诊断，调用 `continue_model(root, root / "continued")` 并复制 `calendar_schedule.parquet`，再运行 `trade_research.month_turn_eval --output data/research/month_turn_1449/continued`。本次没有超期未退出，未读取额外持仓行情；初始结果保留。两期股票独立按当时输入选择，差值不是同股因果效果，`independent_economic_checks.json` 同时保留独立费用、原始均价、月份均值和抽样区间验证。
