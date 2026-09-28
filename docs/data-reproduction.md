# 数据复算

## 当前保守窗口与首季三版本

命令前缀为`PYTHONPATH=src .venv/bin/python`；模块加`-m`，脚本直接给路径。当前价格机会统一09:31—09:59，旧30根结果不得混用。所有已冻结产物拒绝覆盖，复核阶段可以重读已有产物。

1. 先固定`tail_formula_daily_efficiency_q1`的`control`／`path`两模型及`tail_formula_equal_weight_q1`模型，独立核验后分别`freeze_models`／`freeze_model`；再执行`tail_formula_q1_candidates models_gate`。模型只训练2025，首季此前已暴露，不称盲测。
2. `tail_formula_q1_candidate_inputs`依次`features`、`verify_features`、`native`；`scripts/freeze_tail_formula_q1_candidates.py`依次`scores`、`verify_scores`、`freeze`、`verify`。全部版本名单共同固定后才能继续。
3. `tail_formula_q1_candidate_observations`依次`prepare`、`reuse_evidence`、`raw`、`windows`，运行`scripts/verify_tail_formula_q1_observations.py windows`，再生成`labels30`并由同脚本`labels30`复核。此30根层只用于保持原来源与未知状态。
4. `tail_formula_q1_candidate_boundary build`后由`scripts/verify_tail_formula_q1_boundary.py observations`复核；再生成`labels`并由同脚本`labels`复核。`scripts/analyze_tail_formula_q1_candidates.py analyze --variant <control/path/equal_weight>`及`verify`检查三份完整报告，最后`comparisons`核对共同日期、共享未知和同质量控制复用。输出在`data/research/tail_formula_q1_candidates/`。

同日配对排序接续复用2024—2025的原48项和已核准29根标签。模块`tail_formula_pairwise model --fold <2024/recent>`后运行`scripts/verify_tail_formula_pairwise.py --fold <同段>`；随后模块的`scores`、`verify_scores`、`freeze`、`verify`。两段完成后`--fold combined`执行`freeze`与`verify`，才能对两段及全年`analyze`，每份结果用`scripts/verify_tail_formula_before1000.py analysis --root <对应目录>`独立核准。这一接续不读取新2026价格。

先后目标先运行`trade_research.tail_formula_path_order_labels`的`target`／`verify`；`trade_research.tail_formula_path_order`按两折依次`model`、`verify_model`、`scores`、`verify_scores`、`freeze`、`verify`，再共同冻结全年后`analyze`三份。评价必须用原29根标签，训练目标文件不可用于效果汇总。配对与先后两项的共同日期及共享未知均由`compare_tail_formula_same_dates.py`和`compare_tail_formula_shared_unknowns.py`复算，对应协议带各自前缀。

联合输入使用`trade_research.tail_formula_stock_library`的`features`、`verify_features`、`control_inputs`，只复用核准表；控制和联合表字节相同，预测列分别48／82。`trade_research.tail_formula_stock_library_model --arm <joint/control> --fold <2024/recent>`依次`model`、`verify_model`、`scores`、`verify_scores`、`freeze`、`verify`；两臂各用`--fold combined`冻结并核准全年，六份选择齐备才允许`analyze`。逐份分析用上述保守窗口验证器核准，不重做旧原始窗口抽取。所有命令在仓库根目录执行。

60日价格位置的输入模块`trade_research.tail_formula_quarter_position`依次运行`features`、`verify_features`、`native`。原始停牌占位诊断与证据保留；原生重放明确遵守另行固定的停牌占位协议，不能拿通过结果冒充客户端数据核准。模型模块`trade_research.tail_formula_quarter_position_model`按`--fold <2024/recent>`依次`model`、`verify_model`、`scores`、`verify_scores`、`freeze`、`verify`，随后用`--fold combined`冻结并核准全年。三份完整选择共同固定后，分别`analyze`并运行上述29根验证器；原有效交集未变，48项控制直接复用。

单指标分段加分直接用`trade_research.tail_formula_stumps`，两折及全年阶段顺序与60日模型相同，但使用原48项，不重复生成输入。`verify_model`专门核准256棵单分裂树，不能使用原64棵深度3模型验证器代替；全量评分仍由独立SQL重建。模型、阈值和名单不得据2025评价覆盖。

整段净价格面积先用`trade_research.tail_formula_path_area_labels target`与`verify`，再运行`trade_research.tail_formula_path_area`的两折／全年模型与选择阶段。该训练表保留原二元机会，另增`path_area15`；拟合使用`path_area`目标，评价仍连接原29根全池标签。共用模型代码新增分支后，原两折正机会模型证明字节保持；不重拟合原控制。

相邻分钟顺序用`trade_research.tail_formula_serial_price`的`features`、`verify_features`、`native`，只复用既有30个收盘宽表。模型模块`trade_research.tail_formula_serial_price_model`仍按两折依次拟合、核准、评分、核准、冻结、核准，再共同冻结全年后评价；相同有效范围的48项控制直接复用。

截面离散度用`trade_research.tail_formula_cross_dispersion`的`features`、`verify_features`、`native`；两个模型用`trade_research.tail_formula_cross_dispersion_model`，同样完成两折与全年冻结再评价。主要复用已核准等权50项控制，另复用两份48项控制；辅助指标为`YJCS20.tdx`，与每份数值核心配套。

历史累计位置用`trade_research.tail_formula_history_cdf`的`features`、`verify_features`、`native`，以及`trade_research.tail_formula_history_cdf_model`相同两折与全年流程。原20日价量宽表复用，新增原档仅64个固定尾盘价点；不重做已失败历史均价版。

此前5／20日波幅比例用`trade_research.tail_formula_range_change`的`features`、`verify_features`、`native`，模型模块为`trade_research.tail_formula_range_change_model`，两折及全年先共同固定再评价。验证器保留空历史SUM为空、COUNT为0，原输入文件保持不变。

末端目标使用`trade_research.tail_formula_endpoint_robust`。旧主协议在任何拟合前因参考缺失失败，执行以`tail_formula_endpoint_robust_v2_protocol.json`为准，旧错误收据保留。先按两折运行`verify_inputs --arm squared`，独立重建连续及二元同样本目标；随后对`squared/huber/binary`三臂、`2024/recent`两折分别依次`model`、`verify_model`、`scores`、`verify_scores`、`freeze`、`verify`。三臂各自`--fold combined`冻结并核准全年，九份选择齐备后才能评价任一臂。评估仍用原29根完整标签，训练参考有限约束不得进入推断或评价过滤。固定scikit-learn1.7.2源码与本地安装一致，专用模型复核器另外核准Huber每轮分位与叶更新；不是普通残差均值验证器的别名。

日期衰减权重使用`trade_research.tail_formula_recency_weight`，原48项和标签不重建。两折及全年同样先模型、验证、评分、验证、冻结、验证，再共同评价；63日半衰期在协议内固定。日期权重只用于训练，评分门槛仍原逐股训练分数q995，评估仍日期等权。上述每份分析继续用`scripts/verify_tail_formula_before1000.py analysis --root <目录>`核准；共同日期／共享未知工具不变，不新增2026价格。

绝对参考目标入口为`trade_research.tail_formula_endpoint_absolute`，共用已核准损失引擎，按`squared/huber`两臂完成两折及全年六份选择后评价。两折先各执行一次`verify_inputs`，不得使用原中心化目标的训练证明替代。固定末端正事件入口为`trade_research.tail_formula_endpoint_positive`，默认`binary`臂，两折和全年共三份选择；训练目标严格`mark_0959_return15>0`，其余阶段顺序不变。这两项分别用独立主协议和产物目录，不互相混接模型或参考样本。

`scripts/verify_tail_formula_endpoint_reference.py`复核第一轮三臂评价的参考覆盖诊断：已知二元机会可以没有09:59活跃参考价，相关候选保留在机会分母；参考均值只汇总有效参考标记，不能将这种字段缺失当已知负事件或零收益。该脚本不重选名单、不改变评价标签。

自然零门槛入口为`trade_research.tail_formula_absolute_zero`，`freeze`／`verify`一次处理两臂六份名单；原绝对模型不重训，只改选择为score>0。共同冻结后分别对`squared/huber`及`2024/recent/combined`运行`analyze`，严格29根分析核查器仍必需。无候选q995结果保持独立，不覆盖。

`scripts/audit_tail_formula_reference_coverage.py --root <年度目录>`从原选择与标签分别用Pandas／SQL检查两成本、两质量口径的参考缺失、末端严格正事件和日均参考价。它写独立证明，不修改既有分析；参考不全的已知机会仍保留，全未知日期存null。两轮新同日比较分别由`tail_formula_absolute_zero_shared_unknowns_protocol.json`及`tail_formula_endpoint_positive_shared_unknowns_protocol.json`固定，沿用共同日期与共享未知工具。

双评分交集用`trade_research.tail_formula_joint_reference freeze`和`verify`，一次固定两臂各五份选择（两半年、全年、被过滤、被整日过滤）。十份共同核准提交后，逐个`--arm squared/huber --fold 2024/recent/2025/removed/removed_dates analyze`并运行严格29根分析核查器；不生成被过滤组的反向原生公式。

逐轮整日抽样用`trade_research.tail_formula_date_subsample`。两折先各`verify_inputs`，独立SQL重建原训练交集、标签与64轮SHA日程；随后按`model`、`verify_model`、`scores`、`verify_scores`、`freeze`、`verify`，再共同固定全年后评价。原训练241日每轮固定取120日，评分门槛仍完整训练q995；不调用旧独立树平均验证器。

周内日期追加用`trade_research.tail_formula_weekday features`、`verify_features`，后者同时独立核准所有日期的原生数学表达。模型入口`trade_research.tail_formula_weekday_model`复用原特征扩展引擎，53项，两折及全年同顺序先核准、共同冻结再评价。原价格历史与有效交集完全不变；`DATE`表示信号周期日期，不用运行当天系统日期回填历史。

尾段最长连续段用`trade_research.tail_formula_max_run`依次`features`、`verify_features`、`native`；运行`python -m unittest discover -s tests -p test_tail_formula_max_run.py`核验窗口与零成交边界。模型入口`trade_research.tail_formula_max_run_model`复用相同先模型、核准、评分、选择、两折及全年共同冻结后评价的顺序，50项；通用严格29根分析核对与共同日期／共享未知、参考覆盖审计保持。

尾段有序最大幅度用`trade_research.tail_formula_path_excursion`依次`features`、`verify_features`、`native`；`test_tail_formula_path_excursion.py`核对先后顺序及窗口边界。模型入口`trade_research.tail_formula_path_excursion_model`沿用相同50项两折流程。相同完整名单可用`scripts/reuse_tail_formula_selected_analysis.py`生成复用收据，避免重算既有分析；新名单仍完整核准。

按时间选择轮数用`trade_research.tail_formula_chrono_rounds`。每折依次`verify_inputs`、`calibration_model`、`verify_calibration_model`、`calibrate`、`verify_calibration`，先核准0至256阶段全部日期误差和选择，再`model`、`verify_model`进行完整一年重训；后续`scores`、`verify_scores`、`freeze`、`verify`及两折／全年共同冻结、严格29根评价流程保持。0轮有单独的常数原生核心与SQL评分核准；`test_tail_formula_chrono_rounds.py`覆盖零轮、舍入平手和日期权重。

可执行机会入口`trade_research.tail_formula_executable_opportunity`：两折先`verify_inputs`核准原执行拒绝状态、未知保留及新训练事件，再依次`model`、`verify_model`、`scores`、`verify_scores`、`freeze`、`verify`；两折与全年共同冻结后才评价。使用专用模型／评分复核器，训练分位必须包括原明确未买入的0事件；评价仍用原保守29根标签，不能用训练事件替换。`test_tail_formula_executable_opportunity.py`覆盖事件转换边界。

## 原48项的2026首季接续

尾段价格／成交额输入先运行`trade_research.tail_formula_price_impact_amount`，只补2024—2025的29根金额；已有金额收据及分批文件拒绝覆盖，价格／量复用原核准缓存。再依次运行`trade_research.tail_formula_price_impact`的`features`、`verify_features`、`native`，原48项与有效交集必须核对。模型模块`trade_research.tail_formula_price_impact_model`按两折依次执行`model`、`verify_model`、`scores`、`verify_scores`、`freeze`、`verify`；两折和`combined`的完整名单共同固定，才进行三份`analyze`及保守29根独立核算。两项边界测试位于`test_tail_formula_price_impact.py`，完整客户端行情一致性不在数学核准范围内。

协议为 `config/tail_formula_forward_2026q1_protocol.json`，输出独立保存于 `data/research/tail_formula_forward_2026q1/`。依次运行 `trade_research.tail_formula_forward` 的 `model`、`verify_model`、`freeze_model`，再运行 `scripts/verify_tail_formula_replay48_legacy.py`。所有模块使用 `PYTHONPATH=src .venv/bin/python -m`，脚本使用相同前缀直接运行；既有冻结报告拒绝覆盖。

模型固定后运行 `trade_research.tail_formula_forward_inputs prepare`，`scripts/collect_tail_formula_forward_indices.py daily` 和 `minutes`，然后输入模块的 `features`、`scripts/verify_tail_formula_forward_inputs.py`、`scripts/freeze_tail_formula_forward_selections.py`。原完整名单与同分保留的前五位短名单共同冻结，在读取股票买入／次晨结果前提交 `f1f95d6`。

接着运行 `trade_research.tail_formula_forward_observations prepare` 和 `catalog`；供应商同日多条分配四股保留原错误与原始记录，`catalog_dates` 仅核准完整日期覆盖，不声称现金条款核准。运行 `scripts/verify_tail_formula_forward_catalog.py`。原始买入和次晨窗口由 `trade_research.tail_formula_forward_raw` 的 `raw`、`windows`、`quality` 生成，之后 `scripts/verify_tail_formula_forward_windows.py` 独立重建。质量审计限制首季，4月只新增3月31日必要的09:31—10:00原始价格。

最后运行 `trade_research.tail_formula_forward_labels`、`scripts/verify_tail_formula_forward_labels.py`，再分别运行 `scripts/analyze_tail_formula_forward.py all` 和 `shortlist`，同时生成及核准每月和首季统计。`scripts/compare_tail_formula_same_dates.py --left data/research/tail_formula_forward_2026q1/shortlist --right data/research/tail_formula_forward_2026q1/all --output data/research/tail_formula_forward_2026q1/same_dates_comparison.json --periods 2026-01 2026-02 2026-03 2026Q1` 比较同日两份名单。未知独占日期不删除，不把价格机会或10点参考估值当作执行卖出的收益。

Python 3.12；原始行情、锁定版本、审计表和研究输出均放在被 Git 忽略的 `data/`。各历史实验按对应协议限定训练年代；当前次晨公式用2024全年或2024H2—2025H1训练，更早行情只初始化指标。策略评价从2024年开始，2025年非盲测，2026年尽量留给事前冻结验证。

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

以下模块命令均在仓库根目录运行，前缀为 `PYTHONPATH=src .venv/bin/python -m`；各模块的 `--help` 列出输入、输出和可选参数。研究结果及门槛见[策略结果](strategy-results.md)和[输入停止记录](input-gates.md)，数据验收见[研究状态](research-status.md)。历史上已结束的逐项命令仍可在 Git 历史中查阅，不继续累加到本页。

## 原生公式与次日十点前机会

本节所有命令使用`PYTHONPATH=src .venv/bin/python`，模块前加`-m`。原始数据复用已有SHA256清单，不重复下载；已有冻结结果禁止覆盖。当前范围包含沪深主板，按用户补充排除创业板；旧含创业板输入仅保留在`tail_formula_1000_superseded_chinext/`。

1. 模块`trade_research.tail_formula_1000 features`，脚本`scripts/verify_tail_formula_1000_inputs.py`，再运行模块的`observation_keys`、`observations`及脚本`scripts/verify_tail_formula_1000_observations.py`。
2. 模块`trade_research.tail_formula_1000_analysis training`，脚本`scripts/verify_tail_formula_1000_analysis.py training`；随后模块`freeze`与脚本`selection`，仅训练2024H1已完成标签并固定所有时期名单。
3. 同一分析模块的`full`与核查脚本`full`，再运行`analyze`与核查脚本`analysis`。产物在`data/research/tail_formula_1000/`。
4. 固定的选叶修正使用模块`trade_research.tail_formula_1000_daily`依次`freeze`、`verify`、`analyze`，随后核查脚本`analysis --root data/research/tail_formula_1000_daily`。复用已核准标签，明确为探索性接续。
5. 同日机会差模型使用模块`trade_research.tail_formula_1000_residual`依次`model`、`verify_model`、`freeze`、`verify`、`analyze`，随后核查脚本`analysis --root data/research/tail_formula_1000_residual`。只增加一次固定回归树，不扫描参数。
6. 午后路径先运行模块`trade_research.tail_formula_intraday features`及脚本`scripts/verify_tail_formula_intraday_inputs.py`；再运行模块`trade_research.tail_formula_intraday_study`依次`model`、`verify_model`、`freeze`、`verify`、`analyze`，最后核查脚本`analysis --root data/research/tail_formula_intraday`。只读13:01–14:49，原始20项输入与固定原档抽样独立重建；原始提取模块哈希也记录在分片收据中。
7. 少量候选研究使用模块`trade_research.tail_formula_selective`依次`model`、`verify_model`、`calibrate`，再运行`scripts/verify_tail_formula_selective.py`。固定2024H1建树、H2校准及全部质量门槛；只有合格条件才可运行`analyze`并按第3步核准新目录的2025结果，无合格条件禁止补选或展开该分组的2025评价。
8. 历史处境联合研究使用模块`trade_research.tail_formula_joint`依次`features`、`verify_features`、`model`、`verify_model`、`calibrate`；核查脚本为`scripts/verify_tail_formula_selective.py --root data/research/tail_formula_joint --features-root data/research/tail_formula_joint --protocol config/tail_formula_joint_protocol.json`。仅新增4个此前价格位置特征，其余设计沿用第7步；软件分钟历史与日线字段的对齐尚未确认，不冒充可直接上线。

当前48项输入由`trade_research.tail_formula_float features`生成，`scripts/verify_tail_formula_float_inputs.py`独立复建；前置45项、原始指数来源和以前输入的核验不可跳过。后续二元机会、软盈利空间、同日百分位三个家族的模块分别为`tail_formula_float`、`tail_formula_margin`、`tail_formula_rank`。前者语法是`<stage> --fold <fold>`，后两者是`<fold> <stage>`。每个家族先对`2024`和`recent`依次运行`model`、`verify_model`、`scores`；分数用`scripts/verify_tail_formula_additive.py scores --root data/research/<本段目录> --features-root data/research/tail_formula_float --protocol config/<本段协议>.json`重建，再运行各段`freeze`、`verify`。两段均核准后，对`combined`运行`freeze`、`verify`，先保存全年名单，才允许三组`analyze`。每份分析以`scripts/verify_tail_formula_1000_analysis.py analysis --root data/research/<目录>`核准；已冻结文件不重拟合或替换。

开盘联合版先执行`trade_research.tail_formula_opening features`与`scripts/verify_tail_formula_opening_inputs.py`；原始窗口复用已核准的`opening_cash_history/window_parts`，不复用其策略标签。该模块采用`<stage> --fold <fold>`，两段模型／分数／名单和全年衔接的核验顺序同上，但`--features-root`改为`data/research/tail_formula_opening`。另外执行`freeze_control`、`verify_control`固定旧48项在同一有效交集的对照；两段和全年名单冻结前不运行`analyze_control`。对照分析也须运行通用`analysis --root data/research/tail_formula_opening_control`核验。

兼容文本用`scripts/export_tail_formula_native_compatible.py --root data/research/tail_formula_float_recent --features-root data/research/tail_formula_float`导出，不覆盖原冻结核心；全部IF路径和数值叶子另行重放，H2选择标记保持。该版本数值核心已在通达信macOS 3.61编译，完整条件选股、硬过滤及客户端逐点数据一致性仍未核准。各版完整条件与结果见[原生公式研究](selection-formula.md)；研究中的浮盈机会不代表按观察价成交或保证收益。

## 昨日炸板与历史题材支持

按`config/prior_theme_broken_protocol.json`及`prior_theme_date_proof.json`、`prior_theme_amount_semantics.json`，先执行模块`trade_research.prior_theme_broken freeze`、`fetch`，必要时按固定协议执行一次`retry-transports`；再执行`trade_research.prior_theme_broken_inputs history`、`inputs`。下载收据、身份白名单、金额日期锚点及不含收益的前日状态保存在`data/research/prior_theme_broken/`；已存在的输入拒绝覆盖，下载可复用核准缓存。请求间隔至少3秒，鉴权或限流错误停止该来源请求。全部金额均等于日终的旧假设已撤下，原输入保留在superseded_amount_semantics。

使用`PYTHONPATH=src .venv/bin/python scripts/verify_prior_theme_inputs.py`核准所有输入后，才执行模块`trade_research.prior_theme_broken_analysis`，随后用同一Python前缀执行`scripts/verify_prior_theme_analysis.py`。经济标签直接复用已有严格T1及分期质量版本，不重新挑样本或改成交规则。全部昨日状态×题材强弱、来源缺失及两档费用完整保留；供应商历史归档不能替代首次发布时间证据。

同一题材名单的早盘接续另存于`data/research/prior_theme_morning/`：模块`trade_research.prior_theme_morning`依次执行`freeze`、`raw`，脚本`scripts/verify_tick_morning_exit.py windows --root data/research/prior_theme_morning`核准窗口，再执行模块的`labels`与同脚本`labels --root data/research/prior_theme_morning`核准会计。最后模块执行`analyze`、`pairs`，脚本`scripts/verify_prior_theme_morning_analysis.py`独立核对所有分组及同买单配对。共用函数只参数化路径，旧分笔样本结果不替换。

## 历史聚合分笔方向

固定来源样本由`scripts/probe_historical_ticks.py`及其`--sessions`依次取得，`scripts/verify_historical_tick_source.py`独立核准。所有命令使用`PYTHONPATH=src .venv/bin/python`；历史公开接口的当前可达性不保证未来相同。原始协议字节、来源源码以及官方定义链接留存在`data/research/tick_source_probe/`；官方原件的本地下载状态见`official/manifest.json`。

完整研究依次执行模块`trade_research.tick_flow_winner --cohort`、脚本`scripts/collect_tick_flow_sample.py`、模块`trade_research.tick_flow_winner`、脚本`scripts/verify_tick_flow_inputs.py`，再执行模块`trade_research.tick_flow_winner_analysis`与脚本`scripts/verify_tick_flow_analysis.py`。模块命令加`-m`；已冻结输入和结果禁止覆盖，下载仅在输入冻结前可续传。数据保存在`data/research/tick_flow_winner/`，每个股票日保留原始请求／响应、解析序号、来源失败和全日质量标记。

只用至14:48的方向数据，不能把完整日核对状态回填当时选择；原经济标签和新增质量敏感性并列，不把来源缺失、排队未知或未买入记成零收益。

## 流通规模与尾段换手代理

`float_turnover_winner`生成严格前日分母、三个可见代理及原值／末位精度两情景分组；`scripts/verify_float_turnover_winner_inputs.py`独立复现所有来源连接、排名及原始分钟。输入核准后运行`float_turnover_winner_analysis`与`scripts/verify_float_turnover_winner_analysis.py`，输出所有单变量和联合组。完整数据保存于`data/research/float_turnover_winner/`，协议为`config/float_turnover_winner_protocol.json`；官方前端与文档原件在`source/`，不调用当前全日换手率回填历史盘中输入。

## 实际尾盘经济赢家与组合预测

原经济画像按`economic_winner`的`freeze`、`raw`准备输入，再用`scripts/fetch_economic_winner_catalog.py`补齐目录；该续传器保留同次分配金额冲突，日期并集只作权益未知标记。依次运行`scripts/verify_economic_winner_labels.py`的`catalog`与`windows`，`economic_winner_labels`，再核准`labels`。所有以下脚本都用`PYTHONPATH=src .venv/bin/python`；模块用同前缀加`-m trade_research.`。完整产物已存在时禁止覆盖。

- `economic_winner_analysis`与`scripts/verify_economic_winner_analysis.py`：原29项完整画像。
- `economic_winner_quality`与`scripts/verify_economic_winner_quality.py`：保留原版，在`data/research/economic_winner/period_quality`叠加既有全日质量疑问。
- 两份画像／核查命令加`--root data/research/economic_winner/period_quality`：完整敏感性复算。
- `economic_winner_prediction`与`scripts/verify_economic_winner_prediction.py`：复用原39列已核准特征，按固定时期拟合、校准与选股，再独立核准。只有核准后才能连接2025结果。 随后运行`economic_winner_prediction_analysis`与`scripts/verify_economic_winner_prediction_analysis.py`核准全部预测分组、空仓结果和候选。

## 次日训练目标与同风险五日目标对照

固定协议为`config/short_horizon_target_protocol.json`。直接复用既有18列输入和T+5训练原始事实，仅重算T+1训练成交；逐股进度可续传，完成的标签与名单不可覆盖。两种目标最后都按T+1尾盘评价。

```sh
PYTHONPATH=src .venv/bin/python -m trade_research.short_horizon_labels raw
PYTHONPATH=src .venv/bin/python -m trade_research.short_horizon_labels assemble
PYTHONPATH=src .venv/bin/python scripts/verify_short_horizon_labels.py
PYTHONPATH=src .venv/bin/python -m trade_research.short_horizon_target
PYTHONPATH=src .venv/bin/python scripts/verify_short_horizon_inputs.py
PYTHONPATH=src .venv/bin/python -m trade_research.short_horizon_target_eval execute
PYTHONPATH=src .venv/bin/python -m trade_research.short_horizon_target_eval compare
PYTHONPATH=src .venv/bin/python scripts/verify_lhb_institutional_results.py --root data/research/short_horizon_target/t1_target
PYTHONPATH=src .venv/bin/python scripts/verify_lhb_institutional_results.py --root data/research/short_horizon_target/t5_target
PYTHONPATH=src .venv/bin/python scripts/verify_short_horizon_comparison.py
```

最后三步独立复核同样的主板尾盘费用、价格范围、排队标记和统计；不足两周的周块区间为未知，训练惩罚不替代测试未知收益。

## 机构席位净买入的短线接续

复用上交所历史原档，单日多原因仅金额一致时合并；先核准14:49输入与完整名单，再以同一买单比较T+1／T+3尾盘退出。结果及未知边界见[输入记录](input-gates.md#龙虎榜机构席位净买入的短线接续)。原始缓存和源哈希必须一致，已冻结名单不可覆盖。

```sh
PYTHONPATH=src .venv/bin/python -m trade_research.lhb_institutional_short
PYTHONPATH=src .venv/bin/python scripts/verify_lhb_institutional_inputs.py
PYTHONPATH=src .venv/bin/python -m trade_research.lhb_institutional_short_eval execute
PYTHONPATH=src .venv/bin/python -m trade_research.lhb_institutional_short_eval compare
PYTHONPATH=src .venv/bin/python scripts/verify_lhb_institutional_results.py
```

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

旧检验由对应模块重建输入，按该版本 `outcome_gate_passed` 和 `repricing_signals.parquet` 决定是否进入其收益步骤。失败记录保留；研究者可以在读取新结果前登记不同的样本设计，不把旧数量门槛当作当前用户约束。已完成的专项规则及结果留在各专项文档。

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

## 首板后尾盘承接与次日兑现

先运行`trade_research.first_limitup_overnight`，再以`PYTHONPATH=src .venv/bin/python scripts/verify_first_limitup_inputs.py`独立核准日历前态、整数分涨停、排序、冷却、全部匹配边和分配目录覆盖。初始规则提交`b33259b`，核准名单、两时段执行及排队未知补充提交`7e53b8f`；输出`first_limitup_overnight/`。已有输入报告会阻止覆盖名单。

核准后依次运行`trade_research.first_limitup_overnight_eval execute`、同模块`compare`，再运行`scripts/verify_first_limitup_results.py`。两种退出的原始分钟、延迟持仓、公司行动、费用及排队诊断分别在`continued/morning/`和`continued/close/`；同一自然日的上午与尾盘价格、成交量不可合并使用。触及不利涨跌停的窗口只标记排队未知，不根据日后是否成交重选股票，条件账本与原实际未知保留。比较报告输出收益、胜率、盈亏比、尾部、强制延期和同股时段差；这是已暴露年份的失败探索，不是可发布公式。

## Alpha158的2026时间留出检验

协议与权重为`config/alpha158_pool_2026_protocol.json`、`config/alpha158_frozen_2026_model.json`，规则提交`47073f3`。依次运行`trade_research.alpha158_pool_2026 prepare`和`features`，然后`scripts/verify_alpha158_validation_features.py`，再运行同模块`freeze`；全程没有拟合函数。执行`trade_research.alpha158_2026_catalog`补齐固定名单的2026分配目录，之后`scripts/verify_alpha158_validation_inputs.py`独立核准全部评分、排序、冷却、匹配与目录覆盖。

输出在`alpha158_pool_2026/`，两版`positive_pool`和`highest_score`，与2024–2025产物完全分开。名单核准并提交后运行`trade_research.alpha158_2026_eval execute`；它使用本次显式日期质量审计、原始分钟、到期未退续查、分配目录和两档费用，不借用2025的目录或审计范围。随后同模块`compare`输出全期及事前固定季度／短7月段、共同日期增量、50万元账簿和数值门槛。原实际未知保持；目录情景和资金收益不能自动变成可发布公式。8月6日后的行情不进入持有追踪，未结清权益阻止完整本金回报。

## 正分池固定分散选择

运行 `trade_research.positive_pool_1449`，先复现四份原最高分选择，再按固定正分与SHA256顺序生成四份新名单。规则提交`112e710`，名单双重核准提交`4db7029`；输出`positive_pool_1449/`。逐版按`input_report.json`中的名单指纹调用公共`reprice`、`continue_model`、目录会计与半分钱成本评价，原最高分账本只读复用。

然后运行 `trade_research.positive_pool_eval`，保存共同信号日差及周块区间，并以同一50万元政策运行四份新候选／对照资金账簿。`capital/`与原`fixed_capital_1449/`分开，原资金报告必须通过指纹核对；两个成本档、四个模型完整报告。独立选择、经济、价格范围、共同日期比较和现金流重建脚本及结果在本地输出目录，不用新结果重选哈希或覆盖原实验。

## 同一固定本金的资金政策

依次运行 `trade_research.fixed_capital_1449 prepare` 与 `evaluate`。规则提交 `37cb9c2`，输入／实现提交 `d906dc0`；四组源目录、既存名单／账本指纹以及完整订单合并容量写入 `fixed_capital_1449/manifest.json`。准备阶段以原始价格上限计算14:49预留，再核对原股数、实际费用、分配时点和T+5来源；评价阶段按50万元现金逐日决定整笔授权，不重训、不读新行情、不改变原退出。

`orders.parquet` 是全部原始候选及条件会计输入，`order_decisions.parquet` 保存每笔授权前可用现金与所需预留，`cash_ledger.parquet` 逐自然日记账。`report.json` 只有在期末库存、应收股息和税款全部结清时输出完整情景收益；不把现金余额当作带持仓估值的净值。独立输入及历史现金流重建脚本保存在相同本地输出目录。该模块替代不了旧失败版的研究结论，也不修改旧 `portfolio_curve` 的有条件完成样本诊断或事后最低资金需求档案。

## 触跌停后打开

触跌停后打开的完整候选入口为 `trade_research.limit_down_recovery_1449`，保留旧 `limit_down_reopen/`，新输出 `limit_down_recovery_1449/`。名单双重核准及提交 `7e7b10b` 后，按 `11abacaf66347ac5e868ede2fd25880b555e3eb855bf87bca38d18d91de33112` 调用公共 `reprice`、`continue_model` 和目录核算。随后调用 `risk_removal_eval.evaluate(root / "continued", primary_horizon=1)`；只有主终点标记改为T+1，费用、未知值保留和辅助T+5核算共用既有实现。目录、价格范围及完整观察窗覆盖的独立检查保存在本地输出中。

## 标准因子库与同缩放对照

Alpha158 定义固定于 `config/alpha158_definition.json`，许可位于 `licenses/qlib-MIT.txt`。先运行 `trade_research.alpha158_inputs`，把此前完整日线与当日 14:49 临时K线组合成指标；按证券缓存，源指纹改变时拒绝沿用。随后运行 `PYTHONPATH=src .venv/bin/python scripts/verify_alpha158_features.py` 核准独立逐窗计算，再运行 `trade_research.alpha158_models` 拟合，最后运行 `scripts/verify_alpha158_inputs.py` 核准落盘指标、全部评分和选择。各脚本均在仓库根目录使用相同 Python 前缀。已有成交后不重新生成名单。

输出在 `alpha158_1449/`，两版为 `robust18`／`alpha158`，名单提交 `4d867ee`。按根目录 `input_report.json` 中各自指纹调用公共 `reference_gain_eval.reprice`，再调用 `continue_model(source, root / "continued" / name)`；对每个续查目录依次调用 `reference_gain_accounting.evaluate` 与 `risk_removal_eval.evaluate`。最后运行 `trade_research.alpha158_eval`，从每侧至少半分钱的成本账本比较共同信号日，不以固定比例成本替代主终点。含未知股票的日期仍未知，不能通过分组均值跳过；原实际会计列保留。

## 月末跨月持有

先运行 `trade_research.month_turn_1449`，固定 23 个月末及各自提前五交易日的日期和独立选择。2024 年末可跨到 2025 年；2025 年末因需要留出年价格而排除。按 `month_turn_1449/signals.parquet` 指纹调用公共 `reprice`，规则提交 `ef2911a`、名单提交 `6bba4a5`、名义 2 万元。然后执行 `trade_research.month_turn_eval`；它关闭通用按周区间，另按月份与固定五槽位核算，使用连续两个月的循环块区间。

如需统一超期诊断，调用 `continue_model(root, root / "continued")` 并复制 `calendar_schedule.parquet`，再运行 `trade_research.month_turn_eval --output data/research/month_turn_1449/continued`。本次没有超期未退出，未读取额外持仓行情；初始结果保留。两期股票独立按当时输入选择，差值不是同股因果效果，`independent_economic_checks.json` 同时保留独立费用、原始均价、月份均值和抽样区间验证。


## 当日触板先后路径与双退出

`config/touch_sequence_protocol.json`先于收益冻结完整触板池、最近未触板对照、分钟分类和两个退出窗口。执行`PYTHONPATH=src .venv/bin/python -m trade_research.touch_sequence freeze`、`extract`，再运行`scripts/verify_touch_sequence_inputs.py`、模块的`inputs`和`prepare_exits`。原始前缀含09:30，14:20归入尾段；对照先配定再核对同一前缀质量，失败不递补。

对`data/research/touch_sequence/morning`依次调用`tick_morning_exit.raw(ROOT=..., PROTOCOL=Path('config/touch_sequence_protocol.json'))`、`scripts/verify_tick_morning_exit.py windows --root ...`、`tick_morning_exit.labels(ROOT=...)`和同一核对脚本的`labels --root ...`。原尾盘66列须先完整复现。然后运行`trade_research.touch_sequence_analysis`及`scripts/verify_touch_sequence_analysis.py`；结果后拆解用`scripts/diagnose_touch_sequence_loss.py`，不能作为选股条件。

`config/touch_sequence_pair_accounting_repair.json`说明初版未配者半年字段缺失导致的分组漏计。原输入和选择保留；当前分析从每条信号日期重建半年，1,846个未配者必须进入4,143个主组分母，原分析归档`invalid_pair_denominator/`。所有原件、输入、未知路径、费用情景、配对和大涨／大亏反向画像保存在该研究目录，本轮不读取新2026价格。


## 早盘历史同时间金额与尾盘保持

协议`config/opening_cash_history_protocol.json`与画像合同`config/opening_cash_history_analysis_contract.json`在新结果前固定。依次运行`trade_research.opening_cash_history freeze`、`extract`、`features`，再运行`scripts/verify_opening_cash_history.py features`，随后模块`pair`与核对脚本`pairs`。均使用`PYTHONPATH=src .venv/bin/python -m`运行模块或相同前缀直接运行脚本。历史只读2023-12-04至2025-12-30同一开盘窗口，按完整交易日历位移，不跳过无效窗口。

接着运行`trade_research.opening_cash_history_analysis strategy_inputs`、`scripts/verify_opening_cash_history_analysis.py strategy_inputs`和模块`exits`、`raw`。对`data/research/opening_cash_history/strategy/morning`运行`scripts/verify_tick_morning_exit.py windows --root ...`，再运行新模块`labels`及通用核对脚本`labels --root ...`。这只复制完整主组和原固定对照，不改变选择。

模块`portrait`及其独立核对脚本`portrait`报告完整124.6万输入的原始经济标签和输入质量敏感性；模块`analyze`及核对脚本`analyze`报告主组双退出与配对，并核对主组尾盘和全池画像完全相同。四个精简画像情景文件各保留全部输入，特征未知不等同利润未知；完整匹配分母21,112个，不省去12,123个未配者。原始数据与报告在`opening_cash_history/`，已有报告阻止覆盖。

## 已披露首次实际增持的短线接续

协议为`config/first_insider_short_protocol.json`。依次运行`PYTHONPATH=src .venv/bin/python -m trade_research.first_insider_short sources`、`inputs`和`scripts/verify_first_insider_short_inputs.py`，再运行同一模块的`exits`。只复用`first_insider_buy`既有109份原件，不下载月度回购PDF；原文重新提取，日历及最近对照独立SQL核对，全部未配者保留。

对`data/research/first_insider_short/morning`依次调用`tick_morning_exit.raw(ROOT=...,PROTOCOL=Path('config/first_insider_short_protocol.json'))`、通用核对脚本`windows --root ...`、`tick_morning_exit.labels(ROOT=...)`和`labels --root ...`。然后运行`first_insider_short analyze`及`scripts/verify_first_insider_short_analysis.py`。两种退出同时固定，空结果类别的比例存为未知，不补零。通用分析器扩展事件字段后，原触板研究全部报告字段及产物哈希回归一致。
