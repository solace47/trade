# 当前环境与复算

当前为[外部四条件盘中适配入口](../scripts/run_tail_formula_shakeout_literal.py)。旧历史模型严格次晨补齐已完成，结果不重做。Python 3.12虚拟环境及依赖见requirements.txt；完成的阶段禁止重复或覆盖结果。

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts .venv/bin/python -m pytest -q
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts .venv/bin/python scripts/run_tail_formula_shakeout_literal.py --help
```

匹配模型轮prefit、八次fit、freeze、analyze、finish已经完整完成，禁止重复拟合或覆盖结果。直接规则freeze、analyze、finish也全部结束，零拟合、两年度及六比较完整核准；不得重复或更改已固定条件追求正结果。输入／模型／评价协议均先提交，全部名单共同固定后才读经济分组；全量窗口及现金、模型节点与分数、四年度和四比较已独立核准。年度报告已含半年，不重复聚合。旧控制只读复用，完整名单、元数据、适用标签与统计全部相同才复用。

当前规则在data/research/tail_formula_shakeout_literal/；协议与两年名单已固定，待严格次晨一次评价。此前直接规则在tail_formula_cost_history_literal/，原来源与输入在tail_formula_cost_history/；32股试点在tail_formula_cost_history_probe/。旧完整补齐在tail_formula_history_boundary_recheck/；旧两个模型在tail_formula_morning_history_2024/及tail_formula_morning_history_recent/，完整原名单在tail_formula_morning_history_2025/。旧评价09:31—10:00不能混作09:59；2024缺对应旧模型，空表不代表低风险或好质量。

原始日线／分钟在data/baostock/和data/hf/，全部结果、原件和查重缓存在忽略的data/research/。2025暴露探索；不新增2026经济评价、卖出、多日持有或月度回购研究。

已结束方案从对应Git版本独立检出恢复，禁止重做：已删11个单次时间门及延续执行／测试文件1b9f085，稳健入口e13a9b5，平均实现24a883f，八个后过滤／诊断执行文件f6abbcb，同容量整体／分段1e5793f、情绪绝对bbde01d、情绪相对bfbe762、分钟开价ceca953、日内均幅46004ad、ETF总量6bfed42、尾段区间884391c。完整清理前版本4b3bdda5db6470df4d8d61298501852f60b985a2；字节归档映射与当前收据见[固定收据](selection-formula.md)。
