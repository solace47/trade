# 当前环境与复算

当前为[五日含成本历史报价输入入口](../scripts/prepare_tail_formula_cost_history.py)。旧历史模型严格次晨补齐已完成，结果不重做。Python 3.12虚拟环境及依赖见requirements.txt；完成的阶段禁止重复或覆盖结果。

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts .venv/bin/python -m pytest -q
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts .venv/bin/python scripts/prepare_tail_formula_cost_history.py --help
```

当前顺序为calendar、windows、inputs，仅来源及输入：协议先提交；保留原股票日历；32股原窗口精确复用，其余窗口独立SQL／Pandas核对；五日标量另由SQL核准。此协议禁止模型拟合；有效域改变须先固定同质量控制和新模型协议，再共同固定完整名单后评价。原48控制与原50只读复用；年度报告已含半年，不重复聚合。完整名单、元数据、适用标签与统计精确一致才复用。

当前来源与输入在data/research/tail_formula_cost_history/；32股试点在tail_formula_cost_history_probe/。旧完整补齐在tail_formula_history_boundary_recheck/；旧两个模型在tail_formula_morning_history_2024/及tail_formula_morning_history_recent/，完整原名单在tail_formula_morning_history_2025/。旧评价09:31—10:00不能混作09:59；2024缺对应旧模型，空表不代表低风险或好质量。

原始日线／分钟在data/baostock/和data/hf/，全部结果、原件和查重缓存在忽略的data/research/。2025暴露探索；不新增2026经济评价、卖出、多日持有或月度回购研究。

已结束方案从对应Git版本独立检出恢复，禁止重做：已删11个单次时间门及延续执行／测试文件1b9f085，稳健入口e13a9b5，平均实现24a883f，八个后过滤／诊断执行文件f6abbcb，同容量整体／分段1e5793f、情绪绝对bbde01d、情绪相对bfbe762、分钟开价ceca953、日内均幅46004ad、ETF总量6bfed42、尾段区间884391c。完整清理前版本4b3bdda5db6470df4d8d61298501852f60b985a2；字节归档映射与当前收据见[固定收据](selection-formula.md)。
