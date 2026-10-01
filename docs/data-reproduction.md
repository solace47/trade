# 当前环境与复算

当前为同候选库盈利可行性检验：[选择器入口](../scripts/run_tail_formula_rule_reference_gate.py)、[名单与评价入口](../scripts/finish_tail_formula_rule_reference_gate.py)。原三条件及公开分时规则已经完成，禁止重复拟合或覆盖收据。使用已有Python 3.12虚拟环境，依赖见requirements.txt。

```sh
PYTHONPATH=src:scripts .venv/bin/python -m pytest -q
PYTHONPATH=src:scripts .venv/bin/python scripts/run_tail_formula_rule_reference_gate.py --help
PYTHONPATH=src:scripts .venv/bin/python scripts/finish_tail_formula_rule_reference_gate.py --help
```

输入与实际执行协议先提交再fit；四个选择器核准并固定评价实现后才freeze；四份完整年度名单共同提交后才analyze、finish。阶段禁止覆盖已完成产物。原50与原三条件的完整名单和经济统计只读复用；新名单若与旧完整表、适用标签及统计精确相同，也必须复用。当前指纹见[固定收据](selection-formula.md)。

当前产物位于data/research/tail_formula_rule_reference_gate/：原50输入、资格、决策股数及原三条件候选库保持不变，成熟训练净参考独立SQL核准，不新增原始分钟或2026行情。旧结果位于tail_formula_rule_search/与tail_formula_external_pattern/；原件、八组训练反思与十分支诊断均保留。

原始日线／分钟在data/baostock/和data/hf/；旧结果、原件和查重缓存保留于忽略的data/research/。完整名单、元数据、适用标签及统计精确相同才复用，年度已含半年时不重复聚合。

已结束方案从对应Git版本独立检出恢复，禁止重做：同容量整体／分段1e5793f、情绪绝对bbde01d、情绪相对bfbe762、分钟开价ceca953、日内均幅46004ad、ETF总量6bfed42、尾段区间884391c。十七个分段执行文件的原字节已逐项核准，恢复映射保存在当前运行收据。完整清理前版本4b3bdda5db6470df4d8d61298501852f60b985a2；旧14:50快照、10:00窗口、多日和月度回购不能替代当前边界。
