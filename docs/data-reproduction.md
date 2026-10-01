# 当前环境与复算

当前[输入入口](../scripts/run_tail_formula_external_pattern.py)及[固定名单／评价入口](../scripts/finish_tail_formula_external_pattern.py)已全部完成，禁止重复实验或覆盖收据。使用已有Python 3.12虚拟环境，依赖见requirements.txt。

```sh
PYTHONPATH=src:scripts .venv/bin/python -m pytest -q
PYTHONPATH=src:scripts .venv/bin/python scripts/run_tail_formula_external_pattern.py --help
PYTHONPATH=src:scripts .venv/bin/python scripts/finish_tail_formula_external_pattern.py --help
```

输入协议先提交再prepare；来源门与实际执行协议先提交再freeze；四份完整年度名单共同提交后才analyze、finish。所有阶段禁止覆盖已完成产物。十分支诊断也已固定并完成，仅作事后失败解释；当前指纹见[固定收据](selection-formula.md)。

产物位于data/research/tail_formula_external_pattern/：原50资格／决策股数复用已核准完整键，14:49前原始230分钟独立校验；两组施加同样前缀质量，坏来源仍保留未知。只解码2024—2025对应前缀，不新增2026模型行情。

原始日线／分钟在data/baostock/和data/hf/；旧结果、原件和查重缓存保留于忽略的data/research/。完整名单、元数据、适用标签及统计精确相同才复用，年度已含半年时不重复聚合。

已结束方案从对应Git版本独立检出恢复，禁止重做：同容量整体／分段1e5793f、情绪绝对bbde01d、情绪相对bfbe762、分钟开价ceca953、日内均幅46004ad、ETF总量6bfed42、尾段区间884391c。十七个分段执行文件的原字节已逐项核准，恢复映射保存在当前运行收据。完整清理前版本4b3bdda5db6470df4d8d61298501852f60b985a2；旧14:50快照、10:00窗口、多日和月度回购不能替代当前边界。
