# 当前环境与复算

当前[研究入口](../scripts/run_tail_formula_phase_split.py)为同容量整体／分段预测。使用已有Python 3.12虚拟环境，依赖见`requirements.txt`。

```sh
PYTHONPATH=src:scripts .venv/bin/python -m pytest -q
PYTHONPATH=src:scripts .venv/bin/python scripts/run_tail_formula_phase_split.py --help
```

输入协议先提交，再prepare；实际协议先提交，再lookup；十二分折配置与查重收据先提交，再逐折fit。所有模型freeze后先提交联合名单，才analyze、finish。已完成阶段不能覆盖或重做；当前进度及固定指纹见[当前方案](selection-formula.md)。

原50输入精确复用`data/research/tail_formula_emotion_transition/inputs/`；新目标仅从已核准的2023严格窗口及`tail_formula_before1000`的2024—2025标签构造。两方案共用同一训练交集，缺参考保持未知；早于2024的历史只训练或初始化。新产物位于`data/research/tail_formula_phase_split/`，不扫描原分钟、不新增2026经济结果。

原始日线／分钟在`data/baostock/`和`data/hf/`，旧结果和查重缓存保留在忽略的`data/research/`。完整名单、全部元数据、适用标签及统计精确相同才复用；年度报告已含半年时不重复聚合，比较也查重。

失败来源、模型及评价的原字节协议可从相应Git恢复：情绪绝对目标`bbde01d`、情绪相对`bfbe762`、分钟开价`ceca953`、日内均幅`46004ad`、ETF总量`6bfed42`、尾段区间`884391c`。禁止重做。清理前完整源码与记录取`4b3bdda5db6470df4d8d61298501852f60b985a2`，复算须使用对应版本独立检出；旧14:50快照、10:00窗口、多日与月度回购不能替代当前边界。
