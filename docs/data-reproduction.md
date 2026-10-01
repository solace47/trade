# 当前环境与复算

当前只从[唯一入口](../scripts/run_tail_formula_emotion_transition.py)运行市场情绪来源核准。仓库根目录已有Python 3.12虚拟环境；依赖版本见`requirements.txt`，已有环境无需重装。

```sh
PYTHONPATH=src:scripts .venv/bin/python -m pytest -q
PYTHONPATH=src:scripts .venv/bin/python scripts/run_tail_formula_emotion_transition.py --help
PYTHONPATH=src:scripts .venv/bin/python scripts/run_tail_formula_emotion_transition.py source-audit
```

来源协议及运行指纹先提交，再核准原日线状态和筛选前价格缓存。输出在`data/research/tail_formula_emotion_transition/`；该阶段不能拟合或读新分组经济结果。完成产物禁止覆盖或重做。来源门通过后另提交实际输入和模型协议。

原始日线／分钟保留在`data/baostock/`和`data/hf/`，全部旧结果与查重缓存在忽略的`data/research/`。分钟开价代码／协议从Git `ceca953`取回，日内均幅取`46004ad`，ETF总量取`6bfed42`，尾段区间取`884391c`。这些实验已完整失败；不要再拟合或评价。

清理前完整源码和详细记录取`4b3bdda5db6470df4d8d61298501852f60b985a2`，历史复算在对应版本独立检出中进行。旧14:50快照、10:00窗口、多日持有和月度回购结果不替代当前边界。运行协议保留历次原字节收据与确切归档版本，仅精确一致的完整训练值、目标、权重、名单、元数据和统计才可复用。
