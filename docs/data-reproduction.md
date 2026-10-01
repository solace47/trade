# 当前环境与复算

当前[唯一入口](../scripts/run_tail_formula_emotion_absolute.py)为市场情绪绝对机会对照。使用已有Python 3.12虚拟环境，依赖见`requirements.txt`。

```sh
PYTHONPATH=src:scripts .venv/bin/python -m pytest -q
PYTHONPATH=src:scripts .venv/bin/python scripts/run_tail_formula_emotion_absolute.py --help
PYTHONPATH=src:scripts .venv/bin/python scripts/run_tail_formula_emotion_absolute.py protocols
```

实际输入／目标、模型与运行协议先提交，再生成八分折查重。分折配置及查重收据先提交才fit；八模型freeze后先提交联合名单，再analyze、finish。本轮已完整结束且失败，完成产物不能覆盖或重做；后续须独立协议及查重。

不重新生成情绪输入，不扫描原分钟；精确使用`data/research/tail_formula_emotion_transition/inputs/`中的50／52与原训练标签。新产物位于`data/research/tail_formula_emotion_absolute/`。目标改变须有同目标控制，不能复用相对控制代替绝对控制；未知不补零，所有训练观察严格早于评价起点。

原始日线／分钟在`data/baostock/`和`data/hf/`，旧结果和查重缓存均在忽略的`data/research/`。已失败的情绪相对源码／配置取Git `bfbe762`，分钟开价取`ceca953`，日内均幅取`46004ad`，ETF总量取`6bfed42`，尾段区间取`884391c`；禁止重做。

清理前完整源码及记录取`4b3bdda5db6470df4d8d61298501852f60b985a2`，历史复算使用对应版本独立检出。旧14:50快照、10:00窗口、多日持有及月度回购不能替代当前边界。运行协议保留原字节收据和确切归档版本，只有完整训练值、目标、权重、名单、元数据和适用统计精确一致才复用。
