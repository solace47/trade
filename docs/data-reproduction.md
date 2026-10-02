# 环境与复算

使用Python 3.12虚拟环境及[requirements.txt](../requirements.txt)。现有32项共用检查通过，命令：

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts .venv/bin/python -m pytest -q -p no:cacheprovider
```

保留的代码负责特征编码、模型／名单核验、成本参考、未知区间、统计和精确查重；旧协议默认值只用于复算，不是可用策略。结束实验不重训、不覆盖冻结名单或结果。新实验先读[状态](research-status.md)、[失败索引](strategy-results.md)和[输入门槛](input-gates.md)。

原始日线／分钟在`data/baostock/`与`data/hf/`；模型、名单、完整报告、原文、失败日志和查重缓存均保存在忽略的`data/research/`。本次只删除31份已结束实验的专用代码／协议／测试，共6,088行；没有删除行情或研究结果。

| 恢复版本 | 内容 |
| --- | --- |
| `8c6c816` | 本次删除文件：成长历史、外部分时规则、两类AND搜索、结束诊断及旧情绪输入协议；亦含精简前完整文档。 |
| `64a0dbb`／`f23b0ef` | 上午量集中度／跨板块身份来源门。 |
| `6a5af27`／`1a09ca5` | 上市范围扩展／主事件稳定性及训练基准诊断。 |
| `17ac302`／`80314a4`／`79d4916` | 空间与风险／方向来源／方向执行和反思。 |
| `4b3bdda` | 更早大规模清理前的完整研究代码与记录。 |

可用`git show <版本>:<路径>`读取旧文件；完整复算宜另建检出目录，并使用独立输出目录。逐文件指纹与精确恢复版本见[运行清单](../config/research-runtime.json)；本次删除清单和清理前环境存于`data/research/cleanup_20261002/`。当前环境指纹见[当前方案](selection-formula.md)。
