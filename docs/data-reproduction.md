# 环境与复算

在仓库根目录执行。当前状态见[研究入口](research-status.md)，结果表见[策略结果](strategy-results.md)，各版本原始冻结步骤见[公式研究索引](selection-formula.md#已完成研究索引)。

## 环境

已有`.venv`可直接使用。新环境按`requirements.txt`固定版本安装：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

模块统一使用`PYTHONPATH=src .venv/bin/python -m trade_research.<模块>`，脚本使用`PYTHONPATH=src .venv/bin/python scripts/<文件>.py`。已有冻结产物拒绝覆盖；复核不需要重新下载、拟合或聚合。

## 数据位置

| 内容 | 本地位置 |
| --- | --- |
| 日线、历史状态、原始分钟及下载清单 | `data/`各来源目录；锁定版本与下载SHA以原收据为准 |
| 旧14:50全市场快照／持有期／问题日 | `data/research/market_snapshots_ci/`、`market_outcomes_ci/`、`market_issues_ci/` |
| 当前14:49原始前缀 | `data/research/minute_prefix_1449/` |
| 原生公式各版本 | `data/research/tail_formula_*/` |
| 最新前股日联合输入与完整年度结果 | `data/research/tail_formula_prior_bar/` |
| 分钟量额各版／当前接续 | `data/research/tail_formula_minute_vwap*/`；实际阶段及产物以[当前状态](research-status.md)为准 |

`data/`受Git忽略，仓库本身不包含完整行情和研究产物。不能删除本地收据再运行旧命令来重选版本；文件名、数量相同也不证明来源或结果相同。

## 最近完成版本的核对

前股日版本的协议位于`config/tail_formula_prior_bar_*`，输入实现为`src/trade_research/tail_formula_prior_bar.py`，模型实现为同名`_model.py`，共同冻结和评价入口为`scripts/freeze_tail_formula_prior_bar.py`、`scripts/evaluate_tail_formula_prior_bar.py`。

先读取现有报告，不重复执行生产阶段：

```sh
PYTHONPATH=src .venv/bin/python - <<'PYCODE'
import json
from pathlib import Path
from trade_research.corporate_cash import sha
root = Path('data/research/tail_formula_prior_bar')
for name in ['joint_selection_freeze.json', 'prior_bar_gate.json', 'complete_results_manifest.json']:
    path = root / name
    report = json.loads(path.read_text())
    assert report['passed']
    print(name, sha(path))
print('支持2024扩展：', json.loads((root / 'prior_bar_gate.json').read_text())['supports_2024_extension'])
PYCODE
```

报告中的`passed`表示阶段核准，不能解释为策略通过。完整来源复核可调用`scripts/evaluate_tail_formula_prior_bar.py`中的`checked_joint()`；它核对协议、代码、联合收据和完整名单，不运行新经济汇总。

需要复算已有版本时，先核对该版协议／源码／输入指纹及既有阶段报告。不同家族的命令和验证器不能互换；例如上午分布独立核准和高点保持原生核准须用其冻结包装脚本，不能改用曾失败的直接入口。

## 新研究顺序

先固定输入，再独立核准全部键、特征和原生表达；使用已经结束的训练标签，固定模型和完整名单，提交联合收据后才评价。复用前比较完整选择表、元数据、全部适用标签、摘要和日表指纹；相同则绑定旧报告，不重做汇总。年度报告已含两半年，不另重复聚合。

当前评价只读09:31—09:59；旧30根窗口、2026已暴露结果和旧固定退出实验不能当成当前的新验证。当前接续阶段只在[研究状态](research-status.md)维护，不能把历史命令当新任务执行。

## 历史复算

清理前固定版本为`96d34f694d06c1e317c29e10643013478fb34ece`。完整旧文档、精确冻结参数、异常修正记录和旧复算命令均保留在Git历史；不另复制一套归档文件：

```sh
git show 96d34f694d06c1e317c29e10643013478fb34ece:docs/selection-formula.md
git show 96d34f694d06c1e317c29e10643013478fb34ece:docs/input-gates.md
git show 96d34f694d06c1e317c29e10643013478fb34ece:docs/data-reproduction.md
```

[旧公式过程稿](https://github.com/solace47/trade/blob/96d34f694d06c1e317c29e10643013478fb34ece/docs/selection-formula.md)、[旧输入过程稿](https://github.com/solace47/trade/blob/96d34f694d06c1e317c29e10643013478fb34ece/docs/input-gates.md)和[旧复算命令](https://github.com/solace47/trade/blob/96d34f694d06c1e317c29e10643013478fb34ece/docs/data-reproduction.md)可直接查阅。已结束专题的结果已合并到[策略结果](strategy-results.md#旧固定退出实验的口径与门槛)；公告、分笔、题材和大涨画像保留在[反向研究](next-day-winner.md)。

研究代码、冻结配置、许可证和原始数据均保留。它们被来源指纹或旧产物引用，删除／修改会使历史复核失效；当前精简的是重复说明和已结束过程日志。
