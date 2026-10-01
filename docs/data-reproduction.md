# 当前环境与复算

当前拟合研究只从[唯一入口](../scripts/run_tail_formula_intraday_scale.py)运行，使用仓库根目录、Python 3.12及`requirements.txt`固定版本。首次准备环境：

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

已有环境无需重装。基础检查：

```sh
PYTHONPATH=src:scripts .venv/bin/python -m pytest -q
PYTHONPATH=src:scripts .venv/bin/python scripts/run_tail_formula_intraday_scale.py --help
PYTHONPATH=src:scripts .venv/bin/python -c 'from trade_research.research_io import check_runtime; check_runtime()'
```

最后一项只读核对当前固定源码、配置和已有输入指纹，不拟合或生成新结果。旧控制的完整训练值、目标、权重和评分在四折复用阶段核准。

## 当前产物与阶段

原始日线及分钟在`data/baostock/`、`data/hf/`。当前日内尺度产物写入`data/research/tail_formula_intraday_scale/inputs/`；只从固定日线原件重建此前20日日内均幅，原50源与训练标签精确复用。7列训练标签与69列经济标签不可混用。已结束区间方案及参考价审计的原件／收据仍在data，源码可取Git `884391c`；数据及机器产物由Git忽略。

当前日内尺度已完成`prepare`、`protocols`、全部四控制复用／四新模型及`freeze`，不得重做或覆盖。联合SHA提交后再运行`analyze`及`finish`；完成阶段不可重复拟合或评价。

`control/2024h1`评分定义校验曾中断，已按修复协议仅补独立评分验证；原模型、模型验证、评分和分折协议原字节保持，不得重新`fit`。

[运行协议](../config/research-runtime.json)同时保留首次清理协议的原字节收据和当前研究绑定。ETF量级研究已完整失败，其最后源码／配置在Git `6bfed42`，完整产物仍在`data/research/tail_formula_etf_quantity/`；不得重复拟合或评价。

## 历史复算

清理前版本由运行协议的`archive_revision`指明，也记录于[当前方案](selection-formula.md)。旧实验代码、配置和详细记录从该版本读取：

```sh
git show 4b3bdda:docs/selection-formula.md
git show 4b3bdda:scripts/run_tail_formula_volume_memory.py
```

历史脚本应在对应版本的独立检出中复算，不能混入当前运行目录或覆盖已核验产物。原始行情、完成收据和查重缓存仍保留；旧14:50快照、10:00窗口及多日持有结果不替代当前边界。
