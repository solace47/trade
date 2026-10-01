# 当前环境与复算

当前研究只从[唯一入口](../scripts/run_tail_formula_retained_interval.py)运行，使用仓库根目录、Python 3.12及`requirements.txt`固定版本。首次准备环境：

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

已有环境无需重装。基础检查：

```sh
PYTHONPATH=src:scripts .venv/bin/python -m pytest -q
PYTHONPATH=src:scripts .venv/bin/python scripts/run_tail_formula_retained_interval.py --help
PYTHONPATH=src:scripts .venv/bin/python -c 'from trade_research.research_io import check_runtime; check_runtime()'
```

最后一项只读核对当前固定源码、配置和已有输入指纹，不拟合或生成新结果。旧控制的完整训练值、目标、权重和评分在四折复用阶段核准。

## 当前产物与阶段

原始日线及分钟在`data/baostock/`、`data/hf/`。当前区间结构产物将写入`data/research/tail_formula_retained_interval/inputs/`，2024／2025尾段窗口复用`tail_formula_minute_pressure/parts/`，2023未缓存尾段在实际协议提交后提取。原50源为`tail_formula_stock_2024`与`tail_formula_morning_range`；7列训练标签与69列经济标签不可混用。数据及机器产物由Git忽略。

`prepare`、`protocols`已完成且不得覆盖；下一步四折`fit --arm control|memory --fold 2024h1|2024h2|2025h1|2025h2`。控制只能精确复用，最多四个新模型。`freeze`后先提交联合SHA，才可`analyze`及`finish`，已完成阶段不可覆盖。

[运行协议](../config/research-runtime.json)同时保留首次清理协议的原字节收据和当前研究绑定。ETF量级研究已完整失败，其最后源码／配置在Git `6bfed42`，完整产物仍在`data/research/tail_formula_etf_quantity/`；不得重复拟合或评价。

## 历史复算

清理前版本由运行协议的`archive_revision`指明，也记录于[当前方案](selection-formula.md)。旧实验代码、配置和详细记录从该版本读取：

```sh
git show 4b3bdda:docs/selection-formula.md
git show 4b3bdda:scripts/run_tail_formula_volume_memory.py
```

历史脚本应在对应版本的独立检出中复算，不能混入当前运行目录或覆盖已核验产物。原始行情、完成收据和查重缓存仍保留；旧14:50快照、10:00窗口及多日持有结果不替代当前边界。
