# 当前环境与复算

当前研究只从[唯一入口](../scripts/run_tail_formula_etf_quantity.py)运行，使用仓库根目录、Python 3.12及`requirements.txt`固定版本。首次准备环境：

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

已有环境无需重装。基础检查：

```sh
PYTHONPATH=src:scripts .venv/bin/python -m pytest -q
PYTHONPATH=src:scripts .venv/bin/python scripts/run_tail_formula_etf_quantity.py --help
PYTHONPATH=src:scripts .venv/bin/python scripts/verify_research_runtime.py
```

最后一项只读核对清理后代码、清理前固定定义、输入指纹及原控制模型评分，不拟合、不生成新选股名单、不重算选股经济结果。

## 当前产物与阶段

原始日线及分钟在`data/baostock/`、`data/hf/`；当前52项产物在`data/research/tail_formula_etf_quantity/inputs/`。旧控制依赖`tail_formula_stock_2024`与`tail_formula_morning_range`；7列训练标签与69列经济标签是不同文件，不能混用。全部数据与机器产物由Git忽略，不会随源码推送。

`prepare`已经完成且禁止覆盖，`protocols`、四控制复用、四个新`fit`及`freeze`也已完成；下一步仅接续`analyze`与`finish`，不得重拟合。控制只能精确复用；最多四个新增模型。`freeze`完成后须先把联合SHA记入当前方案并提交，随后才能`analyze`与`finish`。

清理后[运行协议](../config/research-runtime.json)绑定当前源码及既有输入；此前输入／模型协议保持原字节。其引用的旧源码通过Git历史验证，当前代码另有完整指纹及等价核验，避免删除旧入口后失去来源追踪或重做已完成输入。

## 历史复算

清理前版本由运行协议的`archive_revision`指明，也记录于[当前方案](selection-formula.md)。旧实验代码、配置和详细记录从该版本读取：

```sh
git show 4b3bdda:docs/selection-formula.md
git show 4b3bdda:scripts/run_tail_formula_volume_memory.py
```

历史脚本应在对应版本的独立检出中复算，不能混入当前运行目录或覆盖已核验产物。原始行情、完成收据和查重缓存仍保留；旧14:50快照、10:00窗口及多日持有结果不替代当前边界。
