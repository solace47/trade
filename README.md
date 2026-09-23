# A 股尾盘短线策略研究

研究交易日 14:50 可见的行情信号、14:52–14:55 买入及 T+1 至 T+5 卖出的实际可执行性。通达信和同花顺公式须等待多年全市场及样本外验证。

## 运行

需要 Python 3.12；原始行情和本地报告保存在被 Git 忽略的 `data/`。

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
export PYTHONPATH=src
.venv/bin/python -m trade_research.ingest --workers 4
.venv/bin/python -m trade_research.hf_download --workers 8
.venv/bin/python -m trade_research.hf_audit
.venv/bin/python -m trade_research.hf_snapshot
.venv/bin/python -m trade_research.hf_outcomes
.venv/bin/python -m trade_research.pilot_study
```

数据来源、验收结果和当前限制见[研究状态](docs/research-status.md)。
