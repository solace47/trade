# 外部项目的复用判断

2026-09-26 针对当前研究瓶颈阅读源码；优先复用可核对的组件和方法，不以项目声称的收益替代本地检验。检索包括 GitHub、skills.sh 及 `npx skills find backtesting`，未安装新的交易框架或 skill。

| 来源 | 与本项目有关的内容 | 采用方式及边界 |
| --- | --- | --- |
| [MyTT 的 DMA/REF 实现](https://github.com/mpquant/MyTT/blob/main/MyTT.py) | 动态均价递推与通达信式指标映射 | 已把同一历史价格换算到固定参考单位，抽取 32 个实际输入与其单倍初始化 DMA 对照，最大误差 `2.84e-14` 元。仅执行审阅过的 DMA 函数；没有验证通达信客户端的实际输出。该实现把缺失平滑权重替换成 1，本项目仍显式处理未知历史；其 `CONST` 取整个传入序列的末值，不能直接放入历史选股输入。 |
| [RQAlpha 股票持仓源码](https://github.com/ricequant/rqalpha/blob/master/rqalpha/mod/rqalpha_mod_sys_accounts/position_model.py) | T+1 可卖数量、应收股息、实际到账、送转、退市 | 值得作为持仓状态与测试案例的参考。所读版本默认关闭股息税、允许退市按市值返还现金，不能原样用于本研究；应收股息与可用现金分开这一设计与现有记账一致。此次没有运行其完整撮合器；项目主页声明限非商业使用。 |
| [Qlib 滚动任务生成器](https://github.com/microsoft/qlib/blob/main/qlib/workflow/task/gen.py) | 滑动／扩展训练窗和 `trunc_segments` | 复用其明确训练、验证、测试边界的设计，已在新模型对照中逐条核对训练标签退出早于测试起点。默认 `trunc_days=None` 不会自动消除标签重叠；当前仍用自行实现的固定时段训练，未宣称运行了 Qlib 的滚动任务。 |
| [backtesting-frameworks skill](https://skills.sh/wshobson/agents/backtesting-frameworks) | 前视偏差、成本、滚动验证的指导和示例 | 检索时显示约 1.54 万安装，来源仓库约 4 万星；这些只表示传播度。读过说明和 `references/details.md`：示例市价单按下一根开盘足量成交、按股收费，没有本项目所需的 A 股涨跌停与分钟量约束。方法可作参考，暂不安装或替换现有执行器。 |

当前最直接的收益是确认历史参考价与常见动态均价实现的数学对应，并明确可复用组件的默认假设；尚未从外部项目取得经本地验证的盈利策略。下载来源、内容 SHA 和 DMA 对照保存在本地忽略的 `data/research/external_reuse/`。GitHub 元数据接口遇到限流，使用公开源码页完成查阅，没有把未成功获取的提交信息写成已核验版本。
