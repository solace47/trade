# 外部项目的复用判断

2026-09-26 针对当前研究瓶颈阅读源码；优先复用可核对的组件和方法，不以项目声称的收益替代本地检验。检索包括 GitHub、skills.sh 及 `npx skills find backtesting`，未安装新的交易框架或 skill。

| 来源 | 与本项目有关的内容 | 采用方式及边界 |
| --- | --- | --- |
| [可转债统一数据接口](https://github.com/reskfa/cb_with_any_api/tree/8687b749d0c2cc735f81fb5d3496b1f9c2d2922d) | 2026-09-27核对README、核心类及AKShare适配器，MIT许可；统一正股映射、转股价值及多源日级字段 | 可参考字段适配，当前不接入14:49信号。所读适配器调用`bond_zh_cov_value_analysis`及日线；`Amt`用`volume × close × 10`估算，不能当原始成交额，更不提供已经核准的2024–2025分钟历史。没有运行外部代码或载入其pickle。 |
| [tdx2db](https://github.com/jing2uo/tdx2db/tree/80b287bf65019c79ff8db40ee42db5389f874731) | 2026-09-27核对固定提交README；本地日线、分钟导入和板块映射 | 免费部分说明历史分钟需自行补齐，超过30天须手补，macOS预编译版本跳过分钟下载；2024年起历史包出现在另一个付费产品说明中，尚未实测内容。不能把此README当作免费完整分钟来源，也不能把其3秒成交记录误作逐笔委托队列。当前不安装或采购。 |
| [BigQuant可转债分钟表目录](https://bigquant.com/data/datasources/cn_cbond_bar1m) | 2026-09-27核对官方字段：时间、OHLC、昨收、量额及笔数 | 确认目录存在，尚未取得数据样本；目录的“起始时间”含16:20时分秒，不等于已核准最早行情日。完整覆盖、时间标签、历史映射及下载权限均未实测，不能据目录声明已具备转债领先正股检验能力。 |
| [Qlib Alpha158 加载器及算子](https://github.com/microsoft/qlib/blob/a7d5a9b500de5df053e32abf00f6a679546636eb/qlib/contrib/data/loader.py) | 固定 158 项表达式、5–60 日滚动窗口及训练期稳健缩放 | 已锁定提交 `a7d5a9b500de5df053e32abf00f6a679546636eb`，同时阅读算子、滚动回归与处理器源码，定义和 MIT 许可纳入仓库。14:49 临时K线适配及 100 万行计算完成，64 个实际样本的全部指标独立核准。固定线性模型 2025 年相对同缩放18列 −0.333 个百分点、区间跨零，未支持收益增量；保留指标组件，不使用其默认标签、撮合器或展示收益，不声称运行了完整 Qlib。 |
| [MyTT 的 DMA/REF 实现](https://github.com/mpquant/MyTT/blob/main/MyTT.py) | 动态均价递推与通达信式指标映射 | 已把同一历史价格换算到固定参考单位，抽取 32 个实际输入与其单倍初始化 DMA 对照，最大误差 `2.84e-14` 元。仅执行审阅过的 DMA 函数；没有验证通达信客户端的实际输出。该实现把缺失平滑权重替换成 1，本项目仍显式处理未知历史；其 `CONST` 取整个传入序列的末值，不能直接放入历史选股输入。 |
| [RQAlpha 股票持仓源码](https://github.com/ricequant/rqalpha/blob/master/rqalpha/mod/rqalpha_mod_sys_accounts/position_model.py) | T+1 可卖数量、应收股息、实际到账、送转、退市 | 值得作为持仓状态与测试案例的参考。所读版本默认关闭股息税、允许退市按市值返还现金，不能原样用于本研究；应收股息与可用现金分开这一设计与现有记账一致。此次没有运行其完整撮合器；项目主页声明限非商业使用。 |
| [Qlib 滚动任务生成器](https://github.com/microsoft/qlib/blob/main/qlib/workflow/task/gen.py) | 滑动／扩展训练窗和 `trunc_segments` | 复用其明确训练、验证、测试边界的设计，已在新模型对照中逐条核对训练标签退出早于测试起点。默认 `trunc_days=None` 不会自动消除标签重叠；当前仍用自行实现的固定时段训练，未宣称运行了 Qlib 的滚动任务。 |
| [Qlib 截面处理器](https://github.com/microsoft/qlib/blob/main/qlib/data/dataset/processor.py) | `CSZScoreNorm`／`CSRankNorm` 按日期跨股票处理字段 | 已核对源码，借鉴其按日期比较的思路，另行冻结同日训练评分去均值对照。只去均值，保留收益量纲，不声称等同于原处理器或复用其策略收益；未加载整个外部模块。公开 main 文件及 SHA 已缓存，未声称锁定 Git 提交。 |
| [backtesting-frameworks skill](https://skills.sh/wshobson/agents/backtesting-frameworks) | 前视偏差、成本、滚动验证的指导和示例 | 检索时显示约 1.54 万安装，来源仓库约 4 万星；这些只表示传播度。读过说明和 `references/details.md`：示例市价单按下一根开盘足量成交、按股收费，没有本项目所需的 A 股涨跌停与分钟量约束。方法可作参考，暂不安装或替换现有执行器。 |

当前最直接的收益是确认历史参考价与常见动态均价实现的数学对应，并明确可复用组件的默认假设；尚未从外部项目取得经本地验证的盈利策略。下载来源、内容 SHA 和 DMA 对照保存在本地忽略的 `data/research/external_reuse/`。GitHub 元数据接口遇到限流，使用公开源码页完成查阅，没有把未成功获取的提交信息写成已核验版本。


2026-09-27另核对[AKShare资金流源码](https://github.com/akfamily/akshare/blob/main/akshare/stock/stock_fund_em.py)、[同花顺适配器](https://github.com/akfamily/akshare/blob/main/akshare/stock_feature/stock_fund_flow.py)与[快讯适配器](https://github.com/akfamily/akshare/blob/main/akshare/stock_feature/stock_info.py)：前者个股接口注明近期日级数据，同花顺大单追踪无历史日期参数，快讯默认只取最近一页；这不证明上游完全没有历史，但封装本身不能提供已核准的2024–2025盘中快照。源码仅阅读，未整体执行。文件SHA分别`04a4677103fe8faa02515944af8e45e6c6200ef45a6f6f68a4bf8f0c95d049cd`、`d9c658a6c47186f06f1094e127e5aa75284d23cd57a82ca4e9b6d2ccd368a0f0`、`af5222de4c1396b2b1ecc14ff40c13e5cdc32ae8be738523b1c2f7a395bedb90`，保存在本地`news_source_probe/`。

[Tushare资金流](https://tushare.pro/document/2?doc_id=170)按主动单金额分组，不能由此确定机构身份；[THS](https://tushare.pro/document/2?doc_id=348)和[DC](https://tushare.pro/document/2?doc_id=349)历史接口明确盘后更新，当前日不能回填14:49。[历史快讯接口](https://tushare.pro/document/2?doc_id=143)声明有超过六年历史，但需单独权限，尚未取得样本或核准时间字段。公开东财资金流及新浪快讯各一次历史日期探测遇TLS连接失败，未得到数据；不能把传输失败记成无事件或断言历史不存在。下一步需要历史盘中归档或独立时间证据，当前没有购买、安装或增加2026价格检验。

2026-09-27为流通规模／绝对换手核对[BaoStock官方日线定义](https://www.baostock.com/mainContent?file=stockKData.md)：`turn`以当日成交股数除以流通股总股数计百分比。本地只用严格前一日的量与换手构造滞后分母，未声称获得精确自由流通股本或当前日实时换手。旧百科地址已重定向到新前端，公开菜单和Markdown端点由官网脚本确认；文档、菜单与前端指纹保存在`float_turnover_winner/source/`。无需安装新框架，也未访问处于用户接管状态的财联社页面。
