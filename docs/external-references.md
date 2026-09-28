# 外部项目的复用判断

2026-09-27核对[通达信官方帮助第6页](https://zxfile.tdx.com.cn/zx/201612/3185123/3185123_fj.pdf)：B／S表示软件判定的主动买卖，不能依据报价判定的记录有不明类别；同一分笔可能合并多笔成交。官方说明没有在此确认CNEquity作者所称的精确tick-rule算法，因此本项目只称“供应商方向标记”，不声称已验证其生成算法或资金身份。[官方历史分笔接口](https://help.tdx.com.cn/quant/docs/markdown/mindoc-1hho7blr2j340/mindoc-1hi3n0rqkakt4/mindoc-1hhob7ou94li8.html)也明确区别分笔与K线；函数目录有`ISBUYORDER`不等于已证明普通分钟公式能复现完整方向序列。官方说明已通过网页读取；本地原件下载遇TLS失败，链接及失败记录保存在`tick_source_probe/official/manifest.json`，不能称为已落盘原件。本次未安装客户端或新插件。

2026-09-26 针对当前研究瓶颈阅读源码；优先复用可核对的组件和方法，不以项目声称的收益替代本地检验。检索包括 GitHub、skills.sh 及 `npx skills find backtesting`，未安装新的交易框架或 skill。

| 来源 | 与本项目有关的内容 | 采用方式及边界 |
| --- | --- | --- |
| [CNEquity历史分笔](https://github.com/rootSunc/CNEquity/tree/1650e384a3fd1f67a70144a489acc91432f1df27) | 2026-09-27核对固定提交的目录、来源、协议和聚合逻辑；独立小程序实际取得2024–2025的12个固定股票日 | 54个原始响应、48,874条记录逐字节独立解码一致；与日线的全天量差不超过47股，开收盘相符。值得复用历史分笔取数协议；三秒左右聚合不等于交易所逐笔，只有分钟时间标签，方向是供应商推断，金额只能近似，不能证明机构身份或排板成交。新信号只取到14:48，完整日只用于来源核验；不运行外部项目整体。保留tdxpy的MIT及CNEquity的Apache许可和通知。 |
| [a_stock_screener](https://github.com/songhuizhang/a_stock_screener/tree/c5e87e35d14efc3adbef861c8be7778f771953fb) | 2026-09-27核对固定提交的README、主程序和参数；002／003、涨幅3%–5%、近20日大涨、量比、换手、流通规模及均价承接的完整交集 | 值得将完整交集作为外部提出的待检验假设。源码在当日分钟缺失时回退昨日仍可通过分时条件，指数失败记零，日线查询含当日；这些做法不能用于本项目的历史14:49选择。复用条件思路、自行重建当时可见数据，不运行或安装外部程序；README声明MIT。 |
| [stock-predict](https://github.com/hyan1985/stock-predict/tree/94d10759e2dbc11eadecbfad45cc23b25316e542) | 2026-09-27核对README、SKILL、形态函数、回测和数据适配；公开宣称的高胜率只能作待核实主张 | 回测把完整当日分钟直接传入形态函数，没有14:49截断；结果是当日收盘至次日收盘涨跌，未模拟原始买卖、费用或排板；最终买信号直接取形态，换手及规模没有进入回测选择。因此不能拿其胜率验证本任务的完整尾盘规则，也不采用代码中的“主力进货”解释。只审阅，不安装skill或执行外部模块。 |
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

2026-09-27接续检索历史题材来源：[levistock固定提交](https://github.com/fleetinglife/levistock/tree/09f871a77b6e3241b1249d986e7e8a7cbf9a44fe)提供开盘红历史涨停列表、题材成员和盘面事件的日期参数，只有接口实现，不能仅凭参数确认历史分类未被回填。40,280字节源码包SHA256为`d8dcb5838421dad5a996be47cd090bed1b7100ffd5982f605eb830bdcbcec3ca`，仅审阅并借用匿名请求字段，MIT原文保留。固定四日探测的第一笔2024-01-02请求遇TLS EOF，按协议停止，没有HTTP响应或历史数据，不能结论为来源不存在；没有调用其中财联社接口。

[zzshare固定提交](https://github.com/zzquant/zzshare/tree/49888288f9bb0c6231d92e17142ec5df62d3b17a)明确支持低频匿名查询，其历史涨停原因及题材梯队日期接口值得进一步探测。它的当前板块列表不能冒充历史成员；尚未将供应商归因认定为买入前已知信息。[涨停池skill](https://github.com/quantskills/skill-b6-limitup-pool)的完整分类与来源标记可参考，但依赖PandaData账号，且分钟缺失会降级用日线估计炸板，本研究不能将该代理当真实炸板次数；没有安装或执行它。继续用原固定2024–2025日期核查另一路匿名来源，不下载最新行情或追逐网站宣传胜率。


后续传输诊断已更新上述来源状态：同样匿名参数换用系统curl并保持TLS验证，两源都成功取回原四个日期，原Python失败仍归档。历史涨停、日期和成员价已核对，首次发布时间／无历史修订仍未证明。开盘红SDK对精选排行第2列的金额映射与原数据不符，原字段实际与题材评分相同；历史响应还有当前估值标题。zzshare梯队含次日开收盘涨幅，全部禁止进入历史输入。只借用经原始响应核准的请求与身份字段，不采用封装声称的资金、估值和盈利语义；详见[来源核对](input-gates.md#历史题材归属与昨日炸板的新检验)。

补读[Chen等《Daily Price Limits and Destructive Market Behavior》作者公开稿](https://www.princeton.edu/~wxiong/papers/PriceLimit.pdf)：研究2012–2015年深市，利用按此前账户规模划分的真实交易数据，发现大账户在封板日买入、次日卖出的群体行为。可借用其对封板、未封和后续时段的区分；不能把本地聚合B/S标记当作同等身份数据，也不能从收盘封板后的价格表现推出普通尾盘委托可成交。历史样本与本研究时期不同，这篇论文不替代2024–2025原始分钟验证，不据其结果回改已固定规则。


2026-09-28在跨期方向约束训练期间补查两项原始研究，未据其宣传收益更改已固定实验：

- [Zhang、Cheng、Shi：日内流动性与预期收益，2023-12-26作者稿](https://www.cfrn.com.cn/uploads/master/file/20240905/66d9a1f1bccdc.pdf)采用日内开收盘变化与成交额衡量非流动性，主要检验下一月的多空组合；第6页明确指出理论因子回报不能直接实现。可参考其区分量、价格变化和相关关系的方法，不把月度多空溢价等同本项目尾盘做多至次晨10点的可交易优势。本轮未接入新信号。
- [Positive feedback trading, the T+1 rule, and asymmetric return reversals in China](https://doi.org/10.1016/j.econmod.2026.107783)出版商检索摘要描述高换手下跌后的反转及其次日日内集中性，主体是市场指数；详情页本次未成功打开。不能由摘要推出具体股票在10点前有利润，也不重做本仓库已完成的下跌放量反转。仅记录检索层面的参考和访问限制，不声称完整读过论文。

2026-09-28另读[Breiman《Bagging Predictors》1994技术报告](https://www.stat.berkeley.edu/~breiman/bagging.pdf)及[scikit-learn集成学习说明](https://scikit-learn.org/stable/modules/ensemble.html#bagging-meta-estimator)：独立拟合重抽样版本再平均可降低部分不稳定学习器的方差，与逐轮拟合残差的提升法不同；效果依赖问题，不能从其非金融数据结果推出股票盈利。股票日记录并非独立，若接续检验需按完整时间块抽样，不能把同日几千只股票当几千次独立行情。当前市值基线实验已固定，不据此改动该实验。


2026-09-28核对[通达信官方函数列表](https://help.tdx.com.cn/gspt/docs/markdown/redword/functionlist.html)：HYBLOCK使用客户端设置的行业体系，未说明等同本库证监会历史分类，不能直接混接。INSUM支持辅助指标输出的板块求和／均值，可表达本轮等权主板参考；语法能力不证明客户端成员、历史行情和本库逐点一致。INSORT虽提供横向排序，文档未明确同值排名方式，本轮不据此另造未经核准的历史百分位输入。

同日又核对[Qlib处理器固定源码](https://github.com/microsoft/qlib/blob/a7d5a9b500de5df053e32abf00f6a679546636eb/qlib/data/dataset/processor.py)，CSZScoreNorm按datetime分组处理字段，提供截面标准化参考。新离散度版本只借用这个思路，以明确的总体方差和固定1基点分母下限计算两项相对强弱；不执行Qlib、复制其默认训练标签或宣称公开代码证明收益。源码与INSUM摘录保存在`tail_formula_cross_dispersion/source/`，SHA256分别`fd85dbba59b8ec9bdc478e3b1245177db368c9c7cabec5ec987c96c776805f0f`、`9516b0c9360cb9aeb2f182b74545dbc80871862f77284bd64884b9978ae00ab2`；已有Qlib MIT许可记录保留。

### 同日配对排序的可复用方法

微软原作者的[RankNet论文](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf)及[树模型综述](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf)使用同组对象的评分差与对数损失学习排序，并说明如何将成对导数汇总为逐对象梯度。可复用的是这种训练形式；论文研究搜索排序，没有证明A股次晨收益。项目另作固定浅树、日期等权、8遍覆盖式配对与叶位移曲率实现，不声称复现论文实验或使用完整LambdaMART。其部署只计算单股固定树，可沿用既有公式导出，无需实时横向排名。代码中的有限差分测试核对梯度、叶内配对抵消和同日常数偏移不改变训练损失；实际选股仍必须通过独立时期的绝对机会与风险检查。

2026-09-28核对[scikit-learn1.7.2梯度提升源码](https://github.com/scikit-learn/scikit-learn/blob/1.7.2/sklearn/ensemble/_gb.py)、[Huber损失](https://github.com/scikit-learn/scikit-learn/blob/1.7.2/sklearn/_loss/loss.py)及[加权分位实现](https://github.com/scikit-learn/scikit-learn/blob/1.7.2/sklearn/utils/stats.py)。三份官方tag原件与现有安装逐字节相同，仅复用现有库的稳健损失并另行验证，不据通用回归文档推断股票收益。后台Node请求超时后，同一浏览器任务成功读取原始源码；源码和指纹在`tail_formula_endpoint_robust/source/`。官方stable文档当时为1.9.1，执行定义以核准的1.7.2为准。


2026-09-28补读[Qiu 2022博士论文《Essays in Overnight Returns, Intraday Reversals, and Short-Selling Constraints in Chinese Stock Market》](https://unnc.globalimpact.cn/ws/portalfiles/portal/531719892/Thesis_Jiayan.pdf)，核对第2章方法与结论及第3章反转频率定义，并目视核准PDF第26／34／87页。第2章2009—2021样本使用早盘半小时成交量加权价定义一种开盘价，按月构造多空组合；低换手组相对高换手组较好，但两组自身隔夜分量均负。第3章关注过去反转频率相对长期均值的异常程度，并检验下一月收益。可复用的是区分绝对收益与组间差、区分频率水平与自身变化的思路；不能外推为2024—2025尾盘做多至次晨的盈利证据。已有20日高开低走频率及隔夜成分在本仓库检验过，不按新论文重新包装或重复实验。固定原件SHA256为`ab7f4335a7c58e3a95eac57a2b8f558e19fabe9c4aafa92f36ece6e924357c4b`，读取收据为`49aa1ce08193351634b83365f1b419a46ed080626ea056d9fa52e0f7ccaa2024`；没有取得机构账户身份或接入新的未来行情。


2026-09-28核对[NYU V-Lab Historical ILLIQ实现说明](https://vlab.stern.nyu.edu/docs/liquidity/ILLIQ-HIST)。其单日度量为绝对收益除以成交金额，平台将金额缩放为一亿美元，历史指标取22交易日平均。可参考的是价格变动与成交额关系的度量；不能据此声称某笔交易的真实冲击、资金净流入、主力身份或A股次晨盈利。原2002论文未读取，本条只据V-Lab自己的公开实现文档；改为尾段分钟指标将是新的改编试验，需另行固定并独立检验。文档原文及来源收据`2aa9e4dc2a07ff79be4d34523a154c6e31edc4e144f5f77e8265c65e4a0b7e33`保存在忽略目录，未拉取新行情。

2026-09-29核对上交所披露的[富国证券公司ETF 2023-11-29招募说明书](https://www.sse.com.cn/disclosure/fund/announcement/c/new/2023-11-29/515850_20231129_5MFZ.pdf)及[富国银行ETF 2022-11-29招募说明书](https://www.sse.com.cn/disclosure/fund/announcement/c/new/2022-11-29/515280_20221129_7EAK.pdf)：标的分别是中证全指证券公司指数399975和中证银行指数399986。只据此确认历史证券身份，不使用基金净值替代指数，也不把行业分类或指数编制说明当作次晨盈利依据。先固定六个历史会话核查分时与独立日线；尚未判定新输入可用。

同日读取[AKShare指数接口源码](https://github.com/akfamily/akshare/blob/4771e6ea34f6139f226e2bf4fab88ce17579831d/akshare/index/index_stock_zh.py)的stock_zh_index_daily_em：可通过beg/end限制日线日期，sz市场对应0，字段顺序为日期、开、收、高、低。仅参考接口参数，未安装或执行外部代码；原件保存在指数探针source目录。搜索其文档时摘要附带了2026上证指数示例值，未导入行情文件或模型，后续只读取源代码和固定2024—2025日期响应；不据此把2026称为全未暴露。
