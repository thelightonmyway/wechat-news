# 发表2天后，Science论文“中国光伏扩张政策降低鸟类多样性”统计结果被质疑

⬇️⬇️⬇️ 我们之前转发文章的评论区也是一边倒的态度，被科学态度严谨的粉丝们喷成筛子了![](./images/c1df59bf-e35c-4e09-a256-8ec5cabba04b.jpg)

**✓ 简报**

8月21日，**《科学》**杂志在线发表了论文China’s solar expansion policy reduces bird diversity **(中国光伏扩张政策导致鸟类生物多样性下降)**。该研究在生态圈、经济圈等科研界引起巨大反响。近日，**国内学者对论文的统计结果的解释表达了关切（英文版已提交预印本)。**关切文章提到，原文中模型包含了县域固定效应和时间固定效应，**而PSI (光伏政策强度)的独立解释贡献仅占0.0007，而非原文Table 1的第1列提供模型R²为0.2715；**而原文并没有提供与PSI的相关的Within R²为0.0007。因此，学者认为，该论文**过分强调了PSI系数的统计显著性和负向符号，却没有充分说明PSI相对于同一模型中其他变量而言非常小的事实**。

**以下文章来源于：**

原文标题：《关于Science论文“中国光伏扩张政策降低鸟类多样性”提供的统计结果的关切》

作者：赖江山、张霜等

Science最新发表“China’s solar expansion policy reduces bird diversity”（Zhang et al. 2026）论文引起国内极大关注。这一主题对于生物多样性保护和可再生能源政策而言具有潜在的意义。该研究使用光伏政策严格程度指数（Photovoltaic Policy Stringency Index, PSI）量化中国的光伏扩张政策，该指数是省级、市级和县级政策文件的加权总和。然而，我们对该研究核心结论所依据的统计解释存在一些担忧，特别是对Table 1提供信息有严重关切。主要有以下三点：

**1.** **将****“自****变量****”****与****“****控制变量****”****进行区分在统计学上具有误导性**

在表1中，PSI被描述为“自变量（independent variable）”，而其他9个变量被描述为“控制变量（controls）”。这种术语容易使读者认为模型主要估计PSI的影响，而其他变量只是控制变量。

在文章中，作者有这样的表述：

“We estimated the relationship between policy stringency and bird diversity using a high-dimensional, two-way fixed-effects regression model. This approach isolates the policy effect by controlling for time-invariant county characteristics and common temporal shocks, alongside time-varying meteorological, geographic, and socioeconomic covariates (e.g., temperature, wind speed, population density, bird-watching duration, carbon emissions, and land cover proportions).”

利用作者提供的原始数据，我们使用R里面fixest包函数feols()进行了“high-dimensional, two-way fixed-effects”进行全模型复现，所得结果与原文中Table 1第三列结果完全一致。

此模型表达式为：

ShannonBDᵢₜ = β₁PSIᵢₜ + β₂Tempᵢₜ + β₃Windᵢₜ + β₄Popᵢₜ + β₅Durationᵢₜ + β₆Carbonᵢₜ + β₇Waterᵢₜ + β₈Greenᵢₜ + β₉Farmᵢₜ + β₁₀Grassᵢₜ + αᵢ + γₜ + εᵢₜ

其中αᵢ表示县域固定效应，γₜ表示时间固定效应。

从统计建模角度来看，PSI和其他9个变量在统计上处于相同层级，均属于解释变量。不存在某一个变量天然是“independent variable”，而其他变量天然是“controls”的情况。这种区别非常重要，因为目前的术语使用可能会导致读者认为，该模型是在剔除其他变量影响之后，量化了PSI某种特殊性的贡献。然而，统计模型本身并不支持这样的解释。所有10个变量均在同一个回归框架中同时进行估，并且受到相同固定效应的控制。

**2.** **表****1****提供****R****2****具有误导性**

一个更为根本的问题在于表1中报告的决定系数（R²）。R²的呈现方式可能会使读者认为，它代表了所列解释变量的解释能力（例如第1列只有PSI变量，所以目前这一列所列0.2715为容易被认为PSI的解释率）。然而，对于该研究采用的high-dimensional, two-way fixed-effects regression model而言，这种解释是不恰当的。目前拟合模型包含县域固定效应和时间固定效应：∣county+year 。因此，报告所提供的R²，并非仅仅来源于解释变量，很大一部分是来county+year固定效应本身。这一点尤其重要，因为固定效应模型的目的，是在控制这些固定效应之后，利用剩余的变异来估计变量之间的关联。因此，在评估解释变量的解释贡献时，相关的指标应该是R语言 feols() 得到 “Within R²”，而不是“Adj.R²” （目前作者提供是模型的Adj.R²）。

     Within R²可以理解为：在去除相关固定效应所解释的变异之后，由解释变量所解释的比例。因此，它与评估PSI及其解释变量能够解释多少变异更加直接相关。例如，在表1第1列中，如果模型仅包含PSI这一时间变化解释变量，同时包含县域固定效应和时间固定效应，那么PSI的解释贡献应该使用Within R²进行评价。在我们的复现分析中，该数值约为：0.0007。这么小的决定系数，虽然跟大样本量有关系，但是否有生态意义，值得商榷。而原文Table 1的第1列故意提供模型R²为0.2715, 而没有提供与PSI的相关的Within R²的0.0007，似乎作者故意掩盖PSI解释能力微小的事实。

**3.** **与****Duration****指标****相比，****PSI****的效应非常小****，几乎可以忽略。**

作者在原文写到“Specifically, a one standard-deviation increase in policy stringency corresponds to a 2.10% reduction in the Shannon index (β = −0.0125, SE = 0.0037, P < 0.01).”

相比之下，Duration这个变量在同一模型框架中系数为：βDuration = 0.1638；SE = 0.0057；P<2.2 × 10⁻¹⁶。因此，与PSI相比，Duration具有显著更大的回归系数绝对值，p值更小。

      当然，由于不同变量的量纲不同，直接比较回归系数并不能直观地反映两个变量相对解释重要性的差异。因此，更具有可比性的比较方式应当基于单变量模型Within R²。刚才算的PSI对应的Within R²为 **0.0007**，当采用相同的方法计算Duration**的****Within R²**数值为：0.038838

0.038838 / 0.0007 ≈ 55.48

Duration对应的解释率约为PSI的55倍，也就是PSI对鸟类多样性影响程度还不及Duration影响的2%。当然，这个单变量**Within R²**的并未充分考虑解释变量之间的相关性。因此，更具体的比较单个变量独自贡献，是比较完整模型的Within R²与删除该变量后的模型的Within R²之间的差异。

对于某一个解释变量 X，我们将其对Within R²的独自贡献定义为：

ΔR²within,X = R²within,full − R²within,−X

该指标表示：在控制其他解释变量以及固定效应的条件下，与变量X唯一相关的解释能力。

采用这一方法，我们得到：

ΔR²within,PSI = 0.00046

ΔR²within,Duration = 0.038567

因此，两者的比值为：

0.038567/0.00046=83.84

因此， Duration所贡献的独自解释率约为PSI的**84****倍**。因此，PSI的解释能力对于Duration来说几乎可以忽略。

      对于鸟类Shannon diversity对Duration的解释尤其重要，因为该变量表示每位观鸟者观鸟持续时间。Duration本质上是一个**观测努力（****observation-effort****）变量**。更长的观察时间可以提高发现额外物种的概率，因此可能与观测到的鸟类Shannon多样性存在较强的关联。而PSI的变量相对Duration影响，几乎可以忽略不计。这个也可以理解，鸟类多样性还是取决于观察强度，而跟PSI关系并不大。

结论：

这篇Science论文标题和核心结论为：“China’s solar expansion policy reduces bird diversity.”我们的担忧在于，该论文过分强调了PSI系数的统计显著性和负向符号，却没有充分说明PSI相对于同一模型中其他变量而言非常小的事实。目前表1的呈现方式可能会高估光伏政策严格程度的解释重要性，同时故意忽略同一模型中其他变量所具有的不是一个数量级更大的贡献。

因此，有必要重新评估现有回归结果是否足以支持论文目前所强调的核心结论。

我们的这个英文comments已经提交给bioRixv. 请大家提出提供更多的讨论！
