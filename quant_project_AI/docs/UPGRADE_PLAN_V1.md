# Quant Framework 升级优化计划 v1.0

> **生成日期**: 2026-02-07
> **调研范围**: 2025 年 6 月 — 2026 年 2 月顶级量化研究论文、框架发展、行业实践
> **分析方法**: 前沿研究 × 当前框架能力矩阵 × 5 轮深度推演

---

## 一、调研总结：过去半年量化领域关键进展

### 1.1 AI + 量化融合（最大趋势）

| 方向 | 代表论文/系统 | 核心方法 | 效果 |
|------|--------------|----------|------|
| 混合 AI 交易系统 | arXiv 2601.19504 (2026) | FinBERT 情感 + XGBoost + EMA/MACD + 动态 regime filtering | 24 个月 135.49% 收益，跑赢 S&P 500 |
| LLM Alpha 发现 | FactorMiner (2026), Alpha-R1 (2025) | LLM 自动生成公式化 alpha + RL 评估/筛选 | 解决因子冗余"红海"问题 |
| 集成 RL 期货交易 | FineFT (2025) | 多 Q-learner 集成 + VAE 能力边界 + 风险感知选策略 | 降低 40% 风险，保持盈利 |
| TFT 时序预测 | HAELT (2025), TFT-GNN (2025) | Transformer + ResNet/GNN 混合架构 | 逐 bar 多空分类 F1 > 0.7 |
| 行为金融 DRL | Nature Sci.Rep. (2026) | 损失厌恶 + 过度自信融入 Actor-Critic | 加密+股票风险调整收益提升 |

### 1.2 市场微观结构与执行

| 方向 | 核心进展 |
|------|----------|
| 订单流预测 | Hawkes 过程建模 OFI 延迟依赖，因果诊断替代简单相关 |
| LOB 聚类 | ClusterLOB: K-means++ 分离方向性/做市/机会型参与者 |
| 最优执行 | Almgren-Chriss 框架 + 混合 TWAP-VWAP + ML 自适应分片 |
| DeFi 路由 | MEV-aware 执行 + 多流动性源分片降滑点 |

### 1.3 风险管理演进

| 方向 | 核心进展 |
|------|----------|
| 深度尾部对冲 | 深度神经网络 + CVaR 最小化，考虑交易成本和流动性 |
| 峰度风险平价 | 用峰度代替波动率做因子级风险分配（KFRP） |
| 声明式风险预算 | 深度声明式网络自动风险预算，超越传统风险平价 |
| 高盛实践 | 尾部对冲的核心价值 = 允许核心仓位增加风险敞口 |

### 1.4 Regime 检测与自适应

| 方向 | 核心进展 |
|------|----------|
| 自适应层级 HMM | 牛市/动荡/熊市三态检测 + VIX 联动 (S&P 500 2000-2024) |
| 集成 HMM + XGBoost | 多模型投票替代单一 HMM，解决初始化敏感和过拟合 |
| GMM 宏观 Regime | 高斯混合模型连接可观察宏观变量与隐含经济状态 |
| VIX 新常态 | VIX 地板从 12 升至 15+，频繁触及 30，波动率结构性走高 |

### 1.5 跨资产与另类数据

| 方向 | 核心进展 |
|------|----------|
| 相关性 regime 突变 | 2021 年后股债相关性大幅升高，通胀是核心驱动 |
| 波动率曲面交易 | 深度对冲 + 隐含波动率曲面建模，超越传统 delta-gamma |
| 加密情感分析 | 加密专用 LLM + 影响力加权 + 500ms 内执行，分类准确率 95%+ |
| 链上分析 | NVT/MVRV/交易所流入等 on-chain 指标融入信号 |

---

## 二、当前框架能力矩阵（现状审计）

### 2.1 已有能力（强项）

| 模块 | 能力 | 成熟度 |
|------|------|--------|
| **策略内核** | 18 个 Numba 编译策略，趋势/动量/均值回归/regime/集成全覆盖 | ★★★★★ |
| **回测引擎** | 单 TF + 多 TF 融合，O(n) 内核，~190K combo/s | ★★★★★ |
| **抗过拟合** | 10 层稳健扫描，Walk-Forward + CPCV + DSR + Monte Carlo | ★★★★★ |
| **成本模型** | 买卖分离佣金/滑点，融券借入费率，杠杆资金费用 | ★★★★☆ |
| **实盘适配** | KernelAdapter + MultiTFAdapter，纸交易，价格源 | ★★★★☆ |
| **数据 I/O** | 多层存储（Binary/Arrow/Parquet），极速读取 | ★★★★☆ |
| **RAG 模块** | 非结构化文本接入策略决策 | ★★★☆☆ |
| **Alpha 因子** | OrderFlow (OFI/VPIN)、CrossAsset、Volatility、FeatureEvaluator | ★★★☆☆ |

### 2.2 关键缺口（弱项）

| 缺口 | 影响 | 紧迫度 |
|------|------|--------|
| 无 ML 模型推理管线 | 无法利用 XGBoost/DL 信号 | 🔴 极高 |
| 无 Regime 检测模块 | 策略无法适应牛/熊/震荡转换 | 🔴 极高 |
| 无组合优化引擎 | 仅等权分配，无法做风险平价/均值方差 | 🔴 极高 |
| 无高级风控 | 无 VaR/CVaR/Kelly/相关性约束 | 🟡 高 |
| 无执行算法 | 无 TWAP/VWAP，实盘大单冲击无法控制 | 🟡 高 |
| 无跨资产策略 | 无配对交易/统计套利/价差交易 | 🟡 高 |
| 无实盘 Broker | 仅纸交易，无法接入 IBKR/Alpaca/Binance 实盘 | 🟡 高 |
| 无深度学习推理 | 无 Transformer/LSTM 信号层 | 🟢 中 |
| 无另类数据管线 | 情感/链上数据无标准化接入 | 🟢 中 |
| 无 RL 决策层 | 无强化学习组合管理 | ⚪ 探索 |

---

## 三、五轮深度推演

### 第一轮：缺口 × 前沿研究的交叉影响

将 §1 的前沿趋势与 §2.2 的缺口交叉，找出**最高杠杆点**——升级一个模块能同时补多个缺口：

```
                   Regime 检测 ──┬── 策略自适应切换
                                ├── 风控 regime 感知
                                └── 组合动态再平衡
                   
                   ML 管线    ──┬── XGBoost/LightGBM alpha
                                ├── Transformer 信号
                                └── RL 决策层基础设施
                   
                   组合优化   ──┬── 风险平价/均值方差
                                ├── 跨资产相关性控制
                                └── 动态再平衡
```

**结论**：Regime 检测和 ML 管线是两个核心枢纽，先建好它们，后续升级才有基础。

### 第二轮：架构影响最小化

当前架构分层清晰（Data → Strategy → Backtest → Analysis → Live），每次升级应：
- 新增模块而非修改现有模块
- 遵循 Protocol/ABC 接口约束
- 保持 Numba 内核的纯数值特性不变
- 新信号源通过 adapter 模式注入

最佳架构映射：

| 升级 | 新模块位置 | 依赖方向 |
|------|-----------|----------|
| Regime 检测 | `quant_framework/regime/` | Data → Regime → Strategy/Risk |
| ML 管线 | `quant_framework/model/` (扩展) | Data → Model → Strategy |
| 组合优化 | `quant_framework/portfolio/` (新) | Backtest → Portfolio → Live |
| 高级风控 | `quant_framework/risk/` (新) | 横切：Strategy, Backtest, Live |
| 执行算法 | `live/execution/` | Broker → Execution → Live |
| 跨资产策略 | 新 kernel + `strategy/` | Data → Strategy → Backtest |

### 第三轮：性能预算与约束

| 组件 | 延迟预算 | 实现约束 |
|------|----------|----------|
| Regime 检测 | < 5ms/bar | Numba/NumPy，在线更新（incremental HMM） |
| ML 推理 | < 10ms/prediction | ONNX Runtime 或原生 NumPy 推理 |
| 组合优化 | < 1s/100 资产 | SciPy SLSQP 或 cvxpy |
| VaR/CVaR | < 1ms/bar | 历史分位数或参数法，NumPy 向量化 |
| TWAP/VWAP 分片 | < 1ms/decision | 纯 Python，异步调度 |
| Transformer 推理 | < 50ms/bar | ONNX 预编译，离线训练 + 在线推理 |

### 第四轮：ROI 排序与依赖关系

评估每项升级的投入产出比（工程量 vs 收益）并建立依赖关系图：

```
Phase 1: 基础层（无依赖，收益最大）
  ├── [A] Regime 检测模块          工程量: ★★☆  收益: ★★★★★
  ├── [B] ML 模型管线              工程量: ★★★  收益: ★★★★★
  └── [C] 高级风控引擎            工程量: ★★☆  收益: ★★★★☆

Phase 2: 信号增强（依赖 Phase 1）
  ├── [D] Transformer 信号层      工程量: ★★★★ 收益: ★★★★☆  ← 依赖 [B]
  ├── [E] 跨资产策略内核          工程量: ★★☆  收益: ★★★☆☆
  └── [F] 另类数据管线            工程量: ★★☆  收益: ★★★☆☆  ← 依赖 [B]

Phase 3: 组合智能（依赖 Phase 1+2）
  ├── [G] 组合优化引擎            工程量: ★★★  收益: ★★★★★  ← 依赖 [A][C]
  └── [H] 动态再平衡              工程量: ★★☆  收益: ★★★★☆  ← 依赖 [G]

Phase 4: 执行卓越（依赖 Phase 1-3）
  ├── [I] 执行算法 (TWAP/VWAP)    工程量: ★★★  收益: ★★★★☆
  └── [J] 实盘 Broker 集成        工程量: ★★★  收益: ★★★★★

Phase 5: 前沿探索（依赖全部）
  ├── [K] RL 组合管理              工程量: ★★★★★ 收益: ★★★☆☆  ← 依赖 [B][G]
  └── [L] LLM Alpha 发现代理      工程量: ★★★★ 收益: ★★★☆☆  ← 依赖 [B][F]
```

### 第五轮：风险评估与质量标准

| 风险 | 缓解策略 |
|------|----------|
| ML 过拟合 | 强制使用现有 10 层抗过拟合体系验证所有 ML 信号 |
| Regime 检测滞后 | 多模型投票 + 在线 Bayesian 更新 |
| 组合优化不稳定 | 正则化 + 收缩估计 + Ledoit-Wolf 协方差 |
| 执行滑点偏离 | 引入 Almgren-Chriss 冲击模型到回测成本 |
| 深度学习过重 | ONNX 推理与训练分离，推理部署尽量轻量 |
| 架构膨胀 | 每个新模块通过 Protocol 接口解耦，可选依赖 |

**质量红线**（每项升级必须满足）：
1. 回测-实盘一致性：新模块在两条路径上的行为必须数值一致
2. 回归测试：每个新模块至少 10 个单元测试 + frozen regression
3. 性能基准：不得拖慢现有内核 >1%
4. 文档同步：PROJECT_KNOWLEDGE.md / BEGINNER_GUIDE.md 同步更新

---

## 四、分阶段升级详细计划

### Phase 1：核心基础层

#### [A] 市场 Regime 检测模块

**目标**：实时检测市场状态（牛市/震荡/熊市/高波动/低波动），驱动策略切换和风控调整。

**技术方案**：

| 组件 | 方法 | 实现 |
|------|------|------|
| 统计 Regime | 3-state HMM（收益率 + 波动率） | hmmlearn + Numba 在线更新 |
| 波动率 Regime | GARCH(1,1) 在线估计 + 阈值分类 | arch 库 + NumPy 滚动 |
| 趋势 Regime | MA 斜率 + ADX + 价格位置评分 | 纯 Numba |
| 集成投票 | 多检测器加权投票 | NumPy |
| 变点检测 | CUSUM / Bayesian Online CPD | 增量算法 |

**新文件结构**：

```
quant_framework/regime/
├── __init__.py          # RegimeDetector Protocol, RegimeState enum
├── hmm_detector.py      # HMM-based regime detection
├── volatility_regime.py # GARCH + ATR regime classification
├── trend_regime.py      # Trend strength scoring (Numba)
├── ensemble.py          # Multi-detector voting
└── changepoint.py       # CUSUM / Bayesian change-point
```

**与现有模块的集成**：
- `BacktestConfig` 新增 `regime_aware: bool` 选项
- `KernelAdapter` 可选接入 `RegimeDetector`，根据 regime 调整参数
- `RiskGate` 根据 regime 收紧/放松风控阈值

**验收标准**：
- S&P 500 2000-2024 回测中 regime 分类准确率 > 70%（与 NBER 衰退期对比）
- 在线检测延迟 < 5ms/bar
- Regime-aware VolRegime 策略 Sharpe 提升 > 0.1

---

#### [B] ML 模型管线

**目标**：提供从特征工程到模型训练、验证、推理的标准化管线，让 ML 信号无缝接入策略层。

**技术方案**：

```
quant_framework/model/
├── __init__.py           # ModelRegistry, FeaturePipeline Protocol
├── feature_pipeline.py   # 标准化特征生成 (技术指标 + alpha + regime)
├── model_registry.py     # 注册/版本化/加载模型
├── trainers/
│   ├── tree_trainer.py   # XGBoost / LightGBM 训练
│   ├── nn_trainer.py     # PyTorch 训练 (TFT, LSTM)
│   └── onnx_export.py    # 导出 ONNX 推理模型
├── inference/
│   ├── onnx_predictor.py # ONNX Runtime 快速推理
│   └── numpy_predictor.py# 纯 NumPy 树模型推理 (无依赖)
└── validation/
    ├── purged_cv.py      # Purged K-Fold + Embargo
    └── feature_importance.py # SHAP / permutation importance
```

**FeaturePipeline 设计**（关键接口）：

```python
class FeaturePipeline(Protocol):
    def compute(self, ohlcv: pd.DataFrame, regime: Optional[RegimeState] = None) -> pd.DataFrame:
        """Return feature matrix with columns aligned to bar timestamps."""
        ...

class ModelPredictor(Protocol):
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return signal array: +1 long, 0 flat, -1 short."""
        ...
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return probability array for confidence-weighted sizing."""
        ...
```

**标准特征集**（约 80 个特征）：

| 类别 | 特征 |
|------|------|
| 价格 | 多周期收益率、log return、close/EMA 比值 |
| 动量 | RSI, MACD, ROC, Stochastic, Williams %R |
| 波动 | ATR, Bollinger width, Yang-Zhang vol, Parkinson |
| 成交量 | OBV, VWAP deviation, volume ratio, OFI |
| 微观结构 | VPIN, trade imbalance, bid-ask spread |
| Regime | HMM state, vol regime, trend score |
| 跨资产 | BTC-ETH 相关性, DXY 变化, VIX proxy |

**验收标准**：
- FeaturePipeline 计算 1000 bars × 80 features < 50ms
- ONNX 推理 < 10ms/prediction
- XGBoost 模型在 BTC 1h 上 OOS 准确率 > 55%（三分类）
- 完整 purged CV 验证通过

---

#### [C] 高级风控引擎

**目标**：提供组合级和单标的级的实时风控，支持 VaR/CVaR、Kelly sizing、相关性约束和动态回撤控制。

**技术方案**：

```
quant_framework/risk/
├── __init__.py           # RiskEngine Protocol
├── var_calculator.py     # 历史 VaR, 参数 VaR, CVaR (ES)
├── kelly.py              # Kelly criterion + half-Kelly 仓位
├── correlation_monitor.py# 滚动相关性矩阵 + 集中度预警
├── drawdown_control.py   # 动态回撤限制 (CPPI 风格)
└── position_sizer.py     # 统一仓位计算 (risk%, Kelly, volatility-target)
```

**核心功能**：

| 功能 | 方法 | 用途 |
|------|------|------|
| 实时 VaR | 历史分位数 (rolling 250 bar) | 每 bar 风险度量 |
| CVaR (ES) | 尾部均值 (超过 VaR 的均值) | 尾部风险控制 |
| Kelly 仓位 | f* = (bp - q) / b，带 half-Kelly 保守 | 最优仓位大小 |
| 波动率目标 | σ_target / σ_realized × leverage | 恒定风险敞口 |
| 相关性限制 | 滚动协方差矩阵 + 最大相关性阈值 | 避免过度集中 |
| 动态回撤限制 | CPPI: 允许敞口 = m × (净值 - floor) | 回撤到阈值自动降仓 |

**与现有模块集成**：
- `BacktestConfig` 新增 `risk_engine: Optional[RiskEngine]`
- `backtest_portfolio()` 在每次调仓前询问 `RiskEngine`
- `live/risk.py` 的 `RiskGate` 可选代理到 `RiskEngine`

**验收标准**：
- VaR 计算 < 1ms/bar（250 bar 窗口）
- Kelly sizing 在 18 策略 × 4 资产上一致降低极端亏损 > 20%
- 动态回撤控制将最大 DD 降低 > 30%（vs 固定仓位）

---

### Phase 2：信号增强层

#### [D] Transformer 信号层

**目标**：引入 Temporal Fusion Transformer (TFT) 或轻量 Attention 模型，为多步预测和特征重要性提供深度学习信号。

**技术方案**：
- 离线训练 TFT 模型（PyTorch / PyTorch Lightning）
- 导出 ONNX 进行在线推理
- 输入：FeaturePipeline 生成的 80+ 特征 + regime 状态
- 输出：T+1/T+3/T+5 收益率方向概率
- 注意力权重用于特征归因和可解释性

**关键设计决策**：
- 训练与推理完全分离，推理端无 PyTorch 依赖
- 支持 per-asset 独立模型和 universe-wide 共享模型
- Attention 权重导出为可视化数据

**验收标准**：
- TFT 推理 < 50ms/bar（ONNX Runtime）
- 在 BTC/ETH 1h 数据上方向准确率 > 55%
- 与现有 18 内核的 ensemble 收益优于纯内核 > 10%

---

#### [E] 跨资产策略内核

**目标**：新增配对交易、统计套利、价差交易内核。

**新内核**：

| 内核 | 逻辑 | 输入 |
|------|------|------|
| `PairSpread` | Johansen 协整 + z-score 均值回归 | 两资产 OHLC |
| `StatArb` | PCA 残差 + Ornstein-Uhlenbeck 均值回归 | 多资产收益率 |
| `LeadLag` | 领先-滞后关系 + Granger 因果 + 滚动相关 | 两资产 OHLC |
| `BetaHedge` | β 中性头寸 + α 提取 | 标的 + 基准 OHLC |

**架构考量**：
- 跨资产内核签名与现有 `_eq_*` 不同（多输入数组），需新的 Registry
- `CROSS_ASSET_REGISTRY` 独立于 `KERNEL_REGISTRY`
- `backtest_cross_asset()` 新 API

**验收标准**：
- PairSpread 在 BTC-ETH 上 Sharpe > 0.8
- 回测速度 > 50K combo/s

---

#### [F] 另类数据管线

**目标**：标准化接入情感数据、链上数据、宏观数据，作为特征输入 ML 管线。

**技术方案**：

```
quant_framework/data/alternative/
├── __init__.py
├── sentiment_adapter.py   # Twitter/Reddit 情感分数 (LLM/VADER)
├── onchain_adapter.py     # Glassnode/Dune 链上指标
├── macro_adapter.py       # FRED 宏观数据 (利率, CPI, PMI)
└── news_adapter.py        # 新闻 NLP + FinBERT 情感
```

**与 RAG 模块的关系**：
- RAG 处理非结构化文本 → 上下文注入
- 另类数据管线处理结构化信号 → 特征矩阵
- 两者互补，不重复

**验收标准**：
- 至少接入 3 个数据源（情感 + 链上 + 宏观）
- 数据对齐到 bar 级别时间戳
- 与 FeaturePipeline 无缝集成

---

### Phase 3：组合智能层

#### [G] 组合优化引擎

**目标**：实现机构级组合优化，支持均值方差、风险平价、Black-Litterman、最小方差等方法。

**技术方案**：

```
quant_framework/portfolio/
├── __init__.py             # PortfolioOptimizer Protocol
├── mean_variance.py        # Markowitz MVO + robust estimation
├── risk_parity.py          # 等风险贡献 (ERC)
├── min_variance.py         # 全局最小方差
├── black_litterman.py      # BL with views (来自 ML 信号)
├── hierarchical_rp.py      # 层次风险平价 (HRP, Lopez de Prado)
├── covariance/
│   ├── ledoit_wolf.py      # 收缩估计
│   ├── ewma.py             # 指数加权
│   └── denoised.py         # 随机矩阵理论去噪
└── constraints.py          # 约束集: 权重/行业/相关性/换手率
```

**核心接口**：

```python
class PortfolioOptimizer(Protocol):
    def optimize(
        self,
        expected_returns: np.ndarray,     # N assets
        covariance: np.ndarray,           # N×N
        constraints: PortfolioConstraints,
        current_weights: Optional[np.ndarray] = None,
    ) -> OptimalWeights:
        ...
```

**Black-Litterman 集成**（最有差异化的功能）：
- ML 模型输出收益率预测 → 转化为 "views"
- BL 结合市场均衡收益与 views，输出后验预期收益
- 避免纯 MVO 对输入敏感的问题

**验收标准**：
- 100 资产优化 < 1s
- HRP 在 S&P 500 子集上 OOS Sharpe > 0.6
- BL + ML views 优于等权 > 15% 年化

---

#### [H] 动态再平衡

**目标**：支持日历再平衡、阈值触发再平衡、成本感知再平衡。

**方法**：
- 固定频率（日/周/月）
- 偏离阈值（权重偏离目标 > x% 触发）
- 成本最优（No-trade region: 考虑交易成本的最优再平衡带）
- Regime-triggered（regime 切换时强制再平衡）

**验收标准**：
- 成本感知再平衡降低换手率 > 30% vs 固定频率
- 支持回测和实盘两条路径

---

### Phase 4：执行卓越层

#### [I] 执行算法

**目标**：实现 TWAP、VWAP、自适应分片算法，控制大单市场冲击。

**技术方案**：

```
quant_framework/live/execution/
├── __init__.py          # ExecutionAlgo Protocol
├── twap.py              # 等时间分片
├── vwap.py              # 按成交量分布分片
├── adaptive.py          # ML-adaptive (根据波动率/价差调整)
├── impact_model.py      # Almgren-Chriss 市场冲击估计
└── scheduler.py         # 异步分片调度器
```

**Almgren-Chriss 集成**：
- 在回测中引入市场冲击模型：impact = η × (quantity/ADV)^γ
- 回测成本 = commission + slippage + market_impact
- 缩小回测-实盘差距

**验收标准**：
- TWAP/VWAP 分片延迟 < 1ms/decision
- 市场冲击模型在回测中降低实盘滑点差异 > 50%

---

#### [J] 实盘 Broker 集成

**目标**：接入主流交易所/经纪商 API，实现真实执行。

**优先级**：

| Broker | 资产类别 | 优先级 |
|--------|---------|--------|
| Binance | 加密现货+合约 | P0 |
| Alpaca | 美股 | P1 |
| IBKR | 全球股票/期货/外汇 | P2 |

**架构**：继承现有 `Broker` Protocol，实现 `submit_order`, `cancel_order`, `get_positions`, `get_account_summary`。

**验收标准**：
- Binance Broker 通过 testnet 验证
- 订单提交到确认 < 200ms
- 完整的错误恢复和重连机制

---

### Phase 5：前沿探索层

#### [K] RL 组合管理

**目标**：用强化学习进行动态资产配置，学习最优仓位和换仓时机。

**技术方案**：
- 环境：OpenAI Gym 接口包装回测引擎
- 状态：特征矩阵 + 当前持仓 + regime
- 动作：目标权重向量
- 奖励：Sharpe-aware reward（收益 / 滚动波动率）
- 算法：PPO / SAC（连续动作空间）

**验收标准**：
- RL agent 在 OOS 上 Sharpe > 纯规则策略 0.2+
- 训练收敛 < 2 小时（单 GPU）

---

#### [L] LLM Alpha 发现代理

**目标**：用 LLM 自动生成公式化 alpha 因子，RL 评估筛选，持续扩充因子库。

**技术方案**（参考 FactorMiner / Alpha-R1）：
- LLM 生成候选 alpha 表达式
- 回测验证 IC/IR/周转率
- RL 排序筛选最优因子
- 去冗余（与现有因子库相关性 < 0.5）
- 自动加入 FeaturePipeline

**验收标准**：
- 自动发现 > 10 个 IC > 0.03 的独立 alpha
- 因子库增量更新不影响现有策略

---

## 五、实施路线图

```
2026 Q1 (3-4月)                      Phase 1: 核心基础
├── Month 1: [A] Regime 检测模块
├── Month 1-2: [B] ML 管线 (FeaturePipeline + ModelRegistry + Tree Trainer)
└── Month 2: [C] 高级风控引擎

2026 Q2 (5-6月)                      Phase 2: 信号增强
├── Month 3: [D] TFT 训练 + ONNX 推理
├── Month 3: [E] 跨资产内核 (PairSpread, StatArb)
└── Month 4: [F] 另类数据管线 (情感 + 链上)

2026 Q3 (7-8月)                      Phase 3: 组合智能
├── Month 5: [G] 组合优化 (MVO + Risk Parity + BL + HRP)
└── Month 6: [H] 动态再平衡

2026 Q4 (9-10月)                     Phase 4: 执行卓越
├── Month 7: [I] TWAP/VWAP + Impact Model
└── Month 8: [J] Binance/Alpaca Broker

2027 Q1 (11-12月)                    Phase 5: 前沿探索
├── Month 9: [K] RL 组合管理
└── Month 10: [L] LLM Alpha Agent
```

---

## 六、与竞品差距分析（升级前后）

### 升级前

| 维度 | 本框架 | VectorBT | Qlib | NautilusTrader |
|------|--------|----------|------|----------------|
| 回测速度 | ★★★★★ | ★★★★★ | ★★★ | ★★★ |
| 策略数量 | ★★★★★ (18) | ★★☆ (用户自写) | ★★★ | ★★★ |
| ML 集成 | ★☆ | ★☆ | ★★★★★ | ★★ |
| 组合优化 | ★☆ | ★★☆ | ★★★ | ★★ |
| 风控深度 | ★★☆ | ★☆ | ★★★ | ★★★★ |
| 实盘执行 | ★★☆ (纸盘) | ★☆ | ★★☆ | ★★★★★ |
| RAG/AI | ★★★★ | ☆ | ☆ | ☆ |

### 升级后（预期）

| 维度 | 本框架 | VectorBT | Qlib | NautilusTrader |
|------|--------|----------|------|----------------|
| 回测速度 | ★★★★★ | ★★★★★ | ★★★ | ★★★ |
| 策略数量 | ★★★★★ (22+) | ★★☆ | ★★★ | ★★★ |
| ML 集成 | ★★★★★ | ★☆ | ★★★★★ | ★★ |
| 组合优化 | ★★★★★ | ★★☆ | ★★★ | ★★ |
| 风控深度 | ★★★★★ | ★☆ | ★★★ | ★★★★ |
| 实盘执行 | ★★★★ | ★☆ | ★★☆ | ★★★★★ |
| RAG/AI | ★★★★★ | ☆ | ☆ | ☆ |
| Regime 感知 | ★★★★★ | ☆ | ★★☆ | ☆ |
| 另类数据 | ★★★★ | ☆ | ★★★ | ☆ |
| RL 决策 | ★★★★ | ☆ | ★★★ | ☆ |

**差异化定位**：唯一一个同时具备「极速 Numba 内核 + ML 管线 + RAG AI 融合 + Regime 感知 + 组合优化 + 实盘执行」的全栈量化框架。

---

## 七、风险与缓解

| 风险 | 等级 | 缓解 |
|------|------|------|
| ML 过拟合导致实盘亏损 | 🔴 高 | 强制 purged CV + DSR + MC，OOS 验证通过才上线 |
| 架构膨胀降低可维护性 | 🟡 中 | Protocol 接口解耦，每个模块可独立使用或禁用 |
| 依赖膨胀（PyTorch, cvxpy 等）| 🟡 中 | 核心功能仅依赖 NumPy/Numba，ML/优化为可选依赖 |
| Broker API 变更 | 🟡 中 | Adapter 模式隔离，API 变更只改一个文件 |
| 开发周期过长 | 🟡 中 | Phase 分期交付，每 phase 独立可用 |

---

## 八、依赖管理

### 核心依赖（必须）
- `numpy`, `numba`, `pandas` — 现有
- `scipy` — 优化器、统计检验

### 可选依赖（按需安装）
- `hmmlearn` — Phase 1 [A] Regime
- `xgboost`, `lightgbm` — Phase 1 [B] ML
- `onnxruntime` — Phase 2 [D] Transformer 推理
- `cvxpy` — Phase 3 [G] 组合优化
- `torch`, `pytorch-lightning` — Phase 2 [D] 训练端
- `stable-baselines3` — Phase 5 [K] RL

**原则**：`pip install quant_framework` 只安装核心依赖；`pip install quant_framework[ml]`、`pip install quant_framework[portfolio]` 等 extras 按需。

---

## 九、总结

本升级计划基于 11 个维度的前沿研究调研，经过 5 轮深度推演，规划了 12 个升级项，分 5 个阶段在 10 个月内交付。

**核心理念**：
1. **Regime 感知是枢纽** — 它同时改善策略、风控和组合三个层面
2. **ML 管线是基础设施** — 它是 Transformer、RL、LLM Alpha 等后续升级的前提
3. **保持 Numba 极速内核不动** — 新功能通过 Protocol 接口叠加，不侵入现有代码
4. **回测-实盘一致性是红线** — 每项新功能必须在两条路径上行为一致
5. **可选依赖，渐进采纳** — 用户可以只用 Phase 1 而不装 PyTorch

升级完成后，本框架将成为唯一同时具备「极速 Numba 回测 + ML/DL 信号 + RAG AI 融合 + 机构级组合优化 + Regime 自适应 + 实盘执行」的开源全栈量化框架，在 GitHub 量化生态中占据独特定位。
