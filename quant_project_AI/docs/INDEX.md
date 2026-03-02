# 文档索引

本文档帮助你在 **Quant Framework（量化 + RAG）** 中快速找到对应说明。

---

## 阅读顺序建议

**初学者**：请优先阅读 [docs/BEGINNER_GUIDE.md](BEGINNER_GUIDE.md)，从环境准备、最小示例、数据→策略→回测的流程，到框架设计与优化，步步讲解。

1. **项目根 README.md**  
   框架概览、目录结构、快速开始与文档入口。

2. **docs/DESIGN.md**  
   **设计原则**：极速数据处理 + AI 融合双目标、文档即规范；建议在架构前先读，建立「为何这样设计」的共识。

3. **docs/ARCHITECTURE.md**  
   整体架构：数据层（结构化 + 非结构化）、策略层、回测、RAG 管道如何连贯工作；适合建立全局图景。

4. **docs/RAG.md**  
   RAG 模块的使用方式、配置、与量化融合的写法；需要接入新闻/研报或 LLM 时阅读。

5. **docs/OPTIMIZATION.md**  
   优化说明（易懂版、循环与热路径、极限优化、适用场景）；需要理解「为什么这样设计」或做二次开发时阅读。

6. **docs/COMPARISON.md**  
   与主流 RAG 框架的对比；选型或对外说明时参考。

7. **docs/QUANT_FRAMEWORK_COMPARISON.md**  
   与 GitHub 主流量化框架（Backtrader、backtesting.py、VectorBT、Qlib 等）的对比；**本框架的短板与改进计划**；客观定位（RAG+量化优势 vs 纯回测能力不足）。

---

## 按主题查找

| 主题 | 文档与位置 |
|------|------------|
| 框架整体设计、量化与 RAG 如何衔接 | README.md、docs/ARCHITECTURE.md |
| 数据层：行情、指标、缓存、RAG 上下文 | docs/ARCHITECTURE.md（数据层）、docs/RAG.md（RAG 部分） |
| 策略：on_bar、get_rag_context、回测时间一致 | docs/ARCHITECTURE.md（策略与回测）、examples/example_rag_with_backtest.py |
| **18 策略全览** | docs/STRATEGY_ARSENAL.md |
| **多时间框架融合回测**（trend_filter / consensus / primary） | `backtest_multi_tf()` API、examples/multi_tf_backtest_example.py |
| **多 TF 综合分析报告** | reports/multi_tf_fusion_analysis_v2.md |
| RAG：接入、分块、检索、重排、Prompt 模板 | docs/RAG.md |
| 优化理由、循环与热路径、极限优化、Rust 边界 | docs/OPTIMIZATION.md |
| 与 LangChain / LlamaIndex / Haystack 对比 | docs/COMPARISON.md |
| 与 Backtrader / VectorBT / Qlib 等量化框架对比 | docs/QUANT_FRAMEWORK_COMPARISON.md |
| **回测与实盘差异** | docs/BACKTEST_VS_LIVE.md |
| **最稳健的回测结果** | docs/ROBUST_BACKTEST.md、examples/example_robust_backtest.py |
| **实盘对接与延迟**（Broker 抽象、纸交易） | docs/LIVE_TRADING_AND_LATENCY.md |
| **执行诊断闭环** | docs/EXECUTION_DIAGNOSTICS.md、examples/example_execution_diagnostics.py |
| **Freqtrade 回测与实盘差异** | docs/FREQTRADE_BACKTEST_LIVE_ANALYSIS.md |
| **顶级量化基金框架 vs 本框架** | docs/INSTITUTIONAL_QUANT_VS_FRAMEWORK.md |
| **分钟频支持评估** | docs/FREQUENCY_SUPPORT.md |
| 性能与数据结构（Pandas vs NumPy、热路径优化理由） | docs/PERFORMANCE.md |
| Polars 与 Pandas 对比及本框架选用与改造 | docs/POLARS_VS_PANDAS.md |
| 数据文件极速 I/O（Binary Mmap / Arrow IPC） | docs/IO_FAST.md |
| 项目全面审核（已修问题、待改进、建议） | docs/AUDIT.md |
| **设计原则（极速数据 + AI 融合、文档即规范）** | docs/DESIGN.md |
| **初学者完全指南** | docs/BEGINNER_GUIDE.md |
| **项目知识库（Python 概念 + 结构设计）** | docs/PROJECT_KNOWLEDGE.md |
| **10 层抗过拟合回测报告（最新）** | docs/COMPREHENSIVE_BACKTEST_ANALYSIS_V3.md |

**极速数据路径**：`DataManager(fast_io=True)` → 读 binary/arrow 优先；可选 `CacheManager`；回测热路径见 PERFORMANCE.md。  
**AI 融合路径**：`RagContextProvider` 注入策略 → `get_rag_context(query, symbol, as_of_date=current_date)`；详见 RAG.md。

---

## 代码入口

- **量化**：`quant_framework.data`（DataManager）、`quant_framework.strategy`、`quant_framework.backtest`、`quant_framework.analysis`、`quant_framework.visualization`  
- **RAG**：`quant_framework.rag`（RAGPipeline、RAGConfig、Document、format_prompt 等）  
- **融合**：`quant_framework.data.RagContextProvider`，在策略中通过 `rag_provider` 与 `get_rag_context(..., as_of_date=current_date)` 使用。
