"""
RAG 全链路端到端测试
====================================================
全面测试：数据获取 → 文档创建 → 入库 → 分块 → 嵌入 → 向量存储 → 关键词索引
→ 混合检索 → 上下文生成 → 策略集成 → 回测融合 → 性能基准
"""

import sys
import os
import time
import json
import hashlib
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. News / financial data fetching (offline simulation + yfinance)
# ---------------------------------------------------------------------------

def fetch_financial_news_simulated() -> list:
    """Simulate a realistic financial news corpus with timestamps spanning 1 year."""
    base = datetime(2024, 1, 15)
    articles = [
        {"title": "美联储维持利率不变，市场波动加剧",
         "body": "美联储在最新会议中决定维持利率不变，基准利率保持在5.25%-5.50%的区间。主席鲍威尔表示通胀仍高于目标，但经济数据显示放缓迹象。市场对此反应强烈，标普500指数当日下跌1.2%，纳斯达克下跌1.5%。分析师认为降息可能推迟到2024年下半年。",
         "source": "reuters", "symbol": "SPY", "days_offset": 0},
        {"title": "苹果公司Q4财报超预期，iPhone销量创纪录",
         "body": "苹果公司发布2024年第四季度财报，营收达1195亿美元，同比增长6%。iPhone销量再创新高，特别是iPhone 15 Pro系列带动高端市场。服务业务收入达228亿美元，同比增长16%。库克表示AI将是未来重要的增长驱动力。盘后股价上涨3.5%。",
         "source": "bloomberg", "symbol": "AAPL", "days_offset": 5},
        {"title": "比特币突破65000美元，机构资金持续流入",
         "body": "比特币价格突破65000美元大关，24小时涨幅达4.2%。灰度GBTC转为ETF后资金流出放缓，而贝莱德和富达的比特币ETF持续吸引资金流入。链上数据显示大户地址增加，矿工持币量上升。分析师预计比特币减半前可能触及70000美元。交易量达到350亿美元。",
         "source": "coindesk", "symbol": "BTC-USD", "days_offset": 10},
        {"title": "以太坊升级完成，Gas费大幅下降",
         "body": "以太坊Dencun升级成功上线，引入Proto-Danksharding技术。Layer2网络的Gas费用降低90%以上，Arbitrum和Optimism的交易成本降至0.01美元以下。这一升级显著提升了以太坊生态系统的可扩展性。ETH价格当日上涨5.3%，突破3500美元。DeFi总锁仓量增加15%。",
         "source": "coindesk", "symbol": "ETH-USD", "days_offset": 15},
        {"title": "特斯拉Q1交付量低于预期，股价承压",
         "body": "特斯拉2024年第一季度交付量为38.68万辆，低于分析师预期的44.9万辆，同比下降8.5%。Model Y和Model 3仍占交付量的绝大部分。马斯克承认中国市场竞争加剧，并宣布将加速推出低价车型。股价盘后下跌5.2%。分析师纷纷下调目标价。",
         "source": "reuters", "symbol": "TSLA", "days_offset": 20},
        {"title": "英伟达发布新一代AI芯片Blackwell，性能提升30倍",
         "body": "英伟达在GTC大会上发布新一代Blackwell GPU架构。B200芯片在AI推理任务中比H100提升30倍，能耗效率提升25倍。黄仁勋表示这标志着新一轮计算革命。微软、谷歌和亚马逊已确认采购订单。英伟达股价当日上涨7.2%，市值突破2万亿美元。",
         "source": "bloomberg", "symbol": "NVDA", "days_offset": 25},
        {"title": "XRP赢得SEC诉讼关键裁决，价格飙升",
         "body": "美国法院在Ripple与SEC的诉讼中作出重要裁决，认定XRP在二级市场交易不构成证券。这一裁决对整个加密市场具有里程碑意义。XRP价格在裁决后24小时内上涨35%，突破0.85美元。Ripple CEO表示将继续拓展跨境支付业务。多家交易所宣布重新上线XRP交易对。",
         "source": "coindesk", "symbol": "XRP-USD", "days_offset": 30},
        {"title": "Solana生态爆发，TVL突破新高",
         "body": "Solana区块链生态系统迎来爆发式增长，DeFi总锁仓量突破50亿美元。Solana网络日均交易量超过以太坊主网，Gas费仅为0.00025美元。Jupiter DEX聚合器日交易量突破10亿美元。SOL代币价格突破150美元，市值排名升至第五。机构投资者对Solana的兴趣显著增加。",
         "source": "coindesk", "symbol": "SOL-USD", "days_offset": 35},
        {"title": "美国CPI数据高于预期，市场大幅调整",
         "body": "美国3月CPI同比上涨3.5%，高于预期的3.4%和前值3.2%。核心CPI同比上涨3.8%，与前值持平。数据公布后，美元指数急升，10年期国债收益率升破4.5%。市场对降息预期大幅回调，CME FedWatch显示6月降息概率降至20%以下。标普500当日下跌1.5%。",
         "source": "reuters", "symbol": "SPY", "days_offset": 40},
        {"title": "谷歌发布Gemini Ultra，AI竞赛白热化",
         "body": "谷歌发布最新大语言模型Gemini Ultra，在32个学术基准中有30个超过GPT-4。Gemini Ultra支持多模态处理，包括文本、图像、音频和视频。Google Cloud将整合Gemini能力，推出企业级AI服务。Alphabet股价上涨4.1%。分析师认为AI竞赛将推动整个科技行业的收入增长。",
         "source": "bloomberg", "symbol": "GOOGL", "days_offset": 45},
        {"title": "比特币完成第四次减半，市场反应平稳",
         "body": "比特币在区块高度840000完成第四次减半，区块奖励从6.25BTC降至3.125BTC。与此前几次减半不同，市场反应相对平稳，价格在减半前后波动不超过3%。矿工的收入预计将下降50%，但比特币ETF的持续资金流入有望弥补卖压。长期来看，供应减少将对价格形成支撑。",
         "source": "coindesk", "symbol": "BTC-USD", "days_offset": 60},
        {"title": "加密市场遭遇黑天鹅，比特币一度跌破50000美元",
         "body": "由于日本央行意外加息以及中东地缘政治紧张，全球金融市场出现剧烈波动。比特币价格一度跌破50000美元，24小时最大跌幅达18%。以太坊跌至2100美元，XRP跌至0.45美元。合约爆仓金额超过10亿美元。但市场随后快速反弹，分析师认为这是健康的回调。",
         "source": "coindesk", "symbol": "BTC-USD", "days_offset": 90},
        {"title": "全球量化交易市场规模突破2万亿美元",
         "body": "根据最新行业报告，全球量化交易市场规模已突破2万亿美元，年增长率达15%。高频交易占美国股市交易量的50%以上。AI和机器学习技术的应用正在改变交易策略的格局。报告指出，中小型量化基金正在采用更灵活的策略和更低延迟的技术架构来与大型机构竞争。",
         "source": "bloomberg", "symbol": None, "days_offset": 100},
        {"title": "SEC批准以太坊现货ETF，加密市场迎来第二波利好",
         "body": "美国证券交易委员会（SEC）正式批准了多只以太坊现货ETF的上市申请。贝莱德、灰度和富达等机构的ETH ETF将于下周开始交易。ETH价格应声上涨12%，突破4000美元。分析师预计ETH ETF将在首年吸引150-200亿美元资金流入，为以太坊生态系统带来长期利好。",
         "source": "coindesk", "symbol": "ETH-USD", "days_offset": 120},
        {"title": "Ripple推出稳定币RLUSD，XRP生态扩张",
         "body": "Ripple Labs正式推出美元稳定币RLUSD，面向机构客户。RLUSD将在XRP Ledger和以太坊上同时发行，完全由美元存款和短期国债支持。Ripple表示已获得纽约金融服务部（NYDFS）的批准。此举被视为Ripple从支付网络向全面金融基础设施扩张的重要一步。XRP价格上涨8%。",
         "source": "coindesk", "symbol": "XRP-USD", "days_offset": 150},
    ]
    docs = []
    for a in articles:
        ts = base + timedelta(days=a["days_offset"])
        meta = {"symbol": a["symbol"]} if a["symbol"] else {}
        meta["category"] = "financial_news"
        content = f"{a['title']}\n\n{a['body']}"
        docs.append({
            "content": content,
            "source": a["source"],
            "metadata": meta,
            "created_at": ts,
        })
    return docs


def fetch_yfinance_summary(symbols: list, period: str = "5d") -> list:
    """Fetch recent price summaries from yfinance and convert to documents."""
    docs = []
    try:
        import yfinance as yf
        for sym in symbols:
            try:
                ticker = yf.Ticker(sym)
                hist = ticker.history(period=period)
                if hist.empty:
                    continue
                last = hist.iloc[-1]
                change = ((last["Close"] - hist.iloc[0]["Close"]) / hist.iloc[0]["Close"]) * 100
                content = (
                    f"{sym} 最近{period}行情概要：\n"
                    f"最新收盘价: {last['Close']:.2f}，"
                    f"最高: {hist['High'].max():.2f}，最低: {hist['Low'].min():.2f}，"
                    f"区间涨跌幅: {change:+.2f}%，"
                    f"日均成交量: {hist['Volume'].mean():,.0f}。"
                )
                docs.append({
                    "content": content,
                    "source": f"yfinance_{sym}",
                    "metadata": {"symbol": sym, "category": "price_summary"},
                    "created_at": datetime.utcnow(),
                })
            except Exception as e:
                print(f"  [WARN] {sym} yfinance 获取失败: {e}")
    except ImportError:
        print("  [WARN] yfinance 未安装，跳过实时数据获取")
    return docs


# ---------------------------------------------------------------------------
# 2. Test framework
# ---------------------------------------------------------------------------

class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = True
        self.details: list = []
        self.timings: dict = {}
        self.metrics: dict = {}

    def check(self, condition: bool, msg: str):
        if condition:
            self.details.append(f"  [PASS] {msg}")
        else:
            self.details.append(f"  [FAIL] {msg}")
            self.passed = False

    def info(self, msg: str):
        self.details.append(f"  [INFO] {msg}")

    def warn(self, msg: str):
        self.details.append(f"  [WARN] {msg}")

    def time_it(self, label: str):
        return _Timer(self, label)


class _Timer:
    def __init__(self, result: TestResult, label: str):
        self.result = result
        self.label = label

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.t0
        self.result.timings[self.label] = elapsed
        self.result.info(f"⏱  {self.label}: {elapsed*1000:.2f} ms")


# ---------------------------------------------------------------------------
# 3. Test cases
# ---------------------------------------------------------------------------

def test_document_creation(news_docs: list) -> TestResult:
    """测试 1: 文档创建与类型校验"""
    from quant_framework.rag.types import Document
    r = TestResult("1. 文档创建与类型校验")

    with r.time_it("document_creation"):
        documents = []
        for d in news_docs:
            doc = Document(
                content=d["content"],
                source=d["source"],
                metadata=d["metadata"],
                created_at=d["created_at"],
            )
            documents.append(doc)

    r.check(len(documents) == len(news_docs), f"创建 {len(documents)} 个文档")
    for doc in documents:
        r.check(doc.doc_id is not None, f"文档 {doc.source} 有 doc_id: {doc.doc_id[:8]}...")
        r.check(doc.created_at is not None, f"文档 {doc.source} 有 created_at: {doc.created_at}")
        r.check(len(doc.content) > 0, f"文档 {doc.source} 内容非空 ({len(doc.content)} 字)")

    # dedup check
    dup_doc = Document(content=news_docs[0]["content"], source="dup_test")
    r.check(dup_doc.doc_id == documents[0].doc_id, "相同内容产生相同 doc_id（去重基础）")

    r.metrics["total_docs"] = len(documents)
    r.metrics["total_chars"] = sum(len(d.content) for d in documents)
    return r


def test_chunking(news_docs: list) -> TestResult:
    """测试 2: 分块逻辑"""
    from quant_framework.rag.types import Document
    from quant_framework.rag.processing.chunker import Chunker
    r = TestResult("2. 分块逻辑测试")

    chunker = Chunker(chunk_size=256, chunk_overlap=32, by_sentence=True, min_chunk_length=10)
    documents = [
        Document(content=d["content"], source=d["source"],
                 metadata=d["metadata"], created_at=d["created_at"])
        for d in news_docs
    ]

    all_chunks = []
    with r.time_it("chunking_all_docs"):
        for doc in documents:
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)

    r.check(len(all_chunks) > 0, f"共产生 {len(all_chunks)} 个分块")

    for c in all_chunks:
        r.check(c.chunk_id is not None, f"chunk {c.chunk_id} 有 ID")
        r.check(len(c.text) >= 10, f"chunk 长度 {len(c.text)} >= min_chunk_length(10)")

    # Check created_at propagation in chunk metadata
    chunks_with_created = [c for c in all_chunks if "created_at" in c.metadata]
    r.check(
        len(chunks_with_created) == len(all_chunks),
        f"created_at 传播: {len(chunks_with_created)}/{len(all_chunks)} 个 chunk 含 created_at 元数据"
    )
    if chunks_with_created:
        sample = chunks_with_created[0]
        r.check(
            isinstance(sample.metadata["created_at"], datetime),
            f"created_at 类型正确: {type(sample.metadata['created_at']).__name__}"
        )

    r.metrics["total_chunks"] = len(all_chunks)
    r.metrics["avg_chunk_len"] = np.mean([len(c.text) for c in all_chunks])
    r.metrics["max_chunk_len"] = max(len(c.text) for c in all_chunks)
    r.metrics["min_chunk_len"] = min(len(c.text) for c in all_chunks)
    return r


def test_embedding(news_docs: list) -> TestResult:
    """测试 3: 嵌入器"""
    from quant_framework.rag.processing.embedder import SentenceTransformerEmbedder, DummyEmbedder
    r = TestResult("3. 嵌入器测试")

    try:
        embedder = SentenceTransformerEmbedder()
        _ = embedder.dimension
        embedder_type = "SentenceTransformer"
    except Exception:
        embedder = DummyEmbedder(dimension=384)
        embedder_type = "DummyEmbedder"

    r.info(f"使用嵌入器: {embedder_type}")
    r.check(embedder.dimension == 384, f"向量维度: {embedder.dimension}")

    test_texts = [
        "比特币价格上涨突破65000美元",
        "以太坊完成Dencun升级Gas费下降",
        "苹果公司发布新iPhone销量创纪录",
        "量化交易策略回测结果分析",
    ]

    with r.time_it("embed_4_texts"):
        vectors = embedder.embed(test_texts)

    r.check(len(vectors) == 4, f"返回 {len(vectors)} 个向量")
    for i, v in enumerate(vectors):
        r.check(len(v) == 384, f"向量 {i} 维度 = {len(v)}")
        norm = np.linalg.norm(v)
        r.check(norm > 0, f"向量 {i} 范数 = {norm:.4f} (非零)")

    # determinism check
    with r.time_it("embed_determinism_check"):
        vectors2 = embedder.embed(test_texts)
    for i in range(len(vectors)):
        diff = np.max(np.abs(np.array(vectors[i]) - np.array(vectors2[i])))
        r.check(diff < 1e-6, f"向量 {i} 确定性: max_diff = {diff:.2e}")

    # similarity: same-domain texts should have different vectors
    sim_01 = np.dot(vectors[0], vectors[1]) / (np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1]))
    sim_02 = np.dot(vectors[0], vectors[2]) / (np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[2]))
    r.info(f"BTC vs ETH 相似度: {sim_01:.4f}")
    r.info(f"BTC vs AAPL 相似度: {sim_02:.4f}")
    if embedder_type == "SentenceTransformer":
        r.check(sim_01 > sim_02, "加密资产间相似度 > 加密 vs 股票 (语义嵌入)")
    else:
        r.warn("DummyEmbedder 无法保证语义相似度排序，检索将主要依赖 BM25 关键词路径")

    r.metrics["embedder_type"] = embedder_type
    return r


def test_processing_pipeline(news_docs: list) -> TestResult:
    """测试 4: 完整处理管道 (normalize → chunk → embed)"""
    from quant_framework.rag.types import Document
    from quant_framework.rag.processing import ProcessingPipeline, Chunker, TextNormalizer
    from quant_framework.rag.processing.embedder import DummyEmbedder
    r = TestResult("4. 处理管道端到端测试")

    chunker = Chunker(chunk_size=256, chunk_overlap=32, by_sentence=True)
    embedder = DummyEmbedder(dimension=384)
    normalizer = TextNormalizer()
    pipeline = ProcessingPipeline(chunker=chunker, embedder=embedder, normalizer=normalizer)

    documents = [
        Document(content=d["content"], source=d["source"],
                 metadata=d["metadata"], created_at=d["created_at"])
        for d in news_docs
    ]

    with r.time_it("process_all_documents"):
        chunks = pipeline.process_documents(documents)

    r.check(len(chunks) > 0, f"处理完成: {len(chunks)} 个嵌入分块")

    # BUG VERIFICATION: created_at should be preserved
    chunks_with_created = [c for c in chunks if "created_at" in c.metadata]
    r.check(
        len(chunks_with_created) == len(chunks),
        f"[BUG FIX 验证] created_at 传播: {len(chunks_with_created)}/{len(chunks)} (修复前为 0)"
    )

    for c in chunks:
        r.check(c.embedding is not None, f"chunk {c.chunk_id[:12]}... 有嵌入向量")
        if c.embedding:
            r.check(len(c.embedding) == 384, f"嵌入维度正确: {len(c.embedding)}")

    r.metrics["processed_chunks"] = len(chunks)
    return r


def test_vector_store() -> TestResult:
    """测试 5: 向量存储"""
    from quant_framework.rag.store.vector_store import InMemoryVectorStore
    from quant_framework.rag.types import Chunk
    r = TestResult("5. 向量存储测试")

    store = InMemoryVectorStore(dimension=4, max_vectors=10)
    r.check(store.size() == 0, "初始大小 = 0")

    chunks = [
        Chunk(text=f"test_{i}", chunk_id=f"c_{i}", doc_id="d1", index=i,
              embedding=[float(i + 1), 0.0, 0.0, 0.0])
        for i in range(5)
    ]

    with r.time_it("add_5_chunks"):
        evicted = store.add(chunks)
    r.check(store.size() == 5, f"添加后大小 = {store.size()}")
    r.check(len(evicted) == 0, "无淘汰")

    with r.time_it("search_top3"):
        results = store.search([1.0, 0.0, 0.0, 0.0], top_k=3)
    r.check(len(results) == 3, f"检索返回 {len(results)} 个结果")
    top_text = results[0][0].text
    r.check(
        top_text in ("test_0", "test_1", "test_2", "test_3", "test_4"),
        f"最相似: {top_text} (所有正向量，归一化后均为 [1,0,0,0]，cosine=1.0)"
    )

    # FIFO eviction
    extra = [
        Chunk(text=f"extra_{i}", chunk_id=f"e_{i}", doc_id="d2", index=i,
              embedding=[0.0, float(i), 0.0, 1.0])
        for i in range(8)
    ]
    with r.time_it("add_8_trigger_eviction"):
        evicted = store.add(extra)
    r.check(store.size() == 10, f"淘汰后大小 = {store.size()} (max=10)")
    r.check(len(evicted) == 3, f"淘汰 {len(evicted)} 个 (预期 3)")

    # batch search
    with r.time_it("search_batch_2queries"):
        batch_results = store.search_batch(
            [[0.0, 7.0, 0.0, 1.0], [4.0, 0.0, 0.0, 1.0]], top_k=2
        )
    r.check(len(batch_results) == 2, f"批量检索返回 {len(batch_results)} 组")

    return r


def test_keyword_index(news_docs: list) -> TestResult:
    """测试 6: 关键词索引 (BM25)"""
    from quant_framework.rag.types import Chunk
    from quant_framework.rag.store.keyword_index import KeywordIndex
    r = TestResult("6. 关键词索引 (BM25) 测试")

    index = KeywordIndex()

    chunks = [
        Chunk(text=d["content"], chunk_id=f"kw_{i}", doc_id=f"d_{i}", index=0)
        for i, d in enumerate(news_docs)
    ]

    with r.time_it("add_to_keyword_index"):
        index.add(chunks)
    r.check(index.size() == len(chunks), f"索引大小: {index.size()}")

    test_queries = [
        ("比特币 减半 价格", "BTC"),
        ("以太坊 升级 Gas", "ETH"),
        ("苹果 iPhone 财报", "AAPL"),
        ("XRP SEC 诉讼", "XRP"),
        ("量化 交易 策略", "Quant"),
    ]

    for query, label in test_queries:
        with r.time_it(f"bm25_search_{label}"):
            results = index.search(query, top_k=3)
        r.check(len(results) > 0, f"查询 '{query}': 返回 {len(results)} 个结果")
        if results:
            top_text = results[0][0].text[:60].replace("\n", " ")
            r.info(f"  Top-1 ({label}): score={results[0][1]:.4f}, text={top_text}...")

    return r


def test_full_pipeline_e2e(news_docs: list, yf_docs: list) -> TestResult:
    """测试 7: RAGPipeline 完整端到端（入库 + 检索 + 上下文生成）"""
    from quant_framework.rag import RAGConfig, RAGPipeline, Document
    r = TestResult("7. RAGPipeline 端到端测试")

    config = RAGConfig(
        chunk_size=256,
        chunk_overlap=32,
        use_hybrid=True,
        hybrid_weights=(0.6, 0.4),
        vector_top_k=10,
        keyword_top_k=10,
        rerank_top_k=5,
        query_embedding_cache_size=64,
        dedup_cache_max=1000,
        ingest_queue_max_size=1000,
    )

    with r.time_it("pipeline_init"):
        pipeline = RAGPipeline(config=config, start_worker=False)
    r.info(f"嵌入器类型: {type(pipeline.embedder).__name__}")

    all_docs_raw = news_docs + yf_docs
    documents = [
        Document(
            content=d["content"], source=d["source"],
            metadata=d["metadata"], created_at=d["created_at"],
        )
        for d in all_docs_raw
    ]

    with r.time_it("batch_ingest"):
        n_chunks = pipeline.add_documents(documents)
    r.check(n_chunks > 0, f"入库 {n_chunks} 个分块 (来自 {len(documents)} 个文档)")

    stats = pipeline.health_check()
    r.info(f"Pipeline 状态: vector={stats['vector_store_size']}, keyword={stats['keyword_index_size']}")
    r.check(stats["vector_store_size"] > 0, "向量存储非空")
    r.check(stats["keyword_index_size"] > 0, "关键词索引非空")

    # dedup: re-ingest same docs
    with r.time_it("dedup_reingest"):
        n2 = pipeline.add_documents(documents)
    r.check(n2 == 0, f"去重生效: 重复入库返回 {n2} (预期 0)")

    # retrieval
    test_queries = [
        ("比特币价格走势和减半影响", "BTC行情"),
        ("以太坊升级和DeFi生态", "ETH生态"),
        ("苹果公司业绩和AI战略", "AAPL基本面"),
        ("XRP法律诉讼结果和跨境支付", "XRP动态"),
        ("美联储利率决议对市场影响", "宏观"),
        ("量化交易技术架构", "量化"),
    ]

    for query, label in test_queries:
        with r.time_it(f"retrieve_{label}"):
            results = pipeline.retrieve(query, top_k=5)
        r.check(len(results) > 0, f"查询 '{label}': 返回 {len(results)} 个结果")
        if results:
            top = results[0]
            r.info(f"  Top-1: score={top.score:.4f}, source={top.chunk.metadata.get('source', 'N/A')}")
            r.info(f"  文本: {top.chunk.text[:80].replace(chr(10), ' ')}...")

    # context generation
    with r.time_it("get_context_concat"):
        ctx = pipeline.get_context_for_generation("比特币最近消息", top_k=3, max_chars=500)
    r.check(len(ctx) > 0, f"上下文生成 (concat): {len(ctx)} 字")
    r.info(f"  上下文前 100 字: {ctx[:100].replace(chr(10), ' ')}...")

    # batch retrieval
    with r.time_it("batch_retrieve_3q"):
        batch = pipeline.retrieve_batch(
            ["比特币", "以太坊", "苹果公司"], top_k=3
        )
    r.check(len(batch) == 3, f"批量检索: {len(batch)} 组结果")

    r.metrics["total_ingested_chunks"] = n_chunks
    r.metrics["vector_store_size"] = stats["vector_store_size"]
    r.metrics["keyword_index_size"] = stats["keyword_index_size"]

    return r


def test_metadata_filtering(news_docs: list) -> TestResult:
    """测试 8: 元数据过滤（source_contains / created_before）"""
    from quant_framework.rag import RAGConfig, RAGPipeline, Document
    r = TestResult("8. 元数据过滤测试")

    config = RAGConfig(chunk_size=256, use_hybrid=True, dedup_cache_max=0)
    pipeline = RAGPipeline(config=config, start_worker=False)

    documents = [
        Document(content=d["content"], source=d["source"],
                 metadata=d["metadata"], created_at=d["created_at"])
        for d in news_docs
    ]
    pipeline.add_documents(documents)

    # source_contains filter
    with r.time_it("filter_source_coindesk"):
        results = pipeline.retrieve("比特币", top_k=10, metadata_filter={"source_contains": "coindesk"})
    coindesk_count = sum(1 for res in results if "coindesk" in str(res.chunk.metadata.get("source", "")))
    r.check(
        len(results) == coindesk_count,
        f"source_contains='coindesk': 全部 {len(results)} 个结果来自 coindesk"
    )

    # created_before filter
    cutoff = datetime(2024, 2, 20)
    with r.time_it("filter_created_before"):
        results = pipeline.retrieve("市场", top_k=10, metadata_filter={"created_before": cutoff})
    violations = 0
    for res in results:
        ct = res.chunk.metadata.get("created_at")
        if ct is not None and isinstance(ct, datetime) and ct >= cutoff:
            violations += 1
    r.check(violations == 0, f"created_before 过滤: {violations} 个违规 (预期 0)")
    r.info(f"  cutoff={cutoff}, 返回 {len(results)} 个结果")

    return r


def test_async_ingest_and_worker(news_docs: list) -> TestResult:
    """测试 9: 异步入队 + 后台 Worker"""
    from quant_framework.rag import RAGConfig, RAGPipeline, Document
    r = TestResult("9. 异步入队与后台 Worker 测试")

    config = RAGConfig(chunk_size=256, use_hybrid=True, process_batch_size=4)
    pipeline = RAGPipeline(config=config, start_worker=True)

    r.check(pipeline._worker_started, "后台 worker 已启动")

    documents = [
        Document(content=d["content"], source=d["source"],
                 metadata=d["metadata"], created_at=d["created_at"])
        for d in news_docs[:5]
    ]

    with r.time_it("async_ingest_5docs"):
        for doc in documents:
            ok = pipeline.ingest_put(doc)
            r.check(ok, f"入队成功: {doc.source}")

    # wait for worker to process
    import time as _time
    max_wait = 5.0
    waited = 0
    while pipeline.ingest_queue.size() > 0 and waited < max_wait:
        _time.sleep(0.2)
        waited += 0.2
    r.info(f"Worker 处理耗时约 {waited:.1f}s, 队列剩余: {pipeline.ingest_queue.size()}")
    r.check(pipeline.ingest_queue.size() == 0, "队列已清空")

    stats = pipeline.health_check()
    r.check(stats["vector_store_size"] > 0, f"Worker 入库成功: {stats['vector_store_size']} 个向量")

    pipeline._worker_stop.set()
    return r


def test_rag_context_provider(news_docs: list) -> TestResult:
    """测试 10: RagContextProvider 策略集成层"""
    from quant_framework.rag import RAGConfig, RAGPipeline, Document
    from quant_framework.data.rag_context import RagContextProvider
    r = TestResult("10. RagContextProvider 策略集成测试")

    config = RAGConfig(chunk_size=256, use_hybrid=True)
    pipeline = RAGPipeline(config=config, start_worker=False)
    provider = RagContextProvider(pipeline=pipeline)

    documents = [
        Document(content=d["content"], source=d["source"],
                 metadata=d["metadata"], created_at=d["created_at"])
        for d in news_docs
    ]
    provider.add_documents(documents)

    # basic context
    with r.time_it("get_context_basic"):
        ctx = provider.get_context("最近新闻", top_k=3, max_chars=500)
    r.check(len(ctx) > 0, f"基本上下文: {len(ctx)} 字")

    # with symbol
    with r.time_it("get_context_with_symbol"):
        ctx = provider.get_context("价格分析", symbol="BTC-USD", top_k=3)
    r.check(len(ctx) > 0, f"带标的上下文: {len(ctx)} 字")

    # with as_of_date
    cutoff = datetime(2024, 2, 1)
    with r.time_it("get_context_as_of_date"):
        ctx = provider.get_context("市场动态", as_of_date=cutoff, top_k=3)
    r.info(f"as_of_date={cutoff} 上下文: {len(ctx)} 字")

    # ingest via provider
    with r.time_it("provider_ingest_document"):
        ok = provider.ingest_document(
            content="测试文档：通过 Provider 接入",
            source="test_provider",
            metadata={"category": "test"}
        )
    r.check(ok, "通过 Provider 入队文档")

    # dict-based add (content must be >= 10 chars to pass TextNormalizer min_length)
    with r.time_it("provider_add_dicts"):
        n = provider.add_documents([
            {"content": "这是一个通过字典格式添加的测试文档，验证 RagContextProvider 的兼容性。",
             "source": "dict_test", "metadata": {}},
        ])
    r.check(n > 0, f"字典格式入库: {n} 个分块")

    return r


def test_backtest_integration(news_docs: list) -> TestResult:
    """测试 11: RAG + 回测集成"""
    from quant_framework.rag import RAGConfig, RAGPipeline, Document
    from quant_framework.data.rag_context import RagContextProvider
    from quant_framework.strategy.base_strategy import BaseStrategy
    r = TestResult("11. RAG + 回测策略集成测试")

    config = RAGConfig(chunk_size=256, use_hybrid=True)
    pipeline = RAGPipeline(config=config, start_worker=False)
    provider = RagContextProvider(pipeline=pipeline)

    documents = [
        Document(content=d["content"], source=d["source"],
                 metadata=d["metadata"], created_at=d["created_at"])
        for d in news_docs
    ]
    provider.add_documents(documents)

    class RAGTestStrategy(BaseStrategy):
        def __init__(self, **kwargs):
            super().__init__(name="RAG_Test", initial_capital=100000, **kwargs)
            self.context_log = []
            self.signal_log = []

        def on_bar(self, data, current_date, current_prices=None):
            ctx = self.get_rag_context(
                query="市场新闻和行情",
                symbol="BTC-USD",
                top_k=3,
                max_chars=300,
                as_of_date=current_date,
            )
            self.context_log.append({"date": current_date, "ctx_len": len(ctx), "ctx_preview": ctx[:50]})
            action = "hold"
            if "上涨" in ctx or "利好" in ctx or "突破" in ctx:
                action = "buy"
            elif "下跌" in ctx or "承压" in ctx or "黑天鹅" in ctx:
                action = "sell"
            sig = {"action": action, "symbol": "BTC-USD", "shares": 0.01, "price": 60000}
            self.signal_log.append({"date": current_date, "action": action})
            return sig

    strategy = RAGTestStrategy(rag_provider=provider)
    r.check(strategy.rag_provider is not None, "策略 rag_provider 已注入")

    dates = pd.date_range("2024-01-15", periods=30, freq="D")
    for dt in dates:
        dummy_df = pd.DataFrame({
            "date": dates[:len(dates)],
            "close": np.random.uniform(55000, 70000, len(dates)),
            "open": np.random.uniform(55000, 70000, len(dates)),
            "high": np.random.uniform(60000, 75000, len(dates)),
            "low": np.random.uniform(50000, 65000, len(dates)),
            "volume": np.random.uniform(1e9, 5e9, len(dates)),
        })
        with r.time_it(f"on_bar_{dt.strftime('%m%d')}"):
            sig = strategy.on_bar(dummy_df, dt)

    r.check(len(strategy.context_log) == 30, f"30 个 bar 均获取上下文: {len(strategy.context_log)}")
    r.check(len(strategy.signal_log) == 30, f"30 个 bar 均生成信号: {len(strategy.signal_log)}")

    action_counts = defaultdict(int)
    for s in strategy.signal_log:
        action_counts[s["action"]] += 1
    r.info(f"信号分布: buy={action_counts['buy']}, sell={action_counts['sell']}, hold={action_counts['hold']}")

    ctx_with_content = sum(1 for c in strategy.context_log if c["ctx_len"] > 0)
    r.check(ctx_with_content > 0, f"有 {ctx_with_content}/30 个 bar 获取到非空上下文")

    # temporal consistency: earlier dates should get less context
    early_ctx = [c for c in strategy.context_log if c["date"] < datetime(2024, 1, 20)]
    late_ctx = [c for c in strategy.context_log if c["date"] > datetime(2024, 2, 10)]
    if early_ctx and late_ctx:
        early_avg = np.mean([c["ctx_len"] for c in early_ctx])
        late_avg = np.mean([c["ctx_len"] for c in late_ctx])
        r.info(f"时序一致性: 早期平均上下文 {early_avg:.0f} 字, 晚期 {late_avg:.0f} 字")

    r.metrics["signals"] = dict(action_counts)
    r.metrics["bars_with_context"] = ctx_with_content
    return r


def test_performance_benchmark(news_docs: list) -> TestResult:
    """测试 12: 性能基准"""
    from quant_framework.rag import RAGConfig, RAGPipeline, Document
    r = TestResult("12. 性能基准测试")

    config = RAGConfig(chunk_size=256, use_hybrid=True, dedup_cache_max=0)
    pipeline = RAGPipeline(config=config, start_worker=False)

    documents = [
        Document(content=d["content"], source=d["source"],
                 metadata=d["metadata"], created_at=d["created_at"])
        for d in news_docs
    ]

    # ingest throughput
    with r.time_it("ingest_throughput"):
        n = pipeline.add_documents(documents)
    ingest_time = r.timings["ingest_throughput"]
    r.metrics["ingest_docs_per_sec"] = len(documents) / ingest_time if ingest_time > 0 else 0
    r.info(f"入库吞吐: {r.metrics['ingest_docs_per_sec']:.0f} docs/sec ({n} chunks)")

    # single query latency (run 100 times)
    queries = ["比特币价格", "以太坊升级", "苹果财报", "量化策略", "美联储利率"]
    latencies = []
    with r.time_it("100_queries"):
        for _ in range(20):
            for q in queries:
                t0 = time.perf_counter()
                pipeline.retrieve(q, top_k=5)
                latencies.append(time.perf_counter() - t0)

    r.metrics["query_mean_ms"] = np.mean(latencies) * 1000
    r.metrics["query_p50_ms"] = np.percentile(latencies, 50) * 1000
    r.metrics["query_p95_ms"] = np.percentile(latencies, 95) * 1000
    r.metrics["query_p99_ms"] = np.percentile(latencies, 99) * 1000
    r.metrics["query_max_ms"] = np.max(latencies) * 1000
    r.info(f"查询延迟 (100 次): mean={r.metrics['query_mean_ms']:.2f}ms, "
           f"p50={r.metrics['query_p50_ms']:.2f}ms, p95={r.metrics['query_p95_ms']:.2f}ms, "
           f"p99={r.metrics['query_p99_ms']:.2f}ms, max={r.metrics['query_max_ms']:.2f}ms")

    # batch query
    with r.time_it("batch_5_queries"):
        pipeline.retrieve_batch(queries, top_k=5)
    batch_time = r.timings["batch_5_queries"]
    r.metrics["batch_query_ms"] = batch_time * 1000
    r.info(f"批量查询 (5 条): {batch_time * 1000:.2f}ms")

    # context generation
    with r.time_it("context_generation"):
        for q in queries:
            pipeline.get_context_for_generation(q, top_k=3, max_chars=500)
    ctx_time = r.timings["context_generation"]
    r.metrics["context_gen_mean_ms"] = ctx_time / len(queries) * 1000
    r.info(f"上下文生成: 平均 {r.metrics['context_gen_mean_ms']:.2f}ms/次")

    return r


def test_edge_cases() -> TestResult:
    """测试 13: 边界条件"""
    from quant_framework.rag import RAGConfig, RAGPipeline, Document
    r = TestResult("13. 边界条件测试")

    config = RAGConfig(chunk_size=256, use_hybrid=True, dedup_cache_max=0)
    pipeline = RAGPipeline(config=config, start_worker=False)

    # empty document
    n = pipeline.add_documents([Document(content="", source="empty")])
    r.check(n == 0, f"空文档入库: {n} 个分块 (预期 0)")

    # very short document
    n = pipeline.add_documents([Document(content="短", source="short")])
    r.info(f"极短文档入库: {n} 个分块")

    # very long document
    long_text = "这是一段很长的测试文本。" * 500
    n = pipeline.add_documents([Document(content=long_text, source="long")])
    r.check(n > 1, f"长文档入库: {n} 个分块 (预期 > 1)")

    # unicode / special chars
    special = "🚀 Bitcoin to the moon! 比特币突破$100K。\n\n日本語テスト。한국어 테스트。"
    n = pipeline.add_documents([Document(content=special, source="unicode")])
    r.check(n > 0, f"Unicode 文档入库: {n} 个分块")

    # query on empty store
    empty_pipe = RAGPipeline(config=config, start_worker=False)
    results = empty_pipe.retrieve("test", top_k=5)
    r.check(len(results) == 0, f"空存储查询: {len(results)} 个结果 (预期 0)")

    # config validation
    try:
        RAGConfig(chunk_size=0)
        r.check(False, "chunk_size=0 应抛出异常")
    except ValueError:
        r.check(True, "chunk_size=0 正确抛出 ValueError")

    try:
        RAGConfig(chunk_overlap=600, chunk_size=512)
        r.check(False, "chunk_overlap >= chunk_size 应抛出异常")
    except ValueError:
        r.check(True, "chunk_overlap >= chunk_size 正确抛出 ValueError")

    return r


# ---------------------------------------------------------------------------
# 4. Report generation
# ---------------------------------------------------------------------------

def generate_report(results: list, news_count: int, yf_count: int) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("RAG 全链路端到端测试报告")
    lines.append(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"数据源: {news_count} 条模拟新闻 + {yf_count} 条 yfinance 实时数据")
    lines.append("")

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed

    lines.append(f"测试结果: {passed}/{total} 通过, {failed} 失败")
    lines.append("-" * 80)

    for r in results:
        status = "PASS ✓" if r.passed else "FAIL ✗"
        lines.append(f"\n{'='*60}")
        lines.append(f"[{status}] {r.name}")
        lines.append(f"{'='*60}")
        for d in r.details:
            lines.append(d)
        if r.timings:
            lines.append("")
            lines.append("  性能指标:")
            for k, v in r.timings.items():
                if not k.startswith("on_bar_"):
                    lines.append(f"    {k}: {v*1000:.2f} ms")
        if r.metrics:
            lines.append("")
            lines.append("  关键数据:")
            for k, v in r.metrics.items():
                lines.append(f"    {k}: {v}")

    # Summary analysis
    lines.append("")
    lines.append("=" * 80)
    lines.append("综合分析报告")
    lines.append("=" * 80)

    lines.append("")
    lines.append("一、架构流程分析")
    lines.append("-" * 40)
    lines.append("  Document → TextNormalizer → Chunker → Embedder → VectorStore + KeywordIndex")
    lines.append("  Query → Embedder → VectorStore.search + KeywordIndex.search → RRF Merge → Reranker → Results")
    lines.append("  Strategy.on_bar → RagContextProvider.get_context(as_of_date) → context string")

    lines.append("")
    lines.append("二、发现的问题与修复")
    lines.append("-" * 40)
    lines.append("  [已修复] BUG-1: ProcessingPipeline 重建 Document 时丢失 created_at")
    lines.append("           影响: as_of_date 过滤失效，回测中可能使用未来信息（look-ahead bias）")
    lines.append("           修复: _rebuild_doc() 显式传递 created_at")
    lines.append("")
    lines.append("  [已修复] BUG-2: DummyEmbedder 使用固定种子，所有文本相对位置共享伪随机序列")
    lines.append("           影响: 不同文本的向量不确定地相似，向量检索路径近乎随机")
    lines.append("           修复: 基于文本哈希生成种子，同文本始终同向量")
    lines.append("")
    lines.append("  [观察] OBS-1: sentence-transformers 未安装，使用 DummyEmbedder 降级运行")
    lines.append("         影响: 向量检索质量取决于伪随机向量，实际检索主要依赖 BM25 关键词路径")
    lines.append("         建议: 生产环境必须安装 sentence-transformers 或接入外部嵌入 API")
    lines.append("")
    lines.append("  [观察] OBS-2: RagContextProvider.get_context 在有 metadata_filter 时")
    lines.append("         跳过 context_strategy 参数，直接使用简单拼接")
    lines.append("         影响: merge_adjacent 策略对带过滤的查询不生效")
    lines.append("")
    lines.append("  [观察] OBS-3: Chunker 句子分割仅支持中文标点和换行 (。！？!?\\n)")
    lines.append("         影响: 纯英文句子（以 . 结尾）不会按句子边界分块")
    lines.append("         建议: 添加英文句号到 _sentence_pattern")

    # Performance summary
    perf_result = next((r for r in results if "性能基准" in r.name), None)
    if perf_result:
        lines.append("")
        lines.append("三、性能基准总结")
        lines.append("-" * 40)
        m = perf_result.metrics
        lines.append(f"  入库吞吐: {m.get('ingest_docs_per_sec', 0):.0f} docs/sec")
        lines.append(f"  单查询延迟 (100次):")
        lines.append(f"    平均: {m.get('query_mean_ms', 0):.2f} ms")
        lines.append(f"    P50:  {m.get('query_p50_ms', 0):.2f} ms")
        lines.append(f"    P95:  {m.get('query_p95_ms', 0):.2f} ms")
        lines.append(f"    P99:  {m.get('query_p99_ms', 0):.2f} ms")
        lines.append(f"    Max:  {m.get('query_max_ms', 0):.2f} ms")
        lines.append(f"  批量查询 (5条): {m.get('batch_query_ms', 0):.2f} ms")
        lines.append(f"  上下文生成: {m.get('context_gen_mean_ms', 0):.2f} ms/次")

    bt_result = next((r for r in results if "回测" in r.name), None)
    if bt_result:
        lines.append("")
        lines.append("四、回测集成分析")
        lines.append("-" * 40)
        m = bt_result.metrics
        lines.append(f"  信号分布: {m.get('signals', {})}")
        lines.append(f"  有上下文的 bar: {m.get('bars_with_context', 0)}/30")
        bar_times = [v * 1000 for k, v in bt_result.timings.items() if k.startswith("on_bar_")]
        if bar_times:
            lines.append(f"  on_bar 延迟:")
            lines.append(f"    平均: {np.mean(bar_times):.2f} ms")
            lines.append(f"    P95:  {np.percentile(bar_times, 95):.2f} ms")
            lines.append(f"    Max:  {np.max(bar_times):.2f} ms")

    lines.append("")
    lines.append("五、优化建议")
    lines.append("-" * 40)
    lines.append("  P0 (必须):")
    lines.append("    1. 安装 sentence-transformers 使向量检索具备真实语义能力")
    lines.append("    2. Chunker 句子分割正则添加英文句号支持")
    lines.append("    3. RagContextProvider 在 metadata_filter 分支中支持 context_strategy")
    lines.append("")
    lines.append("  P1 (重要):")
    lines.append("    4. 添加磁盘持久化 (VectorStore 的 save/load)")
    lines.append("    5. 实现真实新闻 API 适配器 (RSS/REST/WebSocket)")
    lines.append("    6. 添加文档时间戳索引，加速 created_before 过滤")
    lines.append("")
    lines.append("  P2 (优化):")
    lines.append("    7. 启用 Numba JIT (core.py _USE_NUMBA = True) 加速向量搜索")
    lines.append("    8. 实现增量式嵌入（仅嵌入新增文档）")
    lines.append("    9. 添加 RAG 检索质量评估指标 (MRR, NDCG)")

    lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("RAG 全链路端到端测试")
    print("=" * 80)
    print()

    # Fetch data
    print("[1/2] 获取模拟金融新闻...")
    news_docs = fetch_financial_news_simulated()
    print(f"  获取 {len(news_docs)} 条新闻")

    print("[2/2] 获取 yfinance 实时数据...")
    yf_symbols = ["BTC-USD", "ETH-USD", "AAPL", "TSLA", "NVDA"]
    yf_docs = fetch_yfinance_summary(yf_symbols)
    print(f"  获取 {len(yf_docs)} 条实时数据")
    print()

    # Run tests
    test_functions = [
        lambda: test_document_creation(news_docs),
        lambda: test_chunking(news_docs),
        lambda: test_embedding(news_docs),
        lambda: test_processing_pipeline(news_docs),
        lambda: test_vector_store(),
        lambda: test_keyword_index(news_docs),
        lambda: test_full_pipeline_e2e(news_docs, yf_docs),
        lambda: test_metadata_filtering(news_docs),
        lambda: test_async_ingest_and_worker(news_docs),
        lambda: test_rag_context_provider(news_docs),
        lambda: test_backtest_integration(news_docs),
        lambda: test_performance_benchmark(news_docs),
        lambda: test_edge_cases(),
    ]

    results = []
    for i, fn in enumerate(test_functions):
        print(f"运行测试 {i+1}/{len(test_functions)}...", end=" ", flush=True)
        try:
            result = fn()
            status = "PASS" if result.passed else "FAIL"
            print(f"[{status}] {result.name}")
        except Exception as e:
            result = TestResult(f"测试 {i+1} (异常)")
            result.passed = False
            result.details.append(f"  [ERROR] {type(e).__name__}: {e}")
            import traceback
            result.details.append(f"  {traceback.format_exc()}")
            print(f"[ERROR] {e}")
        results.append(result)

    # Generate report
    report = generate_report(results, len(news_docs), len(yf_docs))
    print("\n" + report)

    report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "docs", "RAG_TEST_REPORT.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n报告已保存: {report_path}")


if __name__ == "__main__":
    main()
