"""
RAG 实时新闻压力测试
====================================================
模拟每分钟实时新闻推送场景，测量：
1. 持续入库吞吐 (边写边查)
2. 库容增长下检索延迟退化曲线
3. 并发写读安全性
4. 队列背压与丢弃率
5. 内存使用增长
6. 瓶颈定位
"""

import sys
import os
import time
import threading
import random
import tracemalloc
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from quant_framework.rag import RAGConfig, RAGPipeline, Document

# ---------------------------------------------------------------------------
# 1. Realistic news generator
# ---------------------------------------------------------------------------

CRYPTO_SYMBOLS = ["BTC", "ETH", "SOL", "XRP", "BNB", "ADA", "AVAX", "DOT"]
STOCK_SYMBOLS = ["AAPL", "TSLA", "NVDA", "GOOGL", "MSFT", "AMZN", "META"]
NEWS_SOURCES = ["reuters", "bloomberg", "coindesk", "wsj", "cnbc", "ft"]

TEMPLATES_CN = [
    "{symbol}价格突破关键阻力位，24小时涨幅达{pct:.1f}%。交易量显著放大至{vol:.0f}亿美元。分析师认为短期内可能继续上攻{target}美元目标位。技术指标显示RSI进入超买区间，MACD金叉形成。",
    "{symbol}遭遇大幅抛售，日内跌幅达{pct:.1f}%，触及{low:.0f}美元支撑位。链上数据显示鲸鱼地址大量转入交易所，市场恐慌情绪蔓延。合约爆仓金额超过{liq:.0f}万美元。",
    "美联储官员表示可能在下次会议上讨论{action}，市场对此反应强烈。{symbol}等风险资产价格波动加剧。10年期国债收益率{direction}至{yield_pct:.2f}%。标普500指数{sp_dir}{sp_pct:.1f}%。",
    "{symbol}发布重大更新公告，新版本将引入{feature}功能。社区反应积极，{symbol}代币价格应声上涨{pct:.1f}%。开发者活跃度较上月增加{dev_pct:.0f}%。",
    "{symbol}与{partner}达成战略合作协议，将共同开发{area}解决方案。这一消息推动{symbol}价格上涨{pct:.1f}%，市值突破{mcap:.0f}亿美元。",
    "知名分析师预测{symbol}在未来{months}个月内可能达到{target}美元。该预测基于{reason}。当前价格为{price:.0f}美元，潜在上行空间{upside:.0f}%。",
    "{symbol}季度财报超出市场预期，营收同比增长{rev_pct:.0f}%达{rev:.0f}亿美元。每股收益{eps:.2f}美元，高于预期的{eps_exp:.2f}美元。盘后股价上涨{pct:.1f}%。",
    "全球加密货币市场总市值突破{total_mcap:.1f}万亿美元，{symbol}贡献了主要涨幅。日交易量达{daily_vol:.0f}亿美元创近期新高。机构资金持续净流入。",
]

FEATURES = ["零知识证明", "跨链桥接", "账户抽象", "并行执行", "动态分片"]
PARTNERS = ["微软", "谷歌", "亚马逊", "摩根大通", "贝莱德", "Visa", "万事达"]
AREAS = ["去中心化金融", "数字身份", "供应链溯源", "AI推理", "跨境支付"]
REASONS = ["链上指标强劲", "机构持仓增加", "减半效应", "ETF资金流入", "技术突破"]


def generate_news_batch(n: int, base_time: datetime) -> List[Document]:
    """Generate n realistic financial news documents."""
    docs = []
    for i in range(n):
        sym = random.choice(CRYPTO_SYMBOLS + STOCK_SYMBOLS)
        src = random.choice(NEWS_SOURCES)
        tmpl = random.choice(TEMPLATES_CN)
        ts = base_time + timedelta(seconds=random.uniform(0, 60))

        params = {
            "symbol": sym, "pct": random.uniform(0.5, 15.0),
            "vol": random.uniform(1, 50), "target": random.randint(50, 200000),
            "low": random.uniform(10, 100000), "liq": random.uniform(100, 50000),
            "action": random.choice(["加息", "降息", "维持利率"]),
            "direction": random.choice(["升", "降"]),
            "yield_pct": random.uniform(3.5, 5.5),
            "sp_dir": random.choice(["上涨", "下跌"]),
            "sp_pct": random.uniform(0.1, 3.0),
            "feature": random.choice(FEATURES),
            "dev_pct": random.uniform(5, 50),
            "partner": random.choice(PARTNERS),
            "area": random.choice(AREAS),
            "months": random.randint(1, 12),
            "reason": random.choice(REASONS),
            "price": random.uniform(10, 100000),
            "upside": random.uniform(10, 500),
            "rev_pct": random.uniform(5, 80),
            "rev": random.uniform(10, 1000),
            "eps": random.uniform(0.5, 20),
            "eps_exp": random.uniform(0.3, 18),
            "total_mcap": random.uniform(1.5, 4.0),
            "daily_vol": random.uniform(50, 500),
            "mcap": random.uniform(1, 5000),
        }
        try:
            content = tmpl.format(**params)
        except (KeyError, IndexError):
            content = f"{sym} 市场动态更新：价格变动 {params['pct']:.1f}%，交易量 {params['vol']:.0f}亿美元。"

        docs.append(Document(
            content=content,
            source=src,
            metadata={"symbol": sym, "category": "realtime_news"},
            created_at=ts,
        ))
    return docs


# ---------------------------------------------------------------------------
# 2. Stress test scenarios
# ---------------------------------------------------------------------------

def scenario_sustained_ingestion(pipeline: RAGPipeline, duration_sec: float,
                                  docs_per_minute: int) -> Dict:
    """Scenario 1: Sustained ingestion at target rate, measuring actual throughput."""
    print(f"\n  [场景1] 持续入库: 目标 {docs_per_minute} docs/min, 持续 {duration_sec}s")

    interval = 60.0 / docs_per_minute
    start = time.perf_counter()
    base_time = datetime(2024, 6, 1)

    total_put = 0
    total_rejected = 0
    batch_times = []

    while time.perf_counter() - start < duration_sec:
        t0 = time.perf_counter()
        batch = generate_news_batch(1, base_time + timedelta(minutes=total_put))
        ok = pipeline.ingest_put(batch[0])
        elapsed = time.perf_counter() - t0
        batch_times.append(elapsed)
        if ok:
            total_put += 1
        else:
            total_rejected += 1

        sleep_for = max(0, interval - elapsed)
        if sleep_for > 0:
            time.sleep(sleep_for)

    elapsed_total = time.perf_counter() - start
    return {
        "total_put": total_put,
        "total_rejected": total_rejected,
        "actual_rate_per_min": total_put / elapsed_total * 60,
        "put_mean_us": np.mean(batch_times) * 1e6,
        "put_p99_us": np.percentile(batch_times, 99) * 1e6 if batch_times else 0,
        "elapsed_sec": elapsed_total,
    }


def scenario_burst_ingestion(pipeline: RAGPipeline, burst_sizes: List[int]) -> Dict:
    """Scenario 2: Burst ingestion — large batches arriving at once."""
    print(f"\n  [场景2] 突发入库: batch sizes = {burst_sizes}")
    results = {}
    base_time = datetime(2024, 7, 1)

    for batch_size in burst_sizes:
        docs = generate_news_batch(batch_size, base_time)
        t0 = time.perf_counter()
        n_chunks = pipeline.add_documents(docs)
        elapsed = time.perf_counter() - t0
        results[batch_size] = {
            "docs": batch_size,
            "chunks": n_chunks,
            "time_ms": elapsed * 1000,
            "throughput_docs_sec": batch_size / elapsed if elapsed > 0 else 0,
            "throughput_chunks_sec": n_chunks / elapsed if elapsed > 0 else 0,
        }
        print(f"    batch={batch_size}: {n_chunks} chunks in {elapsed*1000:.1f}ms "
              f"({batch_size/elapsed:.0f} docs/s)")

    return results


def scenario_retrieval_under_load(pipeline: RAGPipeline, n_docs_stages: List[int]) -> Dict:
    """Scenario 3: Retrieval latency as store size grows."""
    print(f"\n  [场景3] 库容增长下检索延迟退化曲线: stages = {n_docs_stages}")

    queries = ["比特币价格走势", "以太坊DeFi生态", "苹果公司财报", "美联储利率决议",
               "XRP法律进展", "量化交易策略", "英伟达AI芯片", "Solana网络性能"]
    results = {}
    base_time = datetime(2024, 8, 1)
    total_ingested = 0

    for stage_target in n_docs_stages:
        need = stage_target - total_ingested
        if need > 0:
            docs = generate_news_batch(need, base_time + timedelta(hours=total_ingested))
            pipeline.add_documents(docs)
            total_ingested += need

        store_size = pipeline.health_check()["vector_store_size"]

        latencies = []
        for _ in range(5):
            for q in queries:
                t0 = time.perf_counter()
                pipeline.retrieve(q, top_k=5)
                latencies.append(time.perf_counter() - t0)

        results[stage_target] = {
            "store_size": store_size,
            "n_queries": len(latencies),
            "mean_ms": np.mean(latencies) * 1000,
            "p50_ms": np.percentile(latencies, 50) * 1000,
            "p95_ms": np.percentile(latencies, 95) * 1000,
            "max_ms": np.max(latencies) * 1000,
        }
        print(f"    docs={stage_target}, store={store_size}: "
              f"mean={results[stage_target]['mean_ms']:.3f}ms, "
              f"p95={results[stage_target]['p95_ms']:.3f}ms")

    return results


def scenario_concurrent_rw(pipeline: RAGPipeline, duration_sec: float,
                            writer_rate_per_sec: float) -> Dict:
    """Scenario 4: Concurrent reads + writes — thread safety and read latency under write load."""
    print(f"\n  [场景4] 并发读写: 写 {writer_rate_per_sec}/s + 持续查询, {duration_sec}s")

    stop = threading.Event()
    write_count = [0]
    write_errors = [0]
    read_latencies = []
    read_errors = [0]

    base_time = datetime(2024, 9, 1)

    def writer():
        interval = 1.0 / writer_rate_per_sec
        while not stop.is_set():
            try:
                doc = generate_news_batch(1, base_time + timedelta(seconds=write_count[0]))[0]
                pipeline.ingest_put(doc)
                write_count[0] += 1
            except Exception:
                write_errors[0] += 1
            time.sleep(interval)

    def reader():
        queries = ["BTC", "ETH", "AAPL", "利率", "升级"]
        while not stop.is_set():
            q = random.choice(queries)
            try:
                t0 = time.perf_counter()
                pipeline.retrieve(q, top_k=3)
                read_latencies.append(time.perf_counter() - t0)
            except Exception:
                read_errors[0] += 1
            time.sleep(0.01)

    writers = [threading.Thread(target=writer, daemon=True) for _ in range(2)]
    readers = [threading.Thread(target=reader, daemon=True) for _ in range(3)]
    for t in writers + readers:
        t.start()

    time.sleep(duration_sec)
    stop.set()
    for t in writers + readers:
        t.join(timeout=2)

    # drain queue
    time.sleep(1.0)

    return {
        "writes": write_count[0],
        "write_errors": write_errors[0],
        "reads": len(read_latencies),
        "read_errors": read_errors[0],
        "read_mean_ms": np.mean(read_latencies) * 1000 if read_latencies else 0,
        "read_p95_ms": np.percentile(read_latencies, 95) * 1000 if read_latencies else 0,
        "read_max_ms": np.max(read_latencies) * 1000 if read_latencies else 0,
    }


def scenario_memory_profile(docs_count: int) -> Dict:
    """Scenario 5: Memory usage for a given store size."""
    print(f"\n  [场景5] 内存分析: 入库 {docs_count} 篇文档")

    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()
    mem_before = sum(stat.size for stat in snap_before.statistics("filename"))

    config = RAGConfig(chunk_size=256, use_hybrid=True, dedup_cache_max=0)
    pipeline = RAGPipeline(config=config, start_worker=False)

    docs = generate_news_batch(docs_count, datetime(2024, 10, 1))
    t0 = time.perf_counter()
    n_chunks = pipeline.add_documents(docs)
    ingest_time = time.perf_counter() - t0

    snap_after = tracemalloc.take_snapshot()
    mem_after = sum(stat.size for stat in snap_after.statistics("filename"))
    tracemalloc.stop()

    mem_delta_mb = (mem_after - mem_before) / (1024 * 1024)
    stats = pipeline.health_check()

    return {
        "docs": docs_count,
        "chunks": n_chunks,
        "ingest_time_sec": ingest_time,
        "memory_delta_mb": mem_delta_mb,
        "bytes_per_chunk": (mem_after - mem_before) / max(n_chunks, 1),
        "vector_store_size": stats["vector_store_size"],
        "keyword_index_size": stats["keyword_index_size"],
    }


def scenario_component_breakdown(n_docs: int) -> Dict:
    """Scenario 6: Per-component timing breakdown."""
    print(f"\n  [场景6] 组件耗时分解: {n_docs} 篇文档")

    from quant_framework.rag.processing import Chunker, TextNormalizer, ProcessingPipeline
    from quant_framework.rag.processing.embedder import DummyEmbedder
    from quant_framework.rag.store.vector_store import InMemoryVectorStore
    from quant_framework.rag.store.keyword_index import KeywordIndex

    docs = generate_news_batch(n_docs, datetime(2024, 11, 1))
    results = {}

    # Normalize
    norm = TextNormalizer()
    t0 = time.perf_counter()
    normalized = [(doc, norm.normalize_and_filter(doc.content)) for doc in docs]
    normalized = [(d, n) for d, n in normalized if n]
    results["normalize_ms"] = (time.perf_counter() - t0) * 1000

    # Chunk
    chunker = Chunker(chunk_size=256, chunk_overlap=32, by_sentence=True)
    t0 = time.perf_counter()
    all_chunks = []
    for doc, content in normalized:
        rebuild = Document(content=content, doc_id=doc.doc_id,
                          source=doc.source, metadata=doc.metadata, created_at=doc.created_at)
        all_chunks.extend(chunker.chunk_document(rebuild))
    results["chunk_ms"] = (time.perf_counter() - t0) * 1000
    results["total_chunks"] = len(all_chunks)

    # Embed
    embedder = DummyEmbedder(dimension=384)
    t0 = time.perf_counter()
    texts = [c.text for c in all_chunks]
    vectors = embedder.embed(texts)
    for c, v in zip(all_chunks, vectors):
        c.embedding = v
    results["embed_ms"] = (time.perf_counter() - t0) * 1000

    # Vector store add
    store = InMemoryVectorStore(dimension=384)
    t0 = time.perf_counter()
    store.add(all_chunks)
    results["vector_add_ms"] = (time.perf_counter() - t0) * 1000

    # Keyword index add
    kw_index = KeywordIndex()
    t0 = time.perf_counter()
    kw_index.add(all_chunks)
    results["keyword_add_ms"] = (time.perf_counter() - t0) * 1000

    # Search breakdown
    query_emb = embedder.embed(["比特币价格"])[0]
    t0 = time.perf_counter()
    for _ in range(100):
        store.search(query_emb, top_k=10)
    results["vector_search_100x_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    for _ in range(100):
        kw_index.search("比特币价格", top_k=10)
    results["keyword_search_100x_ms"] = (time.perf_counter() - t0) * 1000

    total = (results["normalize_ms"] + results["chunk_ms"] + results["embed_ms"]
             + results["vector_add_ms"] + results["keyword_add_ms"])
    results["total_ingest_ms"] = total

    return results


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("RAG 实时新闻推送压力测试")
    print(f"模拟场景: 每分钟 N 条新闻持续推送 + 实时查询")
    print("=" * 80)

    # Setup pipeline
    config = RAGConfig(
        chunk_size=256,
        chunk_overlap=32,
        use_hybrid=True,
        hybrid_weights=(0.6, 0.4),
        vector_top_k=10,
        keyword_top_k=10,
        rerank_top_k=5,
        dedup_cache_max=50000,
        ingest_queue_max_size=10000,
        process_batch_size=16,
    )

    # ---- Scenario 6: Component breakdown (do this first, cleanly) ----
    breakdown_100 = scenario_component_breakdown(100)
    breakdown_1000 = scenario_component_breakdown(1000)
    breakdown_5000 = scenario_component_breakdown(5000)

    # ---- Scenario 2: Burst ingestion ----
    pipeline = RAGPipeline(config=config, start_worker=False)
    burst = scenario_burst_ingestion(pipeline, [10, 50, 100, 500, 1000])

    # ---- Scenario 3: Retrieval latency curve ----
    pipeline2 = RAGPipeline(config=config, start_worker=False)
    latency_curve = scenario_retrieval_under_load(pipeline2, [50, 200, 500, 1000, 2000, 5000])

    # ---- Scenario 1: Sustained ingestion (simulate 1-min frequency) ----
    pipeline3 = RAGPipeline(config=config, start_worker=True)
    # Pre-load some docs
    pipeline3.add_documents(generate_news_batch(100, datetime(2024, 5, 1)))
    sustained = scenario_sustained_ingestion(pipeline3, duration_sec=10, docs_per_minute=120)
    pipeline3._worker_stop.set()
    time.sleep(1)

    # ---- Scenario 4: Concurrent read/write ----
    pipeline4 = RAGPipeline(config=config, start_worker=True)
    pipeline4.add_documents(generate_news_batch(200, datetime(2024, 5, 1)))
    concurrent = scenario_concurrent_rw(pipeline4, duration_sec=8, writer_rate_per_sec=10)
    pipeline4._worker_stop.set()
    time.sleep(1)

    # ---- Scenario 5: Memory ----
    mem_1k = scenario_memory_profile(1000)
    mem_5k = scenario_memory_profile(5000)

    # ---- Generate Report ----
    print("\n" + "=" * 80)
    print("实时新闻推送压力测试报告")
    print("=" * 80)

    print("\n一、组件耗时分解 (DummyEmbedder, 单位: ms)")
    print("-" * 70)
    print(f"{'组件':<25} {'100 docs':>12} {'1000 docs':>12} {'5000 docs':>12}")
    print("-" * 70)
    for key in ["normalize_ms", "chunk_ms", "embed_ms", "vector_add_ms", "keyword_add_ms", "total_ingest_ms"]:
        label = key.replace("_ms", "").replace("_", " ").title()
        print(f"  {label:<23} {breakdown_100[key]:>10.2f}   {breakdown_1000[key]:>10.2f}   {breakdown_5000[key]:>10.2f}")
    print(f"  {'Chunks produced':<23} {breakdown_100['total_chunks']:>10}   {breakdown_1000['total_chunks']:>10}   {breakdown_5000['total_chunks']:>10}")
    print(f"  {'Vector search 100x':<23} {breakdown_100['vector_search_100x_ms']:>10.2f}   {breakdown_1000['vector_search_100x_ms']:>10.2f}   {breakdown_5000['vector_search_100x_ms']:>10.2f}")
    print(f"  {'Keyword search 100x':<23} {breakdown_100['keyword_search_100x_ms']:>10.2f}   {breakdown_1000['keyword_search_100x_ms']:>10.2f}   {breakdown_5000['keyword_search_100x_ms']:>10.2f}")

    print("\n二、突发入库吞吐")
    print("-" * 70)
    print(f"  {'Batch Size':<12} {'Chunks':>8} {'Time (ms)':>12} {'Docs/sec':>12} {'Chunks/sec':>12}")
    for bs, r in burst.items():
        print(f"  {bs:<12} {r['chunks']:>8} {r['time_ms']:>12.1f} {r['throughput_docs_sec']:>12.0f} {r['throughput_chunks_sec']:>12.0f}")

    print("\n三、库容增长 → 检索延迟退化曲线")
    print("-" * 70)
    print(f"  {'Docs':<8} {'Store Size':>12} {'Mean (ms)':>12} {'P50 (ms)':>12} {'P95 (ms)':>12} {'Max (ms)':>12}")
    for stage, r in latency_curve.items():
        print(f"  {stage:<8} {r['store_size']:>12} {r['mean_ms']:>12.3f} {r['p50_ms']:>12.3f} {r['p95_ms']:>12.3f} {r['max_ms']:>12.3f}")

    print("\n四、持续入库测试 (模拟 120 docs/min)")
    print("-" * 70)
    print(f"  入库成功: {sustained['total_put']} 条")
    print(f"  入库拒绝: {sustained['total_rejected']} 条")
    print(f"  实际速率: {sustained['actual_rate_per_min']:.0f} docs/min")
    print(f"  入队延迟: mean={sustained['put_mean_us']:.0f}us, p99={sustained['put_p99_us']:.0f}us")

    print("\n五、并发读写测试 (2 writer + 3 reader, 8s)")
    print("-" * 70)
    print(f"  总写入: {concurrent['writes']} 条, 写入错误: {concurrent['write_errors']}")
    print(f"  总查询: {concurrent['reads']} 次, 查询错误: {concurrent['read_errors']}")
    print(f"  读延迟: mean={concurrent['read_mean_ms']:.3f}ms, "
          f"p95={concurrent['read_p95_ms']:.3f}ms, max={concurrent['read_max_ms']:.3f}ms")

    print("\n六、内存分析")
    print("-" * 70)
    print(f"  {'Docs':<8} {'Chunks':>8} {'Memory (MB)':>14} {'Bytes/Chunk':>14} {'Ingest (s)':>12}")
    for m in [mem_1k, mem_5k]:
        print(f"  {m['docs']:<8} {m['chunks']:>8} {m['memory_delta_mb']:>14.2f} {m['bytes_per_chunk']:>14.0f} {m['ingest_time_sec']:>12.3f}")

    # Projection
    print("\n七、每分钟推送场景推算")
    print("-" * 70)
    rates = [10, 30, 60, 120, 300, 600, 1000]
    per_doc_ingest_ms = breakdown_1000["total_ingest_ms"] / 1000
    per_query_ms = latency_curve[1000]["mean_ms"] if 1000 in latency_curve else 0.1
    bytes_per_chunk = mem_5k["bytes_per_chunk"]
    chunks_per_doc = breakdown_1000["total_chunks"] / 1000

    print(f"  {'Rate (docs/min)':<18} {'Ingest CPU%':>14} {'Query OK?':>12} {'1h RAM (MB)':>14} {'1d RAM (MB)':>14}")
    for rate in rates:
        docs_per_sec = rate / 60.0
        ingest_cpu_pct = docs_per_sec * per_doc_ingest_ms / 10.0
        query_budget_ms = (1000 / docs_per_sec) if docs_per_sec > 0 else 999999
        query_ok = "OK" if per_query_ms < query_budget_ms * 0.5 else "RISK"
        hourly_chunks = rate * 60 * chunks_per_doc
        daily_chunks = hourly_chunks * 24
        hourly_mb = hourly_chunks * bytes_per_chunk / (1024 * 1024)
        daily_mb = daily_chunks * bytes_per_chunk / (1024 * 1024)
        print(f"  {rate:<18} {ingest_cpu_pct:>13.1f}% {query_ok:>12} {hourly_mb:>14.1f} {daily_mb:>14.1f}")

    print("\n八、瓶颈分析与优化建议")
    print("-" * 70)

    embed_pct = breakdown_1000["embed_ms"] / breakdown_1000["total_ingest_ms"] * 100
    kw_pct = breakdown_1000["keyword_add_ms"] / breakdown_1000["total_ingest_ms"] * 100
    vec_pct = breakdown_1000["vector_add_ms"] / breakdown_1000["total_ingest_ms"] * 100
    norm_pct = breakdown_1000["normalize_ms"] / breakdown_1000["total_ingest_ms"] * 100
    chunk_pct = breakdown_1000["chunk_ms"] / breakdown_1000["total_ingest_ms"] * 100

    print(f"  入库耗时占比 (1000 docs):")
    print(f"    嵌入 (Embed):       {embed_pct:5.1f}%  ← {'瓶颈!' if embed_pct > 50 else '正常'}")
    print(f"    关键词索引 (KW):    {kw_pct:5.1f}%  ← {'瓶颈!' if kw_pct > 30 else '正常'}")
    print(f"    向量存储 (Vec):     {vec_pct:5.1f}%  ← {'需关注' if vec_pct > 20 else '正常'}")
    print(f"    规范化 (Norm):      {norm_pct:5.1f}%")
    print(f"    分块 (Chunk):       {chunk_pct:5.1f}%")

    print(f"\n  检索延迟随库容增长:")
    if len(latency_curve) >= 2:
        stages = list(latency_curve.keys())
        first_ms = latency_curve[stages[0]]["mean_ms"]
        last_ms = latency_curve[stages[-1]]["mean_ms"]
        ratio = last_ms / first_ms if first_ms > 0 else 1
        print(f"    {stages[0]} docs → {stages[-1]} docs: {first_ms:.3f}ms → {last_ms:.3f}ms ({ratio:.1f}x)")
        if ratio > 5:
            print(f"    ⚠ 延迟增长 {ratio:.0f}x，需要优化:")
            print(f"      - 启用 max_vectors FIFO 淘汰，控制索引大小")
            print(f"      - 在 core.py 启用 Numba JIT")
            print(f"      - 考虑分区索引 (按 symbol/时间分片)")
        else:
            print(f"    延迟增长可控 ({ratio:.1f}x)")

    print(f"\n  实际使用 SentenceTransformer 后的影响:")
    print(f"    当前 DummyEmbedder: ~{breakdown_1000['embed_ms']/1000:.3f}ms/doc")
    print(f"    SentenceTransformer (CPU): 预计 ~5-20ms/doc (384维)")
    print(f"    SentenceTransformer (GPU): 预计 ~0.5-2ms/doc (384维)")
    print(f"    → 对于 120 docs/min (2/s)，CPU 嵌入可能成为瓶颈")
    print(f"    → 建议: GPU 推理 或 批量嵌入 (当前已支持 batch embed)")

    print("\n  综合结论:")
    print("  ├─ 10-60 docs/min:   轻松应对，无需额外优化")
    print("  ├─ 60-300 docs/min:  需要 SentenceTransformer GPU 或异步批量嵌入")
    print("  ├─ 300-1000 docs/min: 需要 max_vectors 淘汰 + Numba + 分区索引")
    print("  └─ >1000 docs/min:   需要外部向量数据库 (Milvus/Qdrant) + 分布式嵌入")
    print("=" * 80)


if __name__ == "__main__":
    main()
