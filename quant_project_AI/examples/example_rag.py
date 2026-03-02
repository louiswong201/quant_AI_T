"""
RAG 使用示例：实时接入非结构化数据 → 检索 → 生成上下文

运行前可选安装: pip install sentence-transformers
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_framework.rag import RAGConfig, RAGPipeline, Document  # noqa: E402


def main():
    config = RAGConfig(
        chunk_size=256,
        chunk_overlap=32,
        vector_top_k=10,
        use_hybrid=True,
        rerank_top_k=5,
    )
    pipeline = RAGPipeline(config=config)

    # 1) 批量写入
    docs = [
        Document(content="苹果公司发布新款iPhone，搭载A18芯片，续航提升20%。", source="news_1"),
        Document(content="特斯拉Q3交付量创新高，马斯克称将扩大上海工厂产能。", source="news_2"),
        Document(content="美联储维持利率不变，鲍威尔称将依赖数据做决策。", source="news_3"),
    ]
    n = pipeline.add_documents(docs)
    print(f"已写入 {n} 个 chunks")

    # 2) 实时写入（模拟）
    pipeline.ingest_put(Document(content="国内新能源车销量同比增40%，比亚迪领跑。", source="realtime"))
    pipeline.ingest_put(Document(content="英伟达发布新一代AI芯片，算力翻倍。", source="realtime"))
    # 消费队列中的文档并入库
    batch = pipeline.ingest_queue.take(10)
    if batch:
        pipeline.add_documents(batch)
        print(f"实时接入并处理 {len(batch)} 条")

    # 3) 检索
    query = "苹果和特斯拉最近有什么动态？"
    results = pipeline.retrieve(query, top_k=5)
    print(f"\n检索: {query}")
    for r in results:
        print(f"  [{r.rank}] score={r.score:.4f} | {r.chunk.text[:60]}...")

    # 4) 获取生成用上下文（可拼进 LLM prompt）
    context = pipeline.get_context_for_generation(query, top_k=3, max_chars=2000)
    print("\n生成用上下文片段:")
    print(context[:500])


def run_with_file_watcher():
    """使用文件监听实时接入目录内新增/变更文件"""
    import threading
    import time
    from pathlib import Path
    from quant_framework.rag import RAGPipeline, RAGConfig
    from quant_framework.rag.ingestion import FileWatcherIngestAdapter

    config = RAGConfig(chunk_size=256, use_hybrid=True)
    pipeline = RAGPipeline(config=config)
    watch_dir = Path(__file__).parent / "rag_docs"
    watch_dir.mkdir(exist_ok=True)

    stop = threading.Event()

    def on_docs(docs):
        pipeline.add_documents(docs)
        print(f"文件监听: 处理 {len(docs)} 个文档")

    adapter = FileWatcherIngestAdapter(str(watch_dir), poll_interval=2.0)
    t = threading.Thread(target=adapter.run_forever, args=(on_docs, lambda: stop.is_set()))
    t.daemon = True
    t.start()
    print("文件监听已启动，将 rag_docs 目录下放入 .txt/.md 文件即可自动入库。按 Ctrl+C 退出。")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop.set()


if __name__ == "__main__":
    main()
    # run_with_file_watcher()  # 取消注释可运行文件监听示例
