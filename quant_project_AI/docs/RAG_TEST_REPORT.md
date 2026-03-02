================================================================================
RAG 全链路端到端测试报告
测试时间: 2026-02-18 13:23:17
================================================================================

数据源: 15 条模拟新闻 + 5 条 yfinance 实时数据

测试结果: 13/13 通过, 0 失败
--------------------------------------------------------------------------------

============================================================
[PASS ✓] 1. 文档创建与类型校验
============================================================
  [INFO] ⏱  document_creation: 0.03 ms
  [PASS] 创建 15 个文档
  [PASS] 文档 reuters 有 doc_id: cd6f1333...
  [PASS] 文档 reuters 有 created_at: 2024-01-15 00:00:00
  [PASS] 文档 reuters 内容非空 (142 字)
  [PASS] 文档 bloomberg 有 doc_id: b316d000...
  [PASS] 文档 bloomberg 有 created_at: 2024-01-20 00:00:00
  [PASS] 文档 bloomberg 内容非空 (151 字)
  [PASS] 文档 coindesk 有 doc_id: 360db0df...
  [PASS] 文档 coindesk 有 created_at: 2024-01-25 00:00:00
  [PASS] 文档 coindesk 内容非空 (150 字)
  [PASS] 文档 coindesk 有 doc_id: 876dcdcb...
  [PASS] 文档 coindesk 有 created_at: 2024-01-30 00:00:00
  [PASS] 文档 coindesk 内容非空 (171 字)
  [PASS] 文档 reuters 有 doc_id: 8a19413d...
  [PASS] 文档 reuters 有 created_at: 2024-02-04 00:00:00
  [PASS] 文档 reuters 内容非空 (142 字)
  [PASS] 文档 bloomberg 有 doc_id: bfa69ce5...
  [PASS] 文档 bloomberg 有 created_at: 2024-02-09 00:00:00
  [PASS] 文档 bloomberg 内容非空 (155 字)
  [PASS] 文档 coindesk 有 doc_id: 148ac394...
  [PASS] 文档 coindesk 有 created_at: 2024-02-14 00:00:00
  [PASS] 文档 coindesk 内容非空 (155 字)
  [PASS] 文档 coindesk 有 doc_id: 86124b90...
  [PASS] 文档 coindesk 有 created_at: 2024-02-19 00:00:00
  [PASS] 文档 coindesk 内容非空 (164 字)
  [PASS] 文档 reuters 有 doc_id: 26033225...
  [PASS] 文档 reuters 有 created_at: 2024-02-24 00:00:00
  [PASS] 文档 reuters 内容非空 (156 字)
  [PASS] 文档 bloomberg 有 doc_id: b2609ff6...
  [PASS] 文档 bloomberg 有 created_at: 2024-02-29 00:00:00
  [PASS] 文档 bloomberg 内容非空 (180 字)
  [PASS] 文档 coindesk 有 doc_id: 844a6263...
  [PASS] 文档 coindesk 有 created_at: 2024-03-15 00:00:00
  [PASS] 文档 coindesk 内容非空 (151 字)
  [PASS] 文档 coindesk 有 doc_id: b3f6a1ee...
  [PASS] 文档 coindesk 有 created_at: 2024-04-14 00:00:00
  [PASS] 文档 coindesk 内容非空 (151 字)
  [PASS] 文档 bloomberg 有 doc_id: 334d8537...
  [PASS] 文档 bloomberg 有 created_at: 2024-04-24 00:00:00
  [PASS] 文档 bloomberg 内容非空 (140 字)
  [PASS] 文档 coindesk 有 doc_id: 86f8becd...
  [PASS] 文档 coindesk 有 created_at: 2024-05-14 00:00:00
  [PASS] 文档 coindesk 内容非空 (161 字)
  [PASS] 文档 coindesk 有 doc_id: ea293450...
  [PASS] 文档 coindesk 有 created_at: 2024-06-13 00:00:00
  [PASS] 文档 coindesk 内容非空 (173 字)
  [PASS] 相同内容产生相同 doc_id（去重基础）

  性能指标:
    document_creation: 0.03 ms

  关键数据:
    total_docs: 15
    total_chars: 2342

============================================================
[PASS ✓] 2. 分块逻辑测试
============================================================
  [INFO] ⏱  chunking_all_docs: 0.08 ms
  [PASS] 共产生 15 个分块
  [PASS] chunk cd6f133325b98bd6_0 有 ID
  [PASS] chunk 长度 148 >= min_chunk_length(10)
  [PASS] chunk b316d000925d19eb_0 有 ID
  [PASS] chunk 长度 155 >= min_chunk_length(10)
  [PASS] chunk 360db0df415e28c7_0 有 ID
  [PASS] chunk 长度 154 >= min_chunk_length(10)
  [PASS] chunk 876dcdcbdab1b33e_0 有 ID
  [PASS] chunk 长度 176 >= min_chunk_length(10)
  [PASS] chunk 8a19413d5503f2ca_0 有 ID
  [PASS] chunk 长度 149 >= min_chunk_length(10)
  [PASS] chunk bfa69ce562803ecc_0 有 ID
  [PASS] chunk 长度 159 >= min_chunk_length(10)
  [PASS] chunk 148ac3946f305f25_0 有 ID
  [PASS] chunk 长度 159 >= min_chunk_length(10)
  [PASS] chunk 86124b908a0910a3_0 有 ID
  [PASS] chunk 长度 168 >= min_chunk_length(10)
  [PASS] chunk 26033225d33501e4_0 有 ID
  [PASS] chunk 长度 165 >= min_chunk_length(10)
  [PASS] chunk b2609ff65d9b6c4c_0 有 ID
  [PASS] chunk 长度 184 >= min_chunk_length(10)
  [PASS] chunk 844a626329e74c76_0 有 ID
  [PASS] chunk 长度 155 >= min_chunk_length(10)
  [PASS] chunk b3f6a1ee26b17978_0 有 ID
  [PASS] chunk 长度 155 >= min_chunk_length(10)
  [PASS] chunk 334d8537f1e6ded7_0 有 ID
  [PASS] chunk 长度 142 >= min_chunk_length(10)
  [PASS] chunk 86f8becd69cd7b7f_0 有 ID
  [PASS] chunk 长度 163 >= min_chunk_length(10)
  [PASS] chunk ea2934509720095b_0 有 ID
  [PASS] chunk 长度 176 >= min_chunk_length(10)
  [PASS] created_at 传播: 15/15 个 chunk 含 created_at 元数据
  [PASS] created_at 类型正确: datetime

  性能指标:
    chunking_all_docs: 0.08 ms

  关键数据:
    total_chunks: 15
    avg_chunk_len: 160.53333333333333
    max_chunk_len: 184
    min_chunk_len: 142

============================================================
[PASS ✓] 3. 嵌入器测试
============================================================
  [INFO] 使用嵌入器: DummyEmbedder
  [PASS] 向量维度: 384
  [INFO] ⏱  embed_4_texts: 0.18 ms
  [PASS] 返回 4 个向量
  [PASS] 向量 0 维度 = 384
  [PASS] 向量 0 范数 = 1.0000 (非零)
  [PASS] 向量 1 维度 = 384
  [PASS] 向量 1 范数 = 1.0000 (非零)
  [PASS] 向量 2 维度 = 384
  [PASS] 向量 2 范数 = 1.0000 (非零)
  [PASS] 向量 3 维度 = 384
  [PASS] 向量 3 范数 = 1.0000 (非零)
  [INFO] ⏱  embed_determinism_check: 0.07 ms
  [PASS] 向量 0 确定性: max_diff = 0.00e+00
  [PASS] 向量 1 确定性: max_diff = 0.00e+00
  [PASS] 向量 2 确定性: max_diff = 0.00e+00
  [PASS] 向量 3 确定性: max_diff = 0.00e+00
  [INFO] BTC vs ETH 相似度: 0.0548
  [INFO] BTC vs AAPL 相似度: -0.0320
  [WARN] DummyEmbedder 无法保证语义相似度排序，检索将主要依赖 BM25 关键词路径

  性能指标:
    embed_4_texts: 0.18 ms
    embed_determinism_check: 0.07 ms

  关键数据:
    embedder_type: DummyEmbedder

============================================================
[PASS ✓] 4. 处理管道端到端测试
============================================================
  [INFO] ⏱  process_all_documents: 0.46 ms
  [PASS] 处理完成: 15 个嵌入分块
  [PASS] [BUG FIX 验证] created_at 传播: 15/15 (修复前为 0)
  [PASS] chunk cd6f133325b9... 有嵌入向量
  [PASS] 嵌入维度正确: 384
  [PASS] chunk b316d000925d... 有嵌入向量
  [PASS] 嵌入维度正确: 384
  [PASS] chunk 360db0df415e... 有嵌入向量
  [PASS] 嵌入维度正确: 384
  [PASS] chunk 876dcdcbdab1... 有嵌入向量
  [PASS] 嵌入维度正确: 384
  [PASS] chunk 8a19413d5503... 有嵌入向量
  [PASS] 嵌入维度正确: 384
  [PASS] chunk bfa69ce56280... 有嵌入向量
  [PASS] 嵌入维度正确: 384
  [PASS] chunk 148ac3946f30... 有嵌入向量
  [PASS] 嵌入维度正确: 384
  [PASS] chunk 86124b908a09... 有嵌入向量
  [PASS] 嵌入维度正确: 384
  [PASS] chunk 26033225d335... 有嵌入向量
  [PASS] 嵌入维度正确: 384
  [PASS] chunk b2609ff65d9b... 有嵌入向量
  [PASS] 嵌入维度正确: 384
  [PASS] chunk 844a626329e7... 有嵌入向量
  [PASS] 嵌入维度正确: 384
  [PASS] chunk b3f6a1ee26b1... 有嵌入向量
  [PASS] 嵌入维度正确: 384
  [PASS] chunk 334d8537f1e6... 有嵌入向量
  [PASS] 嵌入维度正确: 384
  [PASS] chunk 86f8becd69cd... 有嵌入向量
  [PASS] 嵌入维度正确: 384
  [PASS] chunk ea2934509720... 有嵌入向量
  [PASS] 嵌入维度正确: 384

  性能指标:
    process_all_documents: 0.46 ms

  关键数据:
    processed_chunks: 15

============================================================
[PASS ✓] 5. 向量存储测试
============================================================
  [PASS] 初始大小 = 0
  [INFO] ⏱  add_5_chunks: 0.02 ms
  [PASS] 添加后大小 = 5
  [PASS] 无淘汰
  [INFO] ⏱  search_top3: 1053.71 ms
  [PASS] 检索返回 3 个结果
  [PASS] 最相似: test_0 (所有正向量，归一化后均为 [1,0,0,0]，cosine=1.0)
  [INFO] ⏱  add_8_trigger_eviction: 0.06 ms
  [PASS] 淘汰后大小 = 10 (max=10)
  [PASS] 淘汰 3 个 (预期 3)
  [INFO] ⏱  search_batch_2queries: 125.22 ms
  [PASS] 批量检索返回 2 组

  性能指标:
    add_5_chunks: 0.02 ms
    search_top3: 1053.71 ms
    add_8_trigger_eviction: 0.06 ms
    search_batch_2queries: 125.22 ms

============================================================
[PASS ✓] 6. 关键词索引 (BM25) 测试
============================================================
  [INFO] ⏱  add_to_keyword_index: 0.88 ms
  [PASS] 索引大小: 15
  [INFO] ⏱  bm25_search_BTC: 0.13 ms
  [PASS] 查询 '比特币 减半 价格': 返回 3 个结果
  [INFO]   Top-1 (BTC): score=22.4440, text=比特币完成第四次减半，市场反应平稳  比特币在区块高度840000完成第四次减半，区块奖励从6.25BTC降至3.125...
  [INFO] ⏱  bm25_search_ETH: 0.02 ms
  [PASS] 查询 '以太坊 升级 Gas': 返回 3 个结果
  [INFO]   Top-1 (ETH): score=21.5377, text=以太坊升级完成，Gas费大幅下降  以太坊Dencun升级成功上线，引入Proto-Danksharding技术。Lay...
  [INFO] ⏱  bm25_search_AAPL: 0.01 ms
  [PASS] 查询 '苹果 iPhone 财报': 返回 2 个结果
  [INFO]   Top-1 (AAPL): score=22.6393, text=苹果公司Q4财报超预期，iPhone销量创纪录  苹果公司发布2024年第四季度财报，营收达1195亿美元，同比增长6%...
  [INFO] ⏱  bm25_search_XRP: 0.01 ms
  [PASS] 查询 'XRP SEC 诉讼': 返回 3 个结果
  [INFO]   Top-1 (XRP): score=16.8228, text=XRP赢得SEC诉讼关键裁决，价格飙升  美国法院在Ripple与SEC的诉讼中作出重要裁决，认定XRP在二级市场交易不...
  [INFO] ⏱  bm25_search_Quant: 0.02 ms
  [PASS] 查询 '量化 交易 策略': 返回 3 个结果
  [INFO]   Top-1 (Quant): score=22.6405, text=全球量化交易市场规模突破2万亿美元  根据最新行业报告，全球量化交易市场规模已突破2万亿美元，年增长率达15%。高频交易...

  性能指标:
    add_to_keyword_index: 0.88 ms
    bm25_search_BTC: 0.13 ms
    bm25_search_ETH: 0.02 ms
    bm25_search_AAPL: 0.01 ms
    bm25_search_XRP: 0.01 ms
    bm25_search_Quant: 0.02 ms

============================================================
[PASS ✓] 7. RAGPipeline 端到端测试
============================================================
  [INFO] ⏱  pipeline_init: 0.21 ms
  [INFO] 嵌入器类型: DummyEmbedder
  [INFO] ⏱  batch_ingest: 1.55 ms
  [PASS] 入库 20 个分块 (来自 20 个文档)
  [INFO] Pipeline 状态: vector=20, keyword=20
  [PASS] 向量存储非空
  [PASS] 关键词索引非空
  [INFO] ⏱  dedup_reingest: 0.02 ms
  [PASS] 去重生效: 重复入库返回 0 (预期 0)
  [INFO] ⏱  retrieve_BTC行情: 0.31 ms
  [PASS] 查询 'BTC行情': 返回 5 个结果
  [INFO]   Top-1: score=0.0162, source=coindesk
  [INFO]   文本: 比特币完成第四次减半，市场反应平稳 比特币在区块高度840000完成第四次减半，区块奖励从6. 25BTC降至3. 125BTC。 与此前几次减半不同，市场反应...
  [INFO] ⏱  retrieve_ETH生态: 0.09 ms
  [PASS] 查询 'ETH生态': 返回 5 个结果
  [INFO]   Top-1: score=0.0160, source=coindesk
  [INFO]   文本: SEC批准以太坊现货ETF，加密市场迎来第二波利好 美国证券交易委员会（SEC）正式批准了多只以太坊现货ETF的上市申请。 贝莱德、灰度和富达等机构的ETH E...
  [INFO] ⏱  retrieve_AAPL基本面: 0.07 ms
  [PASS] 查询 'AAPL基本面': 返回 5 个结果
  [INFO]   Top-1: score=0.0161, source=bloomberg
  [INFO]   文本: 全球量化交易市场规模突破2万亿美元 根据最新行业报告，全球量化交易市场规模已突破2万亿美元，年增长率达15%。 高频交易占美国股市交易量的50%以上。 AI和机...
  [INFO] ⏱  retrieve_XRP动态: 0.07 ms
  [PASS] 查询 'XRP动态': 返回 5 个结果
  [INFO]   Top-1: score=0.0164, source=coindesk
  [INFO]   文本: XRP赢得SEC诉讼关键裁决，价格飙升 美国法院在Ripple与SEC的诉讼中作出重要裁决，认定XRP在二级市场交易不构成证券。 这一裁决对整个加密市场具有里程...
  [INFO] ⏱  retrieve_宏观: 0.07 ms
  [PASS] 查询 '宏观': 返回 5 个结果
  [INFO]   Top-1: score=0.0161, source=coindesk
  [INFO]   文本: SEC批准以太坊现货ETF，加密市场迎来第二波利好 美国证券交易委员会（SEC）正式批准了多只以太坊现货ETF的上市申请。 贝莱德、灰度和富达等机构的ETH E...
  [INFO] ⏱  retrieve_量化: 0.07 ms
  [PASS] 查询 '量化': 返回 5 个结果
  [INFO]   Top-1: score=0.0154, source=yfinance_TSLA
  [INFO]   文本: TSLA 最近5d行情概要： 最新收盘价: 410. 63，最高: 436. 35，最低: 400. 51，区间涨跌幅: -3. 43%，日均成交量: 58,9...
  [INFO] ⏱  get_context_concat: 0.06 ms
  [PASS] 上下文生成 (concat): 413 字
  [INFO]   上下文前 100 字: [1] 比特币突破65000美元，机构资金持续流入 比特币价格突破65000美元大关，24小时涨幅达4. 2%。 灰度GBTC转为ETF后资金流出放缓，而贝莱德和富达的比特币ETF持续吸引资金流入。 ...
  [INFO] ⏱  batch_retrieve_3q: 0.18 ms
  [PASS] 批量检索: 3 组结果

  性能指标:
    pipeline_init: 0.21 ms
    batch_ingest: 1.55 ms
    dedup_reingest: 0.02 ms
    retrieve_BTC行情: 0.31 ms
    retrieve_ETH生态: 0.09 ms
    retrieve_AAPL基本面: 0.07 ms
    retrieve_XRP动态: 0.07 ms
    retrieve_宏观: 0.07 ms
    retrieve_量化: 0.07 ms
    get_context_concat: 0.06 ms
    batch_retrieve_3q: 0.18 ms

  关键数据:
    total_ingested_chunks: 20
    vector_store_size: 20
    keyword_index_size: 20

============================================================
[PASS ✓] 8. 元数据过滤测试
============================================================
  [INFO] ⏱  filter_source_coindesk: 0.18 ms
  [PASS] source_contains='coindesk': 全部 5 个结果来自 coindesk
  [INFO] ⏱  filter_created_before: 0.08 ms
  [PASS] created_before 过滤: 0 个违规 (预期 0)
  [INFO]   cutoff=2024-02-20 00:00:00, 返回 5 个结果

  性能指标:
    filter_source_coindesk: 0.18 ms
    filter_created_before: 0.08 ms

============================================================
[PASS ✓] 9. 异步入队与后台 Worker 测试
============================================================
  [PASS] 后台 worker 已启动
  [PASS] 入队成功: reuters
  [PASS] 入队成功: bloomberg
  [PASS] 入队成功: coindesk
  [PASS] 入队成功: coindesk
  [PASS] 入队成功: reuters
  [INFO] ⏱  async_ingest_5docs: 0.01 ms
  [INFO] Worker 处理耗时约 0.2s, 队列剩余: 0
  [PASS] 队列已清空
  [PASS] Worker 入库成功: 5 个向量

  性能指标:
    async_ingest_5docs: 0.01 ms

============================================================
[PASS ✓] 10. RagContextProvider 策略集成测试
============================================================
  [INFO] ⏱  get_context_basic: 0.36 ms
  [PASS] 基本上下文: 325 字
  [INFO] ⏱  get_context_with_symbol: 0.16 ms
  [PASS] 带标的上下文: 828 字
  [INFO] ⏱  get_context_as_of_date: 0.16 ms
  [INFO] as_of_date=2024-02-01 00:00:00 上下文: 656 字
  [INFO] ⏱  provider_ingest_document: 0.03 ms
  [PASS] 通过 Provider 入队文档
  [INFO] ⏱  provider_add_dicts: 0.17 ms
  [PASS] 字典格式入库: 1 个分块

  性能指标:
    get_context_basic: 0.36 ms
    get_context_with_symbol: 0.16 ms
    get_context_as_of_date: 0.16 ms
    provider_ingest_document: 0.03 ms
    provider_add_dicts: 0.17 ms

============================================================
[PASS ✓] 11. RAG + 回测策略集成测试
============================================================
  [PASS] 策略 rag_provider 已注入
  [INFO] ⏱  on_bar_0115: 0.28 ms
  [INFO] ⏱  on_bar_0116: 0.10 ms
  [INFO] ⏱  on_bar_0117: 0.10 ms
  [INFO] ⏱  on_bar_0118: 0.10 ms
  [INFO] ⏱  on_bar_0119: 0.25 ms
  [INFO] ⏱  on_bar_0120: 0.18 ms
  [INFO] ⏱  on_bar_0121: 0.12 ms
  [INFO] ⏱  on_bar_0122: 0.10 ms
  [INFO] ⏱  on_bar_0123: 0.10 ms
  [INFO] ⏱  on_bar_0124: 0.10 ms
  [INFO] ⏱  on_bar_0125: 0.09 ms
  [INFO] ⏱  on_bar_0126: 0.09 ms
  [INFO] ⏱  on_bar_0127: 0.08 ms
  [INFO] ⏱  on_bar_0128: 0.09 ms
  [INFO] ⏱  on_bar_0129: 0.08 ms
  [INFO] ⏱  on_bar_0130: 0.08 ms
  [INFO] ⏱  on_bar_0131: 0.08 ms
  [INFO] ⏱  on_bar_0201: 0.08 ms
  [INFO] ⏱  on_bar_0202: 0.08 ms
  [INFO] ⏱  on_bar_0203: 0.08 ms
  [INFO] ⏱  on_bar_0204: 0.08 ms
  [INFO] ⏱  on_bar_0205: 0.08 ms
  [INFO] ⏱  on_bar_0206: 0.08 ms
  [INFO] ⏱  on_bar_0207: 0.09 ms
  [INFO] ⏱  on_bar_0208: 0.08 ms
  [INFO] ⏱  on_bar_0209: 0.08 ms
  [INFO] ⏱  on_bar_0210: 0.09 ms
  [INFO] ⏱  on_bar_0211: 0.08 ms
  [INFO] ⏱  on_bar_0212: 0.08 ms
  [INFO] ⏱  on_bar_0213: 0.08 ms
  [PASS] 30 个 bar 均获取上下文: 30
  [PASS] 30 个 bar 均生成信号: 30
  [INFO] 信号分布: buy=14, sell=15, hold=1
  [PASS] 有 29/30 个 bar 获取到非空上下文
  [INFO] 时序一致性: 早期平均上下文 122 字, 晚期 164 字

  性能指标:

  关键数据:
    signals: {'hold': 1, 'sell': 15, 'buy': 14}
    bars_with_context: 29

============================================================
[PASS ✓] 12. 性能基准测试
============================================================
  [INFO] ⏱  ingest_throughput: 2.06 ms
  [INFO] 入库吞吐: 7293 docs/sec (15 chunks)
  [INFO] ⏱  100_queries: 5.06 ms
  [INFO] 查询延迟 (100 次): mean=0.05ms, p50=0.05ms, p95=0.07ms, p99=0.10ms, max=0.23ms
  [INFO] ⏱  batch_5_queries: 0.65 ms
  [INFO] 批量查询 (5 条): 0.65ms
  [INFO] ⏱  context_generation: 0.30 ms
  [INFO] 上下文生成: 平均 0.06ms/次

  性能指标:
    ingest_throughput: 2.06 ms
    100_queries: 5.06 ms
    batch_5_queries: 0.65 ms
    context_generation: 0.30 ms

  关键数据:
    ingest_docs_per_sec: 7293.208369878772
    query_mean_ms: 0.050502110000039124
    query_p50_ms: 0.047416999999772
    query_p95_ms: 0.06913720000030206
    query_p99_ms: 0.10429791999930837
    query_max_ms: 0.23279200000025924
    batch_query_ms: 0.6474580000004337
    context_gen_mean_ms: 0.05967499999997017

============================================================
[PASS ✓] 13. 边界条件测试
============================================================
  [PASS] 空文档入库: 0 个分块 (预期 0)
  [INFO] 极短文档入库: 0 个分块
  [PASS] 长文档入库: 34 个分块 (预期 > 1)
  [PASS] Unicode 文档入库: 1 个分块
  [PASS] 空存储查询: 0 个结果 (预期 0)
  [PASS] chunk_size=0 正确抛出 ValueError
  [PASS] chunk_overlap >= chunk_size 正确抛出 ValueError

================================================================================
综合分析报告
================================================================================

一、架构流程分析
----------------------------------------
  Document → TextNormalizer → Chunker → Embedder → VectorStore + KeywordIndex
  Query → Embedder → VectorStore.search + KeywordIndex.search → RRF Merge → Reranker → Results
  Strategy.on_bar → RagContextProvider.get_context(as_of_date) → context string

二、发现的问题与修复
----------------------------------------
  [已修复] BUG-1: ProcessingPipeline 重建 Document 时丢失 created_at
           影响: as_of_date 过滤失效，回测中可能使用未来信息（look-ahead bias）
           修复: _rebuild_doc() 显式传递 created_at

  [已修复] BUG-2: DummyEmbedder 使用固定种子，所有文本相对位置共享伪随机序列
           影响: 不同文本的向量不确定地相似，向量检索路径近乎随机
           修复: 基于文本哈希生成种子，同文本始终同向量

  [观察] OBS-1: sentence-transformers 未安装，使用 DummyEmbedder 降级运行
         影响: 向量检索质量取决于伪随机向量，实际检索主要依赖 BM25 关键词路径
         建议: 生产环境必须安装 sentence-transformers 或接入外部嵌入 API

  [观察] OBS-2: RagContextProvider.get_context 在有 metadata_filter 时
         跳过 context_strategy 参数，直接使用简单拼接
         影响: merge_adjacent 策略对带过滤的查询不生效

  [观察] OBS-3: Chunker 句子分割仅支持中文标点和换行 (。！？!?\n)
         影响: 纯英文句子（以 . 结尾）不会按句子边界分块
         建议: 添加英文句号到 _sentence_pattern

三、性能基准总结
----------------------------------------
  入库吞吐: 7293 docs/sec
  单查询延迟 (100次):
    平均: 0.05 ms
    P50:  0.05 ms
    P95:  0.07 ms
    P99:  0.10 ms
    Max:  0.23 ms
  批量查询 (5条): 0.65 ms
  上下文生成: 0.06 ms/次

四、回测集成分析
----------------------------------------
  信号分布: {'hold': 1, 'sell': 15, 'buy': 14}
  有上下文的 bar: 29/30
  on_bar 延迟:
    平均: 0.10 ms
    P95:  0.22 ms
    Max:  0.28 ms

五、优化建议
----------------------------------------
  P0 (必须):
    1. 安装 sentence-transformers 使向量检索具备真实语义能力
    2. Chunker 句子分割正则添加英文句号支持
    3. RagContextProvider 在 metadata_filter 分支中支持 context_strategy

  P1 (重要):
    4. 添加磁盘持久化 (VectorStore 的 save/load)
    5. 实现真实新闻 API 适配器 (RSS/REST/WebSocket)
    6. 添加文档时间戳索引，加速 created_before 过滤

  P2 (优化):
    7. 启用 Numba JIT (core.py _USE_NUMBA = True) 加速向量搜索
    8. 实现增量式嵌入（仅嵌入新增文档）
    9. 添加 RAG 检索质量评估指标 (MRR, NDCG)

================================================================================