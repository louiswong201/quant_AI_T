# Quant Framework 初学者完全指南（详细版）

本文档面向**刚学编程**的初学者，**深入浅出、细致到每个函数的用法与设计原因**：从 Python 项目架构、包与模块关系，到数据层/策略层/回测层**每一个类、每一个方法的签名、参数、返回值、为什么这么写**，再到完整调用链。内容会很多，可按目录跳读，每节末尾有小结。

---

## 目录

**第一部分：概念与跑通**
1. [第 0 章：前置概念](#第-0-章前置概念)
2. [第 1 章：最小示例与运行](#第-1-章最小示例与运行)
3. [第 2 章：整体数据流](#第-2-章整体数据流)

**第二部分：Python 项目架构（细致到模块与函数）**
4. [第 3 章：项目目录与导入关系](#第-3-章项目目录与导入关系)
5. [第 4 章：数据层——逐类逐方法说明](#第-4-章数据层逐类逐方法说明)
6. [第 5 章：策略层——逐类逐方法说明](#第-5-章策略层逐类逐方法说明)
7. [第 6 章：回测层——逐类逐方法说明](#第-6-章回测层逐类逐方法说明)
8. [第 7 章：从入口到跑通——完整调用链](#第-7-章从入口到跑通完整调用链)

**第三部分：扩展与优化**
9. [第 8 章：分析与可视化、RAG 入门](#第-8-章分析与可视化rag-入门)
10. [第 9 章：设计思想与优化](#第-9-章设计思想与优化)
11. [附录：术语表与 API 速查](#附录术语表与-api-速查)

---

# 第一部分：概念与跑通

## 第 0 章：前置概念

### 0.1 量化与回测

- **量化**：用程序和数学规则，根据行情等数据制定买卖规则（策略），在电脑上模拟或实盘执行。
- **回测**：用历史数据按「若当时按该策略交易」模拟一遍，得到收益、回撤等，用于评估策略。

### 0.2 本框架在做什么

- **数据层**：从文件/接口加载行情（开、高、低、收、量），可选缓存与多种存储格式。
- **策略层**：你实现「给定到当天的历史数据，返回今天买/卖/不操作」的逻辑。
- **回测层**：按交易日循环，把数据切片传给策略，根据策略返回的信号模拟成交并记账。
- **AI 融合（可选）**：在策略中按「当前日期之前」的新闻/研报做决策，与行情一起用。

### 0.3 最小编程概念

- **类**：模板；**对象**：按类创建的具体实例。例：`DataManager` 是类，`dm = DataManager(data_dir="data")` 中 `dm` 是对象。
- **方法**：对象上的函数，形如 `dm.load_data(...)`。
- **参数**：函数/方法括号内的输入，如 `start_date="2020-01-01"`。

---

## 第 1 章：最小示例与运行

### 1.1 环境与数据准备

- 安装 Python 3.8+，在项目根目录执行：`pip install -r requirements.txt`。
- 在项目根下建 `data` 目录，放入 `STOCK.csv`（或 `STOCK.parquet`），表头至少：`date, open, high, low, close, volume`，日期格式 `YYYY-MM-DD`。

### 1.2 最小脚本（逐行说明）

```python
# 从框架根包导入：DataManager 负责数据，BacktestEngine 负责回测
from quant_framework import DataManager, BacktestEngine
from quant_framework.strategy import MovingAverageStrategy

# 创建数据管理器：数据目录为 "data"，不启用缓存、不启用 fast_io
dm = DataManager(data_dir="data")

# 创建回测引擎：绑定 dm，手续费比例 0.001（0.1%）
engine = BacktestEngine(dm, commission_rate=0.001)

# 创建均线策略：名称、初始资金 100 万、短期 5 日、长期 20 日
strategy = MovingAverageStrategy(
    name="我的第一个策略",
    initial_capital=1_000_000,
    short_window=5,
    long_window=20
)

# 运行回测：策略、标的 "STOCK"、起止日期；返回 dict
result = engine.run(strategy, "STOCK", "2020-01-01", "2023-12-31")

print("初始资金:", result["initial_capital"])
print("最终资金:", result["final_value"])
```

- `DataManager(data_dir="data")`：内部会创建一个 `Dataset` 和可选的 `CacheManager`；不传 `cache` 时用 `lru_cache` 做进程内缓存。
- `BacktestEngine(dm, commission_rate=0.001)`：引擎只依赖 `DataManager` 取数，不关心数据从文件还是缓存来；手续费用 `BacktestConfig.from_legacy_rate(0.001)` 生成。
- `engine.run(strategy, "STOCK", start, end)`：内部会调用 `dm.load_data("STOCK", start, end)`、`dm.calculate_indicators(data)`，再按日循环调用 `strategy.on_bar(...)`，根据返回值执行买卖并记录。

---

## 第 2 章：整体数据流

```
用户脚本
  → DataManager.load_data(symbol, start_date, end_date)   # 取行情表
  → BacktestEngine.run(strategy, symbol, start_date, end_date)
       → 内部：load_data + calculate_indicators 得到带指标的全表
       → 按日期 for i in range(n):
              strategy.update_portfolio_value(current_prices)
              处理限价/止损 pending
              hist = data.iloc[:i+1]   # 到当天的切片，不 copy
              raw = strategy.on_bar(hist, current_date)
              signals = _normalize_signals(raw)
              按 signals 市价/限价/止损 成交，更新 strategy.cash/positions，记录 trades
              记录当日净值到 results
       → 返回 { "results", "trades", "daily_returns", "portfolio_values", "final_value", "initial_capital" }
```

策略只做一件事：**输入「到当天的历史表」+ 当前日期，输出买/卖/不操作的信号**。其余（读数据、算指标、按日推进、成交、记账）都由框架完成。

---

# 第二部分：Python 项目架构与逐层详解

## 第 3 章：项目目录与导入关系

### 3.1 目录结构（与 Python 包对应）

```
quant_framework/           # 顶层包
├── __init__.py           # 导出 DataManager, RagContextProvider, BacktestConfig, BacktestEngine
├── data/                  # 子包 data
│   ├── __init__.py       # 导出 DataManager, RagContextProvider 等
│   ├── data_manager.py   # 类 DataManager
│   ├── dataset.py        # 类 Dataset
│   ├── cache_manager.py  # 函数 _data_key, _key_to_filename；类 LRUCache, CacheManager
│   ├── adapters/
│   │   ├── base_adapter.py   # 抽象基类 BaseDataAdapter
│   │   └── file_adapter.py   # 类 FileDataAdapter
│   └── storage/
│       ├── parquet_storage.py  # 类 ParquetStorage
│       └── ...
├── strategy/
│   ├── base_strategy.py  # 类 BaseStrategy
│   └── ma_strategy.py    # 类 MovingAverageStrategy
└── backtest/
    ├── backtest_engine.py # 函数 _to_f64, _normalize_signals, _align_multi_symbol_data, _try_fill_pending_for_symbol；类 BacktestEngine
    └── config.py         # 类 BacktestConfig
```

- **为什么有 `data`、`strategy`、`backtest` 分开？** 分层职责：数据只负责读存，策略只负责算信号，回测只负责按日驱动与成交。这样换数据源或换策略时互不影响。
- **为什么 `DataManager` 在 `data` 里但从 `quant_framework` 导入？** 根包 `__init__.py` 里写了 `from .data import DataManager`，所以用户写 `from quant_framework import DataManager` 即可，不必关心在哪个子包。

### 3.2 谁依赖谁（导入与调用）

- `quant_framework/__init__.py` 依赖 `data`、`backtest`（不直接依赖 strategy，用户自己从 strategy 导入策略类）。
- `data_manager.py` 依赖 `dataset`、`cache_manager`（只用 `_data_key` 和 `CacheManager`）。
- `dataset.py` 依赖 `adapters.file_adapter`、`storage.*`、`indicators`。
- `backtest_engine.py` 依赖 `strategy.base_strategy`（BaseStrategy）、`data.data_manager`（DataManager）、`backtest.config`（BacktestConfig）。
- 策略类（如 `ma_strategy.py`）只依赖 `base_strategy`，不依赖 data 或 backtest，这样同一策略可被不同引擎或数据源复用。

---

## 第 4 章：数据层——逐类逐方法说明

本节按**类 → 方法**列出签名、参数、返回值，并说明**为什么这样设计**。

---

### 4.1 DataManager（`quant_framework/data/data_manager.py`）

DataManager 是**你对数据的唯一入口**：所有「按标的+日期加载/保存」都通过它，内部再转给 Dataset 和可选 CacheManager。

#### 4.1.1 `__init__(self, data_dir="data", use_parquet=True, cache=None, fast_io=False)`

- **参数**  
  - `data_dir`：数据根目录，所有按 symbol 查找的文件都在此目录（或适配器约定子目录）。  
  - `use_parquet`：是否优先用 Parquet 存储；为 True 时 Dataset 会创建 ParquetStorage。  
  - `cache`：可选，`CacheManager` 实例。若传入，`load_data` 先查缓存再回源，`save_data` 后按 symbol 前缀失效缓存。  
  - `fast_io`：是否启用极速 I/O；为 True 时 Dataset 会创建 ArrowIpcStorage 和 BinaryMmapStorage，读顺序为 binary → arrow → parquet → 适配器。
- **为什么这样设计**：  
  - 把「数据目录」「是否用 Parquet」「是否用缓存」「是否极速读」都放在入口处，调用方只需构造一次 DataManager，后面所有 load/save 行为一致。  
  - Dataset 在内部根据这些布尔和对象组合出「先试哪几种存储、是否回写」的逻辑，DataManager 不暴露 Dataset 的细节。

**代码对应**：

```python
def __init__(self, data_dir: str = "data", use_parquet: bool = True,
             cache: Optional[CacheManager] = None, fast_io: bool = False):
    self.dataset = Dataset(data_dir=data_dir, use_parquet=use_parquet, fast_io=fast_io)
    self._cache = cache
```

- `self.dataset`：真正执行 load/save 的对象；DataManager 只做「有缓存先查缓存、没缓存调 dataset、必要时失效缓存」。
- `self._cache`：若为 None，后面 `load_data` 会走 `_load_data_cached`（lru_cache），否则走 CacheManager。

---

#### 4.1.2 `load_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]`

- **参数**：`symbol` 标的代码，`start_date` / `end_date` 日期字符串 `YYYY-MM-DD`。  
- **返回**：行情 DataFrame（列至少含 date, open, high, low, close, volume），无数据时返回 `None`。  
- **逻辑**：  
  1. 若 `self._cache` 不为 None：用 `_data_key(symbol, start_date, end_date)` 拼缓存键（形如 `data|STOCK|2020-01-01|2023-12-31`），调用 `self._cache.get(key)`；若命中则对 DataFrame 做 `.copy()` 返回（避免调用方修改污染缓存），否则继续。  
  2. 调用 `self.dataset.load(symbol, start_date, end_date)` 从存储/适配器取数；若取到且非空，则 `self._cache.put(key, data)` 写入缓存。  
  3. 若 `self._cache` 为 None：直接返回 `self._load_data_cached(symbol, start_date, end_date)`，该函数被 `@lru_cache(maxsize=128)` 装饰，相同参数只回源一次。  
- **为什么返回 Optional[pd.DataFrame]**：标的或日期范围内可能没有数据，用 `None` 表示「没有」，调用方必须做 `if data is None or data.empty` 判断。  
- **为什么有 cache 时返回 copy**：缓存里存的是同一份对象；若返回引用且调用方改了 DataFrame，下次从缓存取到的是被改过的数据，会破坏一致性。DataFrame 拷贝成本可接受，故统一返回 copy。

**代码对应**：

```python
def load_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    if self._cache is not None:
        key = _data_key(symbol, start_date, end_date)
        data = self._cache.get(key)
        if data is not None:
            return data.copy() if isinstance(data, pd.DataFrame) else data
        data = self.dataset.load(symbol, start_date, end_date)
        if data is not None and not data.empty:
            self._cache.put(key, data)
        return data
    return self._load_data_cached(symbol, start_date, end_date)
```

---

#### 4.1.3 `_load_data_cached(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]`

- **用途**：在**没有** CacheManager 时，用 Python 标准库 `functools.lru_cache` 做进程内 LRU 缓存，避免同一 (symbol, start_date, end_date) 重复读盘。  
- **装饰器**：`@lru_cache(maxsize=128)` 表示最多缓存 128 个不同参数组合；超过时淘汰最久未用的。  
- **为什么需要**：回测或多次分析可能多次请求同一区间；不设缓存会每次都调 `dataset.load`，重复 I/O。  
- **注意**：有 CacheManager 时不会走这里；无 CacheManager 时，`save_data` 里会调 `self._load_data_cached.cache_clear()` 清空 lru_cache，因为保存后该 symbol 的任意区间可能已变，必须失效。

---

#### 4.1.4 `load_features(self, symbol, start_date, end_date, features=None) -> pd.DataFrame`

- **参数**：前三个同 `load_data`；`features` 可选，为 None 时用默认特征（ma 5/10/20、rsi、macd）。  
- **返回**：带技术指标列的 DataFrame；无基础数据时返回空 DataFrame。  
- **逻辑**：内部先 `self.dataset.load_features(...)`，即先 `dataset.load` 再按 `features` 配置调用 `dataset.indicators.calculate_all`。  
- **为什么放在 DataManager**：对外只暴露「按标的+日期取带特征的数据」，不暴露 Dataset 或 indicators 的细节；且若启用了 cache，`load_features` 内部用的 `load` 会经 DataManager，间接受 cache 影响（因为 Dataset 的 load 可能被 DataManager 的 load_data 调用，而 load_features 内部调的是 dataset.load_features，它内部又调 dataset.load，若用户直接调 dm.load_features，则走的是 dataset.load_features → dataset.load，不会经 dm.load_data 的缓存。注意：当前实现里 load_features 是 `return self.dataset.load_features(...)`，所以**没有**经过 dm 的 cache；若希望特征也走 cache，需要在 DataManager 里对 load_features 也做一层 cache 键与 get/put。此处仅说明「设计意图」：统一入口、将来可在此加缓存）。

---

#### 4.1.5 `save_data(self, symbol: str, data: pd.DataFrame) -> None`

- **逻辑**：先 `self.dataset.save(symbol, data)` 写入磁盘（根据 use_parquet/fast_io 写 parquet/arrow/binary 等）；若有 `self._cache`，再 `self._cache.remove_keys_with_prefix(f"data|{symbol}|")`，把该 symbol 所有日期区间的缓存删掉；若无 cache 则 `self._load_data_cached.cache_clear()`。  
- **为什么按 symbol 失效**：保存后该标的的任意日期区间都可能被更新，只删该 symbol 的 key 即可，不必清空整个缓存；前缀 `data|{symbol}|` 能匹配所有 `data|STOCK|*`。

---

#### 4.1.6 `calculate_indicators(self, data: pd.DataFrame, indicators_config: Optional[Dict]=None) -> pd.DataFrame`

- **用途**：对已有行情表算技术指标（MA、RSI、MACD 等）；回测引擎在 run 开始时对整段数据调一次，不在每 bar 内调。  
- **返回**：原 DataFrame 加上指标列（如 ma5, ma10, rsi, macd_* 等）。  
- **为什么在 DataManager 暴露**：调用方（如 BacktestEngine）只持有一个 DataManager，直接 `dm.calculate_indicators(data)` 即可，不必接触 Dataset 或 indicators 模块。

---

#### 4.1.7 `get_latest_price(self, symbol: str, date: Optional[datetime]=None) -> Optional[float]`

- **用途**：取某标的在指定日期的「最新价」（即该日及之前最后一笔的 close）。  
- **逻辑**：若无 `date` 则用当前日；取 `date` 前 30 天的区间，`load_data` 后按 `data["date"] <= date_ts` 过滤，取最后一行 `close`。  
- **为什么用 `pd.to_datetime(data["date"])` 再比较**：`data["date"]` 可能是字符串或 datetime，统一成 Timestamp 再比较，避免类型混用导致错误过滤。

---

### 4.2 Dataset（`quant_framework/data/dataset.py`）

Dataset 负责**真正执行**「从哪种存储读、写到哪种存储」；DataManager 只做缓存与对外 API。

#### 4.2.1 `__init__(self, data_dir="data", use_parquet=True, fast_io=False)`

- **在内部创建**：  
  - `self.adapter`：`FileDataAdapter(data_dir, preferred_format="parquet" or "csv")`，用于最后回源（读 CSV/Parquet 等）和 use_parquet=False 时的保存。  
  - `self.parquet_storage`：仅当 `use_parquet` 为 True 时创建，用于读/写 Parquet。  
  - `self.arrow_storage` / `self.binary_storage`：仅当 `fast_io` 为 True 时创建，用于极速读/写。  
  - `self.indicators`：`VectorizedIndicators()`，用于 `load_features` 和 `calculate_all`。  
- **为什么按布尔创建存储**：避免未用的存储也初始化（如不装 pyarrow 时可以不建 Arrow 相关），且读顺序在 `load` 里用 if 分支即可表达，清晰。

---

#### 4.2.2 `load(self, symbol, start_date, end_date, fields=None) -> pd.DataFrame`

- **参数**：`fields` 可选，若传则只保留这些列。  
- **读顺序（为什么这样设计）**：  
  1. 若 `fast_io` 且存在 `binary_storage`：先试 `binary_storage.load(...)`，最快（mmap）。  
  2. 若仍无数据且 `fast_io` 且存在 `arrow_storage`：再试 `arrow_storage.load(...)`。  
  3. 若仍无且 `use_parquet` 且存在 `parquet_storage`：再试 `parquet_storage.load(...)`。  
  4. 若仍无：调 `self.adapter.load_data(symbol, start_date, end_date)`（如从 CSV 读）；若拿到数据，则**回写**到 parquet（若有）、arrow（若有）、binary（若有），这样下次同一区间可从更快格式读。  
  5. 若 `fields` 非空，则只保留 `data` 中存在的字段子集再返回。  
- **为什么「仅当从适配器拿到数据才回写」**：若数据是从 binary/arrow/parquet 读到的，说明磁盘上已有，无需再写；只有从 CSV/API 等第一次拉取时才需要写入框架存储，避免重复写入和覆盖问题。

**代码对应（核心顺序）**：

```python
data = None
if self.fast_io and self.binary_storage:
    data = self.binary_storage.load(symbol, start_date, end_date, columns=fields)
if (data is None or data.empty) and self.fast_io and self.arrow_storage:
    data = self.arrow_storage.load(...)
if (data is None or data.empty) and self.use_parquet and self.parquet_storage:
    data = self.parquet_storage.load(...)
if data is None or data.empty:
    data = self.adapter.load_data(symbol, start_date, end_date)
    if data is not None and not data.empty:
        if self.use_parquet and self.parquet_storage:
            self.parquet_storage.save(symbol, data)
        if self.fast_io and self.arrow_storage:
            self.arrow_storage.save(symbol, data)
        if self.fast_io and self.binary_storage:
            self.binary_storage.save(symbol, data)
if data is None or data.empty:
    return pd.DataFrame()
if fields:
    available_fields = [f for f in fields if f in data.columns]
    if available_fields:
        data = data[available_fields]
return data
```

---

#### 4.2.3 `load_features(self, symbol, start_date, end_date, features=None) -> pd.DataFrame`

- **逻辑**：先 `self.load(symbol, start_date, end_date)`；若为空直接返回；否则若 `features` 为 None 用默认 `features_config`（ma/rsi/macd），否则用 `_parse_features(features)` 得到配置字典；再 `self.indicators.calculate_all(data, features_config)` 得到带指标列的表。  
- **为什么单独一个方法**：调用方只需「给我带特征的数据」，不必先 load 再自己调 indicators；特征配置（周期、参数）在 Dataset 内封装成字典，对外简单。

---

#### 4.2.4 `_parse_features(self, features: List[str]) -> dict`

- **用途**：把字符串列表（如 `["ma5", "ma10", "rsi", "macd"]`）转成 `indicators.calculate_all` 需要的配置字典，如 `{"ma": [5, 10], "rsi": {"period": 14}, "macd": {...}}`。  
- **为什么在 Dataset 里**：特征名到参数的映射是「数据/特征层」的约定，放在 Dataset 比放在 DataManager 或 indicators 更合适，便于扩展新特征名。

---

#### 4.2.5 `save(self, symbol: str, data: pd.DataFrame)`

- **逻辑**：若 `use_parquet` 且有 `parquet_storage` 则写 Parquet；否则调 `self.adapter.save_data(symbol, data)`；若 `fast_io` 再分别写 arrow 和 binary。  
- **为什么写多种格式**：fast_io 开启时，下次 load 会优先从 binary/arrow 读；保存时一并写出，保证数据一致且下次读更快。

---

#### 4.2.6 `get_available_symbols(self) -> List[str]`

- **逻辑**：`return self.adapter.get_available_symbols()`。  
- **用途**：列出当前数据目录（或适配器）下有哪些标的可用，便于遍历或校验。

---

### 4.3 缓存相关：_data_key、CacheManager、LRUCache（简要）

- **`_data_key(symbol, start_date, end_date)`**：返回 `f"data|{symbol}|{start_date}|{end_date}"`。用 `|` 分隔便于用 `startswith("data|STOCK|")` 做按 symbol 前缀失效；且键稳定，同一请求总是同一 key。  
- **CacheManager**：内存用 LRUCache + 按字节上限淘汰（`_evict_until`），磁盘用 Parquet（DataFrame）或 pickle；`get(key)` 先内存后磁盘，返回时 DataFrame 会 copy；`put` 时若超过内存上限会先淘汰再放入；`remove_keys_with_prefix(prefix)` 用于 save_data 后按 symbol 失效。  
- **LRUCache**：`OrderedDict` 实现，`get` 时 `move_to_end` 保证 LRU；`pop_oldest()` 用于 CacheManager 淘汰时按「最久未用」弹出，不触发访问。

---

### 4.4 FileDataAdapter（`quant_framework/data/adapters/file_adapter.py`）

- **职责**：从 `data_dir` 下按 `{symbol}.{format}` 找文件（如 STOCK.csv、STOCK.parquet），按列 date/open/high/low/close/volume 读取并过滤日期区间，返回 DataFrame。  
- **load_data(symbol, start_date, end_date)**：依次尝试 preferred_format、parquet、csv、hdf5；若存在则读入，统一 `date` 列为 datetime，过滤 `[start_date, end_date]`，按 date 排序，检查必要列后返回 `df[required_columns]`。  
- **为什么多种格式循环试**：用户可能只有 CSV 或只有 Parquet，适配器统一兜底，避免「只支持一种格式」导致无法读旧数据。  
- **save_data(symbol, data, format=None)**：按指定格式写入 `data_dir` 下 `{symbol}.{fmt}`；若索引是 DatetimeIndex 会 reset_index 成 date 列再写。  
- **get_available_symbols()**：扫描 data_dir 下文件名，去掉扩展名去重后排序返回，供「有哪些标的」查询。

---

### 4.5 BaseDataAdapter（`quant_framework/data/adapters/base_adapter.py`）

- **抽象基类**：定义接口 `load_data(symbol, start_date, end_date) -> Optional[DataFrame]`、`save_data(symbol, data) -> bool`、`check_connection() -> bool`，以及可选 `get_available_symbols()`、`get_latest_date(symbol)`。  
- **为什么用 ABC**：所有数据源（文件、API、数据库）都实现同一接口，Dataset 只依赖「能 load_data/save_data」的适配器，便于扩展新数据源而不改 Dataset 逻辑。

---

## 第 5 章：策略层——逐类逐方法说明

策略层规定：**所有策略继承 BaseStrategy，并实现 on_bar**。引擎只依赖「能接收 data + current_date、返回信号」的接口，不关心策略内部用均线还是 RAG。

---

### 5.1 BaseStrategy（`quant_framework/strategy/base_strategy.py`）

#### 5.1.1 类定义与 `__init__`

```python
class BaseStrategy(ABC):
    def __init__(self, name: str, initial_capital: float = 1000000,
                 rag_provider: Optional["RagContextProvider"] = None):
        self.name = name
        self.initial_capital = initial_capital
        self.positions: Dict[str, int] = {}   # 当前持仓 { symbol: 股数 }
        self.cash = initial_capital
        self.portfolio_value = initial_capital
        self.rag_provider = rag_provider
```

- **为什么用 ABC**：强制子类实现 `on_bar`，否则无法实例化，避免「忘记实现核心方法」。  
- **positions / cash / portfolio_value**：由引擎在每 bar 前调用 `update_portfolio_value`，在收到买卖信号后调用 `buy`/`sell` 更新；策略也可在 on_bar 里用 `can_buy`/`can_sell`/`calculate_position_size` 辅助决策。  
- **rag_provider**：可选；若不传，`get_rag_context` 直接返回空字符串；若传，策略在 on_bar 里可调 `self.get_rag_context(..., as_of_date=current_date)` 取 RAG 上下文。

---

#### 5.1.2 `on_bar(self, data, current_date, current_prices=None) -> Union[Dict, List[Dict]]`

- **抽象方法**：子类必须实现。  
- **参数**：  
  - `data`：单标的时为 `pd.DataFrame`（从第 0 天到当前 bar 的切片）；多标的时为 `Dict[str, pd.DataFrame]`。  
  - `current_date`：当前交易日 `pd.Timestamp`。  
  - `current_prices`：多标的时引擎传入 `{ symbol: 当前 close }`，用于 `update_portfolio_value` 和引擎内部。  
- **返回值**：不操作可返回 `None` 或 `{"action": "hold"}`；买/卖返回一个 dict 或 dict 列表，每个 dict 需含 `action`('buy'/'sell')、`symbol`、`shares`；可选 `order_type`('market'/'limit'/'stop')、`limit_price`、`stop_price`。  
- **为什么约定「不修改 data」**：引擎传的是 `iloc[:i+1]` 的视图，若策略原地改 DataFrame，会污染后续 bar 或同一数据在多处的使用，回测结果不可信；故文档与注释明确「只读不写」。

---

#### 5.1.3 `calculate_position_size(self, price: float, risk_percent: float = 0.02) -> int`

- **公式**：`risk_amount = self.portfolio_value * risk_percent`，`shares = int(risk_amount / price)`，返回 `max(0, shares)`。  
- **用途**：按「总资产的一定比例」换算可买股数；例如 risk_percent=0.95 表示用 95% 资产买该价格下的股数。  
- **为什么用 portfolio_value**：回测中每 bar 会先 `update_portfolio_value`，所以当前总资产是最新的，用其比例算仓位更合理。

---

#### 5.1.4 `can_buy(self, symbol, price, shares) -> bool` / `can_sell(self, symbol, shares) -> bool`

- **can_buy**：`price * shares <= self.cash`。  
- **can_sell**：`self.positions.get(symbol, 0) >= shares`。  
- **用途**：策略在返回 buy/sell 信号前可先检查，避免返回无法执行的量；引擎侧也会按资金和持仓做调整（如买入时按可用资金截断股数）。

---

#### 5.1.5 `buy(self, symbol, price, shares) -> bool` / `sell(self, symbol, price, shares) -> bool`

- **buy**：先 `can_buy`，不通过则返回 False；否则 `self.cash -= cost`，`self.positions[symbol] = self.positions.get(symbol, 0) + shares`。  
- **sell**：先 `can_sell`，不通过则返回 False；否则 `self.cash += revenue`，减少持仓，若持仓为 0 则从 `positions` 删除该 key。  
- **谁调用**：回测引擎在解析到策略返回的 buy/sell 信号后，会按执行价和手续费调用 `strategy.buy(...)` 或 `strategy.sell(...)`，并记录到 trades。  
- **为什么在基类实现**：所有策略的「记账」逻辑一致（现金减、持仓加），不必每个子类都写一遍；子类只关心「何时买、何时卖、买多少」。

---

#### 5.1.6 `update_portfolio_value(self, current_prices: Dict[str, float])`

- **公式**：`self.portfolio_value = self.cash + sum(shares * current_prices.get(symbol, 0) for symbol, shares in self.positions.items())`。  
- **谁调用**：回测引擎在每 bar 开头、处理限价单和 on_bar 之前调用，保证策略和引擎看到的「当前总资产」一致。  
- **为什么每 bar 都更新**：价格每天变，持仓市值随之变；不更新则 `calculate_position_size` 等会用过期的 portfolio_value。

---

#### 5.1.7 `get_rag_context(self, query, symbol=None, top_k=5, max_chars=4000, as_of_date=None) -> str`

- **逻辑**：若 `self.rag_provider is None` 返回 `""`；否则把 `as_of_date` 转成 datetime（若为 Timestamp 用 `to_pydatetime()`），再调 `self.rag_provider.get_context(...)`。  
- **为什么 as_of_date 要转**：RagContextProvider 内部可能用 datetime 做过滤；Timestamp 与 datetime 兼容但类型统一可避免边界问题。  
- **回测时务必传 as_of_date=current_date**：保证只用「当前 bar 之前」的文档，不偷看未来。

---

### 5.2 MovingAverageStrategy（`quant_framework/strategy/ma_strategy.py`）

#### 5.2.1 `__init__(self, name="MA策略", initial_capital=1000000, short_window=5, long_window=20)`

- 调用 `super().__init__(name, initial_capital)`，再保存 `self.short_window`、`self.long_window`。  
- **为什么用 short/long_window**：均线策略需要两个周期判断金叉/死叉；写成参数便于回测时调参（如 5/20 或 10/30）。

---

#### 5.2.2 `on_bar` 逐段说明（为什么这么写）

```python
df = data if isinstance(data, pd.DataFrame) else list(data.values())[0] if data else pd.DataFrame()
```

- **为什么这样取 df**：引擎单标的时传 DataFrame，多标的时传 Dict[symbol, DataFrame]；这里兼容两种：若是 dict 则取第一个标的的 DataFrame（示例策略是单标的写法），若没有则空表。

```python
if len(df) < self.long_window:
    return {"action": "hold"}
```

- **为什么**：长期均线需要至少 long_window 根 K 线才能算出一个值；数据不足时无法判断金叉死叉，直接不操作。

```python
short_ma = df["close"].rolling(window=self.short_window).mean().iloc[-1]
long_ma  = df["close"].rolling(window=self.long_window).mean().iloc[-1]
prev_short_ma = df["close"].rolling(...).mean().iloc[-2] if len(df) > 1 else short_ma
prev_long_ma  = df["close"].rolling(...).mean().iloc[-2] if len(df) > 1 else long_ma
```

- **rolling(window=k).mean()**：对 close 做长度为 k 的滚动均值；`.iloc[-1]` 是当前 bar 的均线值，`.iloc[-2]` 是上一 bar 的均线值。  
- **为什么需要 prev**：金叉定义是「上一 bar 短期 ≤ 长期，当前 bar 短期 > 长期」；死叉相反。只用当前 bar 无法判断「穿越」。

```python
if prev_short_ma <= prev_long_ma and short_ma > long_ma:
    shares = self.calculate_position_size(current_price, risk_percent=0.95)
    if shares > 0 and self.can_buy(symbol, current_price, shares):
        return {'action': 'buy', 'symbol': symbol, 'shares': shares}
elif prev_short_ma >= prev_long_ma and short_ma < long_ma:
    holdings = self.positions.get(symbol, 0)
    if holdings > 0:
        return {'action': 'sell', 'symbol': symbol, 'shares': holdings}
return {'action': 'hold'}
```

- **金叉**：上一 bar 短期 ≤ 长期且当前 bar 短期 > 长期 → 买；用 95% 资产算股数，并检查 `can_buy` 再返回。  
- **死叉**：上一 bar 短期 ≥ 长期且当前 bar 短期 < 长期 → 卖；卖全部持仓。  
- **为什么 return dict**：引擎用 `_normalize_signals(raw)` 把返回值统一成 list of dict，再遍历执行；所以返回单个 dict 或 list 均可。

---

## 第 6 章：回测层——逐类逐方法说明

### 6.1 BacktestConfig（`quant_framework/backtest/config.py`）

- **@dataclass(frozen=True)**：不可变配置，避免回测中途被修改。  
- **字段**：买卖的比例/固定手续费、买卖的滑点（基点或固定金额）；买入价=参考价+滑点，卖出价=参考价-滑点。  
- **from_legacy_rate(commission_rate)**：兼容旧接口，生成「买卖均为该比例手续费」的 Config。  
- **fill_price_buy(ref_price)** / **fill_price_sell(ref_price)**：根据滑点算执行价。  
- **commission_buy(notional)** / **commission_sell(notional)**：根据比例+固定算手续费金额。  
- **为什么买卖分开**：实盘常有买入费率、卖出费率、印花税等不同，分开便于贴近真实成本。

---

### 6.2 辅助函数（`backtest_engine.py`）

#### `_to_f64(arr: np.ndarray) -> np.ndarray`

- **作用**：若 `arr` 不是 float64 或非 C 连续，则 `np.ascontiguousarray(arr, dtype=np.float64)` 返回新数组；否则返回原数组。  
- **为什么**：回测循环内大量用 `close[i]`、`open[i]` 等，统一 float64 且连续可避免隐式转换和 stride 开销，也为将来 Numba/C 扩展留一致类型。

#### `_normalize_signals(out: Any) -> List[Dict]`

- **作用**：把策略返回值统一成「只含 buy/sell 的 dict 的列表」。若 `out` 为 None 返回 []；若为单个 dict 且 action 为 buy/sell 则包成 [out]；若为 list/tuple 则过滤出 action in ('buy','sell') 的 dict。  
- **为什么**：策略可能返回 None、单 dict、list；引擎后续统一 `for sig in signals` 处理，避免多处写类型判断。

#### `_align_multi_symbol_data(data_by_symbol) -> (common_index, aligned)`

- **作用**：多标的时，取各标的日期的交集 `common_index`，对每个标的 DataFrame 按 `common_index` reindex、ffill、bfill 后得到对齐后的表，保证每 bar 各标的都有数据。  
- **为什么**：回测按「公共交易日」逐日推进，没有交集的日期无法算组合净值；缺失日用前/后填充避免 NaN 打断循环。

#### `_try_fill_pending_for_symbol(pending, symbol, bar_open, bar_high, bar_low) -> List[Dict]`

- **作用**：用当前 bar 的 OHLC 检查 pending 里该 symbol 的限价/止损单是否触及；触及则算出 fill_price，加入返回列表，并从 pending 中移除该笔；未触及的留在 pending。  
- **实现细节**：限价买为 bar_low <= limit_price 时成交，成交价 min(open, limit_price)；卖为 bar_high >= limit_price；止损买为 bar_high >= stop_price，卖为 bar_low <= stop_price。最后 `pending.clear()` 再 `pending.extend(remain)`，保证 pending 只保留未成交单。

---

### 6.3 BacktestEngine

#### 6.3.1 `__init__(self, data_manager, config=None, *, commission_rate=None)`

- 保存 `self.data_manager`；若 `config` 不为 None 则用其作为 `self.config`，否则 `self.config = BacktestConfig.from_legacy_rate(commission_rate or 0.001)`。  
- **为什么 commission_rate 用关键字参数**：与 config 区分，旧代码可写 `BacktestEngine(dm, commission_rate=0.001)` 而不传 config。

#### 6.3.2 `run(self, strategy, symbols, start_date, end_date) -> Dict[str, Any]`

**步骤拆解（与源码一一对应）：**

1. **symbol_list**：若 `symbols` 是 str 则包成 `[symbols]`，否则 `list(symbols)`；空则抛 `ValueError`。  
2. **single**：`len(symbol_list) == 1`，后面单标的传 DataFrame，多标的传 Dict。  
3. **加载数据与指标**：  
   - 单标的：`data = self.data_manager.load_data(symbol_list[0], start_date, end_date)`，若空则抛错；`data = self.data_manager.calculate_indicators(data)`；`dates = pd.to_datetime(data["date"]).values`；`data_by_symbol = { symbol_list[0]: data }`；`n = len(dates)`。  
   - 多标的：对每个 symbol load_data + calculate_indicators 放入 `data_by_symbol`；`common_index, aligned = _align_multi_symbol_data(data_by_symbol)`；`data_by_symbol = aligned`；`dates = common_index.values`；`n = len(dates)`。  
4. **预提取数组（为什么）**：  
   - `close_arrays[sym] = _to_f64(df["close"].values)`；若有 open/high/low 则 `ohlc[sym] = (_to_f64(open), _to_f64(high), _to_f64(low))`，否则用 close 充任 O/H/L。  
   - 循环内用 `close_arrays[sym][i]`、`ohlc[sym][0][i]` 等 O(1) 访问，避免每 bar 再从 DataFrame 取列。  
5. **预分配与初始化**：`results = []`，`trades = []`，`portfolio_values_arr = np.empty(n, dtype=np.float64)`，`daily_returns_arr = np.empty(n, dtype=np.float64)`，`pending = []`，`initial = strategy.initial_capital`。  
6. **for i in range(n)**：  
   - `current_date = pd.Timestamp(dates[i])`；`current_prices = { sym: float(close_arrays[sym][i]) for sym in close_arrays }`。  
   - `strategy.update_portfolio_value(current_prices)`。  
   - 处理 pending：对每个 sym 用当前 bar 的 o,h,l 调 `_try_fill_pending_for_symbol`，对返回的已成交单按 config 算执行价和手续费，调用 `strategy.buy`/`sell` 并 append 到 trades。  
   - 调用策略：单标的 `hist = data_by_symbol[sym].iloc[:i+1]`，`raw = strategy.on_bar(hist, current_date)`；多标的 `hist = { s: data_by_symbol[s].iloc[:i+1] for s in data_by_symbol }`，`raw = strategy.on_bar(hist, current_date, current_prices=current_prices)`。  
   - `signals = _normalize_signals(raw)`。  
   - 遍历 signals：若 order_type 为 limit/stop 且带了 limit_price/stop_price 则加入 pending 并 continue；否则按市价：买则按 ref_price 算执行价和手续费，若资金够则按 `min(请求股数, 可用资金能买的股数)` 调用 buy 并记 trades；卖则按持仓截断股数后 sell 并记 trades。  
   - `portfolio_values_arr[i] = strategy.portfolio_value`；`daily_returns_arr[i] = (pv - initial) / initial`（此处为相对初始的累计收益，名称 daily_returns 在文档中已说明）；`results.append({ date, portfolio_value, cash, positions, daily_return })`。  
7. **返回**：`pd.DataFrame(results)` 为 results_df，trades 转为 DataFrame；返回 dict 含 `results`、`trades`、`daily_returns`、`portfolio_values`、`final_value`、`initial_capital`。

**为什么传 iloc 切片不 copy**：`data_by_symbol[sym].iloc[:i+1]` 是视图，不复制数据；策略只读，可大幅省内存与时间。  
**为什么买入时截断股数**：`adj = min(int((strategy.cash - commission) / exec_price), shares)`，保证不超资金，避免策略返回的 shares 因取整或比例过大导致现金为负。

---

## 第 7 章：从入口到跑通——完整调用链

下面用「用户写的一行 → 实际调到的函数」把整条链串起来（单标的、无 RAG、无 CacheManager）。

1. **用户**：`dm = DataManager(data_dir="data")`  
   → `DataManager.__init__` → 创建 `Dataset(data_dir, use_parquet=True, fast_io=False)`，`self._cache = None`。

2. **用户**：`engine = BacktestEngine(dm, commission_rate=0.001)`  
   → `BacktestEngine.__init__` → `self.config = BacktestConfig.from_legacy_rate(0.001)`。

3. **用户**：`result = engine.run(strategy, "STOCK", "2020-01-01", "2023-12-31")`  
   → `BacktestEngine.run`  
   → `self.data_manager.load_data("STOCK", "2020-01-01", "2023-12-31")`  
   → 因 `_cache is None`，走 `_load_data_cached("STOCK", ...)`（lru_cache）  
   → `self.dataset.load("STOCK", ...)`  
   → Dataset 内：无 fast_io，不试 binary/arrow；试 `parquet_storage.load`（若存在）；若仍无则 `adapter.load_data(...)`（FileDataAdapter 读 data/STOCK.csv 或 .parquet），若有数据则可能回写 parquet。  
   → 返回 DataFrame 给 run。  
   → `self.data_manager.calculate_indicators(data)`  
   → `self.dataset.indicators.calculate_all(data, None)`，得到带 ma/rsi/macd 等列的表。  
   → 构建 `dates`、`data_by_symbol`、`close_arrays`、`ohlc`，预分配 `portfolio_values_arr`、`daily_returns_arr`。  
   → `for i in range(n)`：  
   - `strategy.update_portfolio_value(current_prices)`  
   - 处理 pending 限价/止损  
   - `hist = data_by_symbol["STOCK"].iloc[:i+1]`，`raw = strategy.on_bar(hist, current_date)`  
   - MovingAverageStrategy.on_bar 内：算 short_ma/long_ma、prev，判断金叉/死叉，返回 buy/sell/hold  
   - `signals = _normalize_signals(raw)`  
   - 对每个 signal：市价则按 close 成交、调 strategy.buy/sell、append trades  
   - 记录 results、portfolio_values_arr、daily_returns_arr  
   → 组装并返回 `{ results, trades, daily_returns, portfolio_values, final_value, initial_capital }`。

4. **用户**：`print(result["final_value"])`  
   → 即 `strategy.portfolio_value` 的最终值。

这样从「用户一行 run」到「Dataset.load」「FileAdapter.load_data」「on_bar」「buy/sell」的每一步都对应到具体函数和设计原因。

---

## 第 8 章：分析与可视化、RAG 入门

- **PerformanceAnalyzer**（`analysis/performance.py`）：`analyze(portfolio_values, daily_returns, initial_capital)` 算总收益、年化、波动率、夏普、最大回撤等；`analyze_trades(trades_df)` 做简化交易分析；`print_summary` 打印。  
- **Plotter**（`visualization/plotter.py`）：`plot_backtest_results`、`plot_trades` 等，可指定 `save_path` 保存图片。  
- **RAG**：策略构造时传入 `rag_provider=RagContextProvider(pipeline=...)`；在 on_bar 里 `ctx = self.get_rag_context("查询", symbol=symbol, as_of_date=current_date)`；回测时务必传 `as_of_date=current_date` 保证时间一致。详见 [RAG.md](RAG.md)。

---

## 第 9 章：设计思想与优化

- **分层**：数据 / 策略 / 回测 分离，依赖单向（策略不依赖引擎，引擎依赖数据和策略接口），便于换数据源、换策略、换执行逻辑。  
- **极速 I/O**：fast_io 时读顺序 binary → arrow → parquet → 适配器；回测热路径用 NumPy 数组和 iloc 视图、预分配数组；指标在 run 前算一次。详见 [DESIGN.md](DESIGN.md)、[IO_FAST.md](IO_FAST.md)、[PERFORMANCE.md](PERFORMANCE.md)。

---

## 附录：术语表与 API 速查

| 术语 | 解释 |
|------|------|
| bar | 一个时间单位（本框架中通常为一个交易日） |
| DataFrame | Pandas 表格；行=记录，列=字段 |
| on_bar | 策略在每个 bar 被调用的方法，入参 data + current_date，出参信号 |
| 信号 | 策略返回的 buy/sell dict，含 action、symbol、shares 等 |
| 适配器 | 统一「按 symbol+日期 load/save」的数据源抽象 |
| 视图 | 不复制数据、与原 DataFrame 共享内存的切片（如 iloc[:i+1]） |

**API 速查（仅列常用）**

- `DataManager(data_dir, use_parquet, cache, fast_io)`  
- `DataManager.load_data(symbol, start_date, end_date) -> Optional[DataFrame]`  
- `DataManager.save_data(symbol, data)`  
- `DataManager.calculate_indicators(data, indicators_config) -> DataFrame`  
- `Dataset.load(symbol, start_date, end_date, fields) -> DataFrame`  
- `Dataset.save(symbol, data)`  
- `BaseStrategy.__init__(name, initial_capital, rag_provider)`  
- `BaseStrategy.on_bar(data, current_date, current_prices) -> Dict | List[Dict]`  
- `BaseStrategy.calculate_position_size(price, risk_percent) -> int`  
- `BaseStrategy.get_rag_context(query, symbol, top_k, max_chars, as_of_date) -> str`  
- `BacktestEngine(data_manager, config, commission_rate)`  
- `BacktestEngine.run(strategy, symbols, start_date, end_date) -> Dict`  
- `BacktestConfig.from_legacy_rate(rate)`  
- `BacktestConfig.fill_price_buy(ref_price)` / `fill_price_sell(ref_price)`  
- `BacktestConfig.commission_buy(notional)` / `commission_sell(notional)`

延伸阅读： [INDEX.md](INDEX.md)、[ARCHITECTURE.md](ARCHITECTURE.md)、[DESIGN.md](DESIGN.md)、[IO_FAST.md](IO_FAST.md)、[PERFORMANCE.md](PERFORMANCE.md)、[RAG.md](RAG.md)。
