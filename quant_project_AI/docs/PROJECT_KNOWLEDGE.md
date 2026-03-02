# 项目知识库：Python 概念与结构设计（面向初学者）

本文档面向**初学者**，包含两大部分：  
1. **本项目用到的 Python 概念**：装饰器、LRU、抽象基类、类型注解、dataclass 等**是什么、在项目里怎么用、为什么这么用**。  
2. **项目结构与设计**：分层、适配器、多层存储、缓存策略等**为什么这么设计**，以及**优缺点与适用场景**。

**建议**：先看 [BEGINNER_GUIDE.md](BEGINNER_GUIDE.md) 理解「代码在干什么」，再通过本文理解「用到的语法和设计取舍」。遇到不认识的语法可先查下方**快速查阅表**，再跳转到对应章节。

---

## 快速查阅表

| 概念 / 设计 | 一句话 | 在项目中的位置 |
|-------------|--------|----------------|
| 装饰器 | 用 `@xxx` 包装函数/类，不改内部代码即可加行为（缓存、接口约束等） | DataManager：`@lru_cache`；BaseStrategy：`@abstractmethod`；BacktestConfig：`@dataclass(frozen=True)` |
| lru_cache | 按「参数」缓存返回值，超过容量淘汰最久未用；本项目用于无 CacheManager 时的 load_data | `data/data_manager.py`：`_load_data_cached` |
| ABC + abstractmethod | 抽象基类 + 必须实现的方法，子类漏写会实例化报错 | `strategy/base_strategy.py`、`data/adapters/base_adapter.py` |
| 类型注解 (typing) | 参数/返回值标类型，给 IDE 和类型检查用，运行时不强制 | 全项目：`Optional[...]`、`Dict`、`List`、`Union` 等 |
| dataclass + frozen | 自动生成 __init__，frozen 表示实例不可变 | `backtest/config.py`：BacktestConfig；RAG：RAGConfig、Document |
| __slots__ | 限制实例只能有指定属性，省内存 | `data/cache_manager.py`：LRUCache |
| OrderedDict | 保序字典，配合 move_to_end / popitem(last=False) 实现 LRU | `data/cache_manager.py`：LRUCache._cache |
| RLock | 可重入锁，多线程访问同一对象时保证安全 | `data/cache_manager.py`：CacheManager._lock |
| 拷贝 vs 视图 | 缓存返回 copy 防污染；回测传视图省内存，约定策略只读 | DataManager/CacheManager 返回 copy；BacktestEngine 传 iloc 视图 |
| classmethod / property | 类方法（工厂）、属性（只读访问） | BacktestConfig.from_legacy_rate；RagContextProvider.pipeline |
| TYPE_CHECKING | 类型检查时才 import，避免循环导入 | `strategy/base_strategy.py`：RagContextProvider |
| 分层架构 | 数据 / 策略 / 回测 / 分析 分离，依赖单向 | 见 [ARCHITECTURE.md](ARCHITECTURE.md) |
| 适配器模式 | 统一 load_data/save_data 接口，文件/API/DB 可插拔 | BaseDataAdapter → FileDataAdapter 等 |
| 多层存储 | 读：binary → arrow → parquet → 适配器；仅适配器回源时回写 | Dataset.load、各 storage |
| 缓存策略 | 无 cache 用 lru_cache（全清）；有 CacheManager（按 symbol 失效、可落盘） | DataManager.load_data、save_data |
| 配置不可变 | BacktestConfig frozen，回测过程不改配置 | backtest/config.py |
| 策略只读 + 信号约定 | 引擎传 data 视图，策略不修改；返回 buy/sell dict 或 list | BacktestEngine.run、_normalize_signals |
| Numba @njit | 把纯数值 Python 函数编译成机器码，速度接近 C | `backtest/kernels.py`：所有 `_eq_*` 内核函数 |
| 内核注册表 | 用字典 `{名字: 函数}` 分派策略，避免长 if-else | `backtest/kernels.py`：`KERNEL_REGISTRY`、`eval_kernel` |
| 多时间框架融合 | 不同周期（1h/4h/1d）信号对齐后融合，提升稳健性 | `backtest/__init__.py`：`backtest_multi_tf`；`live/kernel_adapter.py`：`MultiTFAdapter` |

---

## 目录

**一、Python 概念（本项目中的用法）**  
1. [装饰器（Decorator）](#1-装饰器decorator)  
2. [lru_cache](#2-lru_cache)  
3. [抽象基类（ABC）与 abstractmethod](#3-抽象基类abc与-abstractmethod)  
4. [类型注解（typing）](#4-类型注解typing)  
5. [dataclass 与 frozen](#5-dataclass-与-frozen)  
6. [__slots__](#6-__slots__)  
7. [OrderedDict](#7-ordereddict)  
8. [线程锁（RLock）](#8-线程锁rlock)  
9. [拷贝与视图](#9-拷贝与视图)  
10. [classmethod 与 property](#10-classmethod-与-property)  
11. [TYPE_CHECKING 与前向引用](#11-type_checking-与前向引用)  

**二、项目结构与设计**  
12. [Numba JIT 编译（@njit）](#12-numba-jit-编译njit)  
13. [内核注册表与分派模式](#13-内核注册表与分派模式)  
14. [分层架构](#14-分层架构)  
15. [适配器模式](#15-适配器模式)  
16. [多层存储与读顺序](#16-多层存储与读顺序)  
17. [缓存策略（CacheManager vs lru_cache）](#17-缓存策略cachemanager-vs-lru_cache)  
18. [配置不可变与入口简化](#18-配置不可变与入口简化)  
19. [策略接口：只读 data 与信号约定](#19-策略接口只读-data-与信号约定)  
20. [多时间框架融合设计](#20-多时间框架融合设计)  

---

# 一、Python 概念（本项目中的用法）

## 1. 装饰器（Decorator）

### 是什么

**装饰器**是一种语法：在函数或类定义的上方写 `@xxx`，表示「用 `xxx` 包装这个函数/类」。  
本质上等价于：先定义函数，再执行 `函数 = xxx(函数)`。装饰器可以给函数增加「缓存」「类型检查」「日志」等行为，而不改函数内部代码。  
（本项目中最典型的装饰器用法是 **lru_cache**，见 [§2](#2-lru_cache)。）

### 本项目中的例子

- **`@lru_cache(maxsize=128)`**（DataManager._load_data_cached）  
  表示：对该函数的**返回值**做 LRU 缓存，相同参数只算一次，后面直接返回缓存结果。  

- **`@abstractmethod`**（BaseStrategy.on_bar、BaseDataAdapter.load_data 等）  
  表示：子类**必须**实现该方法，否则不能实例化；用于定义「接口」。  

- **`@dataclass(frozen=True)`**（BacktestConfig）  
  表示：这个类由「数据字段」构成，自动生成 `__init__` 等，且实例**不可变**。  

- **`@jit(nopython=True, cache=True)`**（indicators 里、numba）  
  表示：把该函数编译成机器码加速，并缓存编译结果。  

- **`@staticmethod`**（VectorizedIndicators.calculate_all 等）  
  表示：该方法不依赖 `self`，可通过类名直接调用，如 `Indicators.calculate_all(data, config)`。

### 为什么这么用

- **lru_cache**：同一 (symbol, start_date, end_date) 多次请求时避免重复读盘。  
- **abstractmethod**：强制所有策略实现 `on_bar`、所有适配器实现 `load_data`，漏写会在创建对象时报错。  
- **dataclass**：BacktestConfig 只是一组配置字段，用 dataclass 少写样板代码，frozen 防止回测中途被改。  
- **@staticmethod**：指标计算只依赖输入数据和配置，不依赖实例状态，用静态方法语义更清晰。

---

## 2. lru_cache

### 是什么

`functools.lru_cache(maxsize=128)` 是 Python 标准库提供的** Least Recently Used 缓存**：  
- 用「函数的参数」作为 key，用「返回值」作为 value；  
- 相同参数再次调用时，不执行函数体，直接返回上次的结果；  
- 当缓存的 key 超过 `maxsize` 时，淘汰**最久未使用**的那条。

### 本项目中的用法

**文件**：`quant_framework/data/data_manager.py`（约第 56–59 行）

```python
@lru_cache(maxsize=128)
def _load_data_cached(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    return self.dataset.load(symbol, start_date, end_date)
```

- **为什么用 128**：一般不会同时请求超过 128 个不同的 (symbol, start_date, end_date) 组合；足够覆盖常见回测/分析场景，又不会占太多内存。  
- **为什么只用在「无 CacheManager」时**：有 CacheManager 时，`load_data` 直接走 `self._cache.get/put`，不需要 lru_cache；无 CacheManager 时，用 lru_cache 做进程内缓存，避免同一区间重复调 `dataset.load`。  
- **注意**：lru_cache 的 key 是「参数组合」，而 `self` 是对象引用，所以同一 DataManager 下相同 (symbol, start_date, end_date) 会命中缓存；若 `save_data` 更新了该 symbol 的数据，需要清缓存，所以代码里在 `save_data` 时调用 `self._load_data_cached.cache_clear()`。

**常见疑问**  
- **为什么是 128？** 常见回测/分析不会同时请求上百个不同日期区间；128 在「命中率」和「内存」之间折中，可按需改大或改小。  
- **什么时候用 lru_cache、什么时候用 CacheManager？** 不传 `cache` 给 DataManager 时自动用 lru_cache；传了 `CacheManager(...)` 则走 CacheManager，两者不会同时生效。

### 优缺点

| 优点 | 缺点 |
|------|------|
| 零额外依赖，实现简单 | 按「参数组合」缓存，无法按 symbol 失效（只能全清） |
| 进程内、无序列化，速度快 | 不限制内存大小，大 DataFrame 多 key 时可能占很多内存 |
| 适合「无 CacheManager」的轻量场景 | 多进程时每进程各有一份缓存，无法共享 |

---

## 3. 抽象基类（ABC）与 abstractmethod

### 是什么

- **ABC（Abstract Base Class）**：不能直接实例化的类，只用来定义「子类必须长什么样」。  
- **abstractmethod**：在抽象基类里标记「子类必须实现的方法」；若子类没实现，实例化时会报错。

### 本项目中的用法

| 类 | 文件 | 必须实现的方法 |
|----|------|----------------|
| BaseStrategy | `strategy/base_strategy.py` | `on_bar` |
| BaseDataAdapter | `data/adapters/base_adapter.py` | `load_data`、`save_data`、`check_connection` |
| VectorStore / Retriever / BaseEmbedder 等 | RAG 模块 | 各自抽象方法 |

子类若漏写抽象方法，在**实例化**时会报错（如 `TypeError: Can't instantiate abstract class ...`）。

### 为什么这么用

- **统一接口**：Dataset 只依赖「能 load_data / save_data 的适配器」，不关心是文件、API 还是数据库；只要继承 BaseDataAdapter 并实现三个抽象方法即可接入。  
- **早报错**：若有人写了新策略但忘了实现 `on_bar`，在 `BacktestEngine.run(strategy, ...)` 时就会因「无法实例化」或「缺少方法」而失败，而不是跑到一半才出错。  
- **文档化**：看 BaseStrategy / BaseDataAdapter 就知道「策略/适配器必须提供哪些方法」，便于扩展和维护。

### 优缺点

| 优点 | 缺点 |
|------|------|
| 接口清晰、强制实现、便于扩展 | 多一层继承，简单场景略重 |
| 与类型检查、IDE 提示配合好 | 运行时检查在实例化时才有，不是导入时 |

---

## 4. 类型注解（typing）

### 是什么

在参数、返回值、变量后写 `: 类型` 或 `-> 类型`，例如：

```python
def load_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
```

- **Optional[X]** ＝ 「X 或 None」（等价于 `Union[X, None]`）。  
- **Dict[K, V]** ＝ 键类型 K、值类型 V 的字典。  
- **List[T]** ＝ 元素类型 T 的列表。  
- **Union[A, B]** ＝ 「A 或 B」（Python 3.10+ 可写 `A | B`）。

Python 解释器**不会**在运行时强制类型，但 IDE 和 mypy 等工具会用它们做补全和查错。  
**建议**：写新函数时顺手加上参数和返回类型，以后读代码和重构都会更轻松。

### 本项目中的用法

- **Optional[pd.DataFrame]**：表示「可能没有数据」，调用方必须做 `if data is None or data.empty`。  
- **Optional[CacheManager]**：表示「可以不传缓存」。  
- **Dict[str, pd.DataFrame]**：多标的时 data 是 { symbol: DataFrame }。  
- **Union[Dict, List[Dict]]**：策略 on_bar 可以返回单个 dict 或 dict 列表。

### 为什么这么用

- **可读性**：一眼能看出参数和返回值类型，减少「传错类型」的 bug。  
- **IDE 与重构**：重命名、查找引用、补全更准确。  
- **文档化**：类型即文档，和 docstring 互补。

---

## 5. dataclass 与 frozen

### 是什么

- **dataclass**：用 `@dataclass` 装饰的类，只需声明字段和默认值，会自动生成 `__init__`、`__repr__` 等，少写样板代码。  
- **frozen=True**：生成的实例**不可变**，赋值会报错（如 `config.commission_pct_buy = 0.002` 会抛异常）。

### 本项目中的用法

```python
# backtest/config.py
@dataclass(frozen=True)
class BacktestConfig:
    commission_pct_buy: float = 0.001
    commission_pct_sell: float = 0.001
    ...
```

- **RAG**：`RAGConfig`、`Document`、`Chunk` 等也用 `@dataclass`（部分未 frozen），用于配置和数据结构。

### 为什么这么用

- **BacktestConfig 不可变**：回测过程中手续费、滑点不应被某段代码改掉，否则结果不可复现；frozen 从语法上禁止修改。  
- **少写 __init__**：配置类字段多，手写 __init__ 容易漏或顺序错，dataclass 自动按字段生成。  
- **清晰**：一眼看出「这个类就是一组配置/数据字段」。

### 优缺点

| 优点 | 缺点 |
|------|------|
| 代码简洁、不可变保证安全 | frozen 后不能就地改，要改只能新建实例 |
| 与类型注解结合好 | 复杂默认值（如 mutable default）要小心，一般用 default_factory |

---

## 6. __slots__

### 是什么

在类里定义 `__slots__ = ("_cache", "_max_size")` 表示：**该类的实例只能有这些属性**，不能动态增加新属性；同时会节省内存（不维护实例的 `__dict__`）。

### 本项目中的用法

**文件**：`quant_framework/data/cache_manager.py`（LRUCache 类）

```python
class LRUCache:
    __slots__ = ("_cache", "_max_size")
    ...
```

**注意**：若子类继承带 `__slots__` 的类，子类通常也要定义自己的 `__slots__`（可包含父类的 slot 名），否则子类实例会恢复使用 `__dict__`。

### 为什么这么用

- **LRUCache** 只会有 `_cache`（OrderedDict）和 `_max_size` 两个属性，用 slots 避免误加属性，并减少每个实例的内存占用；在缓存里可能有很多 key，类本身更轻更好。  
- **限制属性**：防止有人写 `cache.xxx = ...` 导致难以排查的 bug。

### 优缺点

| 优点 | 缺点 |
|------|------|
| 省内存、禁止动态属性 | 不能给实例随意加属性；继承时子类也要定义自己的 __slots__ |

---

## 7. OrderedDict

### 是什么

**OrderedDict** 是「保持插入顺序」的字典：遍历时顺序与插入顺序一致；配合 `move_to_end(key)` 和 `popitem(last=False)` 可以实现 **LRU**：最近访问的移到末尾，淘汰时从头部弹出。

### 本项目中的用法

- **LRUCache**：`self._cache = OrderedDict()`；`get(key)` 时先 `move_to_end(key)` 再返回值，表示「刚被访问」；`put` 时若超容量则 `popitem(last=False)` 弹出最久未用的。  
- **CacheManager** 的按字节淘汰：通过 `LRUCache.pop_oldest()` 依次弹出最久未用的 key，再减掉其占用字节数。

### 为什么不用普通 dict

- Python 3.7+ 的 dict 虽然也保持插入顺序，但**没有** `move_to_end(key)`；要实现「某 key 被访问后移到末尾」必须用 OrderedDict。  
- LRU 的语义是「最久未用的先淘汰」：`get(key)` 时 `move_to_end(key)` 表示刚被访问；淘汰时 `popitem(last=False)` 从头部弹出，即最久未用。

### 优缺点

| 优点 | 缺点 |
|------|------|
| 顺序明确、O(1) 的 get/put/淘汰 | 比普通 dict 略占内存、略慢一丁点 |
| 标准库、无需额外依赖 | 按字节淘汰需自己维护 current_bytes 与 pop_oldest 循环 |

---

## 8. 线程锁（RLock）

### 是什么

**threading.RLock()** 是「可重入锁」：同一线程可以多次 acquire，每次 acquire 都要对应一次 release；用于保护「一段代码同一时间只被一个线程执行」。

### 本项目中的用法

**文件**：`quant_framework/data/cache_manager.py`（CacheManager）

```python
class CacheManager:
    def __init__(...):
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:   # 等价于 acquire → 执行 → release，即使异常也会 release
            ...
```

所有会修改缓存状态的方法（get 中若从磁盘加载并放入内存、put、remove、remove_keys_with_prefix、clear、_evict_until）都在 `with self._lock` 内执行。

- CacheManager 的 `get`、`put`、`remove`、`remove_keys_with_prefix`、`clear`、`_evict_until` 等都在 `with self._lock` 内执行，保证多线程同时访问缓存时不会出现「竞态」（如同时淘汰导致重复删、或字节数算错）。

### 为什么用 RLock 而不是 Lock

- **可重入**：若在持锁时又调了同一对象上另一个也要持锁的方法（如 get 里间接调 put），RLock 允许同一线程再次进入，不会死锁；Lock 会死锁。  
- **为什么需要线程安全**：数据层可能被多线程使用（如 RAG 写入线程 + 回测主线程同时访问），缓存作为共享结构需要加锁。

### 优缺点

| 优点 | 缺点 |
|------|------|
| 保证并发下数据正确 | 锁竞争可能降低吞吐；设计时要注意锁粒度 |
| RLock 避免同一线程内死锁 | 使用不当（如锁内调外部未知代码）可能引发死锁或性能问题 |

---

## 9. 拷贝与视图

### 是什么

- **视图（view）**：不复制数据，新变量指向同一块内存；例如 `df.iloc[:10]` 往往是视图，修改视图可能影响原 DataFrame。  
- **拷贝（copy）**：`df.copy()` 或 `val.copy()` 会复制数据，新旧互不影响。

### 本项目中的用法

- **缓存返回**：`CacheManager.get(key)` 在命中时对 DataFrame 做 `return val.copy()`，避免调用方修改后污染缓存。  
- **回测传 data**：引擎传 `data_by_symbol[sym].iloc[:i+1]` 给策略，这是**视图**，不复制；同时约定**策略不得修改 data**，这样既省内存又保证正确性。  
- **DataManager.load_data**：从 cache 取到数据后 `return data.copy() if isinstance(data, pd.DataFrame) else data`，同样为防止调用方改缓存内容。

### 为什么这么用

- **缓存必须返回 copy**：缓存里存的是「同一份对象」；若返回引用且调用方改了列或值，下次别人从缓存拿到的就是被改过的数据，回测或分析会错。  
- **回测传视图**：若每 bar 都 `df.iloc[:i+1].copy()`，内存和 CPU 开销大；传视图并约定只读，是性能与安全的折中。

### 怎么记

- **从缓存/DataManager 拿到的 DataFrame**：框架会返回 copy，你可以放心改（如加列、过滤），不会影响缓存。  
- **回测里 on_bar 拿到的 data**：是视图，**不要改**；只读即可，否则可能影响后续 bar 或其它逻辑。

### 优缺点

| 做法 | 优点 | 缺点 |
|------|------|------|
| 缓存返回 copy | 调用方随便改也不影响缓存 | 大 DataFrame 每次返回都拷贝，占内存和 CPU |
| 回测传视图、约定只读 | 不拷贝、快 | 依赖策略遵守「不修改 data」，违反时难排查 |

---

## 10. classmethod 与 property

### 是什么

- **@classmethod**：第一个参数是类本身（通常写成 `cls`），用于「工厂方法」或「与类相关、与实例无关」的逻辑；调用时用 `类名.方法(...)`。  
- **@property**：把方法变成「只读属性」，调用时不用括号，如 `obj.pipeline` 而不是 `obj.pipeline()`。

### 本项目中的用法

- **BacktestConfig.from_legacy_rate(commission_rate)**：`@classmethod`，根据一个比例生成 Config 实例，方便旧代码 `BacktestConfig.from_legacy_rate(0.001)` 而不必记所有字段。  
- **RagContextProvider.pipeline**：`@property`，返回 `self._pipeline`，对外像属性一样读，内部可控制只读或延迟初始化。

### 为什么这么用

- **from_legacy_rate**：回测配置字段多，多数人只关心「手续费比例」；用 classmethod 提供简单入口，内部填好买卖比例和固定 0。  
- **property**：对外 API 更简洁（`provider.pipeline`），且若以后改成懒加载或校验，调用方不用改。

---

## 11. TYPE_CHECKING 与前向引用

### 是什么

- **TYPE_CHECKING**：在 `if TYPE_CHECKING:` 块里的 import 只在**类型检查时**执行，运行时**不**执行；用于避免循环导入或减少运行时依赖。  
- **前向引用**：类型注解里写字符串形式的类名，如 `Optional["RagContextProvider"]`，表示「稍后才定义或在别的模块里的类型」。

### 本项目中的用法

```python
# base_strategy.py
from typing import TYPE_CHECKING, ...

if TYPE_CHECKING:
    from ..data.rag_context import RagContextProvider

class BaseStrategy(ABC):
    def __init__(self, ..., rag_provider: Optional["RagContextProvider"] = None):
```

- BaseStrategy 在 strategy 包，RagContextProvider 在 data 包；若直接 `from ..data.rag_context import RagContextProvider`，可能形成循环导入（data 或 backtest 又引用 strategy）。  
- 用 TYPE_CHECKING 后，类型检查器能解析 `RagContextProvider`，但运行时不会执行该 import；参数类型用字符串 `"RagContextProvider"`，运行时不会去查这个类是否存在。

### 为什么这么用

- **避免循环导入**：策略基类需要「声明」自己接受 RAG 提供者，但不一定在运行时立刻加载 data 包；TYPE_CHECKING 满足类型提示又不影响运行顺序。  
- **字符串注解**：在类尚未定义或跨模块时，用 `"ClassName"` 让类型检查器延后解析，避免 NameError。

---

# 二、项目结构与设计

## 12. Numba JIT 编译（@njit）

### 是什么

**Numba** 是一个 Python 库，能把「纯数值计算」的 Python 函数编译成机器码（类似 C 的速度），而不需要离开 Python 生态。`@njit` 是最常用的装饰器，表示「No-Python 模式 JIT 编译」——编译后的函数完全不经过 Python 解释器。

```python
from numba import njit

@njit(cache=True, fastmath=True)
def _eq_ma(c, o, h, l, p1, p2, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    n = len(c)
    # ... 纯数值循环逻辑 ...
    return ret_pct, mdd, nt, eq_arr, fpos, pos_arr
```

- **`cache=True`**：第一次编译后把机器码缓存到磁盘（`__pycache__` 目录），下次启动直接加载，免去编译时间。
- **`fastmath=True`**：允许重排浮点运算顺序以获得更高速度，代价是极端情况下精度可能微差（在量化回测精度范围内可接受）。

### 本项目中的用法

**文件**：`quant_framework/backtest/kernels.py`

本项目的全部 **18 个策略**（MA、RSI、MACD、Drift、RAMOM、Turtle、Bollinger、Keltner、MultiFactor、VolRegime、MESA、KAMA、Donchian、ZScore、MomBreak、RegimeEMA、DualMom、Consensus）各有一个 `@njit` 编译的内核函数（`_eq_ma`、`_eq_rsi` 等）。

除策略内核外，以下辅助函数也用 `@njit` 编译：
- `_fx_lev`、`_sl_exit`、`_mtm_lev`、`_close_pos` — 交易成本和止损逻辑
- `_equity_from_fused_positions` — 从融合仓位序列计算净值曲线

### 约束（初学者必知）

在 `@njit` 函数里**不能**使用：
- Python 对象（dict、list 可以有限使用，但不能存异构类型）
- 字符串操作、print（调试时可用但正式代码中应去掉）
- pandas DataFrame、Python class 实例
- 动态导入或 try/except

只能用：`numpy` 数组、标量（int/float/bool）、简单 for 循环、if/else。

### 为什么这么用

回测需要对每组参数跑一遍全量数据。18 策略 × 数百参数组合 × 数千 bar = 数百万次计算。纯 Python 循环太慢，用 Numba 编译后可达到 **C 级速度**，同时保留 Python 生态的灵活性（不需要写 C 扩展或用 Cython）。

### 性能对比

| 方式 | 18 策略全参数扫描（1000 bars） |
|------|------|
| 纯 Python 循环 | ~300 秒 |
| @njit 编译后 | ~0.3 秒 |
| 加速比 | ~1000x |

### 优缺点

| 优点 | 缺点 |
|------|------|
| 速度接近 C，无需离开 Python | 第一次调用需要编译（`cache=True` 后仅首次） |
| 可直接操作 NumPy 数组 | 函数内不能用 pandas、字符串等 Python 特性 |
| 与 Python 代码无缝混合 | 改函数签名后需清缓存（删 `__pycache__`） |

---

## 13. 内核注册表与分派模式

### 是什么

**注册表模式**（Registry Pattern）：用一个字典把「名字」映射到「函数」，调用时按名字查字典、拿到函数后调用，避免写一长串 `if name == "MA": ... elif name == "RSI": ...`。

### 本项目中的用法

**文件**：`quant_framework/backtest/kernels.py`

```python
KERNEL_REGISTRY = {
    "MA":         _eq_ma,
    "RSI":        _eq_rsi,
    "MACD":       _eq_macd,
    # ... 共 18 个 ...
}

KERNEL_NAMES = tuple(KERNEL_REGISTRY.keys())
```

分派函数 `eval_kernel_detailed` 通过名字查注册表，取出对应的 `@njit` 函数并调用：

```python
def eval_kernel_detailed(name, params, c, o, h, l, sb, ss, cm, lev, dc, sl, pfrac, sl_slip):
    fn = KERNEL_REGISTRY[name]
    return fn(c, o, h, l, *params, sb, ss, cm, lev, dc, sl, pfrac, sl_slip)
```

上层 API（`backtest()`、`optimize()`、`backtest_multi_tf()`）只需传策略名字符串，不直接引用内核函数。

**参数网格**也用同样的注册表模式：

```python
DEFAULT_PARAM_GRIDS = {
    "MA":   [(s, l) for s in (5,10,20) for l in (20,50,100,200) if s < l],
    "RSI":  [(p, os, ob) for p in (7,14,21) for os in (20,25,30) for ob in (70,75,80)],
    # ...
}
```

### 为什么这么用

- **O(1) 分派**：字典查找比 if-else 链更快、更清晰。
- **开放扩展**：新增策略只需写一个 `_eq_xxx` 函数并加一行到 `KERNEL_REGISTRY`，不改任何调用方代码。
- **统一接口**：所有内核函数签名相同 `(c, o, h, l, *params, sb, ss, cm, lev, dc, sl, pfrac, sl_slip) → (ret, mdd, nt, eq_arr, fpos, pos_arr)`，分派函数不需要关心具体策略。

### 优缺点

| 优点 | 缺点 |
|------|------|
| 扩展简单——加一行注册 | 所有内核必须遵循同一签名，灵活性有限 |
| 调用方只依赖名字，解耦 | 名字拼错只能运行时发现（KeyError） |
| 支持动态枚举（`KERNEL_NAMES`） | — |

---

## 14. 分层架构

整体说明见 [ARCHITECTURE.md](ARCHITECTURE.md)，这里只概括与优缺点。

### 设计是什么

- **数据层**（data）：DataManager、Dataset、适配器、存储、缓存；只负责「按标的+日期加载/保存」，不关心谁在用。  
- **策略层**（strategy）：BaseStrategy 与具体策略；只负责「给定历史数据+当前日，返回信号」，不关心数据从哪来、回测怎么执行。  
- **执行层**（backtest）：BacktestEngine、BacktestConfig；负责按日驱动、调用策略、执行订单、记账。  
- **分析/展示**（analysis、visualization）：绩效、画图；只依赖回测返回的 result 结构。

依赖方向：**策略不依赖 backtest**；**backtest 依赖 data 和 strategy 的接口**；**data 不依赖 strategy 和 backtest**。

### 为什么这么设计

- **换数据源**：只换或新增适配器，策略和引擎不用改。  
- **换策略**：只写新的 on_bar，引擎和数据处理不用改。  
- **测试**：可以单独测「Dataset 读出来的数据对不对」、单独测「策略在给定 data 下输出什么信号」、再测「引擎在给定策略下结果是否合理」。

### 优缺点

| 优点 | 缺点 |
|------|------|
| 职责清晰、易扩展、易测 | 层数多，简单脚本要组好几样对象 |
| 依赖单向，无循环 | 新人需要先理解「谁依赖谁」再改代码 |

**建议**：新增功能时先想清楚属于哪一层（数据/策略/回测/分析），再在对应模块里加代码，避免层之间互相引用。

---

## 15. 适配器模式

### 设计是什么

- **BaseDataAdapter** 定义抽象接口：`load_data(symbol, start_date, end_date)`、`save_data(symbol, data)`、`check_connection()`。  
- **FileDataAdapter**、APIDataAdapter、DatabaseDataAdapter 等实现该接口；Dataset 只依赖「能 load_data/save_data 的对象」，不关心是文件、API 还是数据库。

### 为什么这么设计

- **统一入口**：上层（Dataset）只调 `adapter.load_data(...)`，不必写 `if 是文件 then 读文件 elif 是 API then 调 API`。  
- **可扩展**：新增数据源只需新写一个继承 BaseDataAdapter 的类，并在 Dataset 里注入，不改现有文件/API 逻辑。  
- **可测试**：可以用一个「返回固定 DataFrame 的 Mock 适配器」测 Dataset 或回测，不依赖真实文件或网络。

### 优缺点

| 优点 | 缺点 |
|------|------|
| 数据源可插拔、接口统一 | 每种数据源要写一个类，有一定样板代码 |
| 便于 mock 和单测 | 调用链多一层（Dataset → adapter → 具体实现） |

---

## 16. 多层存储与读顺序

### 设计是什么

- **读**：fast_io 时按 **Binary → Arrow → Parquet → 适配器** 依次尝试；先命中哪个用哪个。  
- **写**：保存时根据配置同时写 Parquet（可选）、Arrow、Binary；**仅当本次数据来自适配器（第一次拉取）时才回写**各存储，避免重复写和覆盖。

### 为什么这么设计

- **速度**：Binary（mmap）、Arrow（memory_map）读得快；Parquet 解压稍慢但兼容性好；适配器是最后兜底。  
- **兼容与迁移**：Parquet 通用；Binary/Arrow 为本机加速；同一份数据多种格式，满足「第一次从 CSV 拉、之后从 bin 读」的流程。  
- **回写条件**：若数据已经从 binary/arrow/parquet 读到，说明磁盘已有，无需再写；只有从 CSV/API 新拉的数据才需要写入框架存储，供下次快读。

### 优缺点

| 优点 | 缺点 |
|------|------|
| 同一套 API，兼顾速度与兼容 | 磁盘可能同时存在 .csv/.parquet/.arrow/.bin，占空间 |
| 首次慢、后续快，符合使用习惯 | 回写逻辑要理解「仅适配器回源时写」，否则易困惑 |

---

## 17. 缓存策略（CacheManager vs lru_cache）

与 [§2 lru_cache](#2-lru_cache) 和 [§7 OrderedDict](#7-ordereddict)、[§8 RLock](#8-线程锁rlock) 相关。

### 设计是什么

- **无 CacheManager**：DataManager 用 `@lru_cache` 装饰的 `_load_data_cached`，按 (symbol, start_date, end_date) 缓存，最多 128 条；`save_data` 时 `cache_clear()` 全清。  
- **有 CacheManager**：按 key（如 `data|STOCK|2020-01-01|2023-12-31`）先内存后磁盘；内存按 LRU + 字节上限淘汰；DataFrame 用 Parquet 落盘；`save_data` 时 `remove_keys_with_prefix("data|STOCK|")` 只删该标的的缓存。

### 为什么这么设计

- **lru_cache**：零配置、适合单进程、轻量；但只能全清、不控内存大小、不能按 symbol 失效。  
- **CacheManager**：可配置内存上限、可落盘、可按 symbol 失效，适合「多标的、多次回测、希望跨进程或重启后仍能命中磁盘缓存」的场景。  
- **二选一**：有 cache 就用 CacheManager，没有就用 lru_cache，避免两套缓存同时生效导致逻辑混乱。

### 优缺点

| 方案 | 优点 | 缺点 |
|------|------|------|
| lru_cache | 简单、无依赖、进程内快 | 只能全清、不控内存、不能按 symbol 失效 |
| CacheManager | 按字节淘汰、可磁盘、按 symbol 失效 | 实现复杂、要锁、磁盘 I/O 有开销 |

**建议**：单进程、标的少、不常 save 时用默认（lru_cache）即可；多标的、多次回测、希望重启后仍命中缓存时再传入 `CacheManager(...)`。

---

## 18. 配置不可变与入口简化

### 设计是什么

- **BacktestConfig** 用 `@dataclass(frozen=True)`，创建后不能改字段。  
- **BacktestEngine** 支持只传 `commission_rate`，内部用 `BacktestConfig.from_legacy_rate(rate)` 生成 config，方便「我只要改一个手续费比例」的用法。

### 为什么这么设计

- **不可变**：回测过程中若某处误改 config，结果难以复现；frozen 从语法上禁止。  
- **from_legacy_rate**：大多数人只关心「手续费比例」，不需要配置买卖分开、滑点等；一个 classmethod 提供简单入口，进阶用户再直接构造 BacktestConfig。

### 优缺点

| 优点 | 缺点 |
|------|------|
| 配置不被误改、复现性好 | 要改配置只能新建 Config 实例 |
| 简单场景一行搞定 | 复杂配置要知道所有字段或读文档 |

---

## 19. 策略接口：只读 data 与信号约定

与「拷贝与视图」（[§9](#9-拷贝与视图)）直接相关。

### 设计是什么

- **只读 data**：引擎传 `data.iloc[:i+1]`（视图），约定策略**不得修改** data；文档和注释明确写出。  
- **信号约定**：策略返回 `None` / `{"action":"hold"}` / 单个 dict / list of dict；每个 dict 含 `action`('buy'/'sell')、`symbol`、`shares`；可选 `order_type`、`limit_price`、`stop_price`。引擎用 `_normalize_signals` 统一成 list 再执行。

### 为什么这么设计

- **只读**：传视图不 copy 省内存和 CPU；若允许改，同一 DataFrame 在多处被用会难以排查 bug，故用约定约束。  
- **信号统一**：策略可以返回单 dict 或 list，引擎统一成 list 后同一套循环处理，减少引擎里的分支。

### 优缺点

| 优点 | 缺点 |
|------|------|
| 不 copy、快；接口灵活（单 dict 或 list） | 依赖策略遵守「不修改 data」，违反时难查 |
| 引擎逻辑简单、统一 | 信号格式要在文档写清，否则易传错字段 |

---

## 20. 多时间框架融合设计

### 是什么

**多时间框架融合**（Multi-Timeframe Fusion）是指同时在不同周期（如 1h、4h、1d）运行策略，然后把各周期的仓位信号合并成一个最终决策。这在专业量化交易中很常见——高周期看趋势方向，低周期找精确入场点。

### 本项目中的用法

项目在回测和实盘两条路径上都实现了多 TF 融合：

**回测路径**（`backtest/__init__.py`）：

```python
from quant_framework.backtest import backtest_multi_tf

result = backtest_multi_tf(
    tf_configs={"1h": ("MA", (10, 50)), "1d": ("MACD", (28, 112, 3))},
    tf_data={"1h": df_1h, "1d": df_1d},
    mode="trend_filter",
)
```

**实盘路径**（`live/kernel_adapter.py`）：

```python
adapter = MultiTFAdapter(
    adapters={"1h": adapter_1h, "4h": adapter_4h, "1d": adapter_1d},
    mode="consensus",
)
signal = adapter.on_bar(window_df, symbol="BTC", interval="1h")
```

### 三种融合模式

| 模式 | 逻辑 | 适用场景 |
|------|------|----------|
| **trend_filter** | 最高 TF（如 1d）决定趋势方向；最低 TF（如 1h）提供入场时机；两者方向一致时才开仓 | 趋势跟踪策略，减少逆势交易 |
| **consensus** | 所有 TF 仓位多数表决：多头数 > 半数则做多，空头数 > 半数则做空 | 多策略确认，信号更稳健 |
| **primary** | 指定一个 TF 为主，其余仅做参考 | 只信任特定周期的策略 |

### 技术要点（初学者需理解）

1. **时间戳对齐**：不同周期的 bar 数量不同（1h 一天 24 根，1d 一天 1 根），必须把高周期仓位「前向填充」（forward-fill）到低周期的时间网格上。

2. **前向填充**：用 `np.searchsorted` 找到每个低周期时间点对应的最近高周期 bar，取那根 bar 的仓位。这保证不偷看未来数据。

3. **成本模型适配**：不同周期的成本需要分别计算——1h bar 的资金费用是日费率的 1/24，1d bar 是日费率本身。代码里会根据 `interval` 自动调整。

4. **净值曲线在最细粒度上计算**：融合后的仓位序列按最低周期（如 1h）逐 bar 计算净值，确保精确反映每次换仓的成本和盈亏。

### 为什么这么设计

- **回测与实盘一致**：两条路径用相同的融合逻辑（trend_filter/consensus/primary），回测结果可信赖。
- **模块化**：每个 TF 独立运行自己的内核，融合逻辑只操作仓位数组，不侵入内核代码。
- **可扩展**：新增融合模式只需加一个函数，不改现有代码。

### 优缺点

| 优点 | 缺点 |
|------|------|
| 减少假信号、提高稳健性 | 需要多个时间框架的数据 |
| 回测/实盘逻辑统一 | 融合逻辑增加了理解成本 |
| 融合模式可插拔 | 参数空间增大（每个 TF 各自选策略+参数） |

---

## 概念与设计的对应关系

下面这张表帮助你把「语言/标准库概念」和「项目设计」连起来：哪个概念支撑了哪种设计。

| 项目设计 | 用到的 Python/结构概念 |
|----------|------------------------|
| 分层架构 | 模块与包、依赖方向（谁 import 谁） |
| 适配器模式 | ABC + abstractmethod（统一接口）、多态 |
| 多层存储 | 条件分支、可选对象（None / 有实例） |
| 缓存：lru_cache | 装饰器、functools.lru_cache |
| 缓存：CacheManager | OrderedDict、LRU、RLock、__slots__、copy |
| 配置不可变 | dataclass(frozen=True) |
| 策略接口（只读 + 信号） | 约定（文档 + 注释）、视图与 copy 的取舍 |
| 按 symbol 失效缓存 | 键设计（键前缀可匹配标的）、remove_keys_with_prefix |
| 避免循环导入 | TYPE_CHECKING、前向引用（字符串注解） |
| Numba 内核加速 | @njit 装饰器、NumPy 数组、cache/fastmath 编译选项 |
| 内核注册表分派 | 字典映射 {名字: 函数}、统一函数签名、O(1) 查找 |
| 多时间框架融合 | 时间戳对齐（np.searchsorted）、前向填充、仓位数组合并 |

---

## 小结与延伸

- **Python 概念**：装饰器、lru_cache、ABC、typing、dataclass、__slots__、OrderedDict、RLock、copy/视图、classmethod/property、TYPE_CHECKING 等在本项目中都有对应用法和「为什么这么用」；遇到不熟悉的语法可先查文首**快速查阅表**，再跳转到对应章节。  
- **性能核心**：Numba `@njit` 是本项目的性能基石，18 个策略内核全部编译成机器码，实现 ~1000x 的加速。理解 `@njit` 的约束（只能用 NumPy 和标量）是阅读 `kernels.py` 的前提。
- **结构设计**：分层、适配器、多层存储、缓存策略、配置不可变、策略只读与信号约定、内核注册表、多时间框架融合，都是为了**可扩展、可测试、行为可预期**；每种选择都有优缺点，按项目规模与需求取舍。  
- **延伸阅读**：实现细节 → [BEGINNER_GUIDE.md](BEGINNER_GUIDE.md)；架构与设计原则 → [ARCHITECTURE.md](ARCHITECTURE.md)、[DESIGN.md](DESIGN.md)；策略全览 → [STRATEGY_ARSENAL.md](STRATEGY_ARSENAL.md)；文档索引 → [INDEX.md](INDEX.md)。
