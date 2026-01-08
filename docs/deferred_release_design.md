# KV Cache Deferred Release Mechanism

## 概述

本次改动实现了 KV Cache 的**延迟释放（Deferred Release）**机制，主要目标是：

1. **减少内存碎片**：通过对释放的块进行排序和合并，将相邻的空闲区域聚合在一起
2. **降低高并发时的分配竞争**：释放操作不立即加入空闲池，而是缓存后批量合并
3. **提升 KV Offload 传输效率**：连续的块可以通过更少的 DMA 传输操作完成

## 改动文件清单

| 文件 | 改动行数 | 改动目的 |
|------|----------|----------|
| `vllm/v1/core/kv_cache_utils.py` | +267 | 新增 `ReleaseCache` 类和 `FreeKVCacheBlockQueue` 的延迟释放方法 |
| `vllm/v1/core/block_pool.py` | +142 | `BlockPool` 集成延迟释放逻辑 |
| `vllm/v1/core/kv_cache_coordinator.py` | +24 | 传递延迟释放配置参数 |
| `vllm/v1/core/kv_cache_manager.py` | +4 | 传递延迟释放配置参数 |
| `vllm/v1/core/sched/scheduler.py` | +2 | 从 CacheConfig 读取配置并传递 |
| `vllm/config/cache.py` | +20 | 新增配置选项 |
| `tests/v1/core/test_deferred_release.py` | +450 | 新增单元测试 |

---

## 详细改动说明

### 1. `vllm/v1/core/kv_cache_utils.py`

#### 新增 `ReleaseCache` 类 (第156-220行)

```python
class ReleaseCache:
    """释放缓存，用于暂存待释放的块"""

    def __init__(self, enable_coalescing: bool = True)
    def add(self, block: KVCacheBlock)           # 添加单个块
    def add_batch(self, blocks: Iterable)        # 批量添加
    def pop_all(self) -> list[KVCacheBlock]      # 弹出所有块（可排序）
    def clear()                                   # 清空
```

**目的**：作为释放块的临时缓冲区，支持按 `block_id` 排序以实现块合并。

#### 修改 `FreeKVCacheBlockQueue` 类

**构造函数新增参数**：
- `enable_deferred_release: bool = False` - 是否启用延迟释放
- `enable_coalescing: bool = True` - 是否在合并时排序

**新增方法**：

| 方法 | 说明 |
|------|------|
| `deferred_free(block)` | 延迟释放单个块到缓存 |
| `deferred_free_n(blocks)` | 延迟释放多个块到缓存 |
| `merge_release_cache()` | 将缓存中的块合并到空闲池 |
| `try_allocate_with_merge(n)` | 尝试分配，不够时先合并缓存再重试 |
| `get_total_available_blocks()` | 获取总可用块数（空闲池+缓存） |
| `get_contiguous_free_ranges()` | 获取连续空闲块的范围列表 |
| `force_merge()` | 强制合并缓存 |

---

### 2. `vllm/v1/core/block_pool.py`

#### 构造函数新增参数
```python
def __init__(
    ...
    enable_deferred_release: bool = False,  # 新增
    enable_coalescing: bool = True,         # 新增
)
```

#### 修改 `free_blocks` 方法

原逻辑：直接将释放的块加入空闲池
新逻辑：
- 若 `enable_deferred_release=True`，调用 `deferred_free_n()` 缓存块
- 若 `enable_deferred_release=False`，保持原有行为

#### 新增方法

| 方法 | 说明 |
|------|------|
| `free_blocks_deferred(blocks)` | 强制使用延迟释放（忽略全局设置） |
| `get_num_free_blocks_in_pool()` | 仅获取空闲池中的块数 |
| `get_num_blocks_in_release_cache()` | 获取缓存中的块数 |
| `merge_release_cache()` | 强制合并缓存到空闲池 |
| `get_new_blocks_with_merge(n)` | 分配时自动合并缓存 |
| `get_contiguous_free_ranges()` | 获取连续空闲块范围（用于 KV Offload） |

#### 修改 `get_num_free_blocks` 方法

当启用延迟释放时，返回值包含空闲池和缓存的总和。

---

### 3. `vllm/v1/core/kv_cache_coordinator.py`

所有 Coordinator 类（`KVCacheCoordinator`, `KVCacheCoordinatorNoPrefixCache`, `UnitaryKVCacheCoordinator`, `HybridKVCacheCoordinator`）的构造函数新增参数：

```python
enable_deferred_release: bool = False
enable_coalescing: bool = True
```

并将参数传递给 `BlockPool`。

`get_kv_cache_coordinator` 工厂函数同样新增这两个参数。

---

### 4. `vllm/v1/core/kv_cache_manager.py`

`KVCacheManager.__init__` 新增参数并传递给 `get_kv_cache_coordinator`：

```python
enable_deferred_release: bool = False
enable_coalescing: bool = True
```

---

### 5. `vllm/v1/core/sched/scheduler.py`

从 `cache_config` 读取配置并传递给 `KVCacheManager`：

```python
self.kv_cache_manager = KVCacheManager(
    ...
    enable_deferred_release=self.cache_config.enable_deferred_release,
    enable_coalescing=self.cache_config.enable_block_coalescing,
)
```

---

### 6. `vllm/config/cache.py`

新增两个配置选项：

```python
enable_deferred_release: bool = False
"""启用 KV cache 块的延迟释放机制。
启用后，释放的块会先缓存，分配失败时再合并排序后重试。
可减少碎片化，提升高并发性能和 KV Offload 传输效率。"""

enable_block_coalescing: bool = True
"""启用块合并时的排序。
启用后（需同时启用 enable_deferred_release），合并时按 block_id 排序，
将相邻空闲区域聚合，减少内存碎片，提升 DMA 传输效率。"""
```

这两个配置被排除在 `compute_hash` 之外，不影响计算图。

---

### 7. `tests/v1/core/test_deferred_release.py`（新文件）

完整的单元测试，覆盖：

- `TestReleaseCache`: ReleaseCache 类的所有方法
- `TestFreeKVCacheBlockQueueDeferredRelease`: FreeKVCacheBlockQueue 延迟释放方法
- `TestBlockPoolDeferredRelease`: BlockPool 集成测试
- `TestDeferredReleaseIntegration`: 集成场景测试（碎片化、高并发、KV Offload）

---

## 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                     Block Lifecycle                              │
│                                                                  │
│   Allocated ──free()──► ReleaseCache ──merge()──► FreePool      │
│       ▲                    (Buffer)                    │        │
│       │                                                │        │
│       └────────────── allocate() ◄─────────────────────┘        │
│                                                                  │
│   触发合并的条件:                                                 │
│   1. 分配请求无法由当前空闲池满足                                  │
│   2. 显式调用 merge_release_cache()                              │
│   3. KV Offload 前调用 get_contiguous_free_ranges()              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 使用方式

### 启用延迟释放

在启动 vLLM 时添加参数：

```bash
# 方式1: 通过命令行参数（如果支持）
vllm serve model_name --enable-deferred-release --enable-block-coalescing

# 方式2: 通过代码配置
from vllm import LLM

llm = LLM(
    model="model_name",
    enable_deferred_release=True,
    enable_block_coalescing=True,
)
```

### 配置说明

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `enable_deferred_release` | `False` | 是否启用延迟释放 |
| `enable_block_coalescing` | `True` | 合并时是否按 block_id 排序 |

---

## 测试方式

### 运行单元测试

```bash
# 运行所有延迟释放相关测试
pytest tests/v1/core/test_deferred_release.py -v

# 运行特定测试类
pytest tests/v1/core/test_deferred_release.py::TestReleaseCache -v
pytest tests/v1/core/test_deferred_release.py::TestBlockPoolDeferredRelease -v

# 运行集成测试
pytest tests/v1/core/test_deferred_release.py::TestDeferredReleaseIntegration -v
```

### 运行现有 KV Cache 测试（确保兼容性）

```bash
# 运行现有 kv_cache_utils 测试
pytest tests/v1/core/test_kv_cache_utils.py -v

# 运行现有 prefix caching 测试
pytest tests/v1/core/test_prefix_caching.py -v
```

### 性能测试建议

```bash
# 1. 基准测试（不启用延迟释放）
python benchmarks/benchmark_serving.py --model model_name

# 2. 启用延迟释放后测试
python benchmarks/benchmark_serving.py --model model_name \
    --enable-deferred-release

# 3. 高并发场景测试
python benchmarks/benchmark_serving.py --model model_name \
    --enable-deferred-release \
    --num-prompts 1000 \
    --request-rate 100
```

---

## 后续优化方向

1. **自适应合并策略**：根据分配压力动态决定何时合并
2. **周期性后台合并**：避免分配时的延迟
3. **与 KV Offload 深度集成**：利用连续块信息优化 DMA 传输
4. **监控指标**：添加碎片化率、合并频率等 Prometheus 指标
