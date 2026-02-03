# HNSW_RABITQ COSINE Recall 回退问题分析

## 状态：分析进行中，未开始修复

## 问题背景

分支 `hnsw_rbq` 上最近两条 commit：
- `9aeb8c5d` - Implement HNSW_RABITQ（旧版，recall 正常）
- `58717ff4` - Busfixs for HNSW_RABITQ（新版，recall 下降）

### 性能对比（cohere 数据集）

| 指标 | 旧版 (9aeb8c5d) | 新版 (58717ff4) |
|------|-----------------|-----------------|
| avg_recall | 0.77345 | 0.28401 |
| min_recall | 0.41 | 0 |
| avg_disterr | 1.36608 | 187.917 |
| QPS | 722.333 | 1090.69 |
| dist_recall | 0 | 1 |

## 旧版实现的两个已知问题（修复的动机）

1. **距离不准确**：L2 距离 ≠ 余弦距离。虽然 L2(normalized) 能给出正确的排序，但返回的距离值不是真正的余弦相似度。
2. **原始数据被破坏**：数据在送入索引前被归一化，导致 refine_type=FP32 时，refine 索引中存储的是归一化后的数据而非原始数据。

## 旧版 COSINE 处理方式（commit 9aeb8c5d）

- **Train/Add 阶段**：归一化所有数据 → 使用 `METRIC_L2` + 归一化数据
- **IVF_RABITQ**：接收归一化数据，使用 L2 度量
- **HNSW**：使用 `IndexHNSWFlatCosine`，内部处理归一化
- **Refine**：也接收归一化数据（**这是问题 #2 的根源**）
- **Search 阶段**：归一化查询向量后搜索
- **距离结果**：`||q_norm - x_norm||² = 2 - 2·cos(q,x)` → 排序正确但不是真余弦距离（**问题 #1**）

## 新版 COSINE 处理方式（commit 58717ff4）

- **Train/Add 阶段**：使用原始数据（不归一化），使用 `METRIC_INNER_PRODUCT`
- **IVF_RABITQ**：接收原始数据，使用 IP 度量
- **新增类**：`IndexIVFRaBitQWrapperCosine`、`IndexHNSWRaBitQWrapperCosine`、`CachedHNSWRaBitDistanceComputerCosine`
- **余弦计算**：`IP(q, or) × inv_norm_x × inv_norm_q ≈ cos(q, x)`
- **Search 阶段**：不归一化查询，依赖 distance computer 内部处理
- **序列化**：额外保存 inverse L2 norms 到 BinarySet

## 核心发现

### 发现 1：RaBitQ 的 IP 度量返回值

**文件**：`thirdparty/faiss/faiss/impl/RaBitQuantizer.cpp:125-128, 249-259`

RaBitQ 编码时，对 IP 度量做了特殊处理：
```cpp
fac->or_minus_c_l2sqr = norm_L2sqr;           // ||or - c||²
if (metric_type == METRIC_INNER_PRODUCT) {
    fac->or_minus_c_l2sqr -= or_L2sqr;         // ||or - c||² - ||or||²
}
```

`distance_to_code` 返回值推导：
```
// 对于 IP 度量：
pre_dist = (||or-c||² - ||or||²) + ||q-c||² - 2·(q-c)·(or-c)
         ≈ ||q||² - 2·IP(q, or)

return = -0.5 × (pre_dist - ||q||²)
       = -0.5 × (-2·IP(q, or))
       = +IP(q, or)     ← 正的内积！
```

**关键**：RaBitQ 对 IP 度量返回 **正的** 内积值 `+IP(q, or)`。

而 faiss 标准 IP distance computer（如 `IndexFlatIP`）返回 `+IP(q, x)`（也是正值）。
对 L2 度量返回 `||q - or||²`（正值，越小越相似）。

### 发现 2：HNSW 的 NegativeDistanceComputer 机制

**文件**：`thirdparty/faiss/faiss/IndexHNSW.cpp:72-78`

```cpp
DistanceComputer* storage_distance_computer(const Index* storage) {
    if (is_similarity_metric(storage->metric_type)) {
        return new NegativeDistanceComputer(storage->get_distance_computer());
    } else {
        return storage->get_distance_computer();
    }
}
```

HNSW 对 `METRIC_INNER_PRODUCT` 类型的 storage，会用 `NegativeDistanceComputer` 包装距离计算器，将返回值取反。这保证了 HNSW 的 "越小越好" 约定：
- **L2 度量**：不包装，直接返回正的 L2 距离 → 越小越近 ✓
- **IP 度量**：包装后返回 `-IP(q, x)` → 越小越相似 ✓

### 发现 3：Wrapper 的 metric_type 设置正确

**文件**：`src/index/ivf/ivfrbq_wrapper.cc:180-187`

```cpp
IndexIVFRaBitQWrapper::IndexIVFRaBitQWrapper(std::unique_ptr<faiss::Index>&& index_in)
    : Index{index_in->d, index_in->metric_type}, index{std::move(index_in)} {
```

Wrapper 构造函数从内部 index 正确继承了 `metric_type`。对于 COSINE 场景，内部 IndexPreTransform 使用 `METRIC_INNER_PRODUCT`，所以 wrapper 的 `metric_type = METRIC_INNER_PRODUCT`。

### 发现 4：HNSW 内部搜索约定

**文件**：`thirdparty/faiss/faiss/impl/HNSW.cpp:498`

```cpp
// greedy_update_nearest 永远使用 < 比较
if (dis < d_nearest) {
    d_nearest = dis;
    nearest = v;
}
```

HNSW 的贪心遍历 **永远使用小于比较**（`<`），不区分度量类型。MinimaxHeap 也维护最小值集合。

### 发现 5：IndexHNSW::search() 的距离回转

**文件**：`thirdparty/faiss/faiss/IndexHNSW.cpp:315-335`

```cpp
void IndexHNSW::search(...) {
    // ... 搜索 ...
    if (is_similarity_metric(this->metric_type)) {
        // 将负的距离值回转为正值
        for (size_t i = 0; i < k * n; i++) {
            distances[i] = -distances[i];
        }
    }
}
```

## 未确认的关键问题（分析中断点）

### ⚠️ knowhere 的 HNSW 搜索是否经过 `storage_distance_computer()`？

这是目前分析的核心悬疑点。有两种可能：

**可能 A：knowhere 经过 `storage_distance_computer()`（走 faiss 标准路径）**

如果是这样：
1. `CachedHNSWRaBitDistanceComputerCosine` 返回 `+cos(q, x)`
2. `NegativeDistanceComputer` 取反为 `-cos(q, x)`
3. HNSW greedy `<` 比较：更小（更负）= 更相似 ✓
4. `IndexHNSW::search()` 最后取反回 `+cos(q, x)` ✓
5. **理论上应该正确** → 但实际 recall 仍然很低，需要找其他原因

**可能 B：knowhere 绕过 `storage_distance_computer()`，直接调用 `storage->get_distance_computer()`**

如果是这样：
1. `CachedHNSWRaBitDistanceComputerCosine` 返回 `+cos(q, x)`（正值）
2. **没有** NegativeDistanceComputer 包装
3. HNSW greedy `<` 比较：更小 = **更不相似** ✗
4. HNSW 会向远离查询的方向探索 → **recall 大幅下降**
5. 搜索快速终止（找不到改善候选）→ **QPS 反而升高**
6. 距离不是余弦值而是原始 IP 值 → **距离误差巨大**

**可能 B 与实际症状完全吻合：**
- recall 0.77 → 0.28（搜索方向错误）
- QPS 722 → 1091（提前终止）
- disterr 1.37 → 187.9（返回原始 IP 而非余弦值）

**注意**：旧版用 L2 度量，即使 knowhere 绕过 `storage_distance_computer()`，L2 距离（正值，越小越好）也不需要 NegativeDistanceComputer，所以旧版碰巧是正确的。

### 下一步验证方向

1. **检查 knowhere HNSW 搜索路径**：在 `src/index/hnsw/faiss_hnsw.cc` 中搜索 `BaseFaissRegularIndexHNSWNode::Search` 的实现，确认是否调用 `IndexHNSW::search()` 或自定义搜索
2. **检查 faiss HNSW 搜索代码**：在 `thirdparty/faiss/faiss/IndexHNSW.cpp` 和 `thirdparty/faiss/faiss/impl/HNSW.cpp` 中确认 `storage_distance_computer()` 的调用路径
3. **快速验证**：复制 `.new` 索引，只运行 search，在距离计算器中加 log 确认返回值范围

## 两种修复方案（待确认可能 A/B 后选定）

### 方案一：修复距离符号（如果是可能 B）

在 `CachedHNSWRaBitDistanceComputerCosine::operator()` 中取反距离：
```cpp
distance = distance * inverse_l2_norms[i] * inverse_query_norm;
return -distance;  // 取反，保证 HNSW 兼容性（越小越相似）
```

**优点**：改动小，保留新架构（原始数据、IP 度量、逆范数）
**风险**：如果某些路径经过 `storage_distance_computer()`，会被双重取反

### 方案二：L2 归一化数据给 RaBitQ + 原始数据给 Refine（混合方案）

- IVF_RABITQ：归一化数据 + L2 度量（保证 HNSW 兼容，无需取反）
- Refine：原始数据（修复旧版问题 #2）
- HNSW flat：原始数据（`IndexHNSWFlatCosine` 内部处理）
- 搜索时归一化查询
- 搜索后：将 L2 距离转换为余弦距离 `cos = 1 - L2²/2`（修复旧版问题 #1）

**优点**：
- L2 度量不需要 NegativeDistanceComputer，避免符号问题
- 分离 RaBitQ 和 Refine 的数据路径
- 解决旧版的两个问题

**缺点**：改动较大，需要仔细处理数据路径分离

## 涉及的关键文件

| 文件 | 作用 |
|------|------|
| `src/index/hnsw/faiss_hnsw.cc` | HNSW_RABITQ 的 Train/Add/Search/Serialize/Deserialize |
| `src/index/ivf/ivfrbq_wrapper.cc` | IVF_RABITQ wrapper 实现，包括距离计算器 |
| `src/index/ivf/ivfrbq_wrapper.h` | Wrapper 类声明和继承关系 |
| `thirdparty/faiss/faiss/impl/RaBitQuantizer.cpp` | RaBitQ 编码和距离计算 |
| `thirdparty/faiss/faiss/IndexHNSW.cpp` | HNSW 搜索，storage_distance_computer() |
| `thirdparty/faiss/faiss/IndexCosine.cpp` | WithCosineNormDistanceComputer, L2NormsStorage |
| `thirdparty/faiss/faiss/impl/HNSW.cpp` | HNSW 图遍历算法 |

## 复现命令

```bash
# 编译
cd /home/ubuntu/workspace/vecTool && make kw_repo=CLiqing kw_tag=main && make install && cp bin/vecTool_main vecTool

# 使用旧索引搜索
cp /home/ubuntu/workspace/dataset/cohere/cohere_COSINE_version_9_HNSW_RABITQ_graph_build_algo_vamana_max_degree_48_nlist_256_kw.index.old \
   /home/ubuntu/workspace/dataset/cohere/cohere_COSINE_version_9_HNSW_RABITQ_graph_build_algo_vamana_max_degree_48_nlist_256_kw.index
cd /home/ubuntu/workspace/vecTool && bash runVectool.sh --index_type=HNSW_RABITQ --data=cohere --stage=search

# 使用新索引搜索
cp /home/ubuntu/workspace/dataset/cohere/cohere_COSINE_version_9_HNSW_RABITQ_graph_build_algo_vamana_max_degree_48_nlist_256_kw.index.new \
   /home/ubuntu/workspace/dataset/cohere/cohere_COSINE_version_9_HNSW_RABITQ_graph_build_algo_vamana_max_degree_48_nlist_256_kw.index
cd /home/ubuntu/workspace/vecTool && bash runVectool.sh --index_type=HNSW_RABITQ --data=cohere --stage=search

# 完整构建（耗时）
cd /home/ubuntu/workspace/vecTool && bash runVectool.sh --index_type=HNSW_RABITQ --data=cohere --stage=build,search
```
