# Index Types

## Overview

| Index Family | Type | Best For | Metric Support |
|--------------|------|----------|----------------|
| Flat | Brute force | Small datasets, 100% recall | L2, IP, COSINE |
| HNSW | Graph-based | Low latency, high recall | L2, IP, COSINE |
| IVF | Inverted file | Large datasets, balanced | L2, IP, COSINE |
| SCANN | IVF + Anisotropic | Fast search, good recall | L2, IP, COSINE |
| DISKANN | Disk-based | Billion-scale, limited RAM | L2, IP, COSINE |
| AISAQ | Disk-based + PQ | Billion-scale, optimized I/O | L2, IP, COSINE |
| SPARSE_INVERTED_INDEX | Inverted index | Sparse vectors, text search | IP, BM25 |
| SPARSE_WAND | Inverted index (WAND) | Sparse vectors, optimized top-k | IP, BM25 |
| MinHash | LSH | Jaccard similarity | Jaccard |
| GPU (CUVS) | CUDA-accelerated | High throughput | L2, IP |

## Index Details

### Flat
- Location: `src/index/flat/`
- Brute force search, no training required
- 100% recall, O(n) search complexity

### HNSW Series
- Location: `src/index/hnsw/`
- Hierarchical Navigable Small World graph
- Key params: `M` (connections), `efConstruction`, `ef` (search)

| Variant | Quantization | Use Case |
|---------|--------------|----------|
| HNSW | None | Best recall, higher memory |
| HNSW_SQ | Scalar (8-bit) | Balanced memory/recall |
| HNSW_PQ | Product | Lower memory, good recall |
| HNSW_PRQ | Product Residual | Lowest memory, acceptable recall |

### IVF Series
- Location: `src/index/ivf/`
- Inverted file with clustering
- Key params: `nlist` (clusters), `nprobe` (search clusters)
- Growing variants (concurrent read/write): IVF_FLAT_CC, IVF_SQ_CC support real-time insertion

| Variant | Quantization | Data Type | Use Case |
|---------|--------------|-----------|----------|
| IVF_FLAT | None | Dense float/int8 | Best recall in IVF family |
| IVF_SQ | Scalar | Dense float/int8 | Balanced memory/recall |
| IVF_PQ | Product | Dense float/int8 | Lower memory, training required |
| IVF_RABITQ | RaBitQ | Dense float | Ultra compression (~32x) |
| BIN_IVF_FLAT | None | Binary | Binary vector search |

### SCANN
- Location: `src/index/ivf/ivf.cc` (uses `faiss::IndexScaNN`)
- Score-aware quantization with anisotropic vector quantization
- Based on Google's ScaNN algorithm
- Key params: `nlist`, `nprobe`, `with_raw_data`, `reorder_k`
- SCANN_DVR variant: with DataViewRefiner for optimized reranking

### DISKANN
- Location: `src/index/diskann/`
- Disk-based graph index for billion-scale
- Requires: `-o with_diskann=True` build option
- Key params: `R` (degree), `L` (search list size)

### AISAQ
- Location: `src/index/diskann/diskann_aisaq.cc`
- Enhanced DISKANN with PQ compression and I/O optimization (by KIOXIA)
- Requires: `-o with_diskann=True` build option
- Key params: `vectors_beamwidth`, `inline_pq`, `pq_cache_size`, `num_entry_points`

### Sparse Series
- Location: `src/index/sparse/`
- Inverted index for sparse vectors
- Data type: `sparse_u32_f32`
- Metrics: IP (inner product), BM25 (text relevance)
- BM25 params: `k1`, `b`, `avgdl`

| Variant | Description |
|---------|-------------|
| SPARSE_INVERTED_INDEX | Standard inverted index |
| SPARSE_WAND | WAND algorithm for optimized top-k retrieval |

### MinHash
- Location: `src/index/minhash/`
- MinHash LSH for Jaccard similarity
- For binary/set data

### GPU (CUVS)
- Location: `src/index/gpu_cuvs/`
- CUDA-accelerated implementations via NVIDIA cuVS
- Requires: `-o with_cuvs=True` build option

| Variant | Description | Data Types |
|---------|-------------|------------|
| GPU_CAGRA | Graph-based (CAGRA algorithm) | fp32, fp16, int8, binary |
| GPU_BRUTE_FORCE | Exact search | fp32, fp16 |
| GPU_IVF_FLAT | IVF without quantization | fp32, fp16, int8 |
| GPU_IVF_PQ | IVF with product quantization | fp32, fp16, int8 |

## Quantization Types

| Type | Abbreviation | Description | Memory Reduction |
|------|--------------|-------------|------------------|
| Scalar 4-bit | SQ4U | 4-bit uniform scalar quantization | ~8x |
| Scalar 6-bit | SQ6 | 6-bit scalar quantization | ~5x |
| Scalar 8-bit | SQ8 | 8-bit scalar quantization | ~4x |
| Product | PQ | Vector split into subvectors, each quantized | ~8-32x |
| Product Residual | PRQ | PQ applied to residuals iteratively | ~8-32x |
| RaBitQ | RaBitQ | Random Binary Quantization | ~32x |

## Data Type Support

| Index | fp32 | fp16 | bf16 | int8 | binary | sparse_u32_f32 |
|-------|------|------|------|------|--------|----------------|
| Flat | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| **HNSW Series** |
| HNSW | ✓ | ✓ | ✓ | ✓ | - | - |
| HNSW_SQ | ✓ | ✓ | ✓ | ✓ | - | - |
| HNSW_PQ | ✓ | ✓ | ✓ | ✓ | - | - |
| HNSW_PRQ | ✓ | ✓ | ✓ | ✓ | - | - |
| **IVF Series** |
| IVF_FLAT | ✓ | ✓ | ✓ | ✓ | - | - |
| IVF_SQ | ✓ | ✓ | ✓ | ✓ | - | - |
| IVF_PQ | ✓ | ✓ | ✓ | ✓ | - | - |
| IVF_RABITQ | ✓ | ✓ | ✓ | - | - | - |
| BIN_IVF_FLAT | - | - | - | - | ✓ | - |
| **Others** |
| SCANN | ✓ | ✓ | ✓ | ✓ | - | - |
| DISKANN | ✓ | ✓ | ✓ | - | - | - |
| AISAQ | ✓ | ✓ | ✓ | - | - | - |
| SPARSE_INVERTED_INDEX | - | - | - | - | - | ✓ |
| SPARSE_WAND | - | - | - | - | - | ✓ |
| MinHash | - | - | - | - | ✓ | - |
