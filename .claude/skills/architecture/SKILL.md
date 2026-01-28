---
name: architecture
description: Use when exploring codebase structure, understanding index implementations (HNSW, IVF, DISKANN, Sparse, MinHash), working with third-party libraries (faiss, hnswlib, DiskANN, Cardinal), or locating specific functionality
---

# Knowhere Architecture

## Directory Structure

| Directory | Purpose |
|-----------|---------|
| `include/knowhere/` | Public headers |
| `src/index/` | Index implementations (flat/, hnsw/, ivf/, gpu/, diskann/, sparse/, minhash/) |
| `src/common/` | Threading, tracing, metrics utilities |
| `src/cluster/` | Clustering algorithms (KMeans) |
| `tests/ut/` | Unit tests |
| `thirdparty/` | Forked third-party libraries |

## Core Components

**Index Factory** (`include/knowhere/index/index_factory.h`)
- Singleton factory pattern for creating indexes
- Uses macro-based static registration (`KNOWHERE_SIMPLE_REGISTER_GLOBAL`)

**Index Interface** (`include/knowhere/index/index.h`)
- Template-based `Index<IndexNode>` wrapper
- Operations: Build, Search, RangeSearch

**Configuration** (`include/knowhere/config.h`)
- JSON-based config system
- Compile-time and runtime parameter validation

**Data Types** (`include/knowhere/operands.h`)
- fp32, fp16, bf16, int8, bin1 (binary), sparse_u32_f32

**Error Handling** (`include/knowhere/expected.h`)
- Custom `expected<T>` type with Status enum

## Metric Types

L2, IP (Inner Product), COSINE, Jaccard, Hamming, BM25

## Detailed Reference

- **Index types**: See [index-types.md](index-types.md) for each index's capabilities and use cases
- **Third-party libraries**: See [dependencies.md](dependencies.md) for faiss, hnswlib, DiskANN, Cardinal details
