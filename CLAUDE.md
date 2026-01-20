# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Knowhere is a C++ vector search library that serves as the core engine for Milvus. It provides a unified interface for Approximate Nearest Neighbor (ANN) algorithms supporting multiple data types (fp32, fp16, bf16, int8, binary, sparse vectors) and index implementations (HNSW, IVF, Flat, DISKANN, MinHash).

## Build Commands

### Prerequisites
```bash
# Ubuntu/Debian
sudo apt install build-essential libopenblas-openmp-dev libaio-dev python3-dev python3-pip
pip3 install conan==1.61.0 --user
export PATH=$PATH:$HOME/.local/bin
```

### Building
```bash
mkdir build && cd build
conan remote add default-conan-local https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local

# CPU Release
conan install .. --build=missing -o with_ut=True -s compiler.libcxx=libstdc++11 -s build_type=Release

# CPU Debug
conan install .. --build=missing -o with_ut=True -s compiler.libcxx=libstdc++11 -s build_type=Debug

# GPU (CUVS) build
conan install .. --build=missing -o with_ut=True -o with_cuvs=True -s compiler.libcxx=libstdc++11 -s build_type=Release

# DISKANN support
conan install .. --build=missing -o with_ut=True -o with_diskann=True -s compiler.libcxx=libstdc++11 -s build_type=Release

# Build
conan build ..
```

### macOS
```bash
conan install .. --build=missing -o with_ut=True -s compiler.libcxx=libc++ -s build_type=Release
conan build ..
```

## Running Tests

Uses Catch2 framework:
```bash
# Run all tests
./Release/tests/ut/knowhere_tests
./Debug/tests/ut/knowhere_tests

# Run specific test by name pattern
./Release/tests/ut/knowhere_tests "[float metrics]"
./Release/tests/ut/knowhere_tests "Test Mem Index*"

# List all test names
./Release/tests/ut/knowhere_tests --list-tests
```

## Code Quality

```bash
pip3 install pre-commit
pre-commit install --hook-type pre-commit --hook-type pre-push

# Run checks manually
pre-commit run
```

Code style: Google style with 120 char line limit, 4 space indent (see `.clang-format`).

## Workflow

Before committing changes, always run pre-commit validation:
```bash
pre-commit run --files <changed-files>  # or: pre-commit run --all-files
git add . && git commit -m "message"
```

## Architecture

### Directory Structure
- `include/knowhere/` - Public headers
- `src/index/` - Index implementations (flat/, hnsw/, ivf/, gpu/, diskann/, sparse/, minhash/)
- `src/common/` - Threading, tracing, metrics utilities
- `src/cluster/` - Clustering algorithms (KMeans)
- `tests/ut/` - Unit tests

### Key Components

**Index Factory** (`include/knowhere/index/index_factory.h`): Singleton factory pattern for creating indexes. Uses macro-based static registration (`KNOWHERE_SIMPLE_REGISTER_GLOBAL`).

**Index Interface** (`include/knowhere/index/index.h`): Template-based `Index<IndexNode>` wrapper providing Build, Search, RangeSearch operations.

**Configuration** (`include/knowhere/config.h`): JSON-based config system with compile-time and runtime parameter validation.

**Data Types** (`include/knowhere/operands.h`): Supported types include fp32, fp16, bf16, int8, bin1 (binary), sparse_u32_f32 (sparse vectors).

**Error Handling** (`include/knowhere/expected.h`): Custom `expected<T>` type with Status enum for type-safe error propagation.

### Index Types
- **Flat**: Brute force search
- **HNSW**: Hierarchical Navigable Small World graph
- **IVF**: Inverted File indexes (IVF_FLAT, IVF_PQ, IVF_SQ8)
- **DISKANN**: Disk-based ANN for large datasets
- **Sparse**: Sparse vector indexes (inverted index)
- **MinHash**: MinHash LSH for Jaccard similarity
- **GPU**: CUDA-accelerated implementations (CUVS)

### Metric Types
L2, IP (Inner Product), COSINE, Jaccard, Hamming

### Third-Party Dependencies (`thirdparty/`)

Forked/customized versions of core algorithm libraries:

- **faiss/** - Meta's similarity search library. Provides IVF, PQ, and flat index implementations. Core algorithms for vector quantization and clustering.
- **hnswlib/** - Header-only HNSW implementation. Graph-based approximate nearest neighbor search algorithm.
- **DiskANN/** - Microsoft's disk-based ANN library. Enables billion-scale search on SSDs without loading full index into memory.
- **cardinalv1/, cardinalv2/** - Zilliz's enterprise Cardinal variants. Extended index implementations with additional features.
