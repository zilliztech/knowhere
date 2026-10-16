# Third-Party Dependencies

Located in `thirdparty/`, these are forked/customized versions of core algorithm libraries.

## faiss

- **Source**: Meta's similarity search library
- **Location**: `thirdparty/faiss/`
- **Provides**: HNSW, IVF, PQ, and flat index implementations
- **Used by**: HNSW (`src/index/hnsw/faiss_hnsw.cc`), IVF, and Flat indexes
- **Key features**:
  - Vector quantization (PQ, SQ)
  - Clustering (KMeans)
  - SIMD-optimized distance computations

## hnswlib

- **Source**: Header-only HNSW implementation
- **Location**: `thirdparty/hnswlib/`
- **Provides**: Graph-based approximate nearest neighbor search
- **Used by**: Legacy HNSW implementation and CPU adaptation of GPU CAGRA indexes
- **Key features**:
  - Hierarchical graph structure
  - Fast search with configurable accuracy
  - Support for incremental insertions

## DiskANN

- **Source**: Microsoft's disk-based ANN library
- **Location**: `thirdparty/DiskANN/`
- **Provides**: Billion-scale search on SSDs
- **Used by**: DISKANN, AISAQ indexes
- **Key features**:
  - Graph index stored on disk
  - Minimal memory footprint
  - SSD-optimized I/O patterns
- **AISAQ extension**: KIOXIA's enhancement with PQ compression and optimized I/O

## Build Dependencies

Use `conanfile.py` and `cmake/libs/` for the current versions and platform conditions.
The options below use the repository Makefile entry point.

| Dependency | Required For | Build Option |
|------------|--------------|--------------|
| OpenBLAS | CPU BLAS on non-Apple platforms | Selected by the build configuration |
| System BLAS/LAPACK | CPU BLAS on Apple platforms | Selected by the build configuration |
| libaio | DISKANN, AISAQ on Linux | `WITH_DISKANN=True` |
| CUDA | GPU indexes | `WITH_GPU=True` |
