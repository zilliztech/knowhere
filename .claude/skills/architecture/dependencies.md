# Third-Party Dependencies

Located in `thirdparty/`, these are forked/customized versions of core algorithm libraries.

## faiss

- **Source**: Meta's similarity search library
- **Location**: `thirdparty/faiss/`
- **Provides**: IVF, PQ, and flat index implementations
- **Used by**: IVF_FLAT, IVF_PQ, IVF_SQ8, Flat indexes
- **Key features**:
  - Vector quantization (PQ, SQ)
  - Clustering (KMeans)
  - SIMD-optimized distance computations

## hnswlib

- **Source**: Header-only HNSW implementation
- **Location**: `thirdparty/hnswlib/`
- **Provides**: Graph-based approximate nearest neighbor search
- **Used by**: HNSW index
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

## Cardinal

- **Source**: Zilliz's enterprise extensions
- **Location**: `thirdparty/cardinalv1/`, `thirdparty/cardinalv2/`
- **Provides**: Extended index implementations with additional features
- **Note**: Enterprise/proprietary variants

## Build Dependencies

| Dependency | Required For | Build Option |
|------------|--------------|--------------|
| OpenBLAS | All builds | Always required |
| libaio | DISKANN, AISAQ | `-o with_diskann=True` |
| CUDA | GPU indexes | `-o with_cuvs=True` |
