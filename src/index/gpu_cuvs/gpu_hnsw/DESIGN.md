# GPU_HNSW: GPU-Accelerated HNSW Search Kernel for Knowhere

## Feature Overview

### What

GPU_HNSW is a CUDA implementation of HNSW (Hierarchical Navigable Small World) graph search that runs entirely on GPU. It takes an existing CPU-built HNSW index, converts the graph to a GPU-friendly format at load time, and executes searches on GPU — achieving 10-50x throughput over CPU HNSW at identical recall.

This is not a heuristic approximation — it implements correct HNSW semantics (greedy upper-layer descent + beam search on layer 0) and matches CPU recall at the same `ef` value across all dimensions and metrics.

### Why

CPU HNSW at large scale (100M-1B+ vectors) is bottlenecked by DRAM bandwidth during graph traversal. Each hop loads a neighbor's vector (128-1024 bytes) and adjacency list from random memory locations, saturating CPU cache hierarchy. GPU VRAM provides 10-20x higher bandwidth with massive parallelism across queries.

Existing GPU ANN options (CAGRA, IVF-PQ) have limitations at scale that make HNSW on GPU the best path forward for production workloads requiring high recall on INT8 embeddings.

### Key Properties

- **Correct HNSW semantics** — matches CPU recall at same `ef` (not a heuristic)
- **Zero rebuild cost** — uses existing CPU-built HNSW indexes, converts at load time
- **Supports L2, IP, COSINE** metrics; **INT8, FP16, FP32** data types
- **Batch parallel** — one thread block per query, scales linearly with batch size
- **Eager GPU upload** — segments are GPU-ready at Deserialize time (no cold-start penalty)

### Performance

Tested on RTX PRO Server 6000 (96 GB VRAM), INT8 embeddings, COSINE metric:

| Dataset | Throughput (per node) | R@1 | ef |
|---------|----------------------|------|-----|
| 2M x 384-dim | 12,700 QPS | 99.8% | 64 |
| 8M x 384-dim | 8,500 QPS | 99.6% | 64 |
| 32.9M x 384-dim | 10,500 QPS | 99.8% | 64 |
| 66M x 1024-dim (production) | ~4,400 QPS | 95.0% | 64 |

Throughput scales linearly with querynodes. For 529M vectors across 8 GPU nodes: ~35,000 QPS projected.

---

## Design

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│ GpuHnswIndexNode (faiss_hnsw.cc)                                        │
│   Deserialize() → build_gpu_index() [eager upload]                      │
│   Search() → gpu_hnsw::search() → kernels                              │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ gpu_hnsw_impl.cuh — search() entry point                                │
│   Allocates scratch, launches Phase 1 + Phase 2 kernels                 │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────┐    ┌─────────────────────────────────────────┐
│ Phase 1:               │    │ Phase 2:                                  │
│ upper_layer_search     │───▶│ layer0_beam_search_kernel                 │
│ (1 warp per query,     │    │ (1 block per query, 128 threads,          │
│  greedy descent)       │    │  OCQ + cooperative distances)             │
└────────────────────────┘    └─────────────────────────────────────────┘
```

### File Layout

| File | Purpose |
|------|---------|
| `gpu_hnsw_search_kernel.cuh` | CUDA kernels (Phase 1 + Phase 2) |
| `gpu_hnsw_impl.cuh` | Host-side search orchestration |
| `gpu_hnsw_interface.cu/hpp` | Public API (`build_gpu_index`, `search`) |
| `gpu_hnsw_faiss_build.hpp` | Convert FAISS HNSW index to GPU format |
| `gpu_hnsw_types.hpp` | Data structures (`search_params`, `search_scratch`) |

### Graph Conversion (CPU → GPU)

At load time, `build_gpu_index()` converts the FAISS HNSW adjacency-list format to GPU-friendly flat arrays:

```
FAISS format:              GPU format (CSR-like):
  node 0: [3, 7, 12]        offsets:  [0, 3, 5, 9, ...]
  node 1: [0, 5]            neighbors: [3, 7, 12, 0, 5, ...]
  node 2: [1, 4, 8, 11]     
```

This eliminates pointer chasing on GPU and enables coalesced memory access patterns.

---

## Search Algorithm

### Phase 1: Upper-Layer Greedy Search

One warp (32 threads) per query. Greedy descent from the global entry point through upper layers down to layer 0, producing a single entry point for the beam search.

Each lane independently computes distance to one neighbor; warp-level reduction finds the best. Iterates until no neighbor improves the current best.

### Phase 2: Layer-0 Beam Search with OCQ

One thread block (128 threads) per query. Unified main loop:

```
Seeding → [Unified Loop: Step 1 → Step 2 → Step 3 → convergence check]
```

#### Overflow Candidate Queue (OCQ)

The critical correctness mechanism. Without OCQ, the GPU kernel conflates the result buffer with the candidate pool, causing premature pruning:

```
CPU HNSW:  result buffer (top-ef) + separate MinimaxHeap (unbounded candidates)
                                     ↑ recycles slots via pop_min
GPU naive: single buffer (top-ef) = result AND candidate pool
                                     ↑ loses marginal candidates permanently

GPU OCQ:   result buffer (Tier 1, shared memory, size ef)
           + overflow queue (Tier 2, global memory, size overflow_ef)
             ↑ captures evicted/rejected candidates for later expansion
```

**Why OCQ is necessary:** The CPU's MinimaxHeap can hold 2-4x more candidates than `ef` during a search. When the GPU's fixed-size result buffer evicts an entry, that entry may still be worth expanding (its neighbors could lead to better results). The overflow queue preserves these entries.

**Data structures (per query):**
- `overflow_ids[overflow_ef]` — node IDs (global memory)
- `overflow_dists[overflow_ef]` — distances, sorted ascending (global memory)
- `overflow_expanded[overflow_ef]` — expansion flags (global memory)
- `overflow_count` — current valid entries

**Memory overhead:** 4.5 KB/query at ef=128 (vs 4.1 MB/query for visited bitmap).

#### Step 1: Parent Selection

Thread 0 selects up to `search_width` unexpanded parents:

1. **Priority 1:** Best unexpanded entries in result buffer (Tier 1)
2. **Priority 2:** Best unexpanded entries in overflow queue (only when result buffer is fully expanded)

This naturally handles phase transitions: early iterations expand result buffer entries (high quality); later iterations fall through to overflow entries (exploratory).

#### Step 2: Neighbor Expansion (Warp-Cooperative Distances)

All 128 threads cooperate to expand parents' neighbors. Uses **warp-cooperative distance computation** for bandwidth efficiency:

```
threads_per_dist = select_threads_per_dist(dim):
    dim >= 256 → 4 threads per distance
    dim >= 96  → 2 threads per distance
    dim < 96   → 1 thread (no cooperation)

num_groups = blockDim.x / threads_per_dist  (32 groups for dim=384)
```

Each group of `threads_per_dist` threads:
1. Lane 0 reads graph neighbor + does atomic `bitmap_visit`
2. Broadcasts visit result via `__shfl_sync(group_mask, ...)`
3. All lanes cooperatively compute distance (each handles `dim/threads_per_dist` dimensions), reduce via warp shuffle
4. Lane 0 writes (id, dist) to staging buffer

**Sub-warp masks:** Each group operates with its own mask (`((1u << threads_per_dist) - 1) << offset`), allowing groups to diverge independently without full-warp synchronization.

**Why this helps at scale:** At 32.9M+ vectors, graph + dataset exceed L2 cache. Cooperative groups reduce unique cache lines accessed per warp from 128 to 32, dramatically reducing L2 thrashing.

#### Step 3: Merge (Thread 0, Serial)

Thread 0 processes the staging buffer and merges into the result buffer:
- Candidates that beat `result_dists[ef-1]` are inserted (binary search + shift)
- Evicted entries from the result buffer spill to the overflow queue
- Rejected candidates go directly to the overflow queue

#### Early Convergence

Tracks consecutive iterations where `result_dists[ef-1]` doesn't improve. After 4 stale iterations, the search terminates even if overflow entries remain unexpanded. Zero-overhead (single counter check per iteration).

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ef` | 200 | Result buffer size (controls recall vs latency) |
| `search_width` | 4 | Parents expanded per iteration |
| `overflow_factor` | 2 | `overflow_ef = overflow_factor * ef` |
| `max_iterations` | auto | `(ef + overflow_ef) / search_width + 20` |
| `thread_block_size` | 128 | Threads per block (one block per query) |

### Tuning Guidelines

- **ef=64** is sufficient for 99.8% R@1 on real INT8 embeddings (32.9M tested)
- **overflow_factor=2** balances exploration vs iteration count; increase to 3 only for pathological distributions (high-dim, unnormalized, flat distance)
- **search_width=4** is optimal; increasing to 8 was tested and regressed throughput due to larger staging buffer and more per-iteration work

---

## Memory Layout

### Shared Memory (per block)

```
result_ids[ef]           — uint32_t
result_dists[ef]         — float (sorted ascending, best first)
is_expanded[ef]          — uint32_t (0/1 flags)
staging_ids[sw*M0]       — uint32_t (sw=search_width, M0=max_degree0)
staging_dists[sw*M0]     — float
parent_ids[sw]           — uint32_t
meta[4]                  — int (result_count, staging_count, num_parents, stale_count)
```

At ef=64, search_width=4, max_degree0=32: ~3 KB per block.

### Global Memory (per query)

- Visited bitmap: `(N+31)/32` uint32_t words (~4.1 MB at N=32.9M)
- Overflow queue: `overflow_ef * 3` arrays (~1.5 KB at overflow_ef=128)

---

## Correctness Guarantees

The OCQ makes this a **correct HNSW implementation**, not a heuristic:

1. **Separation of concerns:** Result buffer (top-k tracking) is separate from candidate pool (exploration frontier). The CPU does this with MinimaxHeap; GPU does it with the overflow queue.

2. **No premature pruning:** Candidates evicted from the result buffer are preserved in the overflow queue and eventually expanded if the result buffer stagnates.

3. **Convergence guarantee:** The unified loop terminates when both the result buffer AND overflow queue are fully expanded (or stale for 4 iterations). Bounded by `max_iterations`.

4. **Dimension/metric agnostic:** Works identically for any dim, any quantization (INT8/FP16/FP32), and any metric (L2/IP/COSINE). No tuning required per workload.

---

## Optimization History

| Version | Change | Impact |
|---------|--------|--------|
| v38 | OCQ implementation | Recall 0.70 → 0.998 at ef=64 |
| v39 | Eager GPU upload on Deserialize | Eliminated cold-start latency |
| v40 | overflow_factor 3→2, `__ldg` annotations | +25% throughput |
| v41 | Warp-cooperative distances | +27% throughput (cumulative +60% vs v39) |
| v42 | Bitonic sort + search_width=8 | **Regression** (reverted) |
| v43 | Early convergence (4 stale iterations) | +10% throughput on large datasets |

---

## Build Requirements

- CUDA Toolkit 12.x+
- Build with `make WITH_GPU=True` (adds `-o with_cuvs=True` to Conan)
- Requires GPU with compute capability 7.0+ (Volta or newer)

## Dependencies

- **Milvus integration:** [milvus-io/milvus#50653](https://github.com/milvus-io/milvus/pull/50653) — wires GPU_HNSW type through query node segment loading
