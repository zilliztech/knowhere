---
name: index-optimization
description: Use when developing or evaluating Knowhere index performance optimizations, including construction cost, memory use, insertion, recall, and search workload coverage.
---

# Index Optimization

Before changing an index, identify the affected build, insertion, storage, and search paths and define a
baseline and representative workloads. Consider construction time, construction memory, query
performance, and recall together. For queries, account for top-k and filtering; for sparse queries, also account for
query NNZ. Use these dimensions during design, not only when writing the final benchmark report.

Respect the user's requested scope and resource budget. Choose representative combinations first, then
expand where a regression, dispatch boundary, or interaction needs investigation. A full Cartesian product
is not required for every change. State which dimensions were measured and which remain unvalidated.

## Construction and memory

| Dimension | What to examine and measure |
|-----------|----------------------------|
| Build time | End-to-end wall time, including validation, training, counting, allocation, packing, and finalization. Separate data loading from index construction. Use phase timings to identify serial work and parallel scaling when relevant. |
| Build memory | Peak process RSS during construction, alongside input-data residency and final index memory. Track overlapping raw, typed, and packed buffers, worker scratch space, temporary metadata, and allocation capacity. |
| Final storage | In-memory index size and serialized size, reported separately from construction peak memory. |
| Growing indexes | If supported and affected, measure single-row and batched Add at multiple existing index sizes. Check throughput, latency distribution, reallocations, whole-index rescans/repacking, and write-lock duration when relevant. |

Measure baseline and candidate memory in separate fresh processes with equivalent input loading.
Use a construction-phase RSS sampler or a build-only process high-water measurement; state the method,
units, and whether input vectors and shared/mapped data are included. A process-lifetime high-water mark
collected after search or serialization cannot be attributed to construction alone.

`Index::Size()` and serialized bytes do not measure peak RSS. Label buffer-lifetime calculations as
estimates rather than measurements, and include old/new buffer overlap during conversion or reallocation.
A smaller final index can still require more memory to build.

## Query workload coverage

Choose supported values that span the production workload and the implementation's branch thresholds.
The following are starting points, not fixed requirements:

| Axis | Representative coverage |
|------|-------------------------|
| Top-k | Small, typical, and large requests, for example 1, 10, 100, and 1000 where meaningful. |
| Filter ratio | No filtering, light/moderate filtering, and highly selective filtering, for example 0%, 10%, 50%, 90%, and 99% excluded. Define the ratio as excluded documents divided by total documents and report eligible counts. |
| Filter distribution | For changes to filter/window/block handling, compare random masks with clustered or contiguous masks; equal global ratios can produce different local workloads. Include all-filtered and fewer-than-k eligible matches as correctness cases. |
| Sparse query NNZ | Bucket real queries into low, medium, and high NNZ groups using observed ranges or quantiles. Record bucket boundaries and query counts. If pruning or refinement changes the query, report original and effective NNZ. |
| Execution settings | Fix batch size, concurrency, thread pools, CPU affinity, and SIMD dispatch for comparisons. Vary them when the optimization specifically targets those dimensions. |

Check meaningful interactions, such as large top-k with high filtering, or high-NNZ sparse queries with
high filtering. For threshold-based dispatch, include inputs on both sides of the threshold.
Report results by workload group so an overall mean cannot hide a regression in one group.

Bucket existing sparse queries without changing their contents. If synthetic generation, truncation, or
other transformations alter a query, regenerate its ground truth. NNZ alone does not describe posting
work: consider term frequency or posting-list length when it explains differences between query groups.

## Recall and search performance

Report recall@k alongside QPS and latency for each top-k, filter, and sparse query NNZ workload group.
Measure latency at the stated load, including tail latency when the sample count supports it. Define
recall targets and acceptable regression tolerances for the workload before evaluating an optimization.

Compare speed and recall at the same search settings. For approximate search, also compare QPS/latency
at matched recall targets, sweeping relevant search parameters to show the speed/recall tradeoff.
Record the settings and achieved recall for each result; if a target cannot be reached, report that
explicitly. Treat speedups accompanied by recall loss as a tradeoff requiring evaluation against the
workload's quality target, especially for quantization, pruning, and refinement changes.

Use exact ground truth with the same metric, queries, and filters. Compute recall against the valid
neighbors available up to k, and document empty-truth and tied-score conventions. For quantized indexes,
measure recall against the original vectors and distinguish it from correctness of scores computed
from stored values.

## Reproducible comparisons

- Record baseline/candidate revisions, compiler and build flags, hardware, actual SIMD path, dataset
  identity and size, index/search configuration, seeds, and workload buckets.
- Use identical query IDs and filter masks for paired comparisons. Warm up and repeat timed runs;
  keep correctness/oracle work outside the timed search region and report variation.
- Avoid overlapping benchmark runs that compete for CPU, memory bandwidth, or I/O. State whether cache
  conditions represent warm or cold operation.
- Keep raw results and commands. Report build time, build peak memory, final size, and query
  latency/QPS/recall together with workload-specific regressions and untested configurations.
- Choose relevant correctness checks through the [testing skill](../testing/SKILL.md); consult the
  [building skill](../building/SKILL.md) when compiling or changing build options.

## Existing benchmarks

Look for reusable benchmarks in [benchmark/](../../../benchmark/). Inspect current code and options
before reuse, and extend coverage for the chosen workload matrix where needed. Do not assume every
parameter or platform is supported.
