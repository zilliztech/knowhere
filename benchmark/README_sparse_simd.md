# Sparse Inverted Index SIMD Benchmark

Comprehensive benchmark for the AVX512-optimized sparse inverted index implementation.

## Features

- **Multiple dataset sizes**: Small (10K docs), Medium (100K docs), Large (1M docs)
- **Both metrics**: IP (Inner Product) and BM25
- **Realistic data**: Power-law posting list distributions
- **Correctness verification**: Validates AVX512 results against scalar baseline
- **Performance metrics**: Reports speedup, absolute timings, and throughput
- **CI-friendly output**: Clean, parseable output format

## Building

### Option 1: CMake (integrated with main build)

```bash
cd knowhere
mkdir -p build && cd build
cmake ..
make benchmark_sparse_simd
```

The binary will be at: `build/benchmark/benchmark_sparse_simd`

### Option 2: Standalone Makefile (quick testing)

```bash
cd knowhere/benchmark
make -f Makefile.sparse_simd
./benchmark_sparse_simd_standalone
```

**Note**: AVX512 requires a compatible CPU and compiler flags `-mavx512f -mavx512dq`

## Running

### Run all benchmarks
```bash
./benchmark_sparse_simd
```

### Expected Output

```
╔══════════════════════════════════════════════════════════════════╗
║  Sparse Inverted Index SIMD Benchmark                           ║
╚══════════════════════════════════════════════════════════════════╝

=== Small dataset (IP metric) ===
Dataset: 10000 docs, 1000 vocab, query length: 10
Avg posting list length: 50.0
CPU Capabilities: AVX512F=1, AVX2=1

[Scalar Fallback]
  Time: 0.123 ms
  Non-zero scores: 450 / 10000

[AVX512 SIMD]
  Time: 0.045 ms
  Non-zero scores: 450 / 10000

[Verification]
  Max difference: 0.000001
  Avg difference: 0.000000 (over 0 elements)
  Correctness: PASS

[Performance]
  Speedup: 2.73x
  Scalar:  0.123 ms (baseline)
  AVX512:  0.045 ms (36.6% of baseline)
==========================================
```

## Benchmark Details

### Dataset Characteristics

- **Posting lists**: Realistic power-law distribution (common terms have longer lists)
- **Query terms**: Random selection with variable weights
- **Document IDs**: Random distribution (tests random memory access performance)
- **Doc lengths**: Normal distribution around average (for BM25)

### What is Measured

1. **Scalar Baseline**: Simple double-loop implementation matching original code
2. **AVX512 SIMD**: Optimized implementation with:
   - 16-wide vectorization
   - 2x loop unrolling (32 elements/iteration)
   - Hardware gather/scatter operations

### Verification

The benchmark validates correctness by:
- Comparing AVX512 results against scalar baseline
- Checking max absolute difference (should be < 0.001)
- Counting non-zero scores (should match exactly)

### Performance Metrics

- **Time**: Average execution time over 50 runs (after 5 warmup runs)
- **Speedup**: Ratio of scalar time to AVX512 time
- **Throughput**: Queries per second (for multi-query benchmarks)

## Expected Performance

On AVX512-capable CPUs (Intel Skylake-X or newer), expect:

- **IP metric**: 2-4x speedup
- **BM25 metric**: 1.5-2.5x speedup (limited by scalar BM25 computation)
- **Large posting lists**: Better speedup (amortizes gather latency)
- **Short posting lists**: Lower speedup (tail loop overhead)

## CI Integration

The benchmark is designed for CI runs:

1. **Exit code**: Returns 0 on success, 1 on verification failure
2. **Output format**: Easy to parse for regression detection
3. **Quick runtime**: ~1-2 seconds for all configurations
4. **No external data**: Generates synthetic datasets on-the-fly

## Troubleshooting

### "Illegal instruction" error

Your CPU doesn't support AVX512. Check with:
```bash
grep avx512 /proc/cpuinfo
```

### Build fails with "unrecognized command line option '-mavx512f'"

Your compiler is too old. Requires GCC 4.9+ or Clang 3.9+.

### Verification fails

This indicates a bug in the SIMD implementation. Please report with:
- CPU model (`cat /proc/cpuinfo | grep "model name"`)
- Compiler version (`g++ --version` or `clang++ --version`)
- Full benchmark output

## Implementation Details

See `src/simd/sparse_simd.h` for the AVX512 implementation and
`src/index/sparse/sparse_inverted_index.h` for the runtime dispatcher.
