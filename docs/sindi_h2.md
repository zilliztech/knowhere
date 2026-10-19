# Optional SINDI window-local bounds

Set `sindi_h2: true` at build or deserialization to enable window-local
remaining-score bounds. The default is false. Supported indexes are sealed
SINDI IP with FP16 postings and sealed SINDI BM25 with u8/u16 postings (including
auto resolution). Growable indexes, non-SINDI algorithms and packed BM25 U4/LUT
representations reject H2. Existing constructor arguments for refinement and
packed storage retain their positions; H2 is the fifth argument.

Summaries use represented FP16 values for IP, and restored u8 overflow TF,
original document lengths and the build scorer for BM25. Inputs and query
weights must be finite and nonnegative. BM25 search scorer parameters must
match the summary scorer. Summaries are derived after build/load, included in
index memory accounting and excluded from serialization. Misaligned legacy
posting arrays are copied to aligned storage during load.

Search retains global term order and accumulation kernels, replacing only the
remaining-contribution suffix with per-window maxima. Ratios below one remain
approximate. FP32 suffix arithmetic and heap tie behavior retain their existing
limitations; H2 is not a general guarantee of exact ranks at every ratio.

## ARM validation

Build on ARM with the checkout's `compile.sh`, with unit tests and benchmarks
enabled, then source `build/Release/generators/conanrun.sh` before execution.

```sh
build/Release/tests/ut/knowhere_tests '[sparse],[sindi]'
python3 scripts/validate_sparse_current.py \
  --binary build/Release/benchmark/benchmark_sparse \
  --data-root /volume/data_sparse2 --threads 8 \
  --output-dir reports/current_sparse_validation
```

The runner validates current source CSR/truth hashes and k10/k100 prefix
identity, format/negative fixtures, and 4097-document/8-query prefixes across
the dataset inventory. It checks H2 off/on ranked outputs and BinarySet/mmap
reloads with fresh independent float64 prefix truth. Prefix timings are smoke
evidence. Original-FP32 truth recall includes representation effects and
existing omission of zero-score results; it is separate from H2 parity.

Both metrics now read the same little-endian `.gt` layout: uint32 query count
and k, query-major uint32 IDs, then query-major float32 scores. Padding is
`UINT32_MAX` with score zero and is excluded from recall. Ranking occurs before
rounding stored scores, so float32 score equality alone does not determine ties.
Dataset `gt/metadata.json` supplies metric, BM25 parameters, `input_sha256` and
`output_sha256`. Supply explicit `--base-file`, `--query-file` and `--truth`
paths for full-corpus runs. Legacy `.bm25` and int64 truth layouts are rejected.
The historical benchmark's FP32 TAAT truth-generation option now fails with
instructions to use the shared float64 converter. Historical runners using old
metadata or truth contracts need adaptation; use `validate_sparse_current.py`
for the current directory layout.

For full measurements, use all available ARM workers, window 4096, ratio 1.05,
no refinement/query dropping, and separate top-10/top-100 runs. One warmup and
one measured full-query batch match the latest non-laptop protocol. Report
single-batch limitations, recall, ranked-output parity, memory and build/load
costs; exclude I/O and validation from search timing. Keep larger pruning and
shared-metadata investigations separate.
