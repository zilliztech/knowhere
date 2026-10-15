# DiskANN RaBitQ navigation

`DISKANN_RABITQ` uses RaBitQ for the resident navigation model and the existing
DiskANN SSD graph. This version supports FP32 with L2, inner product, or cosine.
AiSAQ, embedding lists, FP16 and BF16 do not support RaBitQ navigation.

For building, use the existing DiskANN paths and graph parameters, plus:

```json
{
  "navigation_codec": "RABITQ",
  "rbq_bits": 1,
  "disk_pq_dims": 0
}
```

`DISKANN_RABITQ` defaults to `RABITQ`. Ordinary `DISKANN` also accepts explicit
`navigation_codec: "RABITQ"`; omission at build time keeps PQ navigation.
`rbq_bits` ranges from 1 to 9, default 1. PQ navigation budgets are not required.
`disk_pq_dims` independently selects raw SSD vectors (0) or SSD-PQ (positive),
including when navigation uses RaBitQ.

Search and iterator requests accept `rbq_bits_query` from 0 to 8, default 4;
0 keeps the query unquantized. Range search uses the same iterator path.
Each request owns its setting, including when iterators share thread slots.
Scoring reuses the Faiss RaBitQ kernels used by HNSW, without threshold pruning.
One-bit codes use the existing four-code kernel; multi-bit full scoring uses
the existing scalar kernel.

The model is stored in `<prefix>_rabitq.index`. `<prefix>_navigation.meta`
identifies its codec. Ordinary `DISKANN` detects this during loading without a
codec parameter. An explicit codec is a constraint and must match the files.
`DISKANN_RABITQ` accepts only RaBitQ. Missing metadata means legacy PQ;
unknown or damaged metadata, missing model files and codec mismatches fail
without falling back. Use the metric originally used to build the SSD graph.

The runtime size includes the encoded vectors, centroid, rotation matrix and
per-thread query buffers. The rotation is a square FP32 matrix and costs
approximately `4 * prepared_dimension^2` bytes; IP adds one prepared dimension.
Before model files are read, load admission conservatively includes the full
serialized payload when the codec is unspecified or RaBitQ. Explicit PQ keeps
the existing estimate.
