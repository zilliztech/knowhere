# Native e-means IVF training

Set `e_means_epochs` in the build configuration to select the native CPU
e-means coarse trainer:

```json
{
  "metric_type": "L2",
  "nlist": 1024,
  "e_means_epochs": 2.5
}
```

Zero (the default) preserves the existing trainer. A positive value takes
precedence over `use_super_kmeans`. CPU numeric IVF variants share this option;
L2 and COSINE are supported. It does not select a GPU trainer or change PQ
codebook training. Binary IVF and IP do not accept positive e-means budgets.

The trainer implements [e-means Algorithm 1](https://arxiv.org/abs/2611.09501)
using the automatic recipes shipped in
[FAISS 1.16.0](https://github.com/facebookresearch/faiss/tree/v1.16.0/contrib/e_means).
It preserves remainder batches, cosine step annealing, weighted donor revival
and final-pass closing means. At dimension <= 16 the quality batch law is used;
at dimension <= 2 revival is disabled. Fractional passes precede a complete
closing pass for budgets >= 1. Budgets below 1 close over that partial pass.

Coordinates and assignment use FP32; accumulation uses FP64. The existing
`max_points_per_centroid`, seed and redo settings apply. Final objective
evaluation adds one assignment pass per redo. Native random permutations are
not bit-identical to NumPy. The implementation is CPU-only and has no Python
dependency. No claim is made that paper-reported GPU speedups apply here.

Centroids use the existing IVF serialization and search paths. The option is
only needed by the builder, and does not change the stored index format.
