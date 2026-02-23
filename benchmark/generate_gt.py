#!/ usr / bin / env python3
"""Generate per-metric ground truth for sparse vector benchmarks.

Computes brute-force BM25 and IP top-k results from CSR data
and saves in the binary format expected by benchmark_sparse_algo.

Uses scipy sparse matrix operations for speed.

Usage:
    python3 generate_gt.py --data-dir ~/data/msmarco_bm25 --k 10 --metric bm25
    python3 generate_gt.py --data-dir ~/data --k 10 --metric ip
"""

import argparse
import struct
import time
import numpy as np
from scipy import sparse
from pathlib import Path


def load_csr(path):
    """Load CSR format: int64 n_rows, n_cols, nnz; int64[n_rows+1] indptr; int32[nnz] indices; float32[nnz] data."""
    with open(path, "rb") as f:
        n_rows, n_cols, nnz = struct.unpack("qqq", f.read(24))
        indptr = np.frombuffer(f.read((n_rows + 1) * 8), dtype=np.int64)
        indices = np.frombuffer(f.read(nnz * 4), dtype=np.int32)
        data = np.frombuffer(f.read(nnz * 4), dtype=np.float32)
    return n_rows, n_cols, nnz, indptr, indices, data


def save_gt(path, gt_ids, nq, k):
    """Save ground truth in binary format: int32 nq, int32 k, then nq*k int32 IDs."""
    with open(path, "wb") as f:
        f.write(struct.pack("ii", nq, k))
        for i in range(nq):
            f.write(gt_ids[i].astype(np.int32).tobytes())
    print(f"  Saved {path} ({nq} queries, k={k})")


def compute_bm25_gt(base_mat, query_mat, k, bm25_k1=1.2, bm25_b=0.75):
    """Compute brute-force BM25 top-k using scipy sparse matmul.

    Uses the same formula as knowhere:
      score(q, d) = sum_t query_weight_t * tf_t * (k1 + 1) / (tf_t + k1 * (1 - b + b * dl / avgdl))
    """
    n_docs = base_mat.shape[0]
    nq = query_mat.shape[0]

#Compute document lengths(row sums)
    doc_lens = np.array(base_mat.sum(axis=1)).flatten().astype(np.float64)
    avgdl = max(np.mean(doc_lens), 1.0)
    print(f"  BM25 params: k1={bm25_k1}, b={bm25_b}, avgdl={avgdl:.2f}")

#Build BM25 - normalized base matrix:
#For each(doc_id, term_id) with value tf:
#normalized = tf * (k1 + 1) / (tf + k1 * (1 - b + b * (dl / avgdl)))
    base_csr = base_mat.tocsr()
    norm_data = base_csr.data.astype(np.float64).copy()

#Vectorized : for each nnz, find its row(doc_id) to get doc_len
#Use indptr to map nnz positions to row indices
    row_indices = np.zeros(len(norm_data), dtype=np.int64)
    for i in range(n_docs):
        start, end = base_csr.indptr[i], base_csr.indptr[i + 1]
        row_indices[start:end] = i

    dl = doc_lens[row_indices]
    tf = norm_data
    norm_data = tf * (bm25_k1 + 1) / (tf + bm25_k1 * (1 - bm25_b + bm25_b * (dl / avgdl)))

    normalized_base = sparse.csr_matrix(
        (norm_data.astype(np.float32), base_csr.indices.copy(), base_csr.indptr.copy()),
        shape=base_csr.shape
    )

#score_matrix = query @normalized_base.T(nq x n_docs)
    print(f"  Computing score matrix ({nq} x {n_docs})...")
    t0 = time.time()
    score_matrix = query_mat @ normalized_base.T
    print(f"  Matmul done in {time.time()-t0:.1f}s")

#Extract top - k from each row
    print(f"  Extracting top-{k}...")
    t0 = time.time()
    gt_ids = np.full((nq, k), -1, dtype=np.int32)
    score_dense = score_matrix.toarray()
    for qi in range(nq):
        scores = score_dense[qi]
        if k < n_docs:
            top_k_idx = np.argpartition(-scores, k)[:k]
            top_k_idx = top_k_idx[np.argsort(-scores[top_k_idx])]
        else:
            top_k_idx = np.argsort(-scores)[:k]
        gt_ids[qi, :len(top_k_idx)] = top_k_idx
    print(f"  Top-k extraction done in {time.time()-t0:.1f}s")

    return gt_ids


def compute_ip_gt(base_mat, query_mat, k):
    """Compute brute-force IP (inner product) top-k using scipy sparse matmul."""
    n_docs = base_mat.shape[0]
    nq = query_mat.shape[0]

    print(f"  Computing score matrix ({nq} x {n_docs})...")
    t0 = time.time()
    score_matrix = query_mat @ base_mat.T
    print(f"  Matmul done in {time.time()-t0:.1f}s")

    print(f"  Extracting top-{k}...")
    t0 = time.time()
    gt_ids = np.full((nq, k), -1, dtype=np.int32)
    score_dense = score_matrix.toarray()
    for qi in range(nq):
        scores = score_dense[qi]
        if k < n_docs:
            top_k_idx = np.argpartition(-scores, k)[:k]
            top_k_idx = top_k_idx[np.argsort(-scores[top_k_idx])]
        else:
            top_k_idx = np.argsort(-scores)[:k]
        gt_ids[qi, :len(top_k_idx)] = top_k_idx
    print(f"  Top-k extraction done in {time.time()-t0:.1f}s")

    return gt_ids


def main():
    parser = argparse.ArgumentParser(description="Generate per-metric ground truth for sparse benchmarks")
    parser.add_argument("--data-dir", required=True, help="Directory with base_small.csr and queries.dev.csr")
    parser.add_argument("--k", type=int, default=10, help="Top-k for ground truth")
    parser.add_argument("--metric", choices=["bm25", "ip", "both"], default="both", help="Which metric(s) to compute")
    parser.add_argument("--nq", type=int, default=0, help="Number of queries (0=all)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("[Loading base vectors]")
    base_n_rows, base_n_cols, base_nnz, base_indptr, base_indices, base_data = \
        load_csr(data_dir / "base_small.csr")
    print(f"  {base_n_rows} docs, {base_n_cols} dims, {base_nnz} nnz")

    print("[Loading queries]")
    query_n_rows, query_n_cols, query_nnz, query_indptr, query_indices, query_data = \
        load_csr(data_dir / "queries.dev.csr")
    print(f"  {query_n_rows} queries, {query_n_cols} dims, {query_nnz} nnz")

    nq = args.nq if args.nq > 0 else query_n_rows
    nq = min(nq, query_n_rows)
    if nq < query_n_rows:
        query_indptr = query_indptr[:nq + 1]
#Slice nnz data to match
        max_nnz = query_indptr[-1]
        query_indices = query_indices[:max_nnz]
        query_data = query_data[:max_nnz]
        query_n_rows = nq
    print(f"  Using {nq} queries, k={args.k}")

#Ensure both matrices have the same number of columns
    n_cols = max(base_n_cols, query_n_cols)

#Build scipy sparse matrices
    base_mat = sparse.csr_matrix(
        (base_data.astype(np.float32), base_indices.astype(np.int32), base_indptr),
        shape=(base_n_rows, n_cols)
    )
    query_mat = sparse.csr_matrix(
        (query_data.astype(np.float32), query_indices.astype(np.int32), query_indptr),
        shape=(query_n_rows, n_cols)
    )

    if args.metric in ("bm25", "both"):
        print("\n[Computing BM25 ground truth]")
        t0 = time.time()
        bm25_gt = compute_bm25_gt(base_mat, query_mat, args.k)
        print(f"  Total BM25 GT time: {time.time()-t0:.1f}s")
        save_gt(data_dir / "base_small.dev.bm25.gt", bm25_gt, nq, args.k)

    if args.metric in ("ip", "both"):
        print("\n[Computing IP ground truth]")
        t0 = time.time()
        ip_gt = compute_ip_gt(base_mat, query_mat, args.k)
        print(f"  Total IP GT time: {time.time()-t0:.1f}s")
        save_gt(data_dir / "base_small.dev.ip.gt", ip_gt, nq, args.k)

    print("\nDone!")


if __name__ == "__main__":
    main()
