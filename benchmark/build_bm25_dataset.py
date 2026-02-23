#!/ usr / bin / env python3
"""Build BM25 sparse vector dataset from text using bm25s library.

Creates CSR-format base/query vectors and ground truth for benchmarking.

Usage:
    python3 build_bm25_dataset.py --collection collection.tsv --queries queries.tsv \
        --output-dir ~/data/bm25 --n-docs 100000 --k 10
"""

import argparse
import struct
import time
import numpy as np
from pathlib import Path


def load_tsv_collection(path, n_docs=None):
    """Load MSMARCO-format collection.tsv (id<tab>text per line)."""
    docs = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if n_docs and i >= n_docs:
                break
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                docs.append(parts[1])
            else:
                docs.append("")
    return docs


def load_tsv_queries(path, n_queries=None):
    """Load MSMARCO-format queries.tsv (id<tab>text per line).
    Returns (query_ids, query_texts)."""
    qids = []
    texts = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if n_queries and i >= n_queries:
                break
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                qids.append(int(parts[0]))
                texts.append(parts[1])
    return qids, texts


def save_csr(path, n_rows, n_cols, indptr, indices, data):
    """Save in CSR binary format: int64 n_rows, n_cols, nnz; int64[] indptr; int32[] indices; float32[] data."""
    nnz = len(indices)
    with open(path, "wb") as f:
        f.write(struct.pack("qqq", n_rows, n_cols, nnz))
        f.write(np.array(indptr, dtype=np.int64).tobytes())
        f.write(np.array(indices, dtype=np.int32).tobytes())
        f.write(np.array(data, dtype=np.float32).tobytes())
    print(f"  Saved {path}: {n_rows} rows, {n_cols} cols, {nnz} nnz")


def save_gt(path, gt_ids, nq, k):
    """Save ground truth: int32 nq, int32 k, then nq*k int32 doc IDs."""
    with open(path, "wb") as f:
        f.write(struct.pack("ii", nq, k))
        for i in range(nq):
            row = np.full(k, -1, dtype=np.int32)
            row[:len(gt_ids[i])] = gt_ids[i][:k]
            f.write(row.tobytes())
    print(f"  Saved {path}: {nq} queries, k={k}")


def token_ids_to_csr(id_lists, n_vocab):
    """Convert list of token-ID lists to CSR format (term frequencies)."""
    indptr = [0]
    indices = []
    data = []
    for ids in id_lists:
        tf = {}
        for tid in ids:
            tf[tid] = tf.get(tid, 0) + 1
        for tid in sorted(tf.keys()):
            indices.append(tid)
            data.append(float(tf[tid]))
        indptr.append(len(indices))
    return len(id_lists), n_vocab, indptr, indices, data


def main():
    import bm25s

    parser = argparse.ArgumentParser(description="Build BM25 sparse vector dataset from text using bm25s")
    parser.add_argument("--collection", required=True, help="Path to collection.tsv (docid<tab>text)")
    parser.add_argument("--queries", required=True, help="Path to queries.tsv (qid<tab>text)")
    parser.add_argument("--output-dir", required=True, help="Output directory for CSR files and GT")
    parser.add_argument("--n-docs", type=int, default=100000, help="Number of documents to use")
    parser.add_argument("--n-queries", type=int, default=0, help="Number of queries (0=all)")
    parser.add_argument("--k", type=int, default=10, help="Top-k for ground truth")
    parser.add_argument("--bm25-k1", type=float, default=1.2, help="BM25 k1 parameter")
    parser.add_argument("--bm25-b", type=float, default=0.75, help="BM25 b parameter")
    parser.add_argument("--no-gt", action="store_true", help="Skip ground truth computation (for large datasets)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

#Load text data
    print("[Loading collection]")
    t0 = time.time()
    docs = load_tsv_collection(args.collection, args.n_docs)
    print(f"  Loaded {len(docs)} documents in {time.time()-t0:.1f}s")

    print("[Loading queries]")
    qids, query_texts = load_tsv_queries(args.queries, args.n_queries if args.n_queries > 0 else None)
    print(f"  Loaded {len(query_texts)} queries")

#Tokenize using bm25s(returns.ids and.vocab)
    print("\n[Tokenizing with bm25s]")
    t0 = time.time()
    corpus_tokenized = bm25s.tokenize(docs, stopwords="en")
    query_tokenized = bm25s.tokenize(query_texts, stopwords="en")
    print(f"  Tokenized in {time.time()-t0:.1f}s")

#bm25s.tokenize returns a Tokenized namedtuple with.ids and.vocab
    corpus_ids = corpus_tokenized.ids
    query_ids = query_tokenized.ids
    vocab = corpus_tokenized.vocab  # {token_str: int_id}
    n_vocab = len(vocab)
    print(f"  Vocabulary size: {n_vocab}")

#Save vocabulary
    vocab_path = output_dir / "vocabulary.tsv"
    with open(vocab_path, "w") as f:
        inv_vocab = {v: k for k, v in vocab.items()}
        for i in range(n_vocab):
            f.write(f"{i}\t{inv_vocab.get(i, '<UNK>')}\n")
    print(f"  Saved vocabulary to {vocab_path}")

#Remap query token IDs to corpus vocabulary
#query_tokenized uses its own vocab, we need to map to corpus vocab
    inv_query_vocab = {v: k for k, v in query_tokenized.vocab.items()}
    remapped_query_ids = []
    for qids in query_ids:
        remapped = []
        for qid in qids:
            token_str = inv_query_vocab.get(qid)
            if token_str and token_str in vocab:
                remapped.append(vocab[token_str])
        remapped_query_ids.append(remapped)
    query_ids = remapped_query_ids

#Convert to CSR format(term frequencies)
    print("\n[Building CSR vectors]")
    t0 = time.time()
    base_n, _, base_indptr, base_indices, base_data = token_ids_to_csr(corpus_ids, n_vocab)
    save_csr(output_dir / "base_small.csr", base_n, n_vocab, base_indptr, base_indices, base_data)

    query_n, _, query_indptr, query_indices, query_data = token_ids_to_csr(query_ids, n_vocab)
    save_csr(output_dir / "queries.dev.csr", query_n, n_vocab, query_indptr, query_indices, query_data)
    print(f"  CSR conversion done in {time.time()-t0:.1f}s")

#Compute avgdl
    doc_lens = []
    for i in range(base_n):
        s, e = base_indptr[i], base_indptr[i+1]
        doc_lens.append(sum(base_data[s:e]))
    avgdl = sum(doc_lens) / len(doc_lens) if doc_lens else 1.0
    print(f"  avgdl: {avgdl:.2f}")

    if not args.no_gt:
#Build BM25 index with bm25s and compute ground truth
        print("\n[Building BM25 index with bm25s]")
        t0 = time.time()
        retriever = bm25s.BM25(k1=args.bm25_k1, b=args.bm25_b)
        retriever.index(corpus_tokenized)
        print(f"  Indexed in {time.time()-t0:.1f}s")

        print("\n[Computing ground truth]")
        t0 = time.time()
        results, scores = retriever.retrieve(query_tokenized, k=args.k)
        print(f"  Retrieved in {time.time()-t0:.1f}s")

#Save ground truth
        save_gt(output_dir / "base_small.dev.bm25.gt", results, query_n, args.k)
    else:
        print("\n[Skipping ground truth computation (--no-gt)]")

#Save metadata
    meta_path = output_dir / "metadata.txt"
    with open(meta_path, "w") as f:
        f.write(f"Source: {args.collection}\n")
        f.write(f"Documents: {base_n}\n")
        f.write(f"Queries: {query_n}\n")
        f.write(f"Vocabulary: {len(vocab)}\n")
        f.write(f"Avg doc length: {avgdl:.2f}\n")
        f.write(f"BM25 k1: {args.bm25_k1}\n")
        f.write(f"BM25 b: {args.bm25_b}\n")
        f.write(f"Top-k: {args.k}\n")
    print(f"  Saved metadata to {meta_path}")

    print(f"\nDone! Output in {output_dir}")
    print(f"  Run benchmark with: ./benchmark_sparse_algo --data-dir {output_dir} --data-type bm25")


if __name__ == "__main__":
    main()
