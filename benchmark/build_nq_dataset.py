#!/usr/bin/env python3
"""Download NQ (Natural Questions) from BEIR and build BM25 CSR vectors.

Usage:
    python3 build_nq_dataset.py --output-dir ~/data/nq_bm25
"""

import argparse
import csv
import json
import os
import struct
import time
import zipfile
import numpy as np
from pathlib import Path
from urllib.request import urlretrieve


def save_csr(path, n_rows, n_cols, indptr, indices, data):
    nnz = len(indices)
    with open(path, "wb") as f:
        f.write(struct.pack("qqq", n_rows, n_cols, nnz))
        f.write(np.array(indptr, dtype=np.int64).tobytes())
        f.write(np.array(indices, dtype=np.int32).tobytes())
        f.write(np.array(data, dtype=np.float32).tobytes())
    print(f"  Saved {path}: {n_rows} rows, {n_cols} cols, {nnz} nnz")


def token_ids_to_csr(id_lists, n_vocab):
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

    parser = argparse.ArgumentParser(description="Build NQ BM25 dataset from BEIR")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download NQ from BEIR via direct URL
    beir_url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip"
    zip_path = output_dir / "nq.zip"
    extract_dir = output_dir / "nq_raw"

    if not (extract_dir / "corpus.jsonl").exists():
        print("[Downloading NQ from BEIR]")
        t0 = time.time()
        urlretrieve(beir_url, zip_path)
        print(f"  Downloaded in {time.time()-t0:.1f}s")

        print("[Extracting]")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)
        # BEIR extracts to nq/ subdirectory
        beir_dir = output_dir / "nq"
        if beir_dir.exists():
            beir_dir.rename(extract_dir)
        os.remove(zip_path)
    else:
        print("[NQ data already extracted]")

    # Load corpus
    print("[Loading corpus]")
    t0 = time.time()
    docs = []
    with open(extract_dir / "corpus.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            title = obj.get("title", "")
            text = obj.get("text", "")
            docs.append(f"{title} {text}" if title else text)
    print(f"  {len(docs)} documents in {time.time()-t0:.1f}s")

    # Load queries
    print("[Loading queries]")
    questions = []
    with open(extract_dir / "queries.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            questions.append(obj["text"])
    print(f"  {len(questions)} queries")

    # Tokenize
    print("\n[Tokenizing with bm25s]")
    t0 = time.time()
    corpus_tokenized = bm25s.tokenize(docs, stopwords="en")
    query_tokenized = bm25s.tokenize(questions, stopwords="en")
    print(f"  Tokenized in {time.time()-t0:.1f}s")

    corpus_ids = corpus_tokenized.ids
    query_ids = query_tokenized.ids
    vocab = corpus_tokenized.vocab
    n_vocab = len(vocab)
    print(f"  Vocabulary size: {n_vocab}")

    # Remap query tokens to corpus vocabulary
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

    # Convert to CSR
    print("\n[Building CSR vectors]")
    t0 = time.time()
    base_n, _, base_indptr, base_indices, base_data = token_ids_to_csr(corpus_ids, n_vocab)
    save_csr(output_dir / "base_small.csr", base_n, n_vocab, base_indptr, base_indices, base_data)

    query_n, _, query_indptr, query_indices, query_data = token_ids_to_csr(query_ids, n_vocab)
    save_csr(output_dir / "queries.dev.csr", query_n, n_vocab, query_indptr, query_indices, query_data)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Compute avgdl
    doc_lens = []
    for i in range(base_n):
        s, e = base_indptr[i], base_indptr[i+1]
        doc_lens.append(sum(base_data[s:e]))
    avgdl = sum(doc_lens) / len(doc_lens) if doc_lens else 1.0

    # Save metadata
    meta_path = output_dir / "metadata.txt"
    with open(meta_path, "w") as f:
        f.write(f"Source: BeIR/nq\n")
        f.write(f"Documents: {base_n}\n")
        f.write(f"Queries: {query_n}\n")
        f.write(f"Vocabulary: {n_vocab}\n")
        f.write(f"Avg doc length: {avgdl:.2f}\n")
    print(f"\n  avgdl: {avgdl:.2f}")
    print(f"  Saved metadata to {meta_path}")
    print(f"\nDone! Output in {output_dir}")


if __name__ == "__main__":
    main()
