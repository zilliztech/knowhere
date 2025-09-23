// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#ifndef INDEX_TABLE_H
#define INDEX_TABLE_H
#include <set>
#include <string>

#include "knowhere/comp/index_param.h"
#include "knowhere/index/index_factory.h"
namespace knowhere {
static std::set<std::pair<std::string, VecType>> legal_knowhere_index = {
    // binary ivf
    {IndexEnum::INDEX_FAISS_BIN_IDMAP, VecType::VECTOR_BINARY},
    {IndexEnum::INDEX_FAISS_BIN_IVFFLAT, VecType::VECTOR_BINARY},

    // faiss index
    {IndexEnum::INDEX_FAISS_IDMAP, VecType::VECTOR_FLOAT},
    {IndexEnum::INDEX_FAISS_IDMAP, VecType::VECTOR_FLOAT16},
    {IndexEnum::INDEX_FAISS_IDMAP, VecType::VECTOR_BFLOAT16},
    // {IndexEnum::INDEX_FAISS_IDMAP, VecType::VECTOR_INT8},

    {IndexEnum::INDEX_FAISS_IVFFLAT, VecType::VECTOR_FLOAT},
    {IndexEnum::INDEX_FAISS_IVFFLAT, VecType::VECTOR_FLOAT16},
    {IndexEnum::INDEX_FAISS_IVFFLAT, VecType::VECTOR_BFLOAT16},
    // {IndexEnum::INDEX_FAISS_IVFFLAT, VecType::VECTOR_INT8},

    {IndexEnum::INDEX_FAISS_IVFFLAT_CC, VecType::VECTOR_FLOAT},
    {IndexEnum::INDEX_FAISS_IVFFLAT_CC, VecType::VECTOR_FLOAT16},
    {IndexEnum::INDEX_FAISS_IVFFLAT_CC, VecType::VECTOR_BFLOAT16},
    // {IndexEnum::INDEX_FAISS_IVFFLAT_CC, VecType::VECTOR_INT8},

    {IndexEnum::INDEX_FAISS_IVFPQ, VecType::VECTOR_FLOAT},
    {IndexEnum::INDEX_FAISS_IVFPQ, VecType::VECTOR_FLOAT16},
    {IndexEnum::INDEX_FAISS_IVFPQ, VecType::VECTOR_BFLOAT16},
    // {IndexEnum::INDEX_FAISS_IVFPQ, VecType::VECTOR_INT8},

    {IndexEnum::INDEX_FAISS_SCANN, VecType::VECTOR_FLOAT},
    {IndexEnum::INDEX_FAISS_SCANN, VecType::VECTOR_FLOAT16},
    {IndexEnum::INDEX_FAISS_SCANN, VecType::VECTOR_BFLOAT16},
    // {IndexEnum::INDEX_FAISS_SCANN, VecType::VECTOR_INT8},

    {IndexEnum::INDEX_FAISS_IVFSQ8, VecType::VECTOR_FLOAT},
    {IndexEnum::INDEX_FAISS_IVFSQ8, VecType::VECTOR_FLOAT16},
    {IndexEnum::INDEX_FAISS_IVFSQ8, VecType::VECTOR_BFLOAT16},
    // {IndexEnum::INDEX_FAISS_IVFSQ8, VecType::VECTOR_INT8},

    {IndexEnum::INDEX_FAISS_IVFSQ_CC, VecType::VECTOR_FLOAT},
    {IndexEnum::INDEX_FAISS_IVFSQ_CC, VecType::VECTOR_FLOAT16},
    {IndexEnum::INDEX_FAISS_IVFSQ_CC, VecType::VECTOR_BFLOAT16},
    // {IndexEnum::INDEX_FAISS_IVFSQ_CC, VecType::VECTOR_INT8},

    {IndexEnum::INDEX_FAISS_IVFRABITQ, VecType::VECTOR_FLOAT},
    {IndexEnum::INDEX_FAISS_IVFRABITQ, VecType::VECTOR_FLOAT16},
    {IndexEnum::INDEX_FAISS_IVFRABITQ, VecType::VECTOR_BFLOAT16},

    // gpu index
    {IndexEnum::INDEX_GPU_BRUTEFORCE, VecType::VECTOR_FLOAT},
    {IndexEnum::INDEX_GPU_BRUTEFORCE, VecType::VECTOR_FLOAT16},
    {IndexEnum::INDEX_GPU_IVFFLAT, VecType::VECTOR_FLOAT},
    {IndexEnum::INDEX_GPU_IVFFLAT, VecType::VECTOR_FLOAT16},
    {IndexEnum::INDEX_GPU_IVFFLAT, VecType::VECTOR_INT8},
    {IndexEnum::INDEX_GPU_IVFPQ, VecType::VECTOR_FLOAT},
    {IndexEnum::INDEX_GPU_IVFPQ, VecType::VECTOR_FLOAT16},
    {IndexEnum::INDEX_GPU_IVFPQ, VecType::VECTOR_INT8},
    {IndexEnum::INDEX_GPU_CAGRA, VecType::VECTOR_FLOAT},
    {IndexEnum::INDEX_GPU_CAGRA, VecType::VECTOR_FLOAT16},
    {IndexEnum::INDEX_GPU_CAGRA, VecType::VECTOR_INT8},
    {IndexEnum::INDEX_GPU_CAGRA, VecType::VECTOR_BINARY},

    // hnsw
    {IndexEnum::INDEX_HNSW, VecType::VECTOR_FLOAT},
    {IndexEnum::INDEX_HNSW, VecType::VECTOR_FLOAT16},
    {IndexEnum::INDEX_HNSW, VecType::VECTOR_BFLOAT16},
    {IndexEnum::INDEX_HNSW, VecType::VECTOR_INT8},
    {IndexEnum::INDEX_HNSW, VecType::VECTOR_BINARY},

    {IndexEnum::INDEX_HNSW_SQ, VecType::VECTOR_FLOAT},
    {IndexEnum::INDEX_HNSW_SQ, VecType::VECTOR_FLOAT16},
    {IndexEnum::INDEX_HNSW_SQ, VecType::VECTOR_BFLOAT16},
    {IndexEnum::INDEX_HNSW_SQ, VecType::VECTOR_INT8},

    {IndexEnum::INDEX_HNSW_PQ, VecType::VECTOR_FLOAT},
    {IndexEnum::INDEX_HNSW_PQ, VecType::VECTOR_FLOAT16},
    {IndexEnum::INDEX_HNSW_PQ, VecType::VECTOR_BFLOAT16},
    {IndexEnum::INDEX_HNSW_PQ, VecType::VECTOR_INT8},

    {IndexEnum::INDEX_HNSW_PRQ, VecType::VECTOR_FLOAT},
    {IndexEnum::INDEX_HNSW_PRQ, VecType::VECTOR_FLOAT16},
    {IndexEnum::INDEX_HNSW_PRQ, VecType::VECTOR_BFLOAT16},
    {IndexEnum::INDEX_HNSW_PRQ, VecType::VECTOR_INT8},

    // diskann
    {IndexEnum::INDEX_DISKANN, VecType::VECTOR_FLOAT},
    {IndexEnum::INDEX_DISKANN, VecType::VECTOR_FLOAT16},
    {IndexEnum::INDEX_DISKANN, VecType::VECTOR_BFLOAT16},

    // aisaq
    {IndexEnum::INDEX_AISAQ, VecType::VECTOR_FLOAT},
    {IndexEnum::INDEX_AISAQ, VecType::VECTOR_FLOAT16},
    {IndexEnum::INDEX_AISAQ, VecType::VECTOR_BFLOAT16},

    // sparse index
    {IndexEnum::INDEX_SPARSE_INVERTED_INDEX, VecType::VECTOR_SPARSE_FLOAT},
    {IndexEnum::INDEX_SPARSE_WAND, VecType::VECTOR_SPARSE_FLOAT},
    //  minhash index
    {IndexEnum::INDEX_MINHASH_LSH, VecType::VECTOR_BINARY},
};

static std::set<std::string> legal_support_mmap_knowhere_index = {
    // binary ivf
    IndexEnum::INDEX_FAISS_BIN_IDMAP,
    IndexEnum::INDEX_FAISS_BIN_IVFFLAT,

    // faiss index
    IndexEnum::INDEX_FAISS_IDMAP,
    IndexEnum::INDEX_FAISS_IVFFLAT,
    IndexEnum::INDEX_FAISS_IVFPQ,
    IndexEnum::INDEX_FAISS_SCANN,
    IndexEnum::INDEX_FAISS_IVFSQ8,
    IndexEnum::INDEX_FAISS_IVFSQ_CC,
    IndexEnum::INDEX_FAISS_IVFRABITQ,

    // hnsw
    IndexEnum::INDEX_HNSW,
    IndexEnum::INDEX_HNSW_SQ,
    IndexEnum::INDEX_HNSW_PQ,
    IndexEnum::INDEX_HNSW_PRQ,

    // sparse index
    IndexEnum::INDEX_SPARSE_INVERTED_INDEX,
    IndexEnum::INDEX_SPARSE_WAND,
};

static std::set<std::string> legal_support_emb_list_knowhere_index = {
    IndexEnum::INDEX_HNSW,
    IndexEnum::INDEX_HNSW_SQ,
    IndexEnum::INDEX_HNSW_PQ,
    IndexEnum::INDEX_HNSW_PRQ,
};

KNOWHERE_SET_STATIC_GLOBAL_INDEX_TABLE(0, KNOWHERE_STATIC_INDEX, legal_knowhere_index)
KNOWHERE_SET_STATIC_GLOBAL_INDEX_TABLE(1, KNOWHERE_SUPPORT_MMAP_INDEX, legal_support_mmap_knowhere_index)
KNOWHERE_SET_STATIC_GLOBAL_INDEX_TABLE(2, KNOWHERE_SUPPORT_EMB_LIST_INDEX, legal_support_emb_list_knowhere_index)

}  // namespace knowhere
#endif /* INDEX_TABLE_H */
