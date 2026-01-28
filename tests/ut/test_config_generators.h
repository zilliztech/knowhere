// Copyright (C) 2019-2024 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#pragma once

#include <functional>
#include <string>
#include <vector>

#include "knowhere/comp/index_param.h"
#include "knowhere/dataset.h"
#include "knowhere/index/index_factory.h"

namespace knowhere::test {

/**
 * @brief Centralized configuration generators for index testing
 *
 * This eliminates the need to define similar config generators in multiple test files.
 * Usage:
 *   ConfigGenerator gen(dim, metric, topk);
 *   auto flat_config = gen.Flat();
 *   auto hnsw_config = gen.HNSW();
 */
class ConfigGenerator {
 public:
    ConfigGenerator(int64_t dim, const std::string& metric, int64_t topk) : dim_(dim), metric_(metric), topk_(topk) {
    }

    // Set optional range search parameters
    ConfigGenerator&
    WithRange(float radius, float range_filter) {
        radius_ = radius;
        range_filter_ = range_filter;
        has_range_ = true;
        return *this;
    }

    // ==================== Base Configurations ====================

    Json
    Base() const {
        Json json;
        json[meta::DIM] = dim_;
        json[meta::METRIC_TYPE] = metric_;
        json[meta::TOPK] = topk_;
        if (has_range_) {
            json[meta::RADIUS] = radius_;
            json[meta::RANGE_FILTER] = range_filter_;
        }
        return json;
    }

    // ==================== Flat Index ====================

    Json
    Flat() const {
        return Base();
    }

    // ==================== IVF Indexes ====================

    Json
    IVFFlat(int64_t nlist = 16, int64_t nprobe = 8) const {
        Json json = Base();
        json[indexparam::NLIST] = nlist;
        json[indexparam::NPROBE] = nprobe;
        return json;
    }

    Json
    IVFFlatCC(int64_t nlist = 16, int64_t nprobe = 8, int64_t ssize = 48) const {
        Json json = IVFFlat(nlist, nprobe);
        json[indexparam::SSIZE] = ssize;
        return json;
    }

    Json
    IVFSQ(int64_t nlist = 16, int64_t nprobe = 8) const {
        return IVFFlat(nlist, nprobe);
    }

    Json
    IVFSQCC(int64_t nlist = 16, int64_t nprobe = 8, int64_t ssize = 48, int64_t code_size = 8) const {
        Json json = IVFFlatCC(nlist, nprobe, ssize);
        json[indexparam::CODE_SIZE] = code_size;
        return json;
    }

    Json
    IVFPQ(int64_t nlist = 16, int64_t nprobe = 8, int64_t m = 4, int64_t nbits = 8) const {
        Json json = IVFFlat(nlist, nprobe);
        json[indexparam::M] = m;
        json[indexparam::NBITS] = nbits;
        return json;
    }

    Json
    IVFRaBitQ(int64_t nlist = 16, int64_t nprobe = 8, uint8_t bits_query = 0) const {
        Json json = IVFFlat(nlist, nprobe);
        json[indexparam::RABITQ_QUERY_BITS] = bits_query;
        return json;
    }

    Json
    IVFRaBitQRefine(int64_t nlist = 16, int64_t nprobe = 8, const std::string& refine_type = "FLAT") const {
        Json json = IVFRaBitQ(nlist, nprobe);
        json["refine"] = true;
        json["refine_type"] = refine_type;
        return json;
    }

    // ==================== SCANN ====================

    Json
    SCANN(int64_t nlist = 16, int64_t nprobe = 14, int64_t reorder_k = 200, bool with_raw_data = true) const {
        Json json = IVFFlat(nlist, nprobe);
        json[indexparam::REORDER_K] = reorder_k;
        json[indexparam::WITH_RAW_DATA] = with_raw_data;
        return json;
    }

    // ==================== HNSW Family ====================

    Json
    HNSW(int64_t m = 32, int64_t ef_construction = 120, int64_t ef = 120) const {
        Json json = Base();
        json[indexparam::HNSW_M] = m;
        json[indexparam::EFCONSTRUCTION] = ef_construction;
        json[indexparam::EF] = ef;
        return json;
    }

    Json
    HNSWSQ(int64_t m = 32, int64_t ef_construction = 120, int64_t ef = 120, const std::string& sq_type = "SQ8") const {
        Json json = HNSW(m, ef_construction, ef);
        json[indexparam::SQ_TYPE] = sq_type;
        return json;
    }

    Json
    HNSWSQRefine(int64_t m = 32, int64_t ef_construction = 120, int64_t ef = 120, const std::string& sq_type = "SQ8",
                 const std::string& refine_type = "FLAT") const {
        Json json = HNSWSQ(m, ef_construction, ef, sq_type);
        json["refine"] = true;
        json["refine_type"] = refine_type;
        return json;
    }

    Json
    HNSWPQ(int64_t m_hnsw = 32, int64_t ef_construction = 120, int64_t ef = 120, int64_t m_pq = 16,
           int64_t nbits = 8) const {
        Json json = HNSW(m_hnsw, ef_construction, ef);
        json[indexparam::M] = m_pq;
        json[indexparam::NBITS] = nbits;
        return json;
    }

    // ==================== Sparse Index ====================

    Json
    SparseInvertedIndex(float drop_ratio_search = 0.0, const std::string& algo = "DAAT_MAXSCORE") const {
        Json json = Base();
        json[indexparam::DROP_RATIO_SEARCH] = drop_ratio_search;
        json[indexparam::INVERTED_INDEX_ALGO] = algo;
        return json;
    }

    Json
    SparseWAND(float drop_ratio_search = 0.0) const {
        return SparseInvertedIndex(drop_ratio_search, "DAAT_WAND");
    }

    Json
    SparseBM25(float k1 = 1.2, float b = 0.75, float avgdl = 100) const {
        Json json = Base();
        json[meta::BM25_K1] = k1;
        json[meta::BM25_B] = b;
        json[meta::BM25_AVGDL] = avgdl;
        return json;
    }

    // ==================== DiskANN ====================

    Json
    DiskANN(int64_t max_degree = 56, int64_t search_list_size = 100, float pq_code_budget_gb = 0.012,
            float build_dram_budget_gb = 4.38) const {
        Json json = Base();
        json[indexparam::MAX_DEGREE] = max_degree;
        json[indexparam::SEARCH_LIST_SIZE] = search_list_size;
        json[indexparam::PQ_CODE_BUDGET_GB] = pq_code_budget_gb;
        json[indexparam::BUILD_DRAM_BUDGET_GB] = build_dram_budget_gb;
        return json;
    }

    // ==================== Binary Index ====================

    Json
    BinaryFlat() const {
        return Base();
    }

    Json
    BinaryIVFFlat(int64_t nlist = 16, int64_t nprobe = 8) const {
        return IVFFlat(nlist, nprobe);
    }

 private:
    int64_t dim_;
    std::string metric_;
    int64_t topk_;
    float radius_ = 0.0f;
    float range_filter_ = 0.0f;
    bool has_range_ = false;
};

/**
 * @brief Get default range parameters for a given metric type
 */
inline std::pair<float, float>
GetDefaultRangeParams(const std::string& metric) {
    if (metric == metric::L2) {
        return {200.0f, 0.0f};  // radius, range_filter
    } else if (metric == metric::COSINE) {
        return {0.99f, 1.01f};
    } else if (metric == metric::IP) {
        return {0.0f, 1000.0f};
    } else if (metric == metric::HAMMING) {
        return {10.0f, 0.0f};
    } else if (metric == metric::JACCARD) {
        return {0.1f, 0.0f};
    }
    return {0.0f, 0.0f};
}

/**
 * @brief Common index-config pairs for testing (reduces GENERATE boilerplate)
 */
using IndexConfigPair = std::tuple<std::string, std::function<Json()>>;

inline std::vector<IndexConfigPair>
GetCommonFloatIndexConfigs(const ConfigGenerator& gen) {
    return {
        {IndexEnum::INDEX_FAISS_IDMAP, [&]() { return gen.Flat(); }},
        {IndexEnum::INDEX_FAISS_IVFFLAT, [&]() { return gen.IVFFlat(); }},
        {IndexEnum::INDEX_FAISS_IVFSQ8, [&]() { return gen.IVFSQ(); }},
        {IndexEnum::INDEX_HNSW, [&]() { return gen.HNSW(); }},
    };
}

inline std::vector<IndexConfigPair>
GetCommonBinaryIndexConfigs(const ConfigGenerator& gen) {
    return {
        {IndexEnum::INDEX_FAISS_BIN_IDMAP, [&]() { return gen.BinaryFlat(); }},
        {IndexEnum::INDEX_FAISS_BIN_IVFFLAT, [&]() { return gen.BinaryIVFFlat(); }},
    };
}

}  // namespace knowhere::test
