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

#include <unordered_set>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_check.h"
#include "knowhere/index/index_factory.h"

using namespace knowhere;

// knowhere/index/index_table.h
TEST_CASE("Test index and data type check", "[IndexCheckTest]") {
    SECTION("Test valid") {
        // binary index
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_FAISS_BIN_IDMAP, VecType::VECTOR_BINARY));
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_FAISS_BIN_IVFFLAT, VecType::VECTOR_BINARY));

        // faiss index
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_FAISS_IDMAP, VecType::VECTOR_FLOAT));
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_FAISS_IDMAP, VecType::VECTOR_FLOAT16));
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_FAISS_IDMAP, VecType::VECTOR_BFLOAT16));

        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_FAISS_IVFFLAT, VecType::VECTOR_FLOAT));
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_FAISS_IVFFLAT, VecType::VECTOR_FLOAT16));
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_FAISS_IVFFLAT, VecType::VECTOR_BFLOAT16));

        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_FAISS_IVFFLAT_CC, VecType::VECTOR_FLOAT));
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_FAISS_IVFFLAT_CC, VecType::VECTOR_FLOAT16));
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_FAISS_IVFFLAT_CC, VecType::VECTOR_BFLOAT16));

        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_FAISS_IVFPQ, VecType::VECTOR_FLOAT));
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_FAISS_IVFPQ, VecType::VECTOR_FLOAT16));
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_FAISS_IVFPQ, VecType::VECTOR_BFLOAT16));

        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_FAISS_SCANN, VecType::VECTOR_FLOAT));
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_FAISS_SCANN, VecType::VECTOR_FLOAT16));
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_FAISS_SCANN, VecType::VECTOR_BFLOAT16));

        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_FAISS_IVFSQ8, VecType::VECTOR_FLOAT));
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_FAISS_IVFSQ8, VecType::VECTOR_FLOAT16));
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_FAISS_IVFSQ8, VecType::VECTOR_BFLOAT16));

        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_FAISS_IVFSQ_CC, VecType::VECTOR_FLOAT));
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_FAISS_IVFSQ_CC, VecType::VECTOR_FLOAT16));
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_FAISS_IVFSQ_CC, VecType::VECTOR_BFLOAT16));

        // gpu index
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_GPU_BRUTEFORCE, VecType::VECTOR_FLOAT));
        CHECK_FALSE(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_GPU_BRUTEFORCE, VecType::VECTOR_FLOAT16));
        CHECK_FALSE(
            KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_GPU_BRUTEFORCE, VecType::VECTOR_BFLOAT16));

        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_GPU_IVFFLAT, VecType::VECTOR_FLOAT));
        CHECK_FALSE(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_GPU_IVFFLAT, VecType::VECTOR_FLOAT16));
        CHECK_FALSE(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_GPU_IVFFLAT, VecType::VECTOR_BFLOAT16));

        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_GPU_IVFPQ, VecType::VECTOR_FLOAT));
        CHECK_FALSE(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_GPU_IVFPQ, VecType::VECTOR_FLOAT16));
        CHECK_FALSE(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_GPU_IVFPQ, VecType::VECTOR_BFLOAT16));

        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_GPU_CAGRA, VecType::VECTOR_FLOAT));
        CHECK_FALSE(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_GPU_CAGRA, VecType::VECTOR_FLOAT16));
        CHECK_FALSE(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_GPU_CAGRA, VecType::VECTOR_BFLOAT16));

        // HNSW
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_HNSW, VecType::VECTOR_FLOAT));
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_HNSW, VecType::VECTOR_FLOAT16));
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_HNSW, VecType::VECTOR_BFLOAT16));

#ifdef KNOWHERE_WITH_CARDINAL
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_HNSW, VecType::VECTOR_BINARY));
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_HNSW, VecType::VECTOR_SPARSE_FLOAT));
#else
        CHECK_FALSE(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_HNSW, VecType::VECTOR_BINARY));
        CHECK_FALSE(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_HNSW, VecType::VECTOR_SPARSE_FLOAT));
#endif

        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_HNSW_SQ8, VecType::VECTOR_FLOAT));
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_HNSW_SQ8, VecType::VECTOR_FLOAT16));
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_HNSW_SQ8, VecType::VECTOR_BFLOAT16));

        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_HNSW_SQ8_REFINE, VecType::VECTOR_FLOAT));
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_HNSW_SQ8_REFINE, VecType::VECTOR_FLOAT16));
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_HNSW_SQ8_REFINE, VecType::VECTOR_BFLOAT16));

        // faiss hnsw
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_FAISS_HNSW_FLAT, VecType::VECTOR_FLOAT));
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_FAISS_HNSW_FLAT, VecType::VECTOR_FLOAT16));
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_FAISS_HNSW_FLAT, VecType::VECTOR_BFLOAT16));

        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_FAISS_HNSW_SQ, VecType::VECTOR_FLOAT));
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_FAISS_HNSW_SQ, VecType::VECTOR_FLOAT16));
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_FAISS_HNSW_SQ, VecType::VECTOR_BFLOAT16));

        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_FAISS_HNSW_PQ, VecType::VECTOR_FLOAT));
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_FAISS_HNSW_PQ, VecType::VECTOR_FLOAT16));
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_FAISS_HNSW_PQ, VecType::VECTOR_BFLOAT16));

        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_FAISS_HNSW_PRQ, VecType::VECTOR_FLOAT));
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_FAISS_HNSW_PRQ, VecType::VECTOR_FLOAT16));
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_FAISS_HNSW_PRQ, VecType::VECTOR_BFLOAT16));

        // diskann
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_DISKANN, VecType::VECTOR_FLOAT));
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_DISKANN, VecType::VECTOR_FLOAT16));
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_DISKANN, VecType::VECTOR_BFLOAT16));

        // sparse index
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_SPARSE_INVERTED_INDEX,
                                                       VecType::VECTOR_SPARSE_FLOAT));
        CHECK(KnowhereCheck::IndexTypeAndDataTypeCheck(IndexEnum::INDEX_SPARSE_WAND, VecType::VECTOR_SPARSE_FLOAT));
    }
}

TEST_CASE("Test support mmap index", "[IndexCheckTest]") {
    SECTION("Test valid") {
        CHECK(KnowhereCheck::SupportMmapIndexTypeCheck(IndexEnum::INDEX_FAISS_BIN_IDMAP));
        CHECK(KnowhereCheck::SupportMmapIndexTypeCheck(IndexEnum::INDEX_FAISS_BIN_IVFFLAT));

        // faiss index
        CHECK(KnowhereCheck::SupportMmapIndexTypeCheck(IndexEnum::INDEX_FAISS_IDMAP));
        CHECK(KnowhereCheck::SupportMmapIndexTypeCheck(IndexEnum::INDEX_FAISS_IVFFLAT));
        CHECK(KnowhereCheck::SupportMmapIndexTypeCheck(IndexEnum::INDEX_FAISS_IVFPQ));
        CHECK(KnowhereCheck::SupportMmapIndexTypeCheck(IndexEnum::INDEX_FAISS_SCANN));
        CHECK(KnowhereCheck::SupportMmapIndexTypeCheck(IndexEnum::INDEX_FAISS_IVFSQ8));
        CHECK(KnowhereCheck::SupportMmapIndexTypeCheck(IndexEnum::INDEX_FAISS_IVFSQ_CC));

        // hnsw
        CHECK(KnowhereCheck::SupportMmapIndexTypeCheck(IndexEnum::INDEX_HNSW));
        CHECK(KnowhereCheck::SupportMmapIndexTypeCheck(IndexEnum::INDEX_HNSW_SQ8));
        CHECK(KnowhereCheck::SupportMmapIndexTypeCheck(IndexEnum::INDEX_HNSW_SQ8_REFINE));

        // faiss hnsw
        CHECK(KnowhereCheck::SupportMmapIndexTypeCheck(IndexEnum::INDEX_FAISS_HNSW_FLAT));
        CHECK(KnowhereCheck::SupportMmapIndexTypeCheck(IndexEnum::INDEX_FAISS_HNSW_SQ));
        CHECK(KnowhereCheck::SupportMmapIndexTypeCheck(IndexEnum::INDEX_FAISS_HNSW_PQ));
        CHECK(KnowhereCheck::SupportMmapIndexTypeCheck(IndexEnum::INDEX_FAISS_HNSW_PRQ));

        // sparse index
        CHECK(KnowhereCheck::SupportMmapIndexTypeCheck(IndexEnum::INDEX_SPARSE_INVERTED_INDEX));
        CHECK(KnowhereCheck::SupportMmapIndexTypeCheck(IndexEnum::INDEX_SPARSE_WAND));

#ifdef KNOWHERE_WITH_CARDINAL
        CHECK(KnowhereCheck::SupportMmapIndexTypeCheck(IndexEnum::INDEX_DISKANN));
#else
        CHECK_FALSE(KnowhereCheck::SupportMmapIndexTypeCheck(IndexEnum::INDEX_DISKANN));
#endif
    }
}

TEST_CASE("Test index has raw data", "[IndexHasRawData]") {
    SECTION("Normal test") {
        auto ver = Version::GetCurrentVersion().VersionNumber();

        // binary index
        CHECK(KnowhereCheck::IndexHasRawData<bin1>(IndexEnum::INDEX_FAISS_BIN_IDMAP, metric::HAMMING, ver, {}));
        CHECK(KnowhereCheck::IndexHasRawData<bin1>(IndexEnum::INDEX_FAISS_BIN_IDMAP, metric::JACCARD, ver, {}));

        CHECK(KnowhereCheck::IndexHasRawData<bin1>(IndexEnum::INDEX_FAISS_BIN_IVFFLAT, metric::HAMMING, ver, {}));
        CHECK(KnowhereCheck::IndexHasRawData<bin1>(IndexEnum::INDEX_FAISS_BIN_IVFFLAT, metric::JACCARD, ver, {}));

        // faiss index
        CHECK(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_IDMAP, metric::L2, ver, {}));
        CHECK(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_IDMAP, metric::IP, ver, {}));
        CHECK(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_IDMAP, metric::COSINE, ver, {}));

        CHECK(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_IVFFLAT, metric::L2, ver, {}));
        CHECK(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_IVFFLAT, metric::IP, ver, {}));
        CHECK(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_IVFFLAT, metric::COSINE, ver, {}));

        CHECK(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_IVFFLAT_CC, metric::L2, ver, {}));
        CHECK(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_IVFFLAT_CC, metric::IP, ver, {}));
        CHECK(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_IVFFLAT_CC, metric::COSINE, ver, {}));

        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_IVFPQ, metric::L2, ver, {}));
        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_IVFPQ, metric::IP, ver, {}));
        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_IVFPQ, metric::COSINE, ver, {}));

        CHECK(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_SCANN, metric::L2, ver, {}));
        CHECK(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_SCANN, metric::IP, ver, {}));
        CHECK(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_SCANN, metric::COSINE, ver, {}));

        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_IVFSQ8, metric::L2, ver, {}));
        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_IVFSQ8, metric::IP, ver, {}));
        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_IVFSQ8, metric::COSINE, ver, {}));

        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_IVFSQ_CC, metric::L2, ver, {}));
        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_IVFSQ_CC, metric::IP, ver, {}));
        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_IVFSQ_CC, metric::COSINE, ver, {}));

        // HNSW
#ifndef KNOWHERE_WITH_CARDINAL
        CHECK(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_HNSW, metric::L2, ver, {}));
        CHECK(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_HNSW, metric::IP, ver, {}));
        CHECK(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_HNSW, metric::COSINE, ver, {}));
#endif

        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_HNSW_SQ8, metric::L2, ver, {}));
        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_HNSW_SQ8, metric::IP, ver, {}));
        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_HNSW_SQ8, metric::COSINE, ver, {}));

        CHECK(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_HNSW_SQ8_REFINE, metric::L2, ver, {}));
        CHECK(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_HNSW_SQ8_REFINE, metric::IP, ver, {}));
        CHECK(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_HNSW_SQ8_REFINE, metric::COSINE, ver, {}));

        // faiss HNSW
        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_HNSW_FLAT, metric::L2, ver, {}));
        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_HNSW_FLAT, metric::IP, ver, {}));
        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_HNSW_FLAT, metric::COSINE, ver, {}));

        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_HNSW_SQ, metric::L2, ver, {}));
        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_HNSW_SQ, metric::IP, ver, {}));
        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_HNSW_SQ, metric::COSINE, ver, {}));

        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_HNSW_PQ, metric::L2, ver, {}));
        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_HNSW_PQ, metric::IP, ver, {}));
        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_HNSW_PQ, metric::COSINE, ver, {}));

        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_HNSW_PRQ, metric::L2, ver, {}));
        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_HNSW_PRQ, metric::IP, ver, {}));
        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_HNSW_PRQ, metric::COSINE, ver, {}));

        // diskann
#ifndef KNOWHERE_WITH_CARDINAL
        CHECK(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_DISKANN, metric::L2, ver, {}));
        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_DISKANN, metric::IP, ver, {}));
        CHECK(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_DISKANN, metric::COSINE, ver, {}));
#endif
        // gpu index
        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_GPU_BRUTEFORCE, metric::L2, ver, {}));
        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_GPU_BRUTEFORCE, metric::IP, ver, {}));
        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_GPU_BRUTEFORCE, metric::COSINE, ver, {}));

        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_GPU_IVFFLAT, metric::L2, ver, {}));
        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_GPU_IVFFLAT, metric::IP, ver, {}));
        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_GPU_IVFFLAT, metric::COSINE, ver, {}));

        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_GPU_IVFPQ, metric::L2, ver, {}));
        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_GPU_IVFPQ, metric::IP, ver, {}));
        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_GPU_IVFPQ, metric::COSINE, ver, {}));

        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_GPU_CAGRA, metric::L2, ver, {}));
        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_GPU_CAGRA, metric::IP, ver, {}));
        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_GPU_CAGRA, metric::COSINE, ver, {}));

        CHECK(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_SPARSE_INVERTED_INDEX, metric::IP, ver, {}));
        CHECK(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_SPARSE_WAND, metric::IP, ver, {}));
    }

    SECTION("Special test") {
        auto min_ver = Version::GetMinimalVersion().VersionNumber();
        auto ver = Version::GetCurrentVersion().VersionNumber();

        knowhere::Json json = {
            {indexparam::WITH_RAW_DATA, true},
        };

        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_IDMAP, metric::COSINE, min_ver, {}));

        CHECK(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_SCANN, metric::L2, ver, json));

        json[indexparam::WITH_RAW_DATA] = false;
        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_SCANN, metric::L2, ver, json));
        json[indexparam::WITH_RAW_DATA] = "true";
        CHECK(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_SCANN, metric::L2, ver, json));
        json[indexparam::WITH_RAW_DATA] = "false";
        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_SCANN, metric::L2, ver, json));
        json[indexparam::WITH_RAW_DATA] = 1;
        CHECK_FALSE(KnowhereCheck::IndexHasRawData<fp32>(IndexEnum::INDEX_FAISS_SCANN, metric::L2, ver, json));
    }
}
