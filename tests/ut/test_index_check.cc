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
        CHECK(knowhere::IndexStaticFaced<bin1>::HasRawData(IndexEnum::INDEX_FAISS_BIN_IDMAP, ver, {}));
        CHECK(knowhere::IndexStaticFaced<bin1>::HasRawData(IndexEnum::INDEX_FAISS_BIN_IVFFLAT, ver, {}));

        // faiss index
        CHECK(knowhere::IndexStaticFaced<fp32>::HasRawData(IndexEnum::INDEX_FAISS_IDMAP, ver, {}));
        CHECK(knowhere::IndexStaticFaced<fp32>::HasRawData(IndexEnum::INDEX_FAISS_IVFFLAT, ver, {}));
        CHECK(knowhere::IndexStaticFaced<fp32>::HasRawData(IndexEnum::INDEX_FAISS_IVFFLAT_CC, ver, {}));
        CHECK_FALSE(knowhere::IndexStaticFaced<fp32>::HasRawData(IndexEnum::INDEX_FAISS_IVFPQ, ver, {}));
        CHECK(knowhere::IndexStaticFaced<fp32>::HasRawData(IndexEnum::INDEX_FAISS_SCANN, ver, {}));
        CHECK_FALSE(knowhere::IndexStaticFaced<fp32>::HasRawData(IndexEnum::INDEX_FAISS_IVFSQ8, ver, {}));
        CHECK_FALSE(knowhere::IndexStaticFaced<fp32>::HasRawData(IndexEnum::INDEX_FAISS_IVFSQ_CC, ver, {}));

        // HNSW
#ifndef KNOWHERE_WITH_CARDINAL
        CHECK(knowhere::IndexStaticFaced<fp32>::HasRawData(IndexEnum::INDEX_HNSW, ver, {}));
#endif

        CHECK_FALSE(knowhere::IndexStaticFaced<fp32>::HasRawData(IndexEnum::INDEX_HNSW_SQ8, ver, {}));
        CHECK(knowhere::IndexStaticFaced<fp32>::HasRawData(IndexEnum::INDEX_HNSW_SQ8_REFINE, ver, {}));

        // faiss HNSW
        CHECK(knowhere::IndexStaticFaced<fp32>::HasRawData(IndexEnum::INDEX_FAISS_HNSW_FLAT, ver, {}));
        CHECK_FALSE(knowhere::IndexStaticFaced<fp32>::HasRawData(IndexEnum::INDEX_FAISS_HNSW_SQ, ver, {}));
        CHECK_FALSE(knowhere::IndexStaticFaced<fp32>::HasRawData(IndexEnum::INDEX_FAISS_HNSW_PQ, ver, {}));
        CHECK_FALSE(knowhere::IndexStaticFaced<fp32>::HasRawData(IndexEnum::INDEX_FAISS_HNSW_PRQ, ver, {}));

        // diskann
#ifndef KNOWHERE_WITH_CARDINAL
        CHECK(knowhere::IndexStaticFaced<fp32>::HasRawData(IndexEnum::INDEX_DISKANN, ver,
                                                           knowhere::Json::parse(R"({"metric_type": "L2"})")));
        CHECK_FALSE(knowhere::IndexStaticFaced<fp32>::HasRawData(IndexEnum::INDEX_DISKANN, ver,
                                                                 knowhere::Json::parse(R"({"metric_type": "IP"})")));
        CHECK(knowhere::IndexStaticFaced<fp32>::HasRawData(IndexEnum::INDEX_DISKANN, ver,
                                                           knowhere::Json::parse(R"({"metric_type": "COSINE"})")));
#endif
        // gpu index

#ifdef KNOWHERE_WITH_RAFT
        CHECK_FALSE(knowhere::IndexStaticFaced<fp32>::HasRawData(IndexEnum::INDEX_GPU_BRUTEFORCE, ver, {}));
        CHECK_FALSE(knowhere::IndexStaticFaced<fp32>::HasRawData(IndexEnum::INDEX_GPU_IVFFLAT, ver, {}));
        CHECK_FALSE(knowhere::IndexStaticFaced<fp32>::HasRawData(IndexEnum::INDEX_GPU_IVFPQ, ver, {}));
        CHECK_FALSE(knowhere::IndexStaticFaced<fp32>::HasRawData(IndexEnum::INDEX_GPU_CAGRA, ver, {}));
#endif
        // sparse index
        CHECK(knowhere::IndexStaticFaced<fp32>::HasRawData(IndexEnum::INDEX_SPARSE_INVERTED_INDEX, ver, {}));
        CHECK(knowhere::IndexStaticFaced<fp32>::HasRawData(IndexEnum::INDEX_SPARSE_WAND, ver, {}));
    }

    SECTION("Special test") {
        auto min_ver = Version::GetMinimalVersion().VersionNumber();
        auto ver = Version::GetCurrentVersion().VersionNumber();

        knowhere::Json json = {
            {indexparam::WITH_RAW_DATA, true},
        };

        CHECK_FALSE(knowhere::IndexStaticFaced<fp32>::HasRawData(
            IndexEnum::INDEX_FAISS_IDMAP, min_ver, knowhere::Json::parse(R"({"metric_type": "COSINE"})")));
        CHECK(knowhere::IndexStaticFaced<fp32>::HasRawData(IndexEnum::INDEX_FAISS_SCANN, ver, json));

        json[indexparam::WITH_RAW_DATA] = false;
        CHECK_FALSE(knowhere::IndexStaticFaced<fp32>::HasRawData(IndexEnum::INDEX_FAISS_SCANN, ver, json));
        json[indexparam::WITH_RAW_DATA] = "true";
        CHECK(knowhere::IndexStaticFaced<fp32>::HasRawData(IndexEnum::INDEX_FAISS_SCANN, ver, json));
        json[indexparam::WITH_RAW_DATA] = "false";
        CHECK_FALSE(knowhere::IndexStaticFaced<fp32>::HasRawData(IndexEnum::INDEX_FAISS_SCANN, ver, json));
        json[indexparam::WITH_RAW_DATA] = 1;
        CHECK_FALSE(knowhere::IndexStaticFaced<fp32>::HasRawData(IndexEnum::INDEX_FAISS_SCANN, ver, json));

        knowhere::Json faiss_hnsw_cfg = {{"refine", true}, {"refine_type", "bf16"}};
        CHECK_FALSE(knowhere::IndexStaticFaced<fp32>::HasRawData(IndexEnum::INDEX_FAISS_HNSW_SQ, ver, faiss_hnsw_cfg));
        CHECK(knowhere::IndexStaticFaced<bf16>::HasRawData(IndexEnum::INDEX_FAISS_HNSW_SQ, ver, faiss_hnsw_cfg));
        CHECK_FALSE(knowhere::IndexStaticFaced<fp16>::HasRawData(IndexEnum::INDEX_FAISS_HNSW_SQ, ver, faiss_hnsw_cfg));

        faiss_hnsw_cfg["refine_type"] = "fp16";
        CHECK_FALSE(knowhere::IndexStaticFaced<fp32>::HasRawData(IndexEnum::INDEX_FAISS_HNSW_SQ, ver, faiss_hnsw_cfg));
        CHECK_FALSE(knowhere::IndexStaticFaced<bf16>::HasRawData(IndexEnum::INDEX_FAISS_HNSW_SQ, ver, faiss_hnsw_cfg));
        CHECK(knowhere::IndexStaticFaced<fp16>::HasRawData(IndexEnum::INDEX_FAISS_HNSW_SQ, ver, faiss_hnsw_cfg));

        faiss_hnsw_cfg["refine_type"] = "sq8";
        CHECK_FALSE(knowhere::IndexStaticFaced<fp32>::HasRawData(IndexEnum::INDEX_FAISS_HNSW_SQ, ver, faiss_hnsw_cfg));
        CHECK_FALSE(knowhere::IndexStaticFaced<bf16>::HasRawData(IndexEnum::INDEX_FAISS_HNSW_SQ, ver, faiss_hnsw_cfg));
        CHECK_FALSE(knowhere::IndexStaticFaced<fp16>::HasRawData(IndexEnum::INDEX_FAISS_HNSW_SQ, ver, faiss_hnsw_cfg));

        faiss_hnsw_cfg["refine_type"] = "sq6";
        CHECK_FALSE(knowhere::IndexStaticFaced<fp32>::HasRawData(IndexEnum::INDEX_FAISS_HNSW_SQ, ver, faiss_hnsw_cfg));
        CHECK_FALSE(knowhere::IndexStaticFaced<bf16>::HasRawData(IndexEnum::INDEX_FAISS_HNSW_SQ, ver, faiss_hnsw_cfg));
        CHECK_FALSE(knowhere::IndexStaticFaced<fp16>::HasRawData(IndexEnum::INDEX_FAISS_HNSW_SQ, ver, faiss_hnsw_cfg));
    }
}
