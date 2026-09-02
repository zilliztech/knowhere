// Copyright (C) 2019-2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
// on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for
// the specific language governing permissions and limitations under the License.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "faiss/IndexRaBitQ.h"
#include "faiss/IndexRaBitQFastScan.h"
#include "faiss/VectorTransform.h"
#include "faiss/cppcontrib/knowhere/IndexHNSWRaBitQ.h"
#include "faiss/cppcontrib/knowhere/index_io.h"
#include "faiss/impl/FaissException.h"
#include "faiss/impl/RaBitQUtils.h"
#include "faiss/impl/io.h"
#include "faiss/index_io.h"
#include "knowhere/bitsetview.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_check.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/utils.h"
#include "utils.h"

namespace {

constexpr int64_t kNb = 256;
constexpr int64_t kNq = 4;
constexpr int64_t kTopk = 8;

knowhere::Json
MakeHnswRaBitQConfig(int64_t dim, const std::string& metric, int rbq_bits = 1) {
    knowhere::Json json;
    json[knowhere::meta::DIM] = dim;
    json[knowhere::meta::ROWS] = kNb;
    json[knowhere::meta::METRIC_TYPE] = metric;
    json[knowhere::meta::TOPK] = kTopk;
    json[knowhere::indexparam::HNSW_M] = 12;
    json[knowhere::indexparam::EFCONSTRUCTION] = 64;
    json[knowhere::indexparam::EF] = 64;
    json[knowhere::indexparam::RABITQ_BITS] = rbq_bits;
    return json;
}

bool
SerializedIndexContainsFourcc(const knowhere::BinarySet& binary_set, const std::string& name, const char* fourcc) {
    const auto binary = binary_set.GetByName(name);
    if (binary == nullptr) {
        return false;
    }
    const auto* begin = binary->data.get();
    const auto* end = begin + binary->size;
    return std::search(begin, end, fourcc, fourcc + 4) != end;
}

void
CheckValidKnnResult(const knowhere::DataSet& result, int64_t nb, int64_t nq, int64_t topk) {
    REQUIRE(result.GetRows() == nq);
    REQUIRE(result.GetDim() == topk);
    const auto* ids = result.GetIds();
    const auto* distances = result.GetDistance();
    for (int64_t i = 0; i < nq * topk; ++i) {
        REQUIRE(ids[i] >= 0);
        REQUIRE(ids[i] < nb);
        REQUIRE(std::isfinite(distances[i]));
    }
}

void
CheckKnnOrder(const knowhere::DataSet& result, const std::string& metric) {
    const auto rows = result.GetRows();
    const auto topk = result.GetDim();
    const auto* distances = result.GetDistance();
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 1; j < topk; ++j) {
            const auto previous = distances[i * topk + j - 1];
            const auto current = distances[i * topk + j];
            if (knowhere::IsMetricType(metric, knowhere::metric::L2)) {
                REQUIRE(previous <= current);
            } else {
                REQUIRE(previous >= current);
            }
        }
    }
}

template <typename DataType>
void
CheckTypedBuildAndSearch(int64_t dim) {
    const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto index = knowhere::IndexFactory::Instance().Create<DataType>(knowhere::IndexEnum::INDEX_HNSW_RABITQ, version);
    REQUIRE(index.has_value());

    const auto train = knowhere::ConvertToDataTypeIfNeeded<DataType>(GenDataSet(kNb, dim, 101));
    const auto query = knowhere::ConvertToDataTypeIfNeeded<DataType>(GenDataSet(kNq, dim, 102));
    auto json = MakeHnswRaBitQConfig(dim, knowhere::metric::L2);

    REQUIRE(index.value().Build(train, json) == knowhere::Status::success);
    const auto result = index.value().Search(query, json, nullptr);
    REQUIRE(result.has_value());
    CheckValidKnnResult(*result.value(), kNb, kNq, kTopk);
}

struct SearchSnapshot {
    std::vector<int64_t> ids;
    std::vector<float> distances;
};

SearchSnapshot
TakeSnapshot(const knowhere::DataSet& result) {
    const auto count = result.GetRows() * result.GetDim();
    return {std::vector<int64_t>(result.GetIds(), result.GetIds() + count),
            std::vector<float>(result.GetDistance(), result.GetDistance() + count)};
}

std::vector<int64_t>
ExactTopK(const float* base, const float* queries, int64_t nb, int64_t nq, int64_t dim, int64_t topk,
          bool inner_product) {
    std::vector<int64_t> result(nq * topk);
    std::vector<std::pair<float, int64_t>> scored(nb);
    for (int64_t q = 0; q < nq; ++q) {
        for (int64_t i = 0; i < nb; ++i) {
            float value = 0.0f;
            for (int64_t j = 0; j < dim; ++j) {
                const float bv = base[i * dim + j];
                const float qv = queries[q * dim + j];
                value += inner_product ? bv * qv : (bv - qv) * (bv - qv);
            }
            scored[i] = {value, i};
        }
        std::partial_sort(scored.begin(), scored.begin() + topk, scored.end(),
                          [inner_product](const auto& lhs, const auto& rhs) {
                              return inner_product ? lhs.first > rhs.first : lhs.first < rhs.first;
                          });
        for (int64_t k = 0; k < topk; ++k) {
            result[q * topk + k] = scored[k].second;
        }
    }
    return result;
}

double
RecallAtK(const int64_t* actual, const std::vector<int64_t>& expected, int64_t nq, int64_t topk) {
    int64_t matches = 0;
    for (int64_t q = 0; q < nq; ++q) {
        for (int64_t k = 0; k < topk; ++k) {
            const auto begin = expected.begin() + q * topk;
            const auto end = begin + topk;
            matches += std::find(begin, end, actual[q * topk + k]) != end;
        }
    }
    return static_cast<double>(matches) / static_cast<double>(nq * topk);
}

}  // namespace

TEST_CASE("HNSW_RABITQ supports the public type and data-type boundary", "[hnsw_rabitq]") {
    const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    const auto type = knowhere::IndexEnum::INDEX_HNSW_RABITQ;
    auto& factory = knowhere::IndexFactory::Instance();

    REQUIRE(factory.Create<knowhere::fp32>(type, version).has_value());
    REQUIRE(factory.Create<knowhere::fp16>(type, version).has_value());
    REQUIRE(factory.Create<knowhere::bf16>(type, version).has_value());
    REQUIRE_FALSE(factory.Create<knowhere::int8>(type, version).has_value());
    REQUIRE_FALSE(factory.Create<knowhere::bin1>(type, version).has_value());

    REQUIRE(factory.FeatureCheck(type, knowhere::feature::FLOAT32));
    REQUIRE(factory.FeatureCheck(type, knowhere::feature::FP16));
    REQUIRE(factory.FeatureCheck(type, knowhere::feature::BF16));
    REQUIRE_FALSE(factory.FeatureCheck(type, knowhere::feature::INT8));
    REQUIRE_FALSE(factory.FeatureCheck(type, knowhere::feature::BINARY));
    REQUIRE_FALSE(factory.FeatureCheck(type, knowhere::feature::MMAP));
    REQUIRE_FALSE(factory.FeatureCheck(type, knowhere::feature::MV));
    REQUIRE_FALSE(factory.FeatureCheck(type, knowhere::feature::EMB_LIST));

    REQUIRE(knowhere::KnowhereCheck::IndexTypeAndDataTypeCheck(type, knowhere::VecType::VECTOR_FLOAT));
    REQUIRE(knowhere::KnowhereCheck::IndexTypeAndDataTypeCheck(type, knowhere::VecType::VECTOR_FLOAT16));
    REQUIRE(knowhere::KnowhereCheck::IndexTypeAndDataTypeCheck(type, knowhere::VecType::VECTOR_BFLOAT16));
    REQUIRE_FALSE(knowhere::KnowhereCheck::IndexTypeAndDataTypeCheck(type, knowhere::VecType::VECTOR_INT8));
    REQUIRE_FALSE(knowhere::KnowhereCheck::IndexTypeAndDataTypeCheck(type, knowhere::VecType::VECTOR_BINARY));
    REQUIRE_FALSE(knowhere::KnowhereCheck::SupportMmapIndexTypeCheck(type));
    REQUIRE_FALSE(knowhere::KnowhereCheck::IndexTypeAndDataTypeCheck(type, knowhere::VecType::VECTOR_FLOAT, true));

    auto fp32_index = factory.Create<knowhere::fp32>(type, version).value();
    REQUIRE_FALSE(fp32_index.IsAdditionalScalarSupported(false));
    REQUIRE_FALSE(fp32_index.IsAdditionalScalarSupported(true));

    CheckTypedBuildAndSearch<knowhere::fp32>(13);
    CheckTypedBuildAndSearch<knowhere::fp16>(13);
    CheckTypedBuildAndSearch<knowhere::bf16>(13);
}

TEST_CASE("HNSW_RABITQ validates metrics and bit widths", "[hnsw_rabitq]") {
    const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    const auto train = GenDataSet(kNb, 32, 201);

    for (const int rbq_bits : {0, 9}) {
        CAPTURE(rbq_bits);
        auto index = knowhere::IndexFactory::Instance()
                         .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_RABITQ, version)
                         .value();
        const auto json = MakeHnswRaBitQConfig(32, knowhere::metric::L2, rbq_bits);
        REQUIRE(index.Build(train, json) == knowhere::Status::out_of_range_in_json);
    }

    {
        auto index = knowhere::IndexFactory::Instance()
                         .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_RABITQ, version)
                         .value();
        const auto json = MakeHnswRaBitQConfig(32, knowhere::metric::JACCARD);
        REQUIRE(index.Build(train, json) == knowhere::Status::invalid_metric_type);
    }

    {
        auto index = knowhere::IndexFactory::Instance()
                         .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_RABITQ, version)
                         .value();
        auto json = MakeHnswRaBitQConfig(32, knowhere::metric::L2, 4);
        REQUIRE(index.Build(train, json) == knowhere::Status::success);
        const auto query = GenDataSet(kNq, 32, 202);
        json[knowhere::indexparam::RABITQ_QUERY_BITS] = 4;
        REQUIRE_FALSE(index.Search(query, json, nullptr).has_value());
    }
}

TEST_CASE("HNSW_RABITQ searches, ranges, iterates, and round-trips", "[hnsw_rabitq]") {
    struct Scenario {
        const char* metric;
        int64_t dim;
        int rbq_bits;
    };
    const std::vector<Scenario> scenarios = {
        {knowhere::metric::L2, 127, 1}, {knowhere::metric::IP, 129, 1},    {knowhere::metric::L2, 129, 2},
        {knowhere::metric::IP, 65, 3},  {knowhere::metric::L2, 64, 4},     {knowhere::metric::IP, 65, 5},
        {knowhere::metric::L2, 64, 6},  {knowhere::metric::COSINE, 65, 4}, {knowhere::metric::COSINE, 128, 8},
        {knowhere::metric::L2, 128, 8},
    };
    const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();

    for (const auto& scenario : scenarios) {
        CAPTURE(scenario.metric, scenario.dim, scenario.rbq_bits);
        const auto train = GenDataSet(kNb, scenario.dim, 301 + scenario.dim + scenario.rbq_bits);
        const auto query = GenDataSet(kNq, scenario.dim, 401 + scenario.dim + scenario.rbq_bits);
        auto json = MakeHnswRaBitQConfig(scenario.dim, scenario.metric, scenario.rbq_bits);

        auto index = knowhere::IndexFactory::Instance()
                         .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_RABITQ, version)
                         .value();
        REQUIRE(index.Build(train, json) == knowhere::Status::success);
        REQUIRE(index.Count() == kNb);

        const auto before = index.Search(query, json, nullptr);
        REQUIRE(before.has_value());
        CheckValidKnnResult(*before.value(), kNb, kNq, kTopk);
        CheckKnnOrder(*before.value(), scenario.metric);

        auto range_json = json;
        if (knowhere::IsMetricType(scenario.metric, knowhere::metric::L2)) {
            range_json[knowhere::meta::RADIUS] = std::numeric_limits<float>::max();
            range_json[knowhere::meta::RANGE_FILTER] = 0.0f;
        } else {
            range_json[knowhere::meta::RADIUS] = -std::numeric_limits<float>::max();
            range_json[knowhere::meta::RANGE_FILTER] = std::numeric_limits<float>::max();
        }
        range_json[knowhere::meta::RANGE_SEARCH_K] = 32;
        const auto range_result = index.RangeSearch(query, range_json, nullptr);
        REQUIRE(range_result.has_value());
        REQUIRE(range_result.value()->GetLims()[kNq] > 0);

        const auto iterators = index.AnnIterator(query, json, nullptr, false);
        REQUIRE(iterators.has_value());
        REQUIRE(iterators.value().size() == kNq);
        for (auto& iterator : iterators.value()) {
            REQUIRE(iterator->HasNext().value());
            const auto next = iterator->Next();
            REQUIRE(next.has_value());
            REQUIRE(next.value().first >= 0);
            REQUIRE(next.value().first < kNb);
            REQUIRE(std::isfinite(next.value().second));
        }

        const auto* labels = before.value()->GetIds();
        const auto one_query = knowhere::GenDataSet(1, scenario.dim, query->GetTensor());
        const bool is_cosine = knowhere::IsMetricType(scenario.metric, knowhere::metric::COSINE);
        const auto distances = index.CalcDistByIDs(one_query, nullptr, labels, kTopk, is_cosine);
        REQUIRE(distances.has_value());
        for (int64_t i = 0; i < kTopk; ++i) {
            REQUIRE(distances.value()->GetDistance()[i] ==
                    Catch::Approx(before.value()->GetDistance()[i]).epsilon(1e-5));
        }

        knowhere::BinarySet binary_set;
        REQUIRE(index.Serialize(binary_set) == knowhere::Status::success);
        REQUIRE(SerializedIndexContainsFourcc(binary_set, index.Type(), is_cosine ? "IHRC" : "IHRK"));
        REQUIRE_FALSE(SerializedIndexContainsFourcc(binary_set, index.Type(), "IHNr"));
        if (is_cosine) {
            REQUIRE(SerializedIndexContainsFourcc(binary_set, index.Type(), "IRKC"));
        }
        const char* storage_fourcc = scenario.rbq_bits == 1 ? "Ixrq" : "Ixrd";
        REQUIRE(SerializedIndexContainsFourcc(binary_set, index.Type(), storage_fourcc));

        const auto serialized = binary_set.GetByName(index.Type());
        REQUIRE(serialized != nullptr);
        faiss::VectorIOReader standard_reader;
        standard_reader.data.assign(serialized->data.get(), serialized->data.get() + serialized->size);
        std::unique_ptr<faiss::Index> standard_loaded(faiss::read_index(&standard_reader));
        REQUIRE(dynamic_cast<faiss::cppcontrib::knowhere::IndexHNSWRaBitQ*>(standard_loaded.get()) != nullptr);
        if (is_cosine) {
            REQUIRE(dynamic_cast<faiss::cppcontrib::knowhere::IndexHNSWRaBitQCosine*>(standard_loaded.get()) !=
                    nullptr);
        }
        const auto* standard_storage = dynamic_cast<const faiss::IndexPreTransform*>(
            dynamic_cast<const faiss::cppcontrib::knowhere::IndexHNSWRaBitQ*>(standard_loaded.get())->storage);
        REQUIRE(standard_storage != nullptr);
        const auto* standard_rabitq = dynamic_cast<const faiss::IndexRaBitQ*>(standard_storage->index);
        REQUIRE(standard_rabitq != nullptr);
        REQUIRE(standard_rabitq->rabitq.dense_layout == (scenario.rbq_bits > 1));

        faiss::VectorIOWriter standard_writer;
        faiss::write_index(standard_loaded.get(), &standard_writer);
        faiss::VectorIOReader contrib_reader;
        contrib_reader.data = standard_writer.data;
        std::unique_ptr<faiss::Index> contrib_loaded(faiss::cppcontrib::knowhere::read_index(&contrib_reader));
        REQUIRE(dynamic_cast<faiss::cppcontrib::knowhere::IndexHNSWRaBitQ*>(contrib_loaded.get()) != nullptr);

        auto loaded = knowhere::IndexFactory::Instance()
                          .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_RABITQ, version)
                          .value();
        REQUIRE(loaded.Deserialize(binary_set, json) == knowhere::Status::success);
        const auto after = loaded.Search(query, json, nullptr);
        REQUIRE(after.has_value());
        const auto before_snapshot = TakeSnapshot(*before.value());
        const auto after_snapshot = TakeSnapshot(*after.value());
        REQUIRE(after_snapshot.ids == before_snapshot.ids);
        REQUIRE(after_snapshot.distances == before_snapshot.distances);

        REQUIRE(index.Add(GenDataSet(8, scenario.dim, 501), json) == knowhere::Status::not_implemented);
    }
}

TEST_CASE("HNSW_RABITQ agrees with an independent FP32 oracle", "[hnsw_rabitq]") {
    const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    for (const auto& [metric, dim, bits] :
         std::vector<std::tuple<const char*, int64_t, int>>{{knowhere::metric::L2, 64, 1},
                                                            {knowhere::metric::IP, 65, 1},
                                                            {knowhere::metric::L2, 65, 4},
                                                            {knowhere::metric::IP, 64, 8}}) {
        CAPTURE(metric, dim, bits);
        const auto train = GenDataSet(kNb, dim, 2101 + dim + bits);
        const auto query = GenDataSet(kNq, dim, 2201 + dim + bits);
        auto json = MakeHnswRaBitQConfig(dim, metric, bits);
        json[knowhere::indexparam::EF] = kNb;

        auto index = knowhere::IndexFactory::Instance()
                         .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_RABITQ, version)
                         .value();
        REQUIRE(index.Build(train, json) == knowhere::Status::success);
        const auto actual = index.Search(query, json, nullptr);
        REQUIRE(actual.has_value());

        const bool is_ip = knowhere::IsMetricType(metric, knowhere::metric::IP);
        const auto expected = ExactTopK(static_cast<const float*>(train->GetTensor()),
                                        static_cast<const float*>(query->GetTensor()), kNb, kNq, dim, kTopk, is_ip);
        const double recall = RecallAtK(actual.value()->GetIds(), expected, kNq, kTopk);
        REQUIRE(recall >= (bits == 1 ? 0.20 : 0.75));
    }
}

TEST_CASE("RaBitQ dense layout preserves packed distances", "[hnsw_rabitq]") {
    constexpr int64_t nb = 64;
    constexpr int64_t nq = 2;
    for (const int64_t dim : {13, 64, 65, 128, 129}) {
        const auto base = GenDataSet(nb, dim, 1701 + dim);
        const auto query = GenDataSet(nq, dim, 1801 + dim);
        const auto* base_data = static_cast<const float*>(base->GetTensor());
        const auto* query_data = static_cast<const float*>(query->GetTensor());

        for (const uint8_t bits : {2, 3, 4, 5, 6, 7, 8}) {
            const size_t ex_bits = bits - 1;
            for (const auto metric : {faiss::METRIC_L2, faiss::METRIC_INNER_PRODUCT}) {
                faiss::IndexRaBitQ packed(dim, metric, bits, false);
                faiss::IndexRaBitQ dense(dim, metric, bits, true);
                packed.qb = 0;
                REQUIRE(dense.qb == 0);
                packed.train(nb, base_data);
                dense.train(nb, base_data);
                packed.add(nb, base_data);
                dense.add(nb, base_data);
                REQUIRE_THROWS(faiss::IndexRaBitQFastScan(dense));

                for (int64_t i = 0; i < nb; ++i) {
                    const uint8_t* packed_code = packed.codes.data() + i * packed.code_size;
                    const uint8_t* dense_code = dense.codes.data() + i * dense.code_size;
                    const uint8_t* extra_code =
                        packed_code + (dim + 7) / 8 + sizeof(faiss::rabitq_utils::SignBitFactorsWithError);
                    for (int64_t j = 0; j < dim; ++j) {
                        const uint16_t sign =
                            (packed_code[j / 8] & (1u << (j % 8))) != 0 ? static_cast<uint16_t>(1u << ex_bits) : 0;
                        const uint16_t extra =
                            static_cast<uint16_t>(faiss::rabitq_utils::extract_code_inline(extra_code, j, ex_bits));
                        REQUIRE(faiss::rabitq_utils::extract_code_inline(dense_code, j, bits) ==
                                static_cast<uint16_t>(sign | extra));
                    }
                }

                std::unique_ptr<faiss::DistanceComputer> packed_dc(packed.get_distance_computer());
                std::unique_ptr<faiss::DistanceComputer> dense_dc(dense.get_distance_computer());
                for (int64_t q = 0; q < nq; ++q) {
                    packed_dc->set_query(query_data + q * dim);
                    dense_dc->set_query(query_data + q * dim);
                    for (int64_t i = 0; i < nb; ++i) {
                        REQUIRE((*dense_dc)(i) == Catch::Approx((*packed_dc)(i)).epsilon(2e-5));
                    }
                    for (int64_t i = 0; i + 3 < nb; i += 4) {
                        float batch[4];
                        dense_dc->distances_batch_4(i, i + 1, i + 2, i + 3, batch[0], batch[1], batch[2], batch[3]);
                        for (int64_t j = 0; j < 4; ++j) {
                            REQUIRE(batch[j] == Catch::Approx((*packed_dc)(i + j)).epsilon(2e-5));
                        }
                    }
                }

                constexpr int64_t topk = 8;
                std::vector<float> packed_distances(nq * topk);
                std::vector<float> dense_distances(nq * topk);
                std::vector<faiss::idx_t> packed_labels(nq * topk);
                std::vector<faiss::idx_t> dense_labels(nq * topk);
                packed.search(nq, query_data, topk, packed_distances.data(), packed_labels.data());
                dense.search(nq, query_data, topk, dense_distances.data(), dense_labels.data());
                REQUIRE(dense_labels == packed_labels);
                for (size_t i = 0; i < dense_distances.size(); ++i) {
                    REQUIRE(dense_distances[i] == Catch::Approx(packed_distances[i]).epsilon(2e-5));
                }
            }
        }
    }
}

TEST_CASE("RaBitQ cosine storage applies post-norm correction", "[hnsw_rabitq]") {
    constexpr int64_t dim = 13;
    constexpr int64_t nb = 8;
    const auto generated_base = GenDataSet(nb, dim, 1901);
    auto query = GenDataSet(2, dim, 1902);
    const auto* generated_base_data = static_cast<const float*>(generated_base->GetTensor());
    std::vector<float> base_values(generated_base_data, generated_base_data + nb * dim);
    auto* base_data = base_values.data();
    const auto* query_data = static_cast<const float*>(query->GetTensor());

    for (int64_t i = 0; i < nb; ++i) {
        const float scale = static_cast<float>(i + 1);
        for (int64_t j = 0; j < dim; ++j) {
            base_data[i * dim + j] *= scale;
        }
    }
    std::fill(base_data, base_data + dim, 0.0f);

    auto* rotation = new faiss::RandomRotationMatrix(dim, dim);
    auto* rabitq = new faiss::IndexRaBitQ(dim, faiss::METRIC_INNER_PRODUCT, 4, true);
    rabitq->qb = 0;
    rabitq->centered = false;
    faiss::cppcontrib::knowhere::IndexPreTransformRaBitQCosine storage(rotation, rabitq);
    storage.own_fields = true;
    storage.train(nb, base_data);
    storage.add(nb, base_data);
    storage.validate_norms();

    std::unique_ptr<faiss::DistanceComputer> raw_dc(storage.faiss::IndexPreTransform::get_distance_computer());
    std::unique_ptr<faiss::DistanceComputer> cosine_dc(storage.get_distance_computer());
    const auto* inverse_norms = storage.get_inverse_l2_norms();

    for (int64_t q = 0; q < 2; ++q) {
        const float* current_query = query_data + q * dim;
        float query_norm_sqr = 0.0f;
        for (int64_t j = 0; j < dim; ++j) {
            query_norm_sqr += current_query[j] * current_query[j];
        }
        const float inverse_query_norm = query_norm_sqr > 0.0f ? 1.0f / std::sqrt(query_norm_sqr) : 1.0f;
        raw_dc->set_query(current_query);
        cosine_dc->set_query(current_query);
        for (int64_t i = 0; i < nb; ++i) {
            const float expected = (*raw_dc)(i)*inverse_norms[i] * inverse_query_norm;
            REQUIRE((*cosine_dc)(i) == Catch::Approx(expected).epsilon(1e-6));
        }

        float raw_batch[4];
        float cosine_batch[4];
        raw_dc->distances_batch_4(0, 1, 2, 3, raw_batch[0], raw_batch[1], raw_batch[2], raw_batch[3]);
        cosine_dc->distances_batch_4(0, 1, 2, 3, cosine_batch[0], cosine_batch[1], cosine_batch[2], cosine_batch[3]);
        for (int64_t i = 0; i < 4; ++i) {
            REQUIRE(cosine_batch[i] ==
                    Catch::Approx(raw_batch[i] * inverse_norms[i] * inverse_query_norm).epsilon(1e-6));
        }
    }
}

TEST_CASE("HNSW_RABITQ supports optional refinement", "[hnsw_rabitq]") {
    constexpr int64_t dim = 32;
    const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    const auto train = GenDataSet(kNb, dim, 701);
    const auto query = GenDataSet(kNq, dim, 702);
    auto json = MakeHnswRaBitQConfig(dim, knowhere::metric::L2);
    json[knowhere::indexparam::REFINE] = true;
    json[knowhere::indexparam::REFINE_TYPE] = "FLAT";
    json[knowhere::indexparam::REFINE_K] = 2.0f;

    auto index = knowhere::IndexFactory::Instance()
                     .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_RABITQ, version)
                     .value();
    REQUIRE(index.Build(train, json) == knowhere::Status::success);
    REQUIRE(index.IsIndexRefineEnabled());
    REQUIRE(index.HasRawData(knowhere::metric::L2));

    const auto result = index.Search(query, json, nullptr);
    REQUIRE(result.has_value());
    CheckValidKnnResult(*result.value(), kNb, kNq, kTopk);

    knowhere::BinarySet binary_set;
    REQUIRE(index.Serialize(binary_set) == knowhere::Status::success);
    REQUIRE(SerializedIndexContainsFourcc(binary_set, index.Type(), "IHRK"));
    REQUIRE_FALSE(SerializedIndexContainsFourcc(binary_set, index.Type(), "IHNr"));
    REQUIRE(SerializedIndexContainsFourcc(binary_set, index.Type(), "Ixrq"));

    auto loaded = knowhere::IndexFactory::Instance()
                      .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_RABITQ, version)
                      .value();
    REQUIRE(loaded.Deserialize(binary_set, json) == knowhere::Status::success);
    const auto loaded_result = loaded.Search(query, json, nullptr);
    REQUIRE(loaded_result.has_value());
    REQUIRE(TakeSnapshot(*loaded_result.value()).ids == TakeSnapshot(*result.value()).ids);
}

TEST_CASE("HNSW_RABITQ cosine refinement returns exact cosine", "[hnsw_rabitq]") {
    constexpr int64_t dim = 32;
    const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    const auto train = GenDataSet(kNb, dim, 2001);
    const auto query = GenDataSet(kNq, dim, 2002);
    auto json = MakeHnswRaBitQConfig(dim, knowhere::metric::COSINE, 4);
    json[knowhere::indexparam::REFINE] = true;
    json[knowhere::indexparam::REFINE_TYPE] = "FLAT";
    json[knowhere::indexparam::REFINE_K] = 2.0f;

    auto index = knowhere::IndexFactory::Instance()
                     .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_RABITQ, version)
                     .value();
    REQUIRE(index.Build(train, json) == knowhere::Status::success);
    const auto result = index.Search(query, json, nullptr);
    REQUIRE(result.has_value());
    CheckValidKnnResult(*result.value(), kNb, kNq, kTopk);
    CheckKnnOrder(*result.value(), knowhere::metric::COSINE);

    const auto* base_data = static_cast<const float*>(train->GetTensor());
    const auto* query_data = static_cast<const float*>(query->GetTensor());
    for (int64_t q = 0; q < kNq; ++q) {
        for (int64_t k = 0; k < kTopk; ++k) {
            const int64_t id = result.value()->GetIds()[q * kTopk + k];
            float dot = 0.0f;
            float base_norm_sqr = 0.0f;
            float query_norm_sqr = 0.0f;
            for (int64_t j = 0; j < dim; ++j) {
                const float bv = base_data[id * dim + j];
                const float qv = query_data[q * dim + j];
                dot += bv * qv;
                base_norm_sqr += bv * bv;
                query_norm_sqr += qv * qv;
            }
            const float expected = (base_norm_sqr == 0.0f || query_norm_sqr == 0.0f)
                                       ? 0.0f
                                       : dot / std::sqrt(base_norm_sqr * query_norm_sqr);
            REQUIRE(result.value()->GetDistance()[q * kTopk + k] == Catch::Approx(expected).epsilon(2e-5));
        }
    }

    knowhere::BinarySet binary_set;
    REQUIRE(index.Serialize(binary_set) == knowhere::Status::success);
    REQUIRE(SerializedIndexContainsFourcc(binary_set, index.Type(), "IHRC"));
    REQUIRE(SerializedIndexContainsFourcc(binary_set, index.Type(), "IRKC"));
    auto loaded = knowhere::IndexFactory::Instance()
                      .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_RABITQ, version)
                      .value();
    REQUIRE(loaded.Deserialize(binary_set, json) == knowhere::Status::success);
    const auto loaded_result = loaded.Search(query, json, nullptr);
    REQUIRE(loaded_result.has_value());
    REQUIRE(TakeSnapshot(*loaded_result.value()).ids == TakeSnapshot(*result.value()).ids);
    REQUIRE(TakeSnapshot(*loaded_result.value()).distances == TakeSnapshot(*result.value()).distances);
}
