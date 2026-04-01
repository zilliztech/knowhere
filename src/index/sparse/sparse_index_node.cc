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

#include <sys/mman.h>

#include <boost/intrusive/pack_options.hpp>
#include <exception>
#include <optional>

#include "index/sparse/block_inverted_index.h"
#include "index/sparse/codec/maskedvbyte.h"
#include "index/sparse/codec/streamvbyte.h"
#include "index/sparse/flatten_inverted_index.h"
#include "index/sparse/growable_inverted_index.h"
#include "index/sparse/inverted_index.h"
#include "index/sparse/sindi_inverted_index.h"
#include "index/sparse/sparse_index_config.h"
#include "io/file_io.h"
#include "io/memory_io.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/config.h"
#include "knowhere/dataset.h"
#include "knowhere/expected.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/index/index_node.h"
#include "knowhere/log.h"
#include "knowhere/operands.h"
#include "knowhere/sparse_utils.h"
#include "knowhere/thread_pool.h"
#include "knowhere/utils.h"

namespace knowhere {

// Peek at the encoding_type stored in serialized index data without fully parsing.
// Returns nullopt if the data is too small or the first section is not POSTING_LISTS.
//
// Binary layout:
//   [0..32)  File header: version(4), nr_rows(4), max_dim(4), nr_inner_dims(4), reserved(16)
//   [32..36) nr_sections (uint32_t)
//   [36..)   Section headers: each is InvertedIndexSectionType(4) + padding(4) + offset(8) + size(8) = 24 bytes
//   At section[0].offset: encoding_type (uint32_t)
static std::optional<sparse::inverted::InvertedIndexEncoding>
peek_encoding_type_from_index_data(const uint8_t* data, size_t size) {
    using sparse::inverted::InvertedIndexEncoding;
    using sparse::inverted::InvertedIndexSectionHeader;
    using sparse::inverted::InvertedIndexSectionType;

    // Need at least: 32 (header) + 4 (nr_sections) + 24 (one section header) = 60 bytes
    constexpr size_t kMinHeaderSize = 32 + 4 + sizeof(InvertedIndexSectionHeader);
    if (size < kMinHeaderSize) {
        return std::nullopt;
    }

    // Read nr_sections at offset 32
    uint32_t nr_sections = 0;
    memcpy(&nr_sections, data + 32, sizeof(uint32_t));
    if (nr_sections == 0) {
        return std::nullopt;
    }

    // Read first section header at offset 36
    InvertedIndexSectionHeader first_section{};
    memcpy(&first_section, data + 36, sizeof(InvertedIndexSectionHeader));

    if (first_section.type != InvertedIndexSectionType::POSTING_LISTS) {
        return std::nullopt;
    }

    // encoding_type is the first uint32_t at the section's data offset
    if (first_section.offset + sizeof(uint32_t) > size) {
        return std::nullopt;
    }

    uint32_t encoding_type_raw = 0;
    memcpy(&encoding_type_raw, data + first_section.offset, sizeof(uint32_t));

    return static_cast<InvertedIndexEncoding>(encoding_type_raw);
}

// Inverted Index impl for sparse vectors.
//
// Not overriding RangeSearch, will use the default implementation in IndexNode.
//
// Thread safety: not thread safe.
template <typename T, bool use_wand>
class SparseInvertedIndexNode : public IndexNode {
    static_assert(std::is_same_v<T, knowhere::sparse_u32_f32>, "SparseInvertedIndexNode only sparse_u32_f32");

    using value_type = typename T::ValueType;

 public:
    explicit SparseInvertedIndexNode(const int32_t& version, const Object& /*object*/)
        : search_pool_(ThreadPool::GetGlobalSearchThreadPool()),
          build_pool_(ThreadPool::GetGlobalBuildThreadPool()),
          index_version_(version) {
    }

    Status
    Train(const DataSetPtr dataset, std::shared_ptr<Config> config, bool use_knowhere_build_pool) override {
        auto cfg = static_cast<const SparseInvertedIndexConfig&>(*config);

        // metric type validation
        if (!IsMetricType(cfg.metric_type.value(), metric::IP) &&
            !IsMetricType(cfg.metric_type.value(), metric::BM25)) {
            LOG_KNOWHERE_ERROR_ << Type() << " only support metric_type IP or BM25";
            return Status::invalid_metric_type;
        }

        if (IsMetricType(cfg.metric_type.value(), metric::BM25)) {
            if (!cfg.bm25_k1.has_value() || !cfg.bm25_b.has_value() || !cfg.bm25_avgdl.has_value()) {
                return Status::invalid_args;
            }
        }

        // create index
        auto index_or = CreateIndex(cfg);
        if (!index_or.has_value()) {
            return index_or.error();
        }

        if (index_ != nullptr) {
            LOG_KNOWHERE_WARNING_
                << Type()
                << " index has already been created, Train() will delete the old index and recreate a new one";
        }
        index_ = std::move(index_or.value());

        return Status::success;
    }

    Status
    Add(const DataSetPtr dataset, std::shared_ptr<Config> config, bool use_knowhere_build_pool) override {
        if (!index_) {
            LOG_KNOWHERE_ERROR_ << "Could not add data to uninitialized " << Type() << " index";
            return Status::empty_index;
        }

        auto build_pool_wrapper = std::make_shared<ThreadPoolWrapper>(build_pool_, use_knowhere_build_pool);
        auto tryObj =
            build_pool_wrapper
                ->push([&] {
                    return index_->add(static_cast<const sparse::SparseRow<value_type>*>(dataset->GetTensor()),
                                       dataset->GetRows(), dataset->GetDim());
                })
                .getTry();
        if (!tryObj.hasValue()) {
            LOG_KNOWHERE_WARNING_ << "Failed to add data to index " << Type() << ": " << tryObj.exception().what();
            return Status::sparse_inner_error;
        }

        return tryObj.value();
    }

    [[nodiscard]] expected<DataSetPtr>
    Search(const DataSetPtr dataset, std::unique_ptr<Config> config, const BitsetView& bitset,
           milvus::OpContext* /*op_context*/
    ) const override {
        if (!index_) {
            LOG_KNOWHERE_ERROR_ << "Could not search uninitialized " << Type() << " index";
            return expected<DataSetPtr>::Err(Status::empty_index, "index is not initialized");
        }

        auto cfg = static_cast<const SparseInvertedIndexConfig&>(*config);

        auto search_params_or = PrepareSearchParams(cfg);
        if (!search_params_or.has_value()) {
            return expected<DataSetPtr>::Err(search_params_or.error(), search_params_or.what());
        }
        auto search_params = search_params_or.value();

        auto queries = static_cast<const sparse::SparseRow<value_type>*>(dataset->GetTensor());
        auto nq = dataset->GetRows();
        auto k = cfg.k.value();
        auto p_id = std::make_unique<sparse::label_t[]>(nq * k);
        auto p_dist = std::make_unique<float[]>(nq * k);

        std::vector<folly::Future<folly::Unit>> futs;
        futs.reserve(nq);
        for (int64_t idx = 0; idx < nq; ++idx) {
            futs.emplace_back(search_pool_->push([&, idx = idx, p_id = p_id.get(), p_dist = p_dist.get()]() {
                index_->search(queries[idx], k, p_dist + idx * k, p_id + idx * k, bitset, search_params);
            }));
        }
        WaitAllSuccess(futs);

        return GenResultDataSet(nq, k, p_id.release(), p_dist.release());
    }

    // TODO: for now inverted index and wand use the same impl for AnnIterator.
    [[nodiscard]] expected<std::vector<IndexNode::IteratorPtr>>
    AnnIterator(const DataSetPtr dataset, std::unique_ptr<Config> config, const BitsetView& bitset,
                bool use_knowhere_search_pool, milvus::OpContext* /*op_context*/
    ) const override {
        if (!index_) {
            LOG_KNOWHERE_WARNING_ << "creating iterator on empty index";
            return expected<std::vector<IndexNode::IteratorPtr>>::Err(Status::empty_index, "index not loaded");
        }

        auto nq = dataset->GetRows();

        auto cfg = static_cast<const SparseInvertedIndexConfig&>(*config);

        auto search_params_or = PrepareSearchParams(cfg);
        if (!search_params_or.has_value()) {
            return expected<std::vector<IndexNode::IteratorPtr>>::Err(search_params_or.error(),
                                                                      search_params_or.what());
        }
        auto search_params = search_params_or.value();

        auto vec = std::vector<std::shared_ptr<IndexNode::iterator>>(nq, nullptr);
        try {
            for (int i = 0; i < nq; ++i) {
                // Heavy computations with `compute_dist_func` will be deferred until the first call to
                // 'Iterator->Next()'.
                auto compute_dist_func = [=]() -> std::vector<DistId> {
                    auto queries = static_cast<const sparse::SparseRow<value_type>*>(dataset->GetTensor());
                    std::vector<float> distances = index_->get_all_distances(queries[i], bitset, search_params);
                    std::vector<DistId> distances_ids;
                    // 30% is a ratio guesstimate of non-zero distances: probability of 2 random sparse splade
                    // vectors(100 non zero dims out of 30000 total dims) sharing at least 1 common non-zero
                    // dimension.
                    distances_ids.reserve(distances.size() * 0.3);
                    for (size_t i = 0; i < distances.size(); i++) {
                        if (distances[i] != 0) {
                            distances_ids.emplace_back((int64_t)i, distances[i]);
                        }
                    }
                    return distances_ids;
                };

                auto it =
                    std::make_shared<PrecomputedDistanceIterator>(compute_dist_func, true, use_knowhere_search_pool);
                vec[i] = it;
            }
        } catch (const std::exception& e) {
            return expected<std::vector<IndexNode::IteratorPtr>>::Err(Status::sparse_inner_error, e.what());
        }

        return vec;
    }

    [[nodiscard]] expected<DataSetPtr>
    GetVectorByIds(const DataSetPtr dataset, milvus::OpContext* op_context) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented, "GetVectorByIds not implemented");
    }

    static bool
    StaticHasRawData(const knowhere::BaseConfig& /*config*/, const IndexVersion& /*version*/) {
        return false;
    }

    [[nodiscard]] bool
    HasRawData(const std::string& metric_type) const override {
        return false;
    }

    [[nodiscard]] expected<DataSetPtr>
    GetIndexMeta(std::unique_ptr<Config> cfg) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented, "GetIndexMeta not supported for current index type");
    }

    Status
    Serialize(BinarySet& binset) const override {
        if (!index_) {
            LOG_KNOWHERE_ERROR_ << "Could not serialize empty " << Type();
            return Status::empty_index;
        }

        MemoryIOWriter writer;
        if (this->version_use_raw_data()) {
            RETURN_IF_ERROR(index_->convert_to_raw_data(writer));
        } else {
            RETURN_IF_ERROR(index_->serialize(writer));
        }
        std::shared_ptr<uint8_t[]> data(writer.data());
        binset.Append(Type(), data, writer.tellg());

        return Status::success;
    }

    Status
    Deserialize(const BinarySet& binset, std::shared_ptr<Config> config) override {
        auto cfg = static_cast<const SparseInvertedIndexConfig&>(*config);

        // get binary data first so we can peek encoding before creating index
        auto binary = binset.GetByName(Type());
        if (binary == nullptr) {
            LOG_KNOWHERE_ERROR_ << "Cannot get binary from BinarySet with name " << Type();
            return Status::invalid_binary_set;
        }

        // Detect encoding_type from the serialized data (only for non-raw-data versions)
        std::optional<sparse::inverted::InvertedIndexEncoding> encoding = std::nullopt;
        if (!this->version_use_raw_data()) {
            encoding = peek_encoding_type_from_index_data(binary->data.get(), binary->size);
            if (encoding.has_value()) {
                LOG_KNOWHERE_INFO_ << "Detected encoding_type=" << static_cast<uint32_t>(encoding.value())
                                   << " from serialized index data";
            }
        }

        // create or recreate index
        if (index_ != nullptr) {
            LOG_KNOWHERE_WARNING_
                << Type()
                << " index has already been created, Deserialize() will delete the old index and recreate a new one";
        }
        auto index_or = CreateIndex(cfg, false, encoding);
        if (!index_or.has_value()) {
            LOG_KNOWHERE_ERROR_ << "Failed to create index from BinarySet with name " << Type();
            return index_or.error();
        }
        index_ = std::move(index_or.value());

        // deserialize index from binary
        MemoryIOReader reader(binary->data.get(), binary->size);

        if (this->version_use_raw_data()) {
            LOG_KNOWHERE_INFO_ << "raw data will be used, rebuild index from raw data";
            return index_->build_from_raw_data(reader, false, "");
        } else {
            binary_ = binary;
            return index_->deserialize(reader);
        }
    }

    Status
    DeserializeFromFile(const std::string& filename, std::shared_ptr<Config> config) override {
        auto cfg = static_cast<const SparseInvertedIndexConfig&>(*config);

        // mmap the file first so we can peek encoding before creating index
        auto file_reader = knowhere::FileReader(filename);
        size_t map_size = file_reader.size();
        int map_flags = MAP_SHARED;
#ifdef MAP_POPULATE
        if (cfg.enable_mmap_pop.has_value() && cfg.enable_mmap_pop.value()) {
            map_flags |= MAP_POPULATE;
        }
#endif
        void* mapped_memory = mmap(nullptr, map_size, PROT_READ, map_flags, file_reader.descriptor(), 0);
        if (mapped_memory == MAP_FAILED) {
            LOG_KNOWHERE_ERROR_ << "Failed to mmap file " << filename << ": " << strerror(errno);
            return Status::disk_file_error;
        }
        auto mmap_guard = std::make_unique<MmapGuard>(map_size, filename, mapped_memory);

        // Detect encoding_type from the mapped data (only for non-raw-data versions)
        std::optional<sparse::inverted::InvertedIndexEncoding> encoding = std::nullopt;
        if (!this->version_use_raw_data()) {
            encoding = peek_encoding_type_from_index_data(reinterpret_cast<const uint8_t*>(mapped_memory), map_size);
            if (encoding.has_value()) {
                LOG_KNOWHERE_INFO_ << "Detected encoding_type=" << static_cast<uint32_t>(encoding.value())
                                   << " from index file " << filename;
            }
        }

        // create or recreate index
        if (index_ != nullptr) {
            LOG_KNOWHERE_WARNING_ << Type()
                                  << " index has already been created, DeserializeFromFile() will delete the old index "
                                     "and recreate a new one";
        }
        auto index_or = CreateIndex(cfg, false, encoding);
        if (!index_or.has_value()) {
            return index_or.error();
        }
        index_ = std::move(index_or.value());

        // deserialize index from mapped memory
        MemoryIOReader map_reader(reinterpret_cast<uint8_t*>(mapped_memory), map_size);

        if (this->version_use_raw_data()) {
            auto supplement_target_filename = filename + ".knowhere_sparse_index_supplement";
            return index_->build_from_raw_data(map_reader, true, supplement_target_filename);
        } else {
            this->mmap_guard_ = std::move(mmap_guard);
            return index_->deserialize(map_reader);
        }
    }

    static std::unique_ptr<BaseConfig>
    StaticCreateConfig() {
        return std::make_unique<SparseInvertedIndexConfig>();
    }

    [[nodiscard]] std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return StaticCreateConfig();
    }

    // note that the Dim of a sparse vector index may change as new vectors are added
    [[nodiscard]] int64_t
    Dim() const override {
        return index_ ? index_->nr_cols() : 0;
    }

    [[nodiscard]] int64_t
    Size() const override {
        return index_ ? index_->size() : 0;
    }

    [[nodiscard]] int64_t
    Count() const override {
        return index_ ? index_->nr_rows() : 0;
    }

    [[nodiscard]] std::string
    Type() const override {
        return use_wand ? knowhere::IndexEnum::INDEX_SPARSE_WAND : knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX;
    }

 protected:
    std::unique_ptr<sparse::inverted::InvertedIndex<value_type>> index_;

    template <typename DType, typename QType, sparse::inverted::IndexScorerType MetricType>
    expected<std::unique_ptr<sparse::inverted::InvertedIndex<value_type>>>
    CreateIndexImpl(const SparseInvertedIndexConfig& cfg, bool is_growable = false,
                    std::optional<sparse::inverted::InvertedIndexEncoding> encoding = std::nullopt) const {
        using IndexPtr = std::unique_ptr<sparse::inverted::InvertedIndex<value_type>>;

        const bool is_bm25 = IsMetricType(cfg.metric_type.value(), metric::BM25);
        const bool is_ip = IsMetricType(cfg.metric_type.value(), metric::IP);

        // Factory for docid-sorted index variants
        auto create_index_before_v10 = [&](const std::string& inverted_index_algo,
                                           const std::string& inverted_index_codec) -> expected<IndexPtr> {
            IndexPtr index;
            if (is_growable) {
                // For concurrent add/search, always use growable index and do not override with codec/flatten types
                index = std::make_unique<sparse::inverted::GrowableInvertedIndex<DType, QType>>();
            } else if (encoding.has_value()) {
                // Deserialization path: create index type matching the encoding found in file
                using sparse::inverted::InvertedIndexEncoding;
                switch (encoding.value()) {
                    case InvertedIndexEncoding::FLAT:
                        LOG_KNOWHERE_INFO_ << "Detected FLAT encoding in index file, using FlattenInvertedIndex";
                        index = std::make_unique<sparse::inverted::FlattenInvertedIndex<DType, QType>>();
                        break;
                    case InvertedIndexEncoding::BLOCK_STREAMVBYTE: {
                        LOG_KNOWHERE_INFO_ << "Detected BLOCK_STREAMVBYTE encoding in index file";
                        auto codec = std::make_shared<sparse::inverted::StreamVByteBlockCodec>();
                        index = std::make_unique<sparse::inverted::BlockInvertedIndex<DType, QType, MetricType>>(codec);
                        break;
                    }
                    case InvertedIndexEncoding::BLOCK_MASKEDVBYTE: {
                        LOG_KNOWHERE_INFO_ << "Detected BLOCK_MASKEDVBYTE encoding in index file";
                        auto codec = std::make_shared<sparse::inverted::MaskedVByteBlockCodec>();
                        index = std::make_unique<sparse::inverted::BlockInvertedIndex<DType, QType, MetricType>>(codec);
                        break;
                    }
                    case InvertedIndexEncoding::FIXED_DOCID_WINDOWS:
                        // SINDI encoding should not reach create_index_before_v10; fall through to default codec
                        LOG_KNOWHERE_WARNING_ << "Unexpected FIXED_DOCID_WINDOWS encoding in create_index_before_v10";
                        index = std::make_unique<sparse::inverted::FlattenInvertedIndex<DType, QType>>();
                        break;
                    default:
                        LOG_KNOWHERE_WARNING_ << "Unknown encoding type " << static_cast<uint32_t>(encoding.value())
                                              << ", falling back to config codec";
                        index = std::make_unique<sparse::inverted::FlattenInvertedIndex<DType, QType>>();
                        break;
                }
            } else {
                if (version_use_raw_data()) {
                    LOG_KNOWHERE_INFO_ << "Using FlattenInvertedIndex without codec";
                    index = std::make_unique<sparse::inverted::FlattenInvertedIndex<DType, QType>>();
                } else {
                    // use different index type based on codec
                    if (inverted_index_codec == "block_streamvbyte" || inverted_index_codec == "block_maskedvbyte") {
                        sparse::inverted::BlockCodecPtr codec;
                        if (inverted_index_codec == "block_streamvbyte") {
                            codec = std::make_shared<sparse::inverted::StreamVByteBlockCodec>();
                        } else {
                            codec = std::make_shared<sparse::inverted::MaskedVByteBlockCodec>();
                        }
                        index = std::make_unique<sparse::inverted::BlockInvertedIndex<DType, QType, MetricType>>(codec);
                    } else {
                        index = std::make_unique<sparse::inverted::FlattenInvertedIndex<DType, QType>>();
                    }
                }
            }
            if (inverted_index_algo == "BLOCK_MAX_MAXSCORE" || inverted_index_algo == "BLOCK_MAX_WAND") {
                index->set_build_algo(inverted_index_algo, cfg.block_max_block_size.value_or(128));
            } else {
                index->set_build_algo(inverted_index_algo);
            }
            if (is_bm25) {
                index->set_build_scorer(sparse::inverted::IndexScorerConfig{
                    .scorer_type = sparse::inverted::IndexScorerType::BM25,
                    .scorer_params = {.bm25 = {.k1 = cfg.bm25_k1.value(),
                                               .b = cfg.bm25_b.value(),
                                               .avgdl = std::max(cfg.bm25_avgdl.value(), 1.0f)}}});
                return index;
            } else if (is_ip) {
                index->set_build_scorer(sparse::inverted::IndexScorerConfig{
                    .scorer_type = sparse::inverted::IndexScorerType::IP,
                });
                return index;
            } else {
                return expected<std::unique_ptr<sparse::inverted::InvertedIndex<value_type>>>::Err(
                    Status::invalid_metric_type, "Unsupported metric type");
            }
        };

        if (version_default_to_daat_maxscore()) {
            const std::string algo = cfg.inverted_index_algo.value_or("DAAT_MAXSCORE");
            const std::string codec = cfg.inverted_index_codec.value_or("block_streamvbyte");
            return create_index_before_v10(algo, codec);
        } else if (is_ip) {
            const std::string algo = cfg.inverted_index_algo.value_or("SINDI");
            const std::string codec = cfg.inverted_index_codec.value_or("block_streamvbyte");
            // When encoding is available and not FIXED_DOCID_WINDOWS, the file was not
            // built with SINDI, so use create_index_before_v10 to match the actual file encoding.
            bool use_sindi =
                algo == "SINDI" && (!encoding.has_value() ||
                                    encoding.value() == sparse::inverted::InvertedIndexEncoding::FIXED_DOCID_WINDOWS);
            if (use_sindi) {
                auto index = std::make_unique<sparse::inverted::SindiInvertedIndexIP>(
                    cfg.sindi_window_size.value_or(sparse::inverted::SindiInvertedIndexIP::max_window_size));
                index->set_build_algo(algo);
                index->set_build_scorer(sparse::inverted::IndexScorerConfig{
                    .scorer_type = sparse::inverted::IndexScorerType::IP,
                });
                return index;
            } else {
                return create_index_before_v10(algo, codec);
            }
        } else if (is_bm25) {
            const std::string algo = cfg.inverted_index_algo.value_or("DAAT_MAXSCORE");
            const std::string codec = cfg.inverted_index_codec.value_or("block_streamvbyte");
            bool use_sindi =
                algo == "SINDI" && (!encoding.has_value() ||
                                    encoding.value() == sparse::inverted::InvertedIndexEncoding::FIXED_DOCID_WINDOWS);
            if (use_sindi) {
                auto index = std::make_unique<sparse::inverted::SindiInvertedIndexBM25>(
                    cfg.sindi_window_size.value_or(sparse::inverted::SindiInvertedIndexBM25::max_window_size));
                index->set_build_algo(algo);
                index->set_build_scorer(sparse::inverted::IndexScorerConfig{
                    .scorer_type = sparse::inverted::IndexScorerType::BM25,
                    .scorer_params = {.bm25 = {.k1 = cfg.bm25_k1.value(),
                                               .b = cfg.bm25_b.value(),
                                               .avgdl = std::max(cfg.bm25_avgdl.value(), 1.0f)}}});
                return index;
            } else {
                return create_index_before_v10(algo, codec);
            }
        } else {
            return expected<std::unique_ptr<sparse::inverted::InvertedIndex<value_type>>>::Err(
                Status::invalid_metric_type, "Unsupported metric type");
        }
    }

    expected<std::unique_ptr<sparse::inverted::InvertedIndex<value_type>>>
    CreateIndex(const SparseInvertedIndexConfig& cfg, bool is_growable = false,
                std::optional<sparse::inverted::InvertedIndexEncoding> encoding = std::nullopt) const {
        auto qt = cfg.quant_type.value_or("");
        using sparse::inverted::IndexScorerType;
        if (IsMetricType(cfg.metric_type.value(), metric::IP)) {
            // version < threshold forces fp32; version >= threshold defaults to fp16, user can override to fp32
            bool use_fp16 = version_support_fp16_quant_for_ip() && (qt == "fp16" || qt.empty());
            if (use_fp16) {
                return CreateIndexImpl<value_type, fp16, IndexScorerType::IP>(cfg, is_growable, encoding);
            } else {
                return CreateIndexImpl<value_type, float, IndexScorerType::IP>(cfg, is_growable, encoding);
            }
        } else {
            // BM25 default: u16
            if (qt == "u32") {
                return CreateIndexImpl<value_type, uint32_t, IndexScorerType::BM25>(cfg, is_growable, encoding);
            } else {
                return CreateIndexImpl<value_type, uint16_t, IndexScorerType::BM25>(cfg, is_growable, encoding);
            }
        }
    }

 private:
    /**
     * @brief Prepare search parameters
     *
     * @param config Search config
     * @return InvertedIndexSearchParams Search parameters
     */
    expected<sparse::inverted::InvertedIndexSearchParams>
    PrepareSearchParams(const SparseInvertedIndexConfig& config) const {
        sparse::inverted::InvertedIndexSearchParams search_params = {
            .approx =
                {
                    .drop_ratio_search = config.drop_ratio_search.value_or(0.0f),
                    .dim_max_score_ratio = config.dim_max_score_ratio.value_or(1.05f),
                },
        };

        auto algo_need_max_scores_per_dim = [&]() {
            return search_params.algo == sparse::inverted::InvertedIndexAlgo::DAAT_WAND ||
                   search_params.algo == sparse::inverted::InvertedIndexAlgo::DAAT_MAXSCORE ||
                   search_params.algo == sparse::inverted::InvertedIndexAlgo::BLOCK_MAX_MAXSCORE ||
                   search_params.algo == sparse::inverted::InvertedIndexAlgo::BLOCK_MAX_WAND;
        };

        auto algo_need_block_max_data = [&]() {
            return search_params.algo == sparse::inverted::InvertedIndexAlgo::BLOCK_MAX_MAXSCORE ||
                   search_params.algo == sparse::inverted::InvertedIndexAlgo::BLOCK_MAX_WAND;
        };

        // prepare and check search algo
        if (config.search_algo.value() == "INHERIT") {
            search_params.algo = index_->get_build_algo();
        } else if (config.search_algo.value() == "DAAT_MAXSCORE") {
            search_params.algo = sparse::inverted::InvertedIndexAlgo::DAAT_MAXSCORE;
        } else if (config.search_algo.value() == "DAAT_WAND") {
            search_params.algo = sparse::inverted::InvertedIndexAlgo::DAAT_WAND;
        } else if (config.search_algo.value() == "BLOCK_MAX_MAXSCORE") {
            search_params.algo = sparse::inverted::InvertedIndexAlgo::BLOCK_MAX_MAXSCORE;
        } else if (config.search_algo.value() == "BLOCK_MAX_WAND") {
            search_params.algo = sparse::inverted::InvertedIndexAlgo::BLOCK_MAX_WAND;
        } else if (config.search_algo.value() == "TAAT_NAIVE") {
            search_params.algo = sparse::inverted::InvertedIndexAlgo::TAAT_NAIVE;
        } else if (config.search_algo.value() == "SINDI") {
            if (index_->get_build_algo() != sparse::inverted::InvertedIndexAlgo::SINDI) {
                return expected<sparse::inverted::InvertedIndexSearchParams>::Err(
                    Status::invalid_args, "search algorithm SINDI is only supported for SINDI index");
            }
            search_params.algo = sparse::inverted::InvertedIndexAlgo::SINDI;
        } else {
            return expected<sparse::inverted::InvertedIndexSearchParams>::Err(Status::invalid_args,
                                                                              "Unsupported search algorithm");
        }
        // SINDI index can only use SINDI search algorithm
        if (index_->get_build_algo() == sparse::inverted::InvertedIndexAlgo::SINDI &&
            search_params.algo != sparse::inverted::InvertedIndexAlgo::SINDI) {
            return expected<sparse::inverted::InvertedIndexSearchParams>::Err(
                Status::invalid_args, "SINDI index can only use SINDI search algorithm");
        }
        if (index_->has_max_scores_per_dim() == false && algo_need_max_scores_per_dim()) {
            return expected<sparse::inverted::InvertedIndexSearchParams>::Err(
                Status::invalid_value_in_json, "search algorithm requires max_score_per_dim");
        }
        if (index_->has_block_max_scores() == false && algo_need_block_max_data()) {
            return expected<sparse::inverted::InvertedIndexSearchParams>::Err(
                Status::invalid_value_in_json, "search algorithm requires block_max_data");
        }

        // prepare and check search metric type
        if (IsMetricType(config.metric_type.value(), metric::BM25)) {
            if (!config.bm25_avgdl.has_value()) {
                return expected<sparse::inverted::InvertedIndexSearchParams>::Err(
                    Status::invalid_args, "BM25 parameter avgdl must be set when searching");
            }
            if (index_->get_scorer_config().scorer_type != sparse::inverted::IndexScorerType::BM25) {
                return expected<sparse::inverted::InvertedIndexSearchParams>::Err(
                    Status::invalid_metric_type, "search metric type must be same as built index");
            }
            search_params.scorer_config.scorer_type = sparse::inverted::IndexScorerType::BM25;
            search_params.scorer_config.scorer_params.bm25.avgdl = std::max(config.bm25_avgdl.value(), 1.0f);
            search_params.scorer_config.scorer_params.bm25.k1 =
                config.bm25_k1.value_or(index_->get_scorer_config().scorer_params.bm25.k1);
            search_params.scorer_config.scorer_params.bm25.b =
                config.bm25_b.value_or(index_->get_scorer_config().scorer_params.bm25.b);
            if (search_params.algo == sparse::inverted::InvertedIndexAlgo::DAAT_WAND ||
                search_params.algo == sparse::inverted::InvertedIndexAlgo::DAAT_MAXSCORE) {
                if (search_params.scorer_config.scorer_params.bm25.k1 !=
                        index_->get_scorer_config().scorer_params.bm25.k1 ||
                    search_params.scorer_config.scorer_params.bm25.b !=
                        index_->get_scorer_config().scorer_params.bm25.b) {
                    return expected<sparse::inverted::InvertedIndexSearchParams>::Err(
                        Status::invalid_value_in_json,
                        "BM25 parameters k1 and b in search config must be same as built index");
                }
            }
        } else if (IsMetricType(config.metric_type.value(), metric::IP)) {
            if (index_->get_scorer_config().scorer_type != sparse::inverted::IndexScorerType::IP) {
                return expected<sparse::inverted::InvertedIndexSearchParams>::Err(
                    Status::invalid_metric_type, "search metric type must be same as built index");
            }
            search_params.scorer_config.scorer_type = sparse::inverted::IndexScorerType::IP;
        } else {
            return expected<sparse::inverted::InvertedIndexSearchParams>::Err(Status::invalid_metric_type,
                                                                              "Unsupported metric type");
        }

        return search_params;
    }

    bool
    version_use_raw_data() const {
        return index_version_ < SPARSE_INDEX_VERSION_USE_RAW_DATA_THRESHOLD;
    }

    bool
    version_support_fp16_quant_for_ip() const {
        return index_version_ >= SPARSE_INDEX_VERSION_SUPPORT_FP16_QUANT_FOR_IP;
    }

    bool
    version_default_to_daat_maxscore() const {
        return index_version_ < 10;
    }

    // used to load index
    BinaryPtr binary_;

    struct MmapGuard {
        size_t map_size;
        std::string filename;
        void* map_addr;

        MmapGuard(size_t size, const std::string& fname, void* addr) : map_size(size), filename(fname), map_addr(addr) {
        }

        ~MmapGuard() {
            if (munmap(map_addr, map_size) != 0) {
                LOG_KNOWHERE_ERROR_ << "Failed to munmap file " << filename << ": " << strerror(errno);
            }
        }
    };

    std::unique_ptr<MmapGuard> mmap_guard_;

    std::shared_ptr<knowhere::ThreadPool> search_pool_;
    std::shared_ptr<knowhere::ThreadPool> build_pool_;
    const int32_t index_version_;
};  // class SparseInvertedIndexNode

// Concurrent version of SparseInvertedIndexNode
//
// Thread safety: only the overridden methods are allowed to be called concurrently.
template <typename T, bool use_wand>
class SparseInvertedIndexNodeCC : public SparseInvertedIndexNode<T, use_wand> {
    using value_type = typename T::ValueType;

 public:
    explicit SparseInvertedIndexNodeCC(const int32_t& version, const Object& object)
        : SparseInvertedIndexNode<T, use_wand>(version, object) {
    }

    Status
    Train(const DataSetPtr dataset, std::shared_ptr<Config> config, bool use_knowhere_build_pool) override {
        auto cfg = static_cast<const SparseInvertedIndexConfig&>(*config);

        // metric type validation
        if (!IsMetricType(cfg.metric_type.value(), metric::IP) &&
            !IsMetricType(cfg.metric_type.value(), metric::BM25)) {
            LOG_KNOWHERE_ERROR_ << Type() << " only support metric_type IP or BM25";
            return Status::invalid_metric_type;
        }

        if (IsMetricType(cfg.metric_type.value(), metric::BM25)) {
            if (!cfg.bm25_k1.has_value() || !cfg.bm25_b.has_value() || !cfg.bm25_avgdl.has_value()) {
                return Status::invalid_args;
            }
        }

        // create index
        auto index_or = this->CreateIndex(cfg, true);

        if (this->index_ != nullptr) {
            LOG_KNOWHERE_WARNING_
                << Type()
                << " index has already been created, Train() will delete the old index and recreate a new one";
        }

        this->index_ = std::move(index_or.value());

        return Status::success;
    }

    Status
    Add(const DataSetPtr dataset, std::shared_ptr<Config> config, bool use_knowhere_build_pool) override {
        std::unique_lock<std::mutex> lock(mutex_);
        uint64_t task_id = next_task_id_++;
        add_tasks_.push(task_id);

        // add task is allowed to run only after all search tasks that come before it have finished.
        cv_.wait(lock, [this, task_id]() { return current_task_id_ == task_id && active_readers_ == 0; });

        auto res = SparseInvertedIndexNode<T, use_wand>::Add(dataset, config, use_knowhere_build_pool);

        auto cfg = static_cast<const SparseInvertedIndexConfig&>(*config);
        if (IsMetricType(cfg.metric_type.value(), metric::IP)) {
            // insert dataset to raw data if metric type is IP
            auto data = static_cast<const sparse::SparseRow<value_type>*>(dataset->GetTensor());
            auto rows = dataset->GetRows();
            raw_data_.insert(raw_data_.end(), data, data + rows);
        }

        add_tasks_.pop();
        current_task_id_++;
        lock.unlock();
        cv_.notify_all();
        return res;
    }

    expected<DataSetPtr>
    Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
           milvus::OpContext* op_context) const override {
        ReadPermission permission(*this);
        return SparseInvertedIndexNode<T, use_wand>::Search(dataset, std::move(cfg), bitset, op_context);
    }

    expected<std::vector<IndexNode::IteratorPtr>>
    AnnIterator(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
                bool use_knowhere_search_pool, milvus::OpContext* op_context) const override {
        ReadPermission permission(*this);
        return SparseInvertedIndexNode<T, use_wand>::AnnIterator(dataset, std::move(cfg), bitset,
                                                                 use_knowhere_search_pool, op_context);
    }

    expected<DataSetPtr>
    RangeSearch(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
                milvus::OpContext* op_context) const override {
        ReadPermission permission(*this);
        return SparseInvertedIndexNode<T, use_wand>::RangeSearch(dataset, std::move(cfg), bitset, op_context);
    }

    int64_t
    Dim() const override {
        ReadPermission permission(*this);
        return SparseInvertedIndexNode<T, use_wand>::Dim();
    }

    int64_t
    Size() const override {
        ReadPermission permission(*this);
        return SparseInvertedIndexNode<T, use_wand>::Size();
    }

    int64_t
    Count() const override {
        ReadPermission permission(*this);
        return SparseInvertedIndexNode<T, use_wand>::Count();
    }

    std::string
    Type() const override {
        return use_wand ? knowhere::IndexEnum::INDEX_SPARSE_WAND_CC
                        : knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX_CC;
    }

    expected<DataSetPtr>
    GetVectorByIds(const DataSetPtr dataset, milvus::OpContext* op_context) const override {
        ReadPermission permission(*this);

        if (raw_data_.empty()) {
            return expected<DataSetPtr>::Err(Status::invalid_args, "GetVectorByIds failed: raw data is empty");
        }

        auto rows = dataset->GetRows();
        auto ids = dataset->GetIds();
        auto data = std::make_unique<sparse::SparseRow<value_type>[]>(rows);
        int64_t dim = 0;

        try {
            for (int64_t i = 0; i < rows; ++i) {
                data[i] = raw_data_[ids[i]];
                dim = std::max(dim, data[i].dim());
            }
        } catch (std::exception& e) {
            return expected<DataSetPtr>::Err(Status::invalid_args, "GetVectorByIds failed: " + std::string(e.what()));
        }

        auto res = GenResultDataSet(rows, dim, data.release());
        res->SetIsSparse(true);

        return res;
    }

    static bool
    StaticHasRawData(const knowhere::BaseConfig& config, const IndexVersion& /*version*/) {
        if (!config.metric_type.has_value()) {
            return false;
        }

        return IsMetricType(config.metric_type.value(), metric::IP);
    }

    [[nodiscard]] bool
    HasRawData(const std::string& metric_type) const override {
        return IsMetricType(metric_type, metric::IP);
    }

    Status
    Serialize(BinarySet& binset) const override {
        return Status::not_implemented;
    }

    Status
    Deserialize(const BinarySet& binset, std::shared_ptr<Config> config) override {
        return Status::not_implemented;
    }

    Status
    DeserializeFromFile(const std::string& filename, std::shared_ptr<Config> config) override {
        return Status::not_implemented;
    }

 private:
    struct ReadPermission {
        ReadPermission(const SparseInvertedIndexNodeCC& node) : node_(node) {
            std::unique_lock<std::mutex> lock(node_.mutex_);
            uint64_t task_id = node_.next_task_id_++;
            // read task may execute only after all add tasks that come before it have finished.
            if (!node_.add_tasks_.empty() && task_id > node_.add_tasks_.front()) {
                node_.cv_.wait(
                    lock, [this, task_id]() { return node_.add_tasks_.empty() || task_id < node_.add_tasks_.front(); });
            }
            // read task is allowed to run, block all add tasks
            node_.active_readers_++;
        }

        ~ReadPermission() {
            std::unique_lock<std::mutex> lock(node_.mutex_);
            node_.active_readers_--;
            node_.current_task_id_++;
            node_.cv_.notify_all();
        }

        const SparseInvertedIndexNodeCC& node_;
    };

    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;
    mutable int64_t active_readers_ = 0;
    mutable std::queue<uint64_t> add_tasks_;
    mutable uint64_t next_task_id_ = 0;
    mutable uint64_t current_task_id_ = 0;
    mutable std::vector<sparse::SparseRow<value_type>> raw_data_ = {};
};  // class SparseInvertedIndexNodeCC

KNOWHERE_SIMPLE_REGISTER_SPARSE_FLOAT_GLOBAL(SPARSE_INVERTED_INDEX, SparseInvertedIndexNode, knowhere::feature::MMAP,
                                             /*use_wand=*/false)
KNOWHERE_SIMPLE_REGISTER_SPARSE_FLOAT_GLOBAL(SPARSE_WAND, SparseInvertedIndexNode, knowhere::feature::MMAP,
                                             /*use_wand=*/true)
KNOWHERE_SIMPLE_REGISTER_SPARSE_FLOAT_GLOBAL(SPARSE_INVERTED_INDEX_CC, SparseInvertedIndexNodeCC,
                                             knowhere::feature::MMAP,
                                             /*use_wand=*/false)
KNOWHERE_SIMPLE_REGISTER_SPARSE_FLOAT_GLOBAL(SPARSE_WAND_CC, SparseInvertedIndexNodeCC, knowhere::feature::MMAP,
                                             /*use_wand=*/true)
}  // namespace knowhere
