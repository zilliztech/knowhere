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

#include "faiss/impl/mapped_io.h"
#include "faiss/index_io.h"
#include "index/emb_list/emb_list_config.h"
#include "index/hnsw/base_hnsw_config.h"
#include "index/ivf/ivf_config.h"
#include "knowhere/config.h"
#include "knowhere/dataset.h"
#include "knowhere/emb_list_utils.h"
#include "knowhere/expected.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/index/index_node.h"
#include "knowhere/index/index_node_data_mock_wrapper.h"
#include "knowhere/log.h"
#include "knowhere/object.h"
#include "knowhere/operands.h"
#include "knowhere/utils.h"

namespace knowhere {

/**
 * @brief EmbListHNSWIndexNode: An embedding-list-based index node utilizing HNSW as the underlying ANN engine.
 *
 * This class is intended for scenarios where vectors are organized into embedding lists (emb_lists).
 * Both the build and search operations are performed at the emb_list level, rather than on individual vectors.
 * The interface remains consistent with other vector-based IndexNode implementations, using DataSetPtr as input.
 * However, an additional Lims array is used to indicate the boundaries of each emb_list within both the train and
 * query dataset.
 *
 * For example, if lims = [0, 5, 9, 100], this represents 3 emb_lists:
 *   - The first emb_list starts at 0 and contains 5 vectors (indices 0-4)
 *   - The second emb_list starts at 5 and contains 4 vectors (indices 5-8)
 *   - The third emb_list starts at 9 and contains 91 vectors (indices 9-99)
 *
 * The class manages the mapping between vector IDs and emb_list IDs, and handles
 * serialization/deserialization for both the emb_list structure and the underlying HNSW index.
 *
 * In Milvus, all vectors within the same emb_list share the same scalar value, and correspond to the same bit in the
 * bitset. Therefore, the bitset length equals the number of emb_lists, which may be much smaller than the total number
 * of vectors.
 *
 * Note: In the future, EmbListIndexNode should support not only HNSW, but also other vector-based indexes such as IVF,
 * DiskANN, etc.
 *
 * @tparam DataType The data type of the vectors (e.g., fp32, fp16, bf16).
 */
template <typename DataType>
class EmbListHNSWIndexNode : public IndexNode {
 public:
    EmbListHNSWIndexNode(const int32_t& version, const Object& object) : IndexNode(version) {
        build_pool_ = ThreadPool::GetGlobalBuildThreadPool();
        search_pool_ = ThreadPool::GetGlobalSearchThreadPool();
        base_index_ = std::move(IndexFactory::Instance().Create<DataType>("HNSW", version, object).value());
        data_format_ = datatype_v<DataType>;
    }

    /**
     * @brief Train the emb_list_hnsw index.
     * @param dataset Training dataset
     * @param cfg     Configuration parameters
     * @param use_knowhere_build_pool Whether to use the global build thread pool
     *
     * Main steps:
     * 1. Check that the dataset contains emb_list offset information (group boundaries).
     * 2. Force the underlying HNSW to use IP as the metric.
     * 3. Build the EmbListOffset structure (group offsets).
     * 4. Call the underlying HNSW's Train.
     */
    Status
    Train(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override {
        LOG_KNOWHERE_INFO_ << "train";
        const size_t* lims = dataset->GetLims();
        if (lims == nullptr) {
            LOG_KNOWHERE_WARNING_ << "Missing emb_list offset, could not train index";
            return Status::emb_list_inner_error;
        }
        auto& hnsw_config = static_cast<BaseConfig&>(*cfg);
        auto sub_metric_type_or = get_sub_metric_type(hnsw_config.metric_type.value());
        if (!sub_metric_type_or.has_value()) {
            LOG_KNOWHERE_WARNING_ << "Invalid metric type: " << hnsw_config.metric_type.value();
            return Status::emb_list_inner_error;
        }
        hnsw_config.metric_type = sub_metric_type_or.value();
        emb_list_offset_ = std::make_unique<EmbListOffset>(lims, static_cast<size_t>(dataset->GetRows()));
        return base_index_.Node()->Train(dataset, cfg, use_knowhere_build_pool);
    }

    Status
    Add(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override {
        LOG_KNOWHERE_INFO_ << "add";
        auto base_index_add_res = base_index_.Node()->Add(dataset, cfg, use_knowhere_build_pool);
        if (base_index_add_res != Status::success) {
            LOG_KNOWHERE_WARNING_ << "base index add failed";
            return base_index_add_res;
        }
        // after Add, we need to set the mapping from internal IDs to emb_list IDs.
        SetBaseIndexIDMap();
        return Status::success;
    }

    /**
     * @brief Establishes the mapping from internal HNSW IDs to emb_list IDs.
     *
     * This mapping is essential for sub indexes to correctly apply bitset filtering using only a 1-hop mapping during
     * search. In some cases, such as with mv-only *relayout*, a sub-HNSW index may have its own
     * (base)internal-to-external ID mapping.
     * However, the emb_list search bitset operates on emb_list IDs, which we refer to as the "most external" IDs.
     * Therefore, we need to create a mapping from the base_internal_id (used by the sub-HNSW) to the most external
     * emb_list_id, ensuring that bitset checks and search results are consistent at the emb_list level.
     */
    void
    SetBaseIndexIDMap() {
        LOG_KNOWHERE_INFO_ << "set base index id map";
        auto base_internal_id_to_external_id_map = base_index_.Node()->GetInternalIdToExternalIdMap();
        size_t base_id_map_size = base_internal_id_to_external_id_map->size();
        assert(base_id_map_size == base_index_.Node()->Count());
        std::vector<uint32_t> base_internal_id_to_most_internal_id_map(base_id_map_size);
        for (size_t i = 0; i < base_id_map_size; i++) {
            base_internal_id_to_most_internal_id_map[i] =
                emb_list_offset_->get_el_id(base_internal_id_to_external_id_map->at(i));
        }
        base_index_.Node()->SetInternalIdToMostExternalIdMap(std::move(base_internal_id_to_most_internal_id_map));
        LOG_KNOWHERE_INFO_ << "set base index id map success";
    }

    /**
     * @brief Search interface supporting two-stage emb_list search.
     * @param dataset Query dataset
     * @param cfg     Search configuration
     * @param bitset  Mask for filtering vectors
     *
     * Search process:
     * 1. Check emb_list offset information and build the query group structure.
     * 2. Stage 1: Call underlying HNSW to retrieve candidate vector IDs for each query emb_list.
     * 3. Stage 2: For each emb_list, collect candidate emb_list IDs, and for each emb_list, perform brute-force
     *    distance calculation to aggregate scores at the emb_list level.
     * 4. Return top-k emb_list results.
     *
     * Note: The emb_list index node does not need to split tasks by nq and dispatch them to the search thread pool for
     * parallel processing, because the sub-index_node already handles this.
     */
    expected<DataSetPtr>
    Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset) const override {
        auto dim = dataset->GetDim();
        const size_t* lims = dataset->GetLims();
        if (lims == nullptr) {
            return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "missing emb_list offset, could not search");
        }
        auto num_q_vecs = static_cast<size_t>(dataset->GetRows());
        EmbListOffset query_emb_list_offset(lims, num_q_vecs);
        auto num_q_el = query_emb_list_offset.num_el();
        auto& config = static_cast<EmbListHNSWConfig&>(*cfg);
        auto metric_type = config.metric_type.value();
        auto el_metric_type_or = get_el_metric_type(metric_type);
        if (!el_metric_type_or.has_value()) {
            LOG_KNOWHERE_WARNING_ << "Invalid metric type: " << metric_type;
            return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "invalid metric type");
        }
        auto el_metric_type = el_metric_type_or.value();
        auto el_agg_func_or = get_emb_list_agg_func(el_metric_type);
        if (!el_agg_func_or.has_value()) {
            LOG_KNOWHERE_ERROR_ << "Invalid emb list aggeration function for metric type: " << el_metric_type;
            return expected<DataSetPtr>::Err(Status::emb_list_inner_error,
                                             "invalid emb list aggeration function for metric type: " + el_metric_type);
        }
        auto el_agg_func = el_agg_func_or.value();

        auto sub_metric_type_or = get_sub_metric_type(metric_type);
        if (!sub_metric_type_or.has_value()) {
            LOG_KNOWHERE_WARNING_ << "Invalid metric type: " << metric_type;
            return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "invalid metric type");
        }
        auto sub_metric_type = sub_metric_type_or.value();
        LOG_KNOWHERE_DEBUG_ << "sub metric_type: " << sub_metric_type;
        auto el_k = config.k.value();

        // Allocate result arrays
        auto ids = std::make_unique<int64_t[]>(num_q_el * el_k);
        auto dists = std::make_unique<float[]>(num_q_el * el_k);

        // Stage 1: HNSW search - top k' = k * retrieval_ann_ratio
        config.metric_type = sub_metric_type;
        int32_t vec_topk = std::max((int32_t)(el_k * config.retrieval_ann_ratio.value()), 1);
        config.k = vec_topk;
        auto ann_search_res = base_index_.Node()->Search(dataset, std::move(cfg), bitset).value();
        // Get vector IDs from stage 1
        const auto stage1_ids = ann_search_res->GetIds();

        // For each query emb_list, perform stage 2 aggregation
        for (size_t i = 0; i < num_q_el; i++) {
            auto start_offset = query_emb_list_offset.offset[i];
            auto end_offset = query_emb_list_offset.offset[i + 1];
            auto nq = end_offset - start_offset;

            // Collect unique emb_list IDs hit in stage 1
            std::set<size_t> el_ids_set;
            for (size_t j = start_offset * vec_topk; j < end_offset * vec_topk; j++) {
                if (stage1_ids[j] < 0) {
                    continue;
                }
                el_ids_set.emplace(emb_list_offset_->get_el_id((size_t)stage1_ids[j]));
            }

            // For each emb_list, perform brute-force calculation and aggregate scores
            std::priority_queue<DistId, std::vector<DistId>, std::greater<>> minheap;
            for (const auto& el_id : el_ids_set) {
                if (el_id >= emb_list_offset_->num_el()) {
                    LOG_KNOWHERE_ERROR_ << "Invalid el_id: " << el_id;
                    return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "invalid emb_list id");
                }
                auto vids = emb_list_offset_->get_vids(el_id);
                // Generate query dataset for the current query_emb_list
                auto tensor = (const char*)dataset->GetTensor();
                size_t tensor_offset;
                if (data_format_ == DataFormatEnum::fp32) {
                    tensor_offset = start_offset * dim * 4;
                } else if (data_format_ == DataFormatEnum::fp16) {
                    tensor_offset = start_offset * dim * 2;
                } else if (data_format_ == DataFormatEnum::bf16) {
                    tensor_offset = start_offset * dim * 2;
                } else if (data_format_ == DataFormatEnum::int8) {
                    tensor_offset = start_offset * dim;
                } else {
                    LOG_KNOWHERE_ERROR_ << "Unsupported data format";
                    return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "unsupported data format");
                }
                // Brute-force compute distances between all vectors in the query emb_list and all vectors in the
                // candidate emb_list
                auto bf_query_dataset = GenDataSet(end_offset - start_offset, dim, tensor + tensor_offset);
                auto bf_search_res =
                    base_index_.Node()->CalcDistByIDs(bf_query_dataset, bitset, vids.data(), vids.size());
                if (!bf_search_res.has_value()) {
                    return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "bf search error");
                }
                const auto bf_dists = bf_search_res.value()->GetDistance();

                // Aggregate score for the emb_list (e.g., sum of max similarities)
                auto score_or = el_agg_func(bf_dists, nq, vids.size());
                if (!score_or.has_value()) {
                    LOG_KNOWHERE_WARNING_ << "get_sum_max_sim failed, nq: " << nq << ", vids.size(): " << vids.size();
                    return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "get_sum_max_sim failed");
                }
                auto score = score_or.value();
                if (minheap.size() < (size_t)el_k) {
                    minheap.emplace((int64_t)el_id, score);
                } else {
                    if (score > minheap.top().val) {
                        minheap.pop();
                        minheap.emplace((int64_t)el_id, score);
                    }
                }
            }
            // Write results in descending order of score
            auto real_el_k = minheap.size();
            for (size_t j = 0; j < real_el_k; j++) {
                auto& a = minheap.top();
                ids[i * el_k + real_el_k - j - 1] = a.id;
                dists[i * el_k + real_el_k - j - 1] = a.val;
                minheap.pop();
            }
            // Fill remaining slots with -1 and minimum score if not enough results
            std::fill(ids.get() + i * el_k + real_el_k, ids.get() + i * el_k + el_k, -1);
            std::fill(dists.get() + i * el_k + real_el_k, dists.get() + i * el_k + el_k,
                      std::numeric_limits<float>::min());
        }

        return GenResultDataSet((int64_t)num_q_el, (int64_t)el_k, std::move(ids), std::move(dists));
    }

    // TODO: support AnnIterator
    expected<std::vector<IteratorPtr>>
    AnnIterator(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
                bool use_knowhere_search_pool) const override {
        return expected<std::vector<IteratorPtr>>::Err(Status::not_implemented,
                                                       "AnnIterator not supported for emb_list based index");
    }

    expected<DataSetPtr>
    GetVectorByIds(const DataSetPtr dataset) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented,
                                         "GetVectorByIds not supported for emb_list based index");
    }

    bool
    HasRawData(const std::string& metric_type) const override {
        return base_index_.Node()->HasRawData(metric_type);
    }

    expected<DataSetPtr>
    GetIndexMeta(std::unique_ptr<Config> cfg) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented, "GetIndexMeta not implemented");
    }

    Status
    Serialize(BinarySet& binset) const override {
        try {
            int64_t size = emb_list_offset_->offset.size() * sizeof(size_t);
            auto data = std::shared_ptr<uint8_t[]>(new uint8_t[size]);
            std::memcpy(data.get(), emb_list_offset_->offset.data(), size);
            binset.Append(Type(), data, size);
            return base_index_.Node()->Serialize(binset);
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "error inner: " << e.what();
            return Status::emb_list_inner_error;
        }
    }

    Status
    Deserialize(const BinarySet& binset, std::shared_ptr<Config> config) override {
        try {
            auto binary_ptr = binset.GetByName(Type());
            std::shared_ptr<uint8_t[]> data = binary_ptr->data;
            const auto size = binary_ptr->size;
            std::vector<size_t> offset(size / sizeof(size_t));
            std::memcpy(offset.data(), data.get(), size);
            emb_list_offset_ = std::make_unique<EmbListOffset>(std::move(offset));
            auto base_index_deserialize_res = base_index_.Node()->Deserialize(binset, config);
            if (base_index_deserialize_res != Status::success) {
                return base_index_deserialize_res;
            }
            SetBaseIndexIDMap();
            return Status::success;
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "error inner: " << e.what();
            return Status::emb_list_inner_error;
        }
    }

    Status
    DeserializeFromFile(const std::string& filename, std::shared_ptr<Config> config) override {
        auto cfg = static_cast<const knowhere::BaseConfig&>(*config);

        int io_flags = 0;
        if (cfg.enable_mmap.value()) {
            io_flags |= faiss::IO_FLAG_MMAP_IFC;
        }

        try {
            if ((io_flags & faiss::IO_FLAG_MMAP_IFC) == faiss::IO_FLAG_MMAP_IFC) {
                auto owner = std::make_shared<faiss::MmappedFileMappingOwner>(filename.data());
                faiss::MappedFileIOReader reader(owner);
                // Read emb_list offset
                std::vector<uint32_t> offset;
                faiss::read_vector(offset, &reader);
                emb_list_offset_ = std::make_unique<EmbListOffset>(offset);
            } else {
                faiss::FileIOReader reader(filename.data());
                // Read emb_list offset
                std::vector<uint32_t> offset;
                faiss::read_vector(offset, &reader);
                emb_list_offset_ = std::make_unique<EmbListOffset>(offset);
            }
            // Read underlying index
            auto base_index_deserialize_file_res = base_index_.Node()->DeserializeFromFile(filename, config);
            if (base_index_deserialize_file_res != Status::success) {
                return base_index_deserialize_file_res;
            }
            SetBaseIndexIDMap();
            return Status::success;
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "error inner: " << e.what();
            return Status::emb_list_inner_error;
        }
    }

    static std::unique_ptr<BaseConfig>
    StaticCreateConfig() {
        return std::make_unique<EmbListHNSWConfig>();
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return StaticCreateConfig();
    }

    int64_t
    Dim() const override {
        return base_index_.Node()->Dim();
    }

    int64_t
    Size() const override {
        return base_index_.Node()->Size() + emb_list_offset_->offset.size() * sizeof(size_t);
    }

    /**
     * @brief Get the number of vectors in the index.
     * note that this is the number of vectors in the base index, not the number of the emb_list.
     */
    int64_t
    Count() const override {
        return base_index_.Node()->Count();
    }

    std::string
    Type() const override {
        return knowhere::IndexEnum::INDEX_EMB_LIST_HNSW;
    }

 protected:
    Index<IndexNode> base_index_;                     ///< Underlying HNSW index node
    std::unique_ptr<EmbListOffset> emb_list_offset_;  ///< emb_list group offset structure
    std::shared_ptr<ThreadPool> build_pool_;
    std::shared_ptr<ThreadPool> search_pool_;
    DataFormatEnum data_format_;  ///< Data format (e.g., fp32, fp16, bf16)
};

KNOWHERE_SIMPLE_REGISTER_DENSE_FLOAT_ALL_GLOBAL(EMB_LIST_HNSW, EmbListHNSWIndexNode,
                                                knowhere::feature::MMAP | knowhere::feature::MV |
                                                    knowhere::feature::EMB_LIST)
KNOWHERE_SIMPLE_REGISTER_DENSE_INT_GLOBAL(EMB_LIST_HNSW, EmbListHNSWIndexNode,
                                          knowhere::feature::MMAP | knowhere::feature::MV | knowhere::feature::EMB_LIST)
}  // namespace knowhere
