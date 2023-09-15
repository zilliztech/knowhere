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

#include "knowhere/feder/HNSW.h"

#include <omp.h>

#include <exception>
#include <new>

#include "common/range_util.h"
#include "hnswlib/hnswalg.h"
#include "hnswlib/hnswlib.h"
#include "index/hnsw/hnsw_config.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/thread_pool.h"
#include "knowhere/comp/time_recorder.h"
#include "knowhere/config.h"
#include "knowhere/expected.h"
#include "knowhere/factory.h"
#include "knowhere/log.h"
#include "knowhere/utils.h"

namespace knowhere {
class HnswIndexNode : public IndexNode {
 public:
    HnswIndexNode(const std::string& /*version*/, const Object& object) : index_(nullptr) {
        search_pool_ = ThreadPool::GetGlobalSearchThreadPool();
    }

    Status
    Train(const DataSet& dataset, const Config& cfg) override {
        auto rows = dataset.GetRows();
        auto dim = dataset.GetDim();
        auto hnsw_cfg = static_cast<const HnswConfig&>(cfg);
        hnswlib::SpaceInterface<float>* space = nullptr;
        if (IsMetricType(hnsw_cfg.metric_type.value(), metric::L2)) {
            space = new (std::nothrow) hnswlib::L2Space(dim);
        } else if (IsMetricType(hnsw_cfg.metric_type.value(), metric::IP)) {
            space = new (std::nothrow) hnswlib::InnerProductSpace(dim);
        } else if (IsMetricType(hnsw_cfg.metric_type.value(), metric::COSINE)) {
            space = new (std::nothrow) hnswlib::CosineSpace(dim);
        } else if (IsMetricType(hnsw_cfg.metric_type.value(), metric::HAMMING)) {
            space = new (std::nothrow) hnswlib::HammingSpace(dim);
        } else if (IsMetricType(hnsw_cfg.metric_type.value(), metric::JACCARD)) {
            space = new (std::nothrow) hnswlib::JaccardSpace(dim);
        } else {
            LOG_KNOWHERE_WARNING_ << "metric type not support in hnsw: " << hnsw_cfg.metric_type.value();
            return Status::invalid_metric_type;
        }
        auto index = new (std::nothrow)
            hnswlib::HierarchicalNSW<float>(space, rows, hnsw_cfg.M.value(), hnsw_cfg.efConstruction.value());
        if (index == nullptr) {
            LOG_KNOWHERE_WARNING_ << "memory malloc error.";
            return Status::malloc_error;
        }
        if (this->index_) {
            delete this->index_;
            LOG_KNOWHERE_WARNING_ << "index not empty, deleted old index";
        }
        this->index_ = index;
        return Status::success;
    }

    Status
    Add(const DataSet& dataset, const Config& cfg) override {
        if (!index_) {
            LOG_KNOWHERE_ERROR_ << "Can not add data to empty HNSW index.";
            expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
        }

        knowhere::TimeRecorder build_time("Building HNSW cost");
        auto rows = dataset.GetRows();
        auto tensor = dataset.GetTensor();
        auto hnsw_cfg = static_cast<const HnswConfig&>(cfg);
        index_->addPoint(tensor, 0);

#pragma omp parallel for
        for (int i = 1; i < rows; ++i) {
            index_->addPoint(((const char*)tensor + index_->data_size_ * i), i);
        }
        build_time.RecordSection("");
        LOG_KNOWHERE_INFO_ << "HNSW built with #points num:" << index_->max_elements_ << " #M:" << index_->M_
                           << " #max level:" << index_->maxlevel_ << " #ef_construction:" << index_->ef_construction_
                           << " #dim:" << *(size_t*)(index_->space_->get_dist_func_param());
        return Status::success;
    }

    expected<DataSetPtr>
    Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
        if (!index_) {
            LOG_KNOWHERE_WARNING_ << "search on empty index";
            expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
        }
        auto nq = dataset.GetRows();
        auto xq = dataset.GetTensor();

        auto hnsw_cfg = static_cast<const HnswConfig&>(cfg);
        auto k = hnsw_cfg.k.value();

        feder::hnsw::FederResultUniq feder_result;
        if (hnsw_cfg.trace_visit.value()) {
            if (nq != 1) {
                return expected<DataSetPtr>::Err(Status::invalid_args, "nq must be 1");
            }
            feder_result = std::make_unique<feder::hnsw::FederResult>();
        }

        auto p_id = new int64_t[k * nq];
        auto p_dist = new float[k * nq];

        hnswlib::SearchParam param{(size_t)hnsw_cfg.ef.value(), hnsw_cfg.for_tuning.value()};
        bool transform =
            (index_->metric_type_ == hnswlib::Metric::INNER_PRODUCT || index_->metric_type_ == hnswlib::Metric::COSINE);

        std::vector<folly::Future<folly::Unit>> futs;
        futs.reserve(nq);
        for (int i = 0; i < nq; ++i) {
            futs.emplace_back(search_pool_->push([&, idx = i]() {
                auto single_query = (const char*)xq + idx * index_->data_size_;
                auto rst = index_->searchKnn(single_query, k, bitset, &param, feder_result);
                size_t rst_size = rst.size();
                auto p_single_dis = p_dist + idx * k;
                auto p_single_id = p_id + idx * k;
                for (size_t idx = 0; idx < rst_size; ++idx) {
                    const auto& [dist, id] = rst[idx];
                    p_single_dis[idx] = transform ? (-dist) : dist;
                    p_single_id[idx] = id;
                }
                for (size_t idx = rst_size; idx < (size_t)k; idx++) {
                    p_single_dis[idx] = float(1.0 / 0.0);
                    p_single_id[idx] = -1;
                }
            }));
        }
        for (auto& fut : futs) {
            fut.wait();
        }

        auto res = GenResultDataSet(nq, k, p_id, p_dist);

        // set visit_info json string into result dataset
        if (feder_result != nullptr) {
            Json json_visit_info, json_id_set;
            nlohmann::to_json(json_visit_info, feder_result->visit_info_);
            nlohmann::to_json(json_id_set, feder_result->id_set_);
            res->SetJsonInfo(json_visit_info.dump());
            res->SetJsonIdSet(json_id_set.dump());
        }
        return res;
    }

 private:
    class iterator : public IndexNode::iterator {
     public:
        iterator(const hnswlib::HierarchicalNSW<float>* index, const char* query, const bool transform,
                 const BitsetView& bitset, const bool for_tuning = false, const size_t seed_ef = kIteratorSeedEf)
            : index_(index),
              transform_(transform),
              workspace_(index_->getIteratorWorkspace(query, seed_ef, for_tuning, bitset)) {
            UpdateNext();
        }

        std::pair<int64_t, float>
        Next() override {
            auto ret = std::make_pair(next_id_, next_dist_);
            UpdateNext();
            return ret;
        }

        [[nodiscard]] bool
        HasNext() const override {
            return has_next_;
        }

     private:
        void
        UpdateNext() {
            auto next = index_->getIteratorNext(workspace_.get());
            if (next.has_value()) {
                auto [dist, id] = next.value();
                next_dist_ = transform_ ? (-dist) : dist;
                next_id_ = id;
                has_next_ = true;
            } else {
                has_next_ = false;
            }
        }
        const hnswlib::HierarchicalNSW<float>* index_;
        const bool transform_;
        std::unique_ptr<hnswlib::IteratorWorkspace> workspace_;
        bool has_next_;
        float next_dist_;
        int64_t next_id_;
    };

 public:
    expected<std::vector<std::shared_ptr<IndexNode::iterator>>>
    AnnIterator(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
        if (!index_) {
            LOG_KNOWHERE_WARNING_ << "creating iterator on empty index";
            expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
        }
        auto nq = dataset.GetRows();
        auto xq = dataset.GetTensor();

        auto hnsw_cfg = static_cast<const HnswConfig&>(cfg);

        bool transform =
            (index_->metric_type_ == hnswlib::Metric::INNER_PRODUCT || index_->metric_type_ == hnswlib::Metric::COSINE);
        auto vec = std::vector<std::shared_ptr<IndexNode::iterator>>(nq, nullptr);
        std::vector<folly::Future<folly::Unit>> futs;
        futs.reserve(nq);
        for (int i = 0; i < nq; ++i) {
            futs.emplace_back(search_pool_->push([&, i]() {
                auto single_query = (const char*)xq + i * index_->data_size_;
                vec[i].reset(new iterator(this->index_, single_query, transform, bitset, hnsw_cfg.for_tuning.value(),
                                          hnsw_cfg.seed_ef.value()));
            }));
        }
        // wait for initial search(in top layers and search for seed_ef in base layer) to finish
        for (auto& fut : futs) {
            fut.wait();
        }

        return vec;
    }

    expected<DataSetPtr>
    RangeSearch(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
        if (!index_) {
            LOG_KNOWHERE_WARNING_ << "range search on empty index";
            return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
        }

        auto nq = dataset.GetRows();
        auto xq = dataset.GetTensor();

        auto hnsw_cfg = static_cast<const HnswConfig&>(cfg);
        bool is_ip =
            (index_->metric_type_ == hnswlib::Metric::INNER_PRODUCT || index_->metric_type_ == hnswlib::Metric::COSINE);
        float range_filter = hnsw_cfg.range_filter.value();

        float radius_for_calc = (is_ip ? -hnsw_cfg.radius.value() : hnsw_cfg.radius.value());
        float radius_for_filter = hnsw_cfg.radius.value();

        feder::hnsw::FederResultUniq feder_result;
        if (hnsw_cfg.trace_visit.value()) {
            if (nq != 1) {
                return expected<DataSetPtr>::Err(Status::invalid_args, "nq must be 1");
            }
            feder_result = std::make_unique<feder::hnsw::FederResult>();
        }

        hnswlib::SearchParam param{(size_t)hnsw_cfg.ef.value()};

        int64_t* ids = nullptr;
        float* dis = nullptr;
        size_t* lims = nullptr;

        std::vector<std::vector<int64_t>> result_id_array(nq);
        std::vector<std::vector<float>> result_dist_array(nq);
        std::vector<size_t> result_size(nq);
        std::vector<size_t> result_lims(nq + 1);

        std::vector<folly::Future<folly::Unit>> futs;
        futs.reserve(nq);
        for (int64_t i = 0; i < nq; ++i) {
            futs.emplace_back(search_pool_->push([&, idx = i]() {
                auto single_query = (const char*)xq + idx * index_->data_size_;
                auto rst = index_->searchRange(single_query, radius_for_calc, bitset, &param, feder_result);
                auto elem_cnt = rst.size();
                result_dist_array[idx].resize(elem_cnt);
                result_id_array[idx].resize(elem_cnt);
                for (size_t j = 0; j < elem_cnt; j++) {
                    auto& p = rst[j];
                    result_dist_array[idx][j] = (is_ip ? (-p.first) : p.first);
                    result_id_array[idx][j] = p.second;
                }
                result_size[idx] = rst.size();
                if (hnsw_cfg.range_filter.value() != defaultRangeFilter) {
                    FilterRangeSearchResultForOneNq(result_dist_array[idx], result_id_array[idx], is_ip,
                                                    radius_for_filter, range_filter);
                }
            }));
        }
        for (auto& fut : futs) {
            fut.wait();
        }

        // filter range search result
        GetRangeSearchResult(result_dist_array, result_id_array, is_ip, nq, radius_for_filter, range_filter, dis, ids,
                             lims);

        auto res = GenResultDataSet(nq, ids, dis, lims);

        // set visit_info json string into result dataset
        if (feder_result != nullptr) {
            Json json_visit_info, json_id_set;
            nlohmann::to_json(json_visit_info, feder_result->visit_info_);
            nlohmann::to_json(json_id_set, feder_result->id_set_);
            res->SetJsonInfo(json_visit_info.dump());
            res->SetJsonIdSet(json_id_set.dump());
        }
        return res;
    }

    expected<DataSetPtr>
    GetVectorByIds(const DataSet& dataset) const override {
        if (!index_) {
            return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
        }

        auto dim = Dim();
        auto rows = dataset.GetRows();
        auto ids = dataset.GetIds();

        char* data = nullptr;
        try {
            data = new char[index_->data_size_ * rows];
            for (int64_t i = 0; i < rows; i++) {
                int64_t id = ids[i];
                assert(id >= 0 && id < (int64_t)index_->cur_element_count);
                std::copy_n(index_->getDataByInternalId(id), index_->data_size_, data + i * index_->data_size_);
            }
            return GenResultDataSet(rows, dim, data);
        } catch (std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "hnsw inner error: " << e.what();
            std::unique_ptr<char> auto_del(data);
            return expected<DataSetPtr>::Err(Status::hnsw_inner_error, e.what());
        }
    }

    bool
    HasRawData(const std::string& metric_type) const override {
        return true;
    }

    expected<DataSetPtr>
    GetIndexMeta(const Config& cfg) const override {
        if (!index_) {
            LOG_KNOWHERE_WARNING_ << "get index meta on empty index";
            return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
        }

        auto hnsw_cfg = static_cast<const HnswConfig&>(cfg);
        auto overview_levels = hnsw_cfg.overview_levels.value();
        feder::hnsw::HNSWMeta meta(index_->ef_construction_, index_->M_, index_->cur_element_count, index_->maxlevel_,
                                   index_->enterpoint_node_, overview_levels);
        std::unordered_set<int64_t> id_set;

        for (int i = 0; i < overview_levels; i++) {
            int64_t level = index_->maxlevel_ - i;
            // do not record level 0
            if (level <= 0) {
                break;
            }
            meta.AddLevelLinkGraph(level);
            UpdateLevelLinkList(level, meta, id_set);
        }

        Json json_meta, json_id_set;
        nlohmann::to_json(json_meta, meta);
        nlohmann::to_json(json_id_set, id_set);
        return GenResultDataSet(json_meta.dump(), json_id_set.dump());
    }

    Status
    Serialize(BinarySet& binset) const override {
        if (!index_) {
            LOG_KNOWHERE_ERROR_ << "Can not serialize empty HNSW index.";
            return Status::empty_index;
        }
        try {
            MemoryIOWriter writer;
            index_->saveIndex(writer);
            std::shared_ptr<uint8_t[]> data(writer.data());
            binset.Append(Type(), data, writer.tellg());
        } catch (std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "hnsw inner error: " << e.what();
            return Status::hnsw_inner_error;
        }
        return Status::success;
    }

    Status
    Deserialize(const BinarySet& binset, const Config& config) override {
        if (index_) {
            delete index_;
        }
        try {
            auto binary = binset.GetByName(Type());
            if (binary == nullptr) {
                LOG_KNOWHERE_ERROR_ << "Invalid binary set.";
                return Status::invalid_binary_set;
            }

            MemoryIOReader reader(binary->data.get(), binary->size);

            hnswlib::SpaceInterface<float>* space = nullptr;
            index_ = new (std::nothrow) hnswlib::HierarchicalNSW<float>(space);
            index_->loadIndex(reader);
            LOG_KNOWHERE_INFO_ << "Loaded HNSW index. #points num:" << index_->max_elements_ << " #M:" << index_->M_
                               << " #max level:" << index_->maxlevel_
                               << " #ef_construction:" << index_->ef_construction_
                               << " #dim:" << *(size_t*)(index_->space_->get_dist_func_param());
        } catch (std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "hnsw inner error: " << e.what();
            return Status::hnsw_inner_error;
        }
        return Status::success;
    }

    Status
    DeserializeFromFile(const std::string& filename, const Config& config) override {
        if (index_) {
            delete index_;
        }
        try {
            hnswlib::SpaceInterface<float>* space = nullptr;
            index_ = new (std::nothrow) hnswlib::HierarchicalNSW<float>(space);
            index_->loadIndex(filename, config);
        } catch (std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "hnsw inner error: " << e.what();
            return Status::hnsw_inner_error;
        }
        return Status::success;
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return std::make_unique<HnswConfig>();
    }

    int64_t
    Dim() const override {
        if (!index_) {
            return 0;
        }
        return (*static_cast<size_t*>(index_->dist_func_param_));
    }

    int64_t
    Size() const override {
        if (!index_) {
            return 0;
        }
        return index_->cal_size();
    }

    int64_t
    Count() const override {
        if (!index_) {
            return 0;
        }
        return index_->cur_element_count;
    }

    std::string
    Type() const override {
        return knowhere::IndexEnum::INDEX_HNSW;
    }

    ~HnswIndexNode() override {
        if (index_) {
            delete index_;
        }
    }

 private:
    void
    UpdateLevelLinkList(int32_t level, feder::hnsw::HNSWMeta& meta, std::unordered_set<int64_t>& id_set) const {
        if (!(level > 0 && level <= index_->maxlevel_)) {
            return;
        }
        if (index_->cur_element_count == 0) {
            return;
        }

        std::vector<hnswlib::tableint> level_elements;

        // get all elements in current level
        for (size_t i = 0; i < index_->cur_element_count; i++) {
            // elements in high level also exist in low level
            if (index_->element_levels_[i] >= level) {
                level_elements.emplace_back(i);
            }
        }

        // iterate all elements in current level, record their link lists
        for (auto curr_id : level_elements) {
            auto data = index_->get_linklist(curr_id, level);
            auto size = index_->getListCount(data);

            hnswlib::tableint* datal = (hnswlib::tableint*)(data + 1);
            std::vector<int64_t> neighbors(size);
            for (int i = 0; i < size; i++) {
                hnswlib::tableint cand = datal[i];
                neighbors[i] = cand;
            }
            id_set.insert(curr_id);
            id_set.insert(neighbors.begin(), neighbors.end());
            meta.AddNodeInfo(level, curr_id, std::move(neighbors));
        }
    }

 private:
    hnswlib::HierarchicalNSW<float>* index_;
    std::shared_ptr<ThreadPool> search_pool_;
};

KNOWHERE_REGISTER_GLOBAL(HNSW, [](const std::string& version, const Object& object) {
    return Index<HnswIndexNode>::Create(version, object);
});

}  // namespace knowhere
