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

#include <queue>

#include "customIdx_config.h"
#include "knowhere/index.h"
#include "knowhere/index_node.h"
#include "knowhere/log.h"

namespace demo {

namespace utils {
struct Candidate {
    int64_t id;
    float dist;
    Candidate(int64_t id, float dist) : id(id), dist(dist) {
    }
};

bool
MinFirst(Candidate a, Candidate b) {
    return a.dist < b.dist;
}

bool
MaxFirst(Candidate a, Candidate b) {
    return a.dist > b.dist;
}

auto
get_metric_dist_comparator(knowhere::MetricType metricType) {
    if (metricType == knowhere::metric::L2) {
        return MinFirst;
    } else if (metricType == knowhere::metric::COSINE || metricType == knowhere::metric::IP) {
        return MaxFirst;
    } else {
        LOG_KNOWHERE_ERROR_ << "Only support {L2, IP, COSINE} metric. " << metricType << " does not support";
        throw knowhere::KnowhereException("Unsupported Metric");
    }
}

struct CandidateBoundedQueue {
    explicit CandidateBoundedQueue(size_t limit, knowhere::MetricType metric) : limit(limit) {
        heap.reserve(limit);
        comparator = get_metric_dist_comparator(metric);
    }

    bool
    push(Candidate candidate) {
        if (full()) {
            if (comparator(candidate, heap[0])) {
                while (full()) {
                    heap_pop();
                }
                heap_push(candidate);
                return true;
            } else {
                return false;
            }
        } else {
            heap_push(candidate);
            return true;
        }
    }

    bool
    pop() {
        if (heap.size() > 0) {
            heap_pop();
            return true;
        }
        return false;
    }

    const Candidate&
    top() const {
        return heap[0];
    }

    const Candidate&
    operator[](size_t index) const {
        return heap[index];
    }

    bool
    full() const {
        return heap.size() >= limit;
    }

    size_t
    size() const {
        return heap.size();
    }

 protected:
    void
    heap_push(const Candidate& candidate) {
        heap.push_back(candidate);
        int idx = heap.size() - 1;

        while (idx > 0) {
            int parent = (idx - 1) / 2;
            if (comparator(heap[parent], heap[idx])) {
                std::swap(heap[parent], heap[idx]);
                idx = parent;
            } else {
                break;
            }
        }
    }

    void
    heap_pop() {
        std::swap(heap[0], heap[heap.size() - 1]);
        heap.pop_back();

        size_t cur = 0;
        while (true) {
            size_t left = 2 * cur + 1;
            size_t right = 2 * cur + 2;
            size_t target_idx = cur;

            if (left < heap.size() && comparator(heap[target_idx], heap[left])) {
                target_idx = left;
            }
            if (right < heap.size() && comparator(heap[target_idx], heap[right])) {
                target_idx = right;
            }
            if (target_idx != cur) {
                std::swap(heap[target_idx], heap[cur]);
                cur = target_idx;
            } else {
                break;
            }
        }
    }

    size_t limit;
    bool (*comparator)(Candidate, Candidate);
    std::vector<Candidate> heap;
};

}  // namespace utils

// Hint: calculator can be accelerated by specific instruction (simd)
namespace calculator {
template <typename VecT>
inline float
vector_norm(const VecT* x, int32_t dim) {
    float x_norm_square = 0.0;
    for (int32_t i = 0; i < dim; i++) {
        x_norm_square += float(x[i]) * float(x[i]);
    }
    return sqrt(x_norm_square);
}

template <typename VecT>
inline float
l2_distance(const VecT* x, const VecT* y, int32_t dim) {
    float ret = 0.0;
    for (int32_t i = 0; i < dim; i++) {
        float diff = float(x[i]) - float(y[i]);
        ret += diff * diff;
    }
    return ret;
}

template <typename VecT>
inline float
ip_distance(const VecT* x, const VecT* y, int32_t dim) {
    float ret = 0.0;
    for (int32_t i = 0; i < dim; i++) {
        float dot = float(x[i]) * float(y[i]);
        ret += dot;
    }
    return ret;
}

template <typename VecT>
inline float
cosine_distance(const VecT* x, const VecT* y, int32_t dim) {
    float x_norm = vector_norm(x, dim);
    float y_norm = vector_norm(y, dim);

    float x_y_inner = ip_distance(x, y, dim);
    if (std::abs(x_norm) > std::numeric_limits<float>::epsilon() &&
        std::abs(y_norm) > std::numeric_limits<float>::epsilon()) {
        return x_y_inner / (x_norm * y_norm);
    }
    return 0.0;
}

template <typename VecT>
auto
get_distance_calculator(knowhere::MetricType metricType) {
    if (metricType == knowhere::metric::L2) {
        return l2_distance<VecT>;
    } else if (metricType == knowhere::metric::COSINE) {
        return cosine_distance<VecT>;
    } else if (metricType == knowhere::metric::IP) {
        return ip_distance<VecT>;
    } else {
        LOG_KNOWHERE_ERROR_ << "Only support {L2, IP, COSINE} metric. " << metricType << " does not support";
        throw knowhere::KnowhereException("Unsupported Metric");
    }
}

bool
l2_distance_range_checker(const float dist, const float radius, const float range_filter) {
    return range_filter <= dist && dist < radius;
}

bool
ip_distance_range_checker(const float dist, const float radius, const float range_filter) {
    return radius < dist && dist <= range_filter;
}

bool
cosine_distance_range_checker(const float dist, const float radius, const float range_filter) {
    return radius < dist && dist <= range_filter;
}

auto
get_distance_range_checker(knowhere::MetricType metricType) {
    if (metricType == knowhere::metric::L2) {
        return l2_distance_range_checker;
    } else if (metricType == knowhere::metric::COSINE) {
        return cosine_distance_range_checker;
    } else if (metricType == knowhere::metric::IP) {
        return ip_distance_range_checker;
    } else {
        LOG_KNOWHERE_ERROR_ << "Only support {L2, IP, COSINE} metric. " << metricType << " does not support";
        throw knowhere::KnowhereException("Unsupported Metric");
    }
}

}  // namespace calculator

template <typename VecT>
class CustomIdx : public knowhere::IndexNode {
 public:
    /**
     * [[must be implemented]]
     * custom index construction method
     * @param version : index version for compatibility
     * @param object : some extensible parameters
     */
    CustomIdx(const int32_t& version, const Object& object) : IndexNode(version) {
    }

    /**
     * [[must be implemented]]
     * extract features from input data (for example : clustering information)
     * @param dataset : training data
     * @param cfg : train config
     * @return : success if no exception
     */
    knowhere::Status
    Train(const knowhere::DataSet& dataset, const knowhere::Config& cfg) override {
        // no acutal training process for current index
        return knowhere::Status::success;
    }

    /**
     * [[must be implemented]]
     * Construct an index based on vector data \\
     * (through clustering, tree or graph structure to link the similar vectors )
     * @param dataset data need to be indexed
     * @param cfg index building config
     * @return success if no exception
     */
    knowhere::Status
    Add(const knowhere::DataSet& dataset, const knowhere::Config& cfg) override {
        auto x = dataset.GetTensor();
        auto row = dataset.GetRows();
        auto dim = dataset.GetDim();

        // construct a flat index
        data_ = std::make_unique<VecT[]>(row * dim);
        std::memcpy(data_.get(), x, row * dim * sizeof(VecT));

        // hold the metadata required for the index
        row_ = row;
        dim_ = dim;
        return knowhere::Status::success;
    }

    /**
     *[[must be implemented]]
     * return  top_k results that meet the filtering condition
     * @param dataset: query vector dataset
     * @param cfg: index search config
     * @param bitset: 0-1 bitmask indicating vectors should be filtered in the final result.
     * 1 indicates the vector should be filtered
     * @return : knn search result
     */
    knowhere::expected<knowhere::DataSetPtr>
    Search(const knowhere::DataSet& dataset, const knowhere::Config& cfg,
           const knowhere::BitsetView& bitset) const override {
        const CustomIdxConfig& index_cfg = static_cast<const demo::CustomIdxConfig&>(cfg);
        auto nq = dataset.GetRows();
        auto k = index_cfg.k.value();
        try {
            auto ret_ids = std::make_unique<int64_t[]>(k * nq);
            auto ret_distances = std::make_unique<float[]>(k * nq);

            auto query_vec_array = dataset.GetTensor();
            auto calculator = calculator::get_distance_calculator<VecT>(index_cfg.metric_type.value());

            // query request dataset will contain multiple query vectors
            for (int64_t qIdx = 0; qIdx < nq; qIdx++) {
                utils::CandidateBoundedQueue candidates(k, index_cfg.metric_type.value());
                auto query_vec = reinterpret_cast<const VecT*>(query_vec_array) + qIdx * dim_;
                for (int64_t id = 0; id < row_; id++) {
                    auto base_vec = reinterpret_cast<const VecT*>(data_.get()) + id * dim_;
                    // only vectors that are not filtered will be returned
                    if (bitset.test(id)) {
                        candidates.push(utils::Candidate(id, calculator(query_vec, base_vec, dim_)));
                    }
                    // access them directly to use custom parameters
                    if (index_cfg.early_terminate.value() && candidates.size() >= index_cfg.k) {
                        break;
                    }
                }
                size_t ret_idx = 0;
                size_t result_limit = std::min(candidates.size(), (size_t)k);
                auto l_ret_ids = ret_ids.get() + qIdx * k;
                auto l_ret_distances = ret_distances.get() + qIdx * k;
                for (; ret_idx < result_limit; ret_idx++) {
                    auto candidate = candidates.top();
                    auto offset = result_limit - 1 - ret_idx;
                    l_ret_ids[offset] = candidate.id;
                    l_ret_distances[offset] = candidate.dist;
                    candidates.pop();
                }
                // fill in invalid values
                for (; ret_idx < (size_t)k; ret_idx++) {
                    l_ret_ids[ret_idx] = -1;
                    l_ret_distances[ret_idx] = 1.0 / 0.0;
                }
            }
            return knowhere::GenResultDataSet(nq, k, ret_ids.release(), ret_distances.release());
        } catch (const std::exception& e) {
            LOG_KNOWHERE_ERROR_ << "Search on CustomIdx error : " << e.what();
            return knowhere::expected<knowhere::DataSetPtr>::Err(knowhere::Status::plugin_index_error, e.what());
        }
    }

    /**
     * [[Optional]] : can be ignored if user does not use range search functionality
     * return results that meet the expression filtering condition and distance range checking
     * @param dataset: query vector dataset
     * @param cfg: index range search config
     * @param bitset: 0-1 bitmask indicating vectors should be filtered in the final result.
     * 1 indicates the vector should be ignored
     * @return : range search result
     */
    knowhere::expected<knowhere::DataSetPtr>
    RangeSearch(const knowhere::DataSet& dataset, const knowhere::Config& cfg,
                const knowhere::BitsetView& bitset) const override {
        const CustomIdxConfig& index_cfg = static_cast<const demo::CustomIdxConfig&>(cfg);
        auto nq = dataset.GetRows();
        try {
            std::vector<std::vector<int64_t>> temp_ids(nq);
            std::vector<std::vector<float>> temp_dists(nq);
            auto result_offsets = std::make_unique<size_t[]>(nq + 1);

            auto query_vec_array = dataset.GetTensor();
            auto calculator = calculator::get_distance_calculator<VecT>(index_cfg.metric_type.value());
            auto range_checker = calculator::get_distance_range_checker(index_cfg.metric_type.value());

            // query request dataset will contain multiple query vectors
            for (int64_t qIdx = 0; qIdx < nq; qIdx++) {
                std::vector<utils::Candidate> candidates;
                auto query_vec = reinterpret_cast<const VecT*>(query_vec_array) + qIdx * dim_;
                for (int64_t id = 0; id < row_; id++) {
                    auto base_vec = reinterpret_cast<const VecT*>(data_.get()) + id * dim_;
                    // only vectors that are not filtered will be returned
                    if (bitset.test(id)) {
                        auto dist = calculator(query_vec, base_vec, dim_);
                        // only vectors with distance values within the given range will be returned
                        if (range_checker(dist, index_cfg.radius.value(), index_cfg.range_filter.value())) {
                            candidates.emplace_back(id, dist);
                        }
                    }
                }
                std::sort(candidates.begin(), candidates.end(),
                          utils::get_metric_dist_comparator(index_cfg.metric_type.value()));

                temp_ids[qIdx].resize(candidates.size());
                temp_dists[qIdx].resize(candidates.size());

                for (size_t idx = 0; idx < candidates.size(); idx++) {
                    temp_ids[qIdx][idx] = candidates[idx].id;
                    temp_dists[qIdx][idx] = candidates[idx].dist;
                }
            }

            result_offsets[0] = 0;
            for (int64_t qIdx = 0; qIdx < nq; qIdx++) {
                result_offsets[qIdx + 1] = result_offsets[qIdx] + temp_ids[qIdx].size();
            }

            size_t total_num = result_offsets[nq];
            auto result_dists = std::make_unique<float[]>(total_num);
            auto result_ids = std::make_unique<int64_t[]>(total_num);
            for (int64_t qIdx = 0; qIdx < nq; qIdx++) {
                std::memcpy(result_ids.get() + result_offsets[qIdx], temp_ids[qIdx].data(),
                            temp_ids[qIdx].size() * sizeof(int64_t));
                std::memcpy(result_dists.get() + result_offsets[qIdx], temp_dists[qIdx].data(),
                            temp_dists[qIdx].size() * sizeof(float));
            }
            return knowhere::GenResultDataSet(nq, result_ids.release(), result_dists.release(),
                                              result_offsets.release());
        } catch (const std::exception& e) {
            LOG_KNOWHERE_ERROR_ << "Search on CustomIdx error : " << e.what();
            return knowhere::expected<knowhere::DataSetPtr>::Err(knowhere::Status::plugin_index_error, e.what());
        }
    }

    /**
     * [[Optional]] : can be ignored if index does not retain the original vector data
     *                No need to worry. If the index cannot provide the original data,
     *                Milvus will fetch and return it from the object storage, just a bit slower
     * return the collection of vectors corresponding to the sppecified ids
     * @param dataset: retrieve vector ids
     * @return : original vector data for the given ids
     */
    knowhere::expected<knowhere::DataSetPtr>
    GetVectorByIds(const knowhere::DataSet& dataset) const override {
        auto dim = Dim();
        auto row = dataset.GetRows();
        auto ids = dataset.GetIds();
        try {
            auto result = std::make_unique<VecT[]>(row * dim);
            for (int64_t qIdx = 0; qIdx < row; qIdx++) {
                int64_t id = ids[qIdx];
                auto base_vec = reinterpret_cast<const VecT*>(data_.get()) + id * dim;
                // copy vevtor raw data from index
                std::memcpy(result.get() + qIdx * dim, base_vec, dim * sizeof(VecT));
            }
            return knowhere::GenResultDataSet(row, dim, result.release());
        } catch (const std::exception& e) {
            LOG_KNOWHERE_ERROR_ << "GetVectorByIds on CustomIdx error : " << e.what();
            return knowhere::expected<knowhere::DataSetPtr>::Err(knowhere::Status::plugin_index_error, e.what());
        }
    }

    /**
     * [[must be implemented]]
     * whether the index provide the ability to retrieve raw data
     * @param metric_type: metric used in current index
     * @return : return true if GetVectorByIds is supported; otherwhise return false
     */
    bool
    HasRawData(const std::string& metric_type) const override {
        return true;
    }

    /**
     * [[ignored]] : used for specific index
     */
    knowhere::expected<knowhere::DataSetPtr>
    GetIndexMeta(const knowhere::Config& cfg) const override {
        return knowhere::expected<knowhere::DataSetPtr>::Err(knowhere::Status::not_implemented,
                                                             "GetIndexMeta not implemented");
    }

    /**
     * [[must be implemented]]
     * serialize the index into a block of contiguous memory
     * @param BinarySet: structure used to hold the serialization memory buffer
     * @return : return success if the index is serialized
     */
    knowhere::Status
    Serialize(knowhere::BinarySet& binset) const override {
        std::shared_ptr<uint8_t[]> binary(new uint8_t[Size()]);
        int offset = 0;

        std::memcpy(binary.get() + offset, &row_, sizeof(int64_t));
        offset += sizeof(int64_t);

        std::memcpy(binary.get() + offset, &dim_, sizeof(int64_t));
        offset += sizeof(int64_t);

        std::memcpy(binary.get() + offset, data_.get(), row_ * dim_ * sizeof(VecT));
        offset += row_ * dim_ * sizeof(VecT);

        binset.Append(Type(), binary, offset);
        return knowhere::Status::success;
    }

    /**
     * [[must be implemented]]
     * deserialize the index from a block of contiguous memory
     * @param BinarySet: sturcture used to get the deserialization memory buffer
     * @param config: index deserialize config
     * @return : return success if the index is deserialized
     */
    knowhere::Status
    Deserialize(const knowhere::BinarySet& binset, const knowhere::Config& config) override {
        auto binary = binset.GetByName(Type());
        int offset = 0;

        std::memcpy(&row_, binary->data.get() + offset, sizeof(int64_t));
        offset += sizeof(int64_t);

        std::memcpy(&dim_, binary->data.get() + offset, sizeof(int64_t));
        offset += sizeof(int64_t);

        if (!data_) {
            data_ = std::make_unique<VecT[]>(row_ * dim_);
        }
        std::memcpy(data_.get(), binary->data.get() + offset, row_ * dim_ * sizeof(VecT));
        offset += row_ * dim_ * sizeof(VecT);

        return knowhere::Status::success;
    }

    /**
     * [[ignored]] used for deserialize index from file
     */
    knowhere::Status
    DeserializeFromFile(const std::string& filename, const knowhere::Config& config) override {
        return knowhere::Status::not_implemented;
    }

    /**
     * [[must be implemented]]
     * create a index configuration object
     * @return : return a created configuration object
     */
    std::unique_ptr<knowhere::BaseConfig>
    CreateConfig() const override {
        return std::make_unique<demo::CustomIdxConfig>();
    }

    /**
     * [[must be implemented]]
     * vector dim for current index
     * @return : vector dim
     */
    int64_t
    Dim() const override {
        return dim_;
    }

    /**
     * [[must be implemented]]
     * memory consumption for current index
     * @return : memory consumption size (in bytes)
     */
    int64_t
    Size() const override {
        return dim_ * row_ * sizeof(VecT) + sizeof(int64_t) * 2;
    }

    /**
     * [[must be implemented]]
     * indexed vector number for current index
     * @return : vector number indexed by index
     */
    int64_t
    Count() const override {
        return row_;
    }

    /**
     * [[must be implemented]]
     * index name for current index
     * @return : index name
     */
    std::string
    Type() const override {
        return "CustomIdx";
    }

 private:
    int64_t row_;
    int64_t dim_;
    std::unique_ptr<VecT[]> data_;
};
}  // namespace demo

extern "C" {

knowhere::PluginIndexList
getPluginIndexList() {
    knowhere::PluginIndexList pluginIndexList;

    pluginIndexList.emplace_back("CustomIdx", knowhere::IndexDataType::FLOAT32,
                                 [](const int32_t& version, const knowhere::Object& object) {
                                     return knowhere::Index<demo::CustomIdx<knowhere::fp32>>::Create(version, object);
                                 });

    pluginIndexList.emplace_back("CustomIdx", knowhere::IndexDataType::BF16,
                                 [](const int32_t& version, const knowhere::Object& object) {
                                     return knowhere::Index<demo::CustomIdx<knowhere::bf16>>::Create(version, object);
                                 });

    pluginIndexList.emplace_back("CustomIdx", knowhere::IndexDataType::FP16,
                                 [](const int32_t& version, const knowhere::Object& object) {
                                     return knowhere::Index<demo::CustomIdx<knowhere::fp16>>::Create(version, object);
                                 });

    return pluginIndexList;
}
}
