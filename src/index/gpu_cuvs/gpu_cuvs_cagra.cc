/**
 * SPDX-FileCopyrightText: Copyright (c) 2023,NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <vector>

#include "common/cuvs/proto/cuvs_index_kind.hpp"
#include "gpu_cuvs.h"
#include "hnswlib/hnswalg.h"
#include "knowhere/dataset.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/index/index_node_thread_pool_wrapper.h"
#include "raft/core/device_resources.hpp"
#include "raft/util/cuda_rt_essentials.hpp"
namespace knowhere {

template <typename DataType>
class GpuCuvsCagraHybridIndexNode : public GpuCuvsCagraIndexNode<DataType> {
 public:
    using DistType = float;
    GpuCuvsCagraHybridIndexNode(int32_t version, const Object& object)
        : GpuCuvsCagraIndexNode<DataType>(version, object) {
    }

    Status
    Train(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override {
        const GpuCuvsCagraConfig& cagra_cfg = static_cast<const GpuCuvsCagraConfig&>(*cfg);
        if (cagra_cfg.adapt_for_cpu.value())
            adapt_for_cpu = true;
        return GpuCuvsCagraIndexNode<DataType>::Train(dataset, cfg, use_knowhere_build_pool);
    }

    expected<DataSetPtr>
    Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset) const override {
        if (!adapt_for_cpu || hnsw_index_ == nullptr)
            return GpuCuvsCagraIndexNode<DataType>::Search(dataset, std::move(cfg), bitset);
        auto nq = dataset->GetRows();
        auto xq = dataset->GetTensor();

        auto cagra_cfg = static_cast<const GpuCuvsCagraConfig&>(*cfg);
        auto k = cagra_cfg.k.value();

        auto p_id = std::make_unique<int64_t[]>(k * nq);
        auto p_dist = std::make_unique<DistType[]>(k * nq);

        hnswlib::SearchParam param{(size_t)cagra_cfg.ef.value()};
        bool transform = (hnsw_index_->metric_type_ == hnswlib::Metric::INNER_PRODUCT ||
                          hnsw_index_->metric_type_ == hnswlib::Metric::COSINE);

        for (int i = 0; i < nq; ++i) {
            auto p_id_ptr = p_id.get();
            auto p_dist_ptr = p_dist.get();
            auto single_query = (const char*)xq + i * hnsw_index_->data_size_;
            auto rst = hnsw_index_->searchKnn(single_query, k, bitset, &param);
            size_t rst_size = rst.size();
            auto p_single_dis = p_dist_ptr + i * k;
            auto p_single_id = p_id_ptr + i * k;
            for (size_t idx = 0; idx < rst_size; ++idx) {
                const auto& [dist, id] = rst[idx];
                p_single_dis[idx] = transform ? (-dist) : dist;
                p_single_id[idx] = id;
            }
            for (size_t idx = rst_size; idx < (size_t)k; idx++) {
                p_single_dis[idx] = DistType(1.0 / 0.0);
                p_single_id[idx] = -1;
            }
        }

        auto res = GenResultDataSet(nq, k, p_id.release(), p_dist.release());

        return res;
    }
    Status
    Serialize(BinarySet& binset) const override {
        if (!adapt_for_cpu)
            return GpuCuvsCagraIndexNode<DataType>::Serialize(binset);
        auto result = Status::success;
        std::stringbuf buf;
        if (!this->index_.is_trained()) {
            result = Status::empty_index;
        } else {
            std::ostream os(&buf);
            try {
                this->index_.serialize_to_hnswlib(os);
                this->index_.synchronize(true);
            } catch (const std::exception& e) {
                LOG_KNOWHERE_ERROR_ << e.what();
                result = Status::cuvs_inner_error;
            }
            os.flush();
        }
        if (result == Status::success) {
            std::shared_ptr<uint8_t[]> index_binary(new (std::nothrow) uint8_t[buf.str().size()]);
            memcpy(index_binary.get(), buf.str().c_str(), buf.str().size());
            // Use the key to differentiate whether CPU search support is needed.
            binset.Append(std::string(this->Type()) + "_cpu", index_binary, buf.str().size());
        }
        return result;
    }

    int64_t
    Count() const override {
        if (!adapt_for_cpu)
            return GpuCuvsCagraIndexNode<DataType>::Count();
        if (!hnsw_index_) {
            return 0;
        }
        return hnsw_index_->cur_element_count;
    }

    Status
    Deserialize(const BinarySet& binset, std::shared_ptr<Config> cfg) override {
        const GpuCuvsCagraConfig& cagra_cfg = static_cast<const GpuCuvsCagraConfig&>(*cfg);
        if (cagra_cfg.adapt_for_cpu.value()) {
            adapt_for_cpu = true;
            if constexpr (std::is_same_v<DataType, std::int8_t>) {
                // TODO: Add HNSW support for INT8
                LOG_KNOWHERE_ERROR_ << "CAGRA+HNSW does not support INT8 data.";
                return Status::invalid_binary_set;
            } else {
                BinaryPtr binary = nullptr;
                hnswlib::SpaceInterface<float>* space = nullptr;
                hnsw_index_.reset(new (std::nothrow) hnswlib::HierarchicalNSW<DataType, float, hnswlib::None>(space));
                try {
                    if ((binary = binset.GetByName(std::string(this->Type()) + "_cpu")) != nullptr) {
                        MemoryIOReader reader(binary->data.get(), binary->size);
                        hnsw_index_->loadIndex(reader);
                    } else if ((binary = binset.GetByName(std::string(this->Type()))) != nullptr) {
                        std::stringbuf buf;
                        buf.sputn((char*)binary->data.get(), binary->size);
                        std::istream stream(&buf);
                        hnsw_index_->loadIndexFromGpuFormat(stream);
                    } else {
                        LOG_KNOWHERE_ERROR_ << "Invalid binary set.";
                        return Status::invalid_binary_set;
                    }
                    hnsw_index_->base_layer_only = true;
                } catch (std::exception& e) {
                    LOG_KNOWHERE_WARNING_ << "hnsw inner error: " << e.what();
                    return Status::hnsw_inner_error;
                }
                return Status::success;
            }
        }

        return GpuCuvsCagraIndexNode<DataType>::Deserialize(binset, std::move(cfg));
    }

    Status
    DeserializeFromFile(const std::string& filename, std::shared_ptr<Config>) override {
        return Status::not_implemented;
    }

 private:
    bool adapt_for_cpu = false;
    std::unique_ptr<hnswlib::HierarchicalNSW<DataType, float, hnswlib::None>> hnsw_index_ = nullptr;
};

KNOWHERE_REGISTER_GLOBAL_WITH_THREAD_POOL(GPU_CUVS_CAGRA, GpuCuvsCagraHybridIndexNode, fp32,
                                          knowhere::feature::GPU_ANN_FLOAT_INDEX, []() {
                                              int count;
                                              RAFT_CUDA_TRY(cudaGetDeviceCount(&count));
                                              return count * cuda_concurrent_size_per_device;
                                          }());
KNOWHERE_REGISTER_GLOBAL_WITH_THREAD_POOL(GPU_CAGRA, GpuCuvsCagraHybridIndexNode, fp32,
                                          knowhere::feature::GPU_ANN_FLOAT_INDEX, []() {
                                              int count;
                                              RAFT_CUDA_TRY(cudaGetDeviceCount(&count));
                                              return count * cuda_concurrent_size_per_device;
                                          }());

KNOWHERE_REGISTER_GLOBAL_WITH_THREAD_POOL(GPU_CUVS_CAGRA, GpuCuvsCagraHybridIndexNode, fp16,
                                          knowhere::feature::GPU | knowhere::feature::FP16, []() {
                                              int count;
                                              RAFT_CUDA_TRY(cudaGetDeviceCount(&count));
                                              return count * cuda_concurrent_size_per_device;
                                          }());
KNOWHERE_REGISTER_GLOBAL_WITH_THREAD_POOL(GPU_CAGRA, GpuCuvsCagraHybridIndexNode, fp16,
                                          knowhere::feature::GPU | knowhere::feature::FP16, []() {
                                              int count;
                                              RAFT_CUDA_TRY(cudaGetDeviceCount(&count));
                                              return count * cuda_concurrent_size_per_device;
                                          }());
// TODO: Add HNSW support for INT8. Not using the hybrid CAGRA+HNSW for INT8
KNOWHERE_REGISTER_GLOBAL_WITH_THREAD_POOL(GPU_CUVS_CAGRA, GpuCuvsCagraIndexNode, int8,
                                          knowhere::feature::GPU | knowhere::feature::INT8, []() {
                                              int count;
                                              RAFT_CUDA_TRY(cudaGetDeviceCount(&count));
                                              return count * cuda_concurrent_size_per_device;
                                          }());
KNOWHERE_REGISTER_GLOBAL_WITH_THREAD_POOL(GPU_CAGRA, GpuCuvsCagraIndexNode, int8,
                                          knowhere::feature::GPU | knowhere::feature::INT8, []() {
                                              int count;
                                              RAFT_CUDA_TRY(cudaGetDeviceCount(&count));
                                              return count * cuda_concurrent_size_per_device;
                                          }());
KNOWHERE_REGISTER_GLOBAL_WITH_THREAD_POOL(GPU_CUVS_CAGRA, GpuCuvsCagraHybridIndexNode, bin1,
                                          knowhere::feature::GPU | knowhere::feature::BINARY, []() {
                                              int count;
                                              RAFT_CUDA_TRY(cudaGetDeviceCount(&count));
                                              return count * cuda_concurrent_size_per_device;
                                          }());
KNOWHERE_REGISTER_GLOBAL_WITH_THREAD_POOL(GPU_CAGRA, GpuCuvsCagraHybridIndexNode, bin1,
                                          knowhere::feature::GPU | knowhere::feature::BINARY, []() {
                                              int count;
                                              RAFT_CUDA_TRY(cudaGetDeviceCount(&count));
                                              return count * cuda_concurrent_size_per_device;
                                          }());

}  // namespace knowhere
