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

#ifndef DISKANN_CONFIG_H
#define DISKANN_CONFIG_H

#include "knowhere/config.h"

namespace knowhere {

namespace {

constexpr const CFG_INT::value_type kSearchListSizeMinValue = 16;
constexpr const CFG_INT::value_type kDefaultSearchListSizeForBuild = 128;

}  // namespace

class DiskANNConfig : public BaseConfig {
 public:
    // This is the degree of the graph index, typically between 60 and 150. Larger R will result in larger indices and
    // longer indexing times, but better search quality.
    CFG_INT max_degree;
    // The size of the search list during the index build or (knn/range) search. Typical values are between 75 to 200.
    // Larger values will take more time to build but result in indices that provide higher recall for the same search
    // complexity. Plz set this value larger than the max_degree unless you need to build indices really quickly and can
    // somewhat compromise on quality.
    CFG_INT search_list_size;
    // Limit the size of the PQ code after the raw vector has been PQ-encoded. PQ code is a (pq_code_budget_gb * 1024 *
    // 1024 * 1024) / row_num)-dimensional uint8 vector. If pq_code_budget_gb is too large, it will be adjusted to the
    // size of dim*row_num.
    CFG_FLOAT pq_code_budget_gb;
    // Limit on the memory allowed for building the index in GB. If you specify a value less than what is required to
    // build the index in one pass, the index is built using a divide and conquer approach so that sub-graphs will fit
    // in the RAM budget. The sub-graphs are overlayed to build the overall index. This approach can be up to 1.5 times
    // slower than building the index in one shot. Allocate as much memory as your RAM allows.
    CFG_FLOAT build_dram_budget_gb;
    // Use 0 to store uncompressed data on SSD. This allows the index to asymptote to 100% recall. If your vectors are
    // too large to store in SSD, this parameter provides the option to compress the vectors using PQ for storing on
    // SSD. This will trade off the recall. You would also want this to be greater than the number of bytes used for the
    // PQ compressed data stored in-memory
    CFG_INT disk_pq_dims;
    // This is the flag to enable fast build, in which we will not build vamana graph by full 2 round. This can
    // accelerate index build ~30% with an ~1% recall regression.
    CFG_BOOL accelerate_build;
    // While serving the index, the entire graph is stored on SSD. For faster search performance, you can cache a few
    // frequently accessed nodes in memory.
    CFG_FLOAT search_cache_budget_gb;
    // Should we do warm-up before searching.
    CFG_BOOL warm_up;
    // Should we use the bfs strategy to cache. We have two cache strategies: 1. use sample queries to do searches and
    // cached the nodes on the search paths; 2. do bfs from the entry point and cache them. The first method is suitable
    // for TopK query heavy circumstances and the second one performed better in range search.
    CFG_BOOL use_bfs_cache;
    // The beamwidth to be used for search. This is the maximum number of IO requests each query will issue per
    // iteration of search code. Larger beamwidth will result in fewer IO round-trips per query but might result in
    // slightly higher total number of IO requests to SSD per query. For the highest query throughput with a fixed SSD
    // IOps rating, use W=1. For best latency, use W=4,8 or higher complexity search.
    CFG_INT beamwidth;
    // DiskANN uses TopK search to simulate range search by double the K in every round. This is the start K.
    CFG_INT min_k;
    // DiskANN uses TopK search to simulate range search by double the K in every round. This is the largest K.
    CFG_INT max_k;
    // The threshold which determines when to switch to PQ + Refine strategy based on the number of bits set. The
    // value should be in range of [0.0, 1.0] which means when greater or equal to x% of the bits are set,
    // use PQ + Refine. Default to -1.0f, negative vlaues will use dynamic threshold calculator given topk.
    CFG_FLOAT filter_threshold;
    KNOHWERE_DECLARE_CONFIG(DiskANNConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(max_degree)
            .description("the degree of the graph index.")
            .set_default(48)
            .set_range(1, 2048)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(search_list_size)
            .description("the size of search list during the index build or search.")
            .allow_empty_without_default()
            .set_range(1, std::numeric_limits<CFG_INT::value_type>::max())
            .for_train()
            .for_search()
            .for_iterator();
        KNOWHERE_CONFIG_DECLARE_FIELD(pq_code_budget_gb)
            .description("the size of PQ compressed representation in GB.")
            .set_range(0, std::numeric_limits<CFG_FLOAT::value_type>::max())
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(build_dram_budget_gb)
            .description("limit on the memory allowed for building the index in GB.")
            .set_range(0, std::numeric_limits<CFG_FLOAT::value_type>::max())
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(disk_pq_dims)
            .description("the dimension of compressed vectors stored on the ssd, use 0 to store uncompressed data.")
            .set_default(0)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(accelerate_build)
            .description("a flag to enbale fast build.")
            .set_default(false)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(search_cache_budget_gb)
            .description("the size of cached nodes in GB.")
            .set_default(0)
            .set_range(0, std::numeric_limits<CFG_FLOAT::value_type>::max())
            .for_train()
            .for_deserialize();
        KNOWHERE_CONFIG_DECLARE_FIELD(warm_up)
            .description("should do warm up before search.")
            .set_default(false)
            .for_deserialize();
        KNOWHERE_CONFIG_DECLARE_FIELD(use_bfs_cache)
            .description("should bfs strategy to cache nodes.")
            .set_default(false)
            .for_deserialize();
        KNOWHERE_CONFIG_DECLARE_FIELD(beamwidth)
            .description("the maximum number of IO requests each query will issue per iteration of search code.")
            .set_default(8)
            .set_range(1, 128)
            .for_search()
            .for_range_search()
            .for_iterator();
        KNOWHERE_CONFIG_DECLARE_FIELD(min_k)
            .description("the min l_search size used in range search.")
            .set_default(100)
            .set_range(1, std::numeric_limits<CFG_INT::value_type>::max())
            .for_range_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(max_k)
            .description("the max l_search size used in range search.")
            .set_default(std::numeric_limits<CFG_INT::value_type>::max())
            .set_range(1, std::numeric_limits<CFG_INT::value_type>::max())
            .for_range_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(filter_threshold)
            .description("the threshold of filter ratio to use PQ + Refine.")
            .set_default(-1.0f)
            .set_range(-1.0f, 1.0f)
            .for_search()
            .for_iterator();
    }

    Status
    CheckAndAdjust(PARAM_TYPE param_type, std::string* err_msg) override {
        switch (param_type) {
            case PARAM_TYPE::TRAIN: {
                if (!search_list_size.has_value()) {
                    search_list_size = kDefaultSearchListSizeForBuild;
                }
                break;
            }
            case PARAM_TYPE::SEARCH: {
                if (!search_list_size.has_value()) {
                    search_list_size = std::max(k.value(), kSearchListSizeMinValue);
                } else if (k.value() > search_list_size.value()) {
                    *err_msg = "search_list_size(" + std::to_string(search_list_size.value()) +
                               ") should be larger than k(" + std::to_string(k.value()) + ")";
                    LOG_KNOWHERE_ERROR_ << *err_msg;
                    return Status::out_of_range_in_json;
                }
                break;
            }
            default:
                break;
        }
        return Status::success;
    }
};
}  // namespace knowhere
#endif /* DISKANN_CONFIG_H */
