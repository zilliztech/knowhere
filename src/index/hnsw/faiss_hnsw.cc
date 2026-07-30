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

#include <faiss/cppcontrib/knowhere/IndexBinaryScalarQuantizer.h>
#include <faiss/cppcontrib/knowhere/IndexCosine.h>
#include <faiss/cppcontrib/knowhere/IndexFlat.h>
#include <faiss/cppcontrib/knowhere/IndexHNSWBinary.h>
#include <faiss/cppcontrib/knowhere/IndexSQ4Uniform.h>
#include <faiss/cppcontrib/knowhere/MetricType.h>
#include <faiss/cppcontrib/knowhere/impl/CountSizeIOWriter.h>
#include <faiss/cppcontrib/knowhere/impl/HnswSearcher.h>
#include <faiss/cppcontrib/knowhere/impl/additional_io.h>
#include <faiss/cppcontrib/knowhere/utils/Bitset.h>
#include <faiss/utils/Heap.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "common/metric.h"
#include "faiss/cppcontrib/knowhere/IndexHNSW.h"
#include "faiss/cppcontrib/knowhere/IndexRefine.h"
#include "faiss/cppcontrib/knowhere/index_io.h"
#include "faiss/impl/mapped_io.h"
#include "index/clustering_config.h"
#include "index/hnsw/faiss_hnsw_config.h"
#include "index/hnsw/hnsw.h"
#include "index/hnsw/impl/DummyVisitor.h"
#include "index/hnsw/impl/FederVisitor.h"
#include "index/hnsw/impl/IndexBruteForceWrapper.h"
#include "index/hnsw/impl/IndexConditionalWrapper.h"
#include "index/hnsw/impl/IndexHNSWWrapper.h"
#include "index/hnsw/impl/IndexWrapperCosine.h"
#include "index/refine/refine_utils.h"
#include "io/memory_io.h"
#include "knowhere/bitsetview_idselector.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/task.h"
#include "knowhere/comp/time_recorder.h"
#include "knowhere/config.h"
#include "knowhere/context.h"
#include "knowhere/expected.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/index/index_node_data_mock_wrapper.h"
#include "knowhere/log.h"
#include "knowhere/range_util.h"
#include "knowhere/utils.h"

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
#include "knowhere/prometheus_client.h"
#endif

namespace knowhere {

//
class BaseFaissIndexNode : public IndexNode {
 public:
    BaseFaissIndexNode(const int32_t& /*version*/, const Object& object) {
        build_pool = ThreadPool::GetGlobalBuildThreadPool();
        search_pool = ThreadPool::GetGlobalSearchThreadPool();
    }

    bool
    IsAdditionalScalarSupported(bool is_mv_only) const override {
        if (is_mv_only) {
            return true;
        }
        return false;
    }

    //
    Status
    Train(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override {
        // config
        const BaseConfig& base_cfg = static_cast<const FaissHnswConfig&>(*cfg);

        // use build_pool_ to make sure the OMP threads spawned by index_->train etc
        // can inherit the low nice value of threads in build_pool_.
        auto tryObj =
            build_pool
                ->push([&] {
                    std::unique_ptr<ThreadPool::ScopedBuildOmpSetter> setter;
                    if (base_cfg.num_build_thread.has_value()) {
                        setter = std::make_unique<ThreadPool::ScopedBuildOmpSetter>(base_cfg.num_build_thread.value());
                    } else {
                        setter = std::make_unique<ThreadPool::ScopedBuildOmpSetter>();
                    }
                    return TrainInternal(dataset, *cfg);
                })
                .getTry();

        if (!tryObj.hasValue()) {
            LOG_KNOWHERE_WARNING_ << "faiss internal error: " << tryObj.exception().what();
            return Status::faiss_inner_error;
        }

        return tryObj.value();
    }

    Status
    Add(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override {
        const BaseConfig& base_cfg = static_cast<const FaissHnswConfig&>(*cfg);

        // use build_pool_ to make sure the OMP threads spawned by index_->train etc
        // can inherit the low nice value of threads in build_pool_.
        auto tryObj =
            build_pool
                ->push([&] {
                    std::unique_ptr<ThreadPool::ScopedBuildOmpSetter> setter;
                    if (base_cfg.num_build_thread.has_value()) {
                        setter = std::make_unique<ThreadPool::ScopedBuildOmpSetter>(base_cfg.num_build_thread.value());
                    } else {
                        setter = std::make_unique<ThreadPool::ScopedBuildOmpSetter>();
                    }
                    return AddInternal(dataset, *cfg);
                })
                .getTry();

        if (!tryObj.hasValue()) {
            LOG_KNOWHERE_WARNING_ << "faiss internal error: " << tryObj.exception().what();
            return Status::faiss_inner_error;
        }

        return tryObj.value();
    }

    expected<DataSetPtr>
    GetIndexMeta(std::unique_ptr<Config> cfg) const override {
        // todo
        return expected<DataSetPtr>::Err(Status::not_implemented, "GetIndexMeta not implemented");
    }

 protected:
    std::shared_ptr<ThreadPool> build_pool;
    std::shared_ptr<ThreadPool> search_pool;

    // train impl
    virtual Status
    TrainInternal(const DataSetPtr dataset, const Config& cfg) = 0;

    // add impl
    virtual Status
    AddInternal(const DataSetPtr dataset, const Config& cfg) = 0;
};

// returns true if the text of FaissException is about non-recognizing fourcc
static inline bool
is_faiss_fourcc_error(const char* what) {
    if (what == nullptr) {
        return false;
    }

    std::string error_msg(what);

    // check if this is fourCC problem
    return (error_msg.find("Index type") != std::string::npos) &&
           (error_msg.find("not recognized") != std::string::npos);
}

//
class BaseFaissRegularIndexNode : public BaseFaissIndexNode {
 public:
    BaseFaissRegularIndexNode(const int32_t& version, const Object& object)
        : BaseFaissIndexNode(version, object), indexes(1, nullptr) {
    }

    Status
    Serialize(BinarySet& binset) const override {
        if (isIndexEmpty()) {
            return Status::empty_index;
        }

        try {
            MemoryIOWriter writer;
            if (indexes.size() > 1) {
                // this is a hack for compatibility, faiss index has 4-byte header to indicate index category
                // create a new one to distinguish MV faiss hnsw from faiss hnsw
                faiss::cppcontrib::knowhere::write_mv(&writer);
                writeHeader(&writer);
                for (const auto& index : indexes) {
                    faiss::cppcontrib::knowhere::write_index(index.get(), &writer);
                }

                std::shared_ptr<uint8_t[]> data(writer.data());
                binset.Append(Type(), data, writer.tellg());
            } else {
                faiss::cppcontrib::knowhere::write_index(indexes[0].get(), &writer);
                std::shared_ptr<uint8_t[]> data(writer.data());
                binset.Append(Type(), data, writer.tellg());
            }
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
            return Status::faiss_inner_error;
        }

        return Status::success;
    }

    Status
    Deserialize(const BinarySet& binset, std::shared_ptr<Config> config) override {
        auto binary = binset.GetByName(Type());
        if (binary == nullptr) {
            LOG_KNOWHERE_ERROR_ << "Invalid binary set.";
            return Status::invalid_binary_set;
        }

        MemoryIOReader reader(binary->data.get(), binary->size);
        try {
            // this is a hack for compatibility, faiss index has 4-byte header to indicate index category
            // create a new one to distinguish MV faiss hnsw from faiss hnsw
            bool is_mv = faiss::cppcontrib::knowhere::read_is_mv(&reader);
            if (is_mv) {
                LOG_KNOWHERE_INFO_ << "start to load index by mv";
                uint32_t v = readHeader(&reader);
                indexes.resize(v);
                LOG_KNOWHERE_INFO_ << "read " << v << " mvs";
                for (uint32_t i = 0; i < v; ++i) {
                    auto read_index = std::unique_ptr<faiss::Index>(faiss::cppcontrib::knowhere::read_index(&reader));
                    indexes[i].reset(read_index.release());
                }
            } else {
                reader.reset();
                auto read_index = std::unique_ptr<faiss::Index>(faiss::cppcontrib::knowhere::read_index(&reader));
                indexes[0].reset(read_index.release());
            }
        } catch (const std::exception& e) {
            if (is_faiss_fourcc_error(e.what())) {
                LOG_KNOWHERE_WARNING_ << "faiss does not recognize the input index: " << e.what();
                return Status::invalid_serialized_index_type;
            } else {
                LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
                return Status::faiss_inner_error;
            }
        }

        return Status::success;
    }

    Status
    DeserializeFromFile(const std::string& filename, std::shared_ptr<Config> config) override {
        auto cfg = static_cast<const knowhere::BaseConfig&>(*config);

        int io_flags = 0;
        if (cfg.enable_mmap.value()) {
            io_flags |= faiss::cppcontrib::knowhere::IO_FLAG_MMAP_IFC;
        }

        try {
            // this is a hack for compatibility, faiss index has 4-byte header to indicate index category
            // create a new one to distinguish MV faiss hnsw from faiss hnsw
            bool is_mv = faiss::cppcontrib::knowhere::read_is_mv(filename.data());
            if (is_mv) {
                auto read_index = [&](faiss::IOReader* r) {
                    LOG_KNOWHERE_INFO_ << "start to load index by mv";
                    faiss::cppcontrib::knowhere::read_is_mv(r);
                    uint32_t v = readHeader(r);
                    LOG_KNOWHERE_INFO_ << "read " << v << " mvs";
                    indexes.resize(v);
                    for (uint32_t i = 0; i < v; ++i) {
                        auto read_index =
                            std::unique_ptr<faiss::Index>(faiss::cppcontrib::knowhere::read_index(r, io_flags));
                        indexes[i].reset(read_index.release());
                    }
                };
                if ((io_flags & faiss::cppcontrib::knowhere::IO_FLAG_MMAP_IFC) ==
                    faiss::cppcontrib::knowhere::IO_FLAG_MMAP_IFC) {
                    // enable mmap-supporting IOReader
                    auto owner = std::make_shared<faiss::MmappedFileMappingOwner>(filename.data());
                    faiss::MappedFileIOReader reader(owner);
                    read_index(&reader);
                } else {
                    faiss::FileIOReader reader(filename.data());
                    read_index(&reader);
                }
            } else {
                auto read_index =
                    std::unique_ptr<faiss::Index>(faiss::cppcontrib::knowhere::read_index(filename.data(), io_flags));
                indexes[0].reset(read_index.release());
            }
        } catch (const std::exception& e) {
            if (is_faiss_fourcc_error(e.what())) {
                LOG_KNOWHERE_WARNING_ << "faiss does not recognize the input index: " << e.what();
                return Status::invalid_serialized_index_type;
            } else {
                LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
                return Status::faiss_inner_error;
            }
        }

        return Status::success;
    }

    //
    int64_t
    Dim() const override {
        if (isIndexEmpty()) {
            return -1;
        }

        return indexes[0]->d;
    }

    int64_t
    Count() const override {
        if (isIndexEmpty()) {
            return -1;
        }
        int64_t count = 0;
        for (const auto& index : indexes) {
            count += index->ntotal;
        }

        // total number of indexed vectors
        return count;
    }

    int64_t
    Size() const override {
        if (isIndexEmpty()) {
            return 0;
        }

        // a temporary yet expensive workaround
        faiss::cppcontrib::knowhere::CountSizeIOWriter writer;
        for (const auto& index : indexes) {
            faiss::cppcontrib::knowhere::write_index(index.get(), &writer);
        }

        // todo
        return writer.total_size;
    }

    std::shared_ptr<std::vector<uint32_t>>
    GetInternalIdToExternalIdMap() const override {
        auto internal_offset_to_label = std::make_shared<std::vector<uint32_t>>();
        assert(indexes.size() > 0);
        if (indexes.size() == 1) {
            // without mv-only labels, the id mapping is the same as the internal offset.
            internal_offset_to_label->resize(Count());
            std::iota(internal_offset_to_label->begin(), internal_offset_to_label->end(), 0);
        } else {
            // faiss_hnsw has stored mv-only labels (id mapping for each mv-index) *separately*, not a contiguous
            // memory block. Note that the mv-only labels have a fixed serialization format; changing the design of
            // labels would affect index version compatibility.
            // so a temporary vector must be created to memcpy all id mappings.
            auto total_size = index_rows_sum[index_rows_sum.size() - 1];
            assert(total_size == Count());
            internal_offset_to_label->resize(total_size);
            for (auto par_idx = 0; par_idx < index_rows_sum.size() - 1; ++par_idx) {
                auto par_size = index_rows_sum[par_idx + 1] - index_rows_sum[par_idx];
                assert(par_size == labels[par_idx]->size());
                std::memcpy(internal_offset_to_label->data() + index_rows_sum[par_idx], labels[par_idx]->data(),
                            par_size * sizeof(uint32_t));
            }
        }
        return internal_offset_to_label;
    }

    Status
    SetInternalIdToMostExternalIdMap(std::vector<uint32_t>&& map) override {
        internal_offset_to_most_external_id = std::move(map);
        return Status::success;
    }

 protected:
    // it is std::shared_ptr, not std::unique_ptr, because it can be
    //    shared with FaissHnswIterator
    std::vector<std::shared_ptr<faiss::Index>> indexes;
    // each index's out ids(label), can be shared with FaissHnswIterator
    std::vector<std::shared_ptr<std::vector<uint32_t>>> labels;

    // index rows, help to locate index id by offset
    std::vector<uint32_t> index_rows_sum;
    // label to locate internal offset
    std::vector<uint32_t> label_to_internal_offset;
    // internal offset to most external id, only for 1-hop bitset check
    std::vector<uint32_t> internal_offset_to_most_external_id;

    int
    getIndexToSearchByScalarInfo(const BitsetView& bitset) const {
        if (indexes.size() == 1) {
            return 0;
        }
        if (bitset.empty()) {
            LOG_KNOWHERE_WARNING_ << "partition key value not correctly set";
            return -1;
        }
        // all data is filtered, just pick the first one
        // this will not happen combined with milvus, which will not call knowhere and just return
        if (bitset.count() == bitset.size()) {
            return 0;
        }
        size_t first_valid_index = bitset.get_first_valid_index();
        if (!bitset.has_out_ids()) {
            first_valid_index = label_to_internal_offset[first_valid_index];
        }
        auto it = std::upper_bound(index_rows_sum.begin(), index_rows_sum.end(), first_valid_index);

        if (it == index_rows_sum.end()) {
            LOG_KNOWHERE_WARNING_ << "can not find vector of offset " << label_to_internal_offset[first_valid_index];
            return -1;
        }
        return std::distance(index_rows_sum.begin(), it) - 1;
    }

    void
    writeHeader(faiss::IOWriter* f) const {
        uint32_t version = 0;
        faiss::cppcontrib::knowhere::write_value(version, f);
        uint32_t size = indexes.size();
        faiss::cppcontrib::knowhere::write_value(size, f);
        uint32_t cluster_size = labels.size();
        faiss::cppcontrib::knowhere::write_value(cluster_size, f);
        for (const auto& label : labels) {
            faiss::cppcontrib::knowhere::write_vector(*label, f);
        }
        faiss::cppcontrib::knowhere::write_vector(index_rows_sum, f);
        faiss::cppcontrib::knowhere::write_vector(label_to_internal_offset, f);
    }

    uint32_t
    readHeader(faiss::IOReader* f) {
        [[maybe_unused]] uint32_t version = faiss::cppcontrib::knowhere::read_value(f);
        const uint32_t size = faiss::cppcontrib::knowhere::read_value(f);
        const uint32_t cluster_size = faiss::cppcontrib::knowhere::read_value(f);
        labels.resize(cluster_size);

        for (uint32_t j = 0; j < cluster_size; ++j) {
            labels[j] = std::make_shared<std::vector<uint32_t>>();
            faiss::cppcontrib::knowhere::read_vector(*labels[j], f);
        }
        faiss::cppcontrib::knowhere::read_vector(index_rows_sum, f);
        faiss::cppcontrib::knowhere::read_vector(label_to_internal_offset, f);
        return size;
    }

    bool
    isIndexEmpty() const {
        if (indexes.empty()) {
            return true;
        }
        for (const auto& index : indexes) {
            if (index == nullptr) {
                return true;
            }
        }
        return false;
    }
};

namespace {

bool
convert_rows_to_fp32(const void* const __restrict src_in, float* const __restrict dst,
                     const DataFormatEnum src_data_format, const uint32_t* const __restrict offsets, const size_t nrows,
                     const size_t dim) {
    if (src_data_format == DataFormatEnum::fp16) {
        const knowhere::fp16* const src = reinterpret_cast<const knowhere::fp16*>(src_in);
        for (size_t i = 0; i < nrows; i++) {
            for (size_t j = 0; j < dim; ++j) {
                dst[i * dim + j] = static_cast<float>(src[offsets[i] * dim + j]);
            }
        }
        return true;
    } else if (src_data_format == DataFormatEnum::bf16) {
        const knowhere::bf16* const src = reinterpret_cast<const knowhere::bf16*>(src_in);
        for (size_t i = 0; i < nrows; i++) {
            for (size_t j = 0; j < dim; ++j) {
                dst[i * dim + j] = static_cast<float>(src[offsets[i] * dim + j]);
            }
        }
        return true;
    } else if (src_data_format == DataFormatEnum::fp32) {
        const knowhere::fp32* const src = reinterpret_cast<const knowhere::fp32*>(src_in);
        for (size_t i = 0; i < nrows; i++) {
            for (size_t j = 0; j < dim; ++j) {
                dst[i * dim + j] = src[offsets[i] * dim + j];
            }
        }
        return true;
    } else if (src_data_format == DataFormatEnum::int8) {
        const knowhere::int8* const src = reinterpret_cast<const knowhere::int8*>(src_in);
        for (size_t i = 0; i < nrows; i++) {
            for (size_t j = 0; j < dim; ++j) {
                dst[i * dim + j] = static_cast<float>(src[offsets[i] * dim + j]);
            }
        }
        return true;
    } else if (src_data_format == DataFormatEnum::bin1) {
        const knowhere::bin1* const src = reinterpret_cast<const knowhere::bin1*>(src_in);
        auto uint8_dim = (dim + 7) / 8;
        for (size_t i = 0; i < nrows; i++) {
            for (size_t j = 0; j < uint8_dim; ++j) {
                dst[i * dim + j] = static_cast<float>(src[offsets[i] * uint8_dim + j]);
            }
        }
        return true;
    } else {
        // unknown
        return false;
    }
}

bool
convert_rows_to_fp32(const void* const __restrict src_in, float* const __restrict dst,
                     const DataFormatEnum src_data_format, const size_t start_row, const size_t nrows,
                     const size_t dim) {
    if (src_data_format == DataFormatEnum::fp16) {
        const knowhere::fp16* const src = reinterpret_cast<const knowhere::fp16*>(src_in);
        for (size_t i = 0; i < nrows * dim; i++) {
            dst[i] = static_cast<float>(src[i + start_row * dim]);
        }
        return true;
    } else if (src_data_format == DataFormatEnum::bf16) {
        const knowhere::bf16* const src = reinterpret_cast<const knowhere::bf16*>(src_in);
        for (size_t i = 0; i < nrows * dim; i++) {
            dst[i] = static_cast<float>(src[i + start_row * dim]);
        }
        return true;
    } else if (src_data_format == DataFormatEnum::fp32) {
        const knowhere::fp32* const src = reinterpret_cast<const knowhere::fp32*>(src_in);
        for (size_t i = 0; i < nrows * dim; i++) {
            dst[i] = src[i + start_row * dim];
        }
        return true;
    } else if (src_data_format == DataFormatEnum::int8) {
        const knowhere::int8* const src = reinterpret_cast<const knowhere::int8*>(src_in);
        for (size_t i = 0; i < nrows * dim; i++) {
            dst[i] = static_cast<float>(src[i + start_row * dim]);
        }
        return true;
    } else if (src_data_format == DataFormatEnum::bin1) {
        // NOTE: This is a little bit weird conversion. The source (`src_in`) is a uint8_t byte stream,
        // where each query_row has ((dim + 7) / 8) * 8 bits, and the total is nrows * ((dim + 7) / 8) * 8 bits.
        // But the final format required is nrows * dim * 32 bits (float).
        // There are actually two conversions happening here:
        // 1. Each uint8_t value must be converted to float (in `BinaryFlatCodesDC::set_query` inside
        //    IndexBinaryScalarQuantizer, it will be converted back to uint8_t). [same as int8]
        // 2. Each row must occupy dim * 32 bits of space, even if not all bits are filled;
        //    this is required by the convention set by IndexBinaryScalarQuantizer::sa_encode.
        const knowhere::bin1* const src = reinterpret_cast<const knowhere::bin1*>(src_in);
        auto uint8_dim = (dim + 7) / 8;
        for (size_t i = 0; i < nrows; i++) {
            for (size_t j = 0; j < uint8_dim; j++) {
                dst[i * dim + j] = static_cast<float>(src[(start_row + i) * uint8_dim + j]);
            }
        }
        return true;
    } else {
        // unknown
        return false;
    }
}

bool
convert_rows_from_fp32(const float* const __restrict src, void* const __restrict dst_in,
                       const DataFormatEnum dst_data_format, const size_t start_row, const size_t nrows,
                       const size_t dim) {
    if (dst_data_format == DataFormatEnum::fp16) {
        knowhere::fp16* const dst = reinterpret_cast<knowhere::fp16*>(dst_in);
        for (size_t i = 0; i < nrows * dim; i++) {
            dst[i + start_row * dim] = static_cast<knowhere::fp16>(src[i]);
        }
        return true;
    } else if (dst_data_format == DataFormatEnum::bf16) {
        knowhere::bf16* const dst = reinterpret_cast<knowhere::bf16*>(dst_in);
        for (size_t i = 0; i < nrows * dim; i++) {
            dst[i + start_row * dim] = static_cast<knowhere::bf16>(src[i]);
        }
        return true;
    } else if (dst_data_format == DataFormatEnum::fp32) {
        knowhere::fp32* const dst = reinterpret_cast<knowhere::fp32*>(dst_in);
        for (size_t i = 0; i < nrows * dim; i++) {
            dst[i + start_row * dim] = src[i];
        }
        return true;
    } else if (dst_data_format == DataFormatEnum::int8) {
        knowhere::int8* const dst = reinterpret_cast<knowhere::int8*>(dst_in);
        for (size_t i = 0; i < nrows * dim; i++) {
            KNOWHERE_THROW_IF_NOT_MSG(src[i] >= std::numeric_limits<knowhere::int8>::min() &&
                                          src[i] <= std::numeric_limits<knowhere::int8>::max(),
                                      "convert float to int8_t overflow");
            dst[i + start_row * dim] = static_cast<knowhere::int8>(src[i]);
        }
        return true;
    } else if (dst_data_format == DataFormatEnum::bin1) {
        knowhere::bin1* const dst = reinterpret_cast<knowhere::bin1*>(dst_in);
        auto uint8_dim = (dim + 7) / 8;
        for (size_t i = 0; i < nrows * uint8_dim; i++) {
            KNOWHERE_THROW_IF_NOT_MSG(src[i] >= std::numeric_limits<knowhere::bin1>::min() &&
                                          src[i] <= std::numeric_limits<knowhere::bin1>::max(),
                                      "convert float to bin1(uint8_t) overflow");
            dst[i + start_row * uint8_dim] = static_cast<knowhere::bin1>(src[i]);
        }
        return true;
    } else {
        // unknown
        return false;
    }
}

//
DataSetPtr
convert_ds_to_float(const DataSetPtr& src, DataFormatEnum data_format) {
    if (data_format == DataFormatEnum::fp32) {
        return src;
    } else if (data_format == DataFormatEnum::fp16) {
        return ConvertFromDataTypeIfNeeded<knowhere::fp16>(src);
    } else if (data_format == DataFormatEnum::bf16) {
        return ConvertFromDataTypeIfNeeded<knowhere::bf16>(src);
    } else if (data_format == DataFormatEnum::int8) {
        return ConvertFromDataTypeIfNeeded<knowhere::int8>(src);
    } else if (data_format == DataFormatEnum::bin1) {
        return ConvertFromDataTypeIfNeeded<knowhere::bin1>(src);
    }
    return nullptr;
}

Status
add_to_index(faiss::Index* const __restrict index, const DataSetPtr& dataset, const DataFormatEnum data_format) {
    const auto* data = dataset->GetTensor();
    const auto rows = dataset->GetRows();
    const auto dim = dataset->GetDim();

    if (data_format == DataFormatEnum::fp32) {
        // add as is
        index->add(rows, reinterpret_cast<const float*>(data));
    } else {
        // convert data into float in pieces and add to the index
        constexpr int64_t n_tmp_rows = 4096;
        std::unique_ptr<float[]> tmp = std::make_unique<float[]>(n_tmp_rows * dim);

        for (int64_t irow = 0; irow < rows; irow += n_tmp_rows) {
            const int64_t start_row = irow;
            const int64_t end_row = std::min(rows, start_row + n_tmp_rows);
            const int64_t count_rows = end_row - start_row;

            if (!convert_rows_to_fp32(data, tmp.get(), data_format, start_row, count_rows, dim)) {
                LOG_KNOWHERE_ERROR_ << "Unsupported data format";
                return Status::invalid_args;
            }

            // add
            index->add(count_rows, tmp.get());
        }
    }

    return Status::success;
}

Status
add_partial_dataset_to_index(faiss::Index* const __restrict index, const DataSetPtr& dataset,
                             const DataFormatEnum data_format, const std::vector<uint32_t>& ids) {
    const auto* data = dataset->GetTensor();

    if (ids.size() > dataset->GetRows()) {
        LOG_KNOWHERE_ERROR_ << "partial ids size larger than whole dataset size";
        return Status::invalid_args;
    }
    const int64_t rows = ids.size();
    const auto dim = dataset->GetDim();

    // convert data into float in pieces and add to the index
    constexpr int64_t n_tmp_rows = 4096;
    std::unique_ptr<float[]> tmp = std::make_unique<float[]>(n_tmp_rows * dim);

    for (int64_t irow = 0; irow < rows; irow += n_tmp_rows) {
        const int64_t start_row = irow;
        const int64_t end_row = std::min(rows, start_row + n_tmp_rows);
        const int64_t count_rows = end_row - start_row;

        if (!convert_rows_to_fp32(data, tmp.get(), data_format, ids.data() + start_row, count_rows, dim)) {
            LOG_KNOWHERE_ERROR_ << "Unsupported data format";
            return Status::invalid_args;
        }

        // add
        index->add(count_rows, tmp.get());
    }

    return Status::success;
}

// IndexFlat and IndexFlatCosine contain raw fp32 data
// IndexScalarQuantizer and IndexScalarQuantizerCosine may contain rar bf16 and fp16 data
//
// returns nullopt if an input index does not contain raw bf16, fp16 or fp32 data
std::optional<DataFormatEnum>
get_index_data_format(const faiss::Index* index) {
    // empty
    if (index == nullptr) {
        return std::nullopt;
    }

    // is it flat?
    // note: IndexFlatCosine preserves the original data, no cosine norm is applied.
    // Cast to baseline ::faiss::IndexFlat so this catches both the knowhere
    // Jaccard-aware subclass (fresh build path) and baseline variants produced
    // by the IxFI/IxF2 deserialization factory.
    auto index_flat = dynamic_cast<const ::faiss::IndexFlat*>(index);
    if (index_flat != nullptr) {
        return DataFormatEnum::fp32;
    }

    // is it binary (1-bit-direct)? Routed through
    // IndexBinaryScalarQuantizer, which replaces the legacy
    // IndexScalarQuantizer(QT_1bit_direct) path.
    if (dynamic_cast<const faiss::cppcontrib::knowhere::IndexBinaryScalarQuantizer*>(index) != nullptr) {
        return DataFormatEnum::bin1;
    }

    // is it sq? All SQ storage produced by knowhere now inherits from
    // baseline faiss::IndexScalarQuantizer (Cosine/SQ4U wrappers,
    // plain IndexHNSWSQ, and refine).
    if (auto* index_sq = dynamic_cast<const faiss::IndexScalarQuantizer*>(index)) {
        switch (index_sq->sq.qtype) {
            case faiss::ScalarQuantizer::QT_bf16:
                return DataFormatEnum::bf16;
            case faiss::ScalarQuantizer::QT_fp16:
                return DataFormatEnum::fp16;
            case faiss::ScalarQuantizer::QT_8bit_direct_signed:
                return DataFormatEnum::int8;
            default:
                return std::nullopt;
        }
    }

    // some other index
    return std::nullopt;
}

// cloned from IndexHNSW.cpp
faiss::DistanceComputer*
storage_distance_computer(const faiss::Index* storage) {
    if (faiss::cppcontrib::knowhere::is_similarity_metric(storage->metric_type)) {
        return new faiss::NegativeDistanceComputer(storage->get_distance_computer());
    } else {
        return storage->get_distance_computer();
    }
}

// there are chances that each partition split by scalar distribution is too small that we could not even train pq on it
// bcz 256 points are needed for a 8-bit pq training in faiss
// combine some small partitions to get a bigger one
// for example: scalar_info= {{1,2}, {3,4,5}, {1}}, base_rows = 3
// we will get {{2, 0}, {1}}, which means we combine the scalar id 0 and 2 together
std::vector<std::vector<int>>
combine_partitions(const std::vector<std::vector<uint32_t>>& scalar_info, const int64_t base_rows) {
    auto scalar_size = scalar_info.size();
    std::vector<int> indices(scalar_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<int> sizes;
    sizes.reserve(scalar_size);
    for (const auto& id_list : scalar_info) {
        sizes.emplace_back(id_list.size());
    }
    std::sort(indices.begin(), indices.end(), [&sizes](size_t i1, size_t i2) { return sizes[i1] < sizes[i2]; });
    std::vector<std::vector<int>> res;
    std::vector<int> cur;
    int64_t cur_size = 0;
    for (auto i : indices) {
        cur_size += sizes[i];
        cur.push_back(i);
        if (cur_size >= base_rows) {
            res.push_back(cur);
            cur.clear();
            cur_size = 0;
        }
    }
    // tail
    if (!cur.empty()) {
        if (res.empty()) {
            res.push_back(cur);
            return res;
        } else {
            res[res.size() - 1].insert(res[res.size() - 1].end(), cur.begin(), cur.end());
        }
    }
    return res;
}

}  // namespace

// Contains an iterator state
struct FaissHnswIteratorWorkspace {
    // hnsw.
    // this pointer is not owned.
    const faiss::cppcontrib::knowhere::HNSW* hnsw = nullptr;

    // nodes that we've already visited
    faiss::cppcontrib::knowhere::Bitset visited_nodes;

    // Computes distances.
    //   This needs to be wrapped with a sign change.
    std::unique_ptr<faiss::DistanceComputer> qdis;
    // Computes refine distances (if refine is available).
    //   This DOES NOT need to be wrapped with a sign change
    std::unique_ptr<faiss::DistanceComputer> qdis_refine;

    // for filtering out nodes
    BitsetView bitset;

    // accumulated alpha
    float accumulated_alpha = 0;

    // visitor
    DummyVisitor graph_visitor;

    // faiss hnsw search params (such as ef)
    faiss::cppcontrib::knowhere::SearchParametersHNSW search_params;

    // the query
    std::unique_ptr<float[]> query;

    // whether the initial search is done or not.
    // basically, upon initialization, we need to traverse to the largest
    //   hnsw layer.
    bool initial_search_done = false;

    // accumulated elements
    std::vector<DistId> dists;

    // TODO test for memory usage of this heap and add a metric monitoring it.
    faiss::cppcontrib::knowhere::IteratorMinHeap to_visit;
};

// Contains an iterator logic
class FaissHnswIterator : public IndexIterator {
 public:
    FaissHnswIterator(const std::shared_ptr<faiss::Index>& index_in,
                      const std::shared_ptr<std::vector<uint32_t>>& labels_in, std::unique_ptr<float[]>&& query_in,
                      const BitsetView& bitset_in, const int32_t ef_in, bool larger_is_closer,
                      const float refine_ratio = 0.5f, const std::vector<uint32_t>& label_to_internal_offset_in = {},
                      const uint32_t mv_base_offset_in = 0, bool use_knowhere_search_pool = true)
        : IndexIterator(larger_is_closer, use_knowhere_search_pool, refine_ratio),
          index{index_in},
          labels{labels_in},
          label_to_internal_offset(label_to_internal_offset_in),
          mv_base_offset(mv_base_offset_in) {
        workspace.accumulated_alpha =
            (bitset_in.count() >= (index->ntotal * HnswSearchThresholds::kHnswSearchKnnBFFilterThreshold))
                ? std::numeric_limits<float>::max()
                : 1.0f;

        // set up a visitor
        workspace.graph_visitor = DummyVisitor();

        // A note about the sign of the result.
        // Our infra is build on structures that track elements with min distance.
        //   So, we multiply distances to (-1) if we need to track max distance,
        //   such as COSINE or IP distances. And, of course, we'll need to multiply
        //   to (-1) again after we're done.

        // TODO: upgrade to refine options && cosine
        const faiss::cppcontrib::knowhere::IndexRefine* index_refine =
            dynamic_cast<const faiss::cppcontrib::knowhere::IndexRefine*>(index.get());
        if (index_refine != nullptr) {
            const faiss::cppcontrib::knowhere::IndexHNSW* index_hnsw =
                dynamic_cast<const faiss::cppcontrib::knowhere::IndexHNSW*>(index_refine->base_index);
            if (index_hnsw == nullptr) {
                // todo: turn constructor into a factory method
                KNOWHERE_THROW_MSG("FaissHnswIterator: refine base index is not an IndexHNSW");
            }

            workspace.hnsw = &index_hnsw->hnsw;

            // wrap a sign, if needed
            workspace.qdis = std::unique_ptr<faiss::DistanceComputer>(storage_distance_computer(index_hnsw));

            if (refine_ratio != 0) {
                // the refine is needed

                // a tricky point here.
                // Basically, if out hnsw index's storage is HasInverseL2Norms, then
                //   this is a cosine index. But because refine always keeps original
                //   data, then we need to use a wrapper over a distance computer
                const faiss::cppcontrib::knowhere::HasInverseL2Norms* has_l2_norms =
                    dynamic_cast<const faiss::cppcontrib::knowhere::HasInverseL2Norms*>(index_hnsw->storage);
                if (has_l2_norms != nullptr) {
                    // add a cosine wrapper over it
                    // DO NOT WRAP A SIGN, by design
                    workspace.qdis_refine = std::unique_ptr<faiss::DistanceComputer>(
                        new faiss::cppcontrib::knowhere::WithCosineNormDistanceComputer(
                            has_l2_norms->get_inverse_l2_norms(), index->d,
                            std::unique_ptr<faiss::DistanceComputer>(
                                index_refine->refine_index->get_distance_computer())));
                } else {
                    // use it as is
                    // DO NOT WRAP A SIGN, by design
                    workspace.qdis_refine =
                        std::unique_ptr<faiss::DistanceComputer>(index_refine->refine_index->get_distance_computer());
                }
            } else {
                // the refine is not needed
                workspace.qdis_refine = nullptr;
            }
        } else {
            const faiss::cppcontrib::knowhere::IndexHNSW* index_hnsw =
                dynamic_cast<const faiss::cppcontrib::knowhere::IndexHNSW*>(index.get());
            if (index_hnsw == nullptr) {
                // todo: turn constructor into a factory method
                KNOWHERE_THROW_MSG("FaissHnswIterator: index is not an IndexHNSW");
            }

            workspace.hnsw = &index_hnsw->hnsw;

            // wrap a sign, if needed
            workspace.qdis = std::unique_ptr<faiss::DistanceComputer>(storage_distance_computer(index_hnsw));
        }

        // set query
        workspace.qdis->set_query(query_in.get());

        if (workspace.qdis_refine != nullptr) {
            workspace.qdis_refine->set_query(query_in.get());
        }

        // set up a buffer that tracks visited points
        workspace.visited_nodes = faiss::cppcontrib::knowhere::Bitset::create_cleared(index->ntotal);

        workspace.search_params.efSearch = ef_in;
        // no need to set this one, use bitsetview directly
        workspace.search_params.sel = nullptr;

        // set up a bitset for filtering database points that we traverse
        workspace.bitset = bitset_in;

        // initial search starts as 'not done'
        workspace.initial_search_done = false;

        // save a query
        workspace.query = std::move(query_in);
    }

 protected:
    template <typename FilterT>
    void
    next_batch(std::function<void(const std::vector<DistId>&)> batch_handler, FilterT& filter) {
        //
        using searcher_type =
            faiss::cppcontrib::knowhere::v2_hnsw_searcher<faiss::DistanceComputer, DummyVisitor,
                                                          faiss::cppcontrib::knowhere::Bitset, FilterT>;

        using storage_idx_t = typename searcher_type::storage_idx_t;
        using idx_t = typename searcher_type::idx_t;

        searcher_type searcher(*workspace.hnsw, *workspace.qdis, workspace.graph_visitor, workspace.visited_nodes,
                               filter, 1.0f, &workspace.search_params);

        // whether to track hnsw stats
        constexpr bool track_hnsw_stats = true;

        // accumulate elements for a new batch?
        if (!workspace.initial_search_done) {
            // yes
            faiss::cppcontrib::knowhere::HNSWStats stats;

            // is the graph empty?
            if (searcher.hnsw.entry_point != -1) {
                // not empty

                // perform a search starting from the initial point
                storage_idx_t nearest = searcher.hnsw.entry_point;
                float d_nearest = searcher.qdis(nearest);

                // iterate through upper levels
                faiss::cppcontrib::knowhere::HNSWStats bottom_levels_stats =
                    searcher.greedy_search_top_levels(nearest, d_nearest);

                // update stats
                if (track_hnsw_stats) {
                    stats.combine(bottom_levels_stats);
                }

                //
                searcher.graph_visitor.visit_level(0);

                // initialize the container for candidates
                const idx_t n_candidates = workspace.search_params.efSearch;
                faiss::cppcontrib::knowhere::NeighborSetDoublePopList retset(n_candidates);

                // initialize retset with a single 'nearest' point
                {
                    if (!searcher.filter.is_member(nearest)) {
                        retset.insert(faiss::cppcontrib::knowhere::Neighbor(
                            nearest, d_nearest, faiss::cppcontrib::knowhere::Neighbor::kInvalid));
                    } else {
                        retset.insert(faiss::cppcontrib::knowhere::Neighbor(
                            nearest, d_nearest, faiss::cppcontrib::knowhere::Neighbor::kValid));
                    }

                    searcher.visited_nodes[nearest] = true;
                }

                // perform the search of the level 0.
                faiss::cppcontrib::knowhere::HNSWStats local_stats =
                    searcher.search_on_a_level(retset, 0, &workspace.to_visit, workspace.accumulated_alpha);
                if (track_hnsw_stats) {
                    stats.combine(local_stats);
                }

                // populate the result
                workspace.dists.reserve(retset.size());
                for (size_t i = 0; i < retset.size(); i++) {
                    workspace.dists.emplace_back(retset[i].id, retset[i].distance);
                }
            }

            workspace.initial_search_done = true;
        } else {
            // the initial batch is accumulated

            workspace.dists.clear();

            // TODO: currently each time iterator.Next() is called, we return 1 result but adds more than 1 results to
            // to_visit. Consider limit the size of visit by searching 1 step only after several Next() calls. Careful:
            // how does such strategy affect the correctness of the search?
            faiss::cppcontrib::knowhere::IteratorMinHeap* to_visit_ptr = &workspace.to_visit;

            while (!to_visit_ptr->empty()) {
                auto top = to_visit_ptr->top();
                to_visit_ptr->pop();

                auto add_search_candidate = [to_visit_ptr](auto neighbor) {
                    to_visit_ptr->push(neighbor);
                    return true;
                };

                searcher.evaluate_single_node(top.id, 0, workspace.accumulated_alpha, add_search_candidate);

                if (searcher.filter.is_member(top.id)) {
                    workspace.dists.emplace_back(top.id, top.distance);
                    break;
                }
            }
        }

        // Multiply distances to (-1) in case of IP and COSINE distances,
        //   because workspace.qdis() does so.
        // We need to ensure that we pass positive distances into batch_handler(),
        //   thus we need to negate the sign from workspace.qdis().
        if (faiss::cppcontrib::knowhere::is_similarity_metric(index->metric_type)) {
            for (auto& p : workspace.dists) {
                p.val = -p.val;
            }
        }

        if (labels != nullptr) {
            for (auto& p : workspace.dists) {
                p.id = p.id < 0 ? p.id : labels->operator[](p.id);
            }
        }

        // pass back to the handler
        batch_handler(workspace.dists);

        // clear the current batch of processed elements
        workspace.dists.clear();
    }

    void
    next_batch(std::function<void(const std::vector<DistId>&)> batch_handler) override {
        if (workspace.bitset.empty()) {
            using filter_type = faiss::IDSelectorAll;
            filter_type sel;

            next_batch(batch_handler, sel);
        } else {
            using filter_type = knowhere::BitsetViewIDSelector;
            filter_type sel(workspace.bitset);

            next_batch(batch_handler, sel);
        }
    }

    float
    raw_distance(int64_t id) override {
        if (label_to_internal_offset.empty()) {
            return workspace.qdis_refine->operator()(id);
        }
        // todo: Currently, next-batch returns quant results that have already been mapped to labels (external_id),
        // but refine requires the internal offset within the mv-index, so we need to map them back.
        // This reverse mapping is actually quite wasteful.
        // The best solution is to detect if need refine in next-batch, and directly return the internal offset of the
        // mv-index. After refine computation, convert it back to the label (external_id).
        // This involves changing the iterator interface in the base class in index_node.h, which will take some time.
        auto mv_internal_offset = label_to_internal_offset[id] - mv_base_offset;
        return workspace.qdis_refine->operator()(mv_internal_offset);
    }

 private:
    std::shared_ptr<faiss::Index> index;
    std::shared_ptr<std::vector<uint32_t>> labels;
    const std::vector<uint32_t>& label_to_internal_offset;  // internal_offset = label_to_internal_offset[label_id];
    const uint32_t mv_base_offset;                          // mv_internal_offset = internal_offset - mv_base_offset;

    FaissHnswIteratorWorkspace workspace;
};

//
class BaseFaissRegularIndexHNSWNode : public BaseFaissRegularIndexNode {
 public:
    BaseFaissRegularIndexHNSWNode(const int32_t& version, const Object& object, DataFormatEnum data_format_in)
        : BaseFaissRegularIndexNode(version, object), data_format{data_format_in} {
    }

    bool
    HasRawData(const std::string& metric_type) const override {
        if (indexes.empty()) {
            return false;
        }

        // check whether there is an index to reconstruct a raw data from
        // only check one is enough
        return (GetIndexToReconstructRawDataFrom(0) != nullptr);
    }

    expected<DataSetPtr>
    GetVectorByIds(const DataSetPtr dataset, milvus::OpContext* op_context) const override {
        if (indexes.empty()) {
            return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
        }
        for (const auto& index : indexes) {
            if (index == nullptr) {
                return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
            }
            if (!index->is_trained) {
                return expected<DataSetPtr>::Err(Status::index_not_trained, "index not trained");
            }
        }

        // an index that is used for reconstruction
        std::vector<const faiss::Index*> indexes_to_reconstruct_from(indexes.size());
        for (auto i = 0; i < indexes.size(); ++i) {
            const faiss::Index* index_to_reconstruct_from = GetIndexToReconstructRawDataFrom(i);

            // check whether raw data is available
            if (index_to_reconstruct_from == nullptr) {
                return expected<DataSetPtr>::Err(
                    Status::invalid_index_error,
                    "The index does not contain a raw data, cannot proceed with GetVectorByIds");
            }
            indexes_to_reconstruct_from[i] = index_to_reconstruct_from;
        }

        // perform reconstruction
        auto dim = Dim();
        auto rows = dataset->GetRows();
        auto ids = dataset->GetIds();

        auto get_vector = [&](int64_t id, float* result) -> bool {
            if (indexes.size() == 1) {
                indexes_to_reconstruct_from[0]->reconstruct(id, result);
            } else {
                auto it =
                    std::lower_bound(index_rows_sum.begin(), index_rows_sum.end(), label_to_internal_offset[id] + 1);
                if (it == index_rows_sum.end()) {
                    return false;
                }
                auto index_id = std::distance(index_rows_sum.begin(), it) - 1;
                indexes_to_reconstruct_from[index_id]->reconstruct(
                    label_to_internal_offset[id] - index_rows_sum[index_id], result);
            }
            return true;
        };

        try {
            // limit the parallel of reconstruction
            ThreadPool::ScopedSearchOmpSetter setter(1);

            if (data_format == DataFormatEnum::fp32) {
                // perform a direct reconstruction for fp32 data
                auto data = std::make_unique<float[]>(dim * rows);
                for (int64_t i = 0; i < rows; i++) {
                    const int64_t id = ids[i];
                    assert(id >= 0 && id < Count());
                    if (!get_vector(id, data.get() + i * dim)) {
                        return expected<DataSetPtr>::Err(Status::invalid_index_error,
                                                         "index inner error, cannot proceed with GetVectorByIds");
                    }
                }
                return GenResultDataSet(rows, dim, std::move(data));
            } else if (data_format == DataFormatEnum::fp16) {
                auto data = std::make_unique<knowhere::fp16[]>(dim * rows);
                // faiss produces fp32 data format, we need some other format.
                // Let's create a temporary fp32 buffer for this.
                auto tmp = std::make_unique<float[]>(dim);
                for (int64_t i = 0; i < rows; i++) {
                    const int64_t id = ids[i];
                    assert(id >= 0 && id < Count());
                    if (!get_vector(id, tmp.get())) {
                        return expected<DataSetPtr>::Err(Status::invalid_index_error,
                                                         "index inner error, cannot proceed with GetVectorByIds");
                    }
                    if (!convert_rows_from_fp32(tmp.get(), data.get(), data_format, i, 1, dim)) {
                        return expected<DataSetPtr>::Err(Status::invalid_args, "Unsupported data format");
                    }
                }
                return GenResultDataSet(rows, dim, std::move(data));
            } else if (data_format == DataFormatEnum::bf16) {
                auto data = std::make_unique<knowhere::bf16[]>(dim * rows);
                // faiss produces fp32 data format, we need some other format.
                // Let's create a temporary fp32 buffer for this.
                auto tmp = std::make_unique<float[]>(dim);
                for (int64_t i = 0; i < rows; i++) {
                    const int64_t id = ids[i];
                    assert(id >= 0 && id < Count());
                    if (!get_vector(id, tmp.get())) {
                        return expected<DataSetPtr>::Err(Status::invalid_index_error,
                                                         "index inner error, cannot proceed with GetVectorByIds");
                    }
                    if (!convert_rows_from_fp32(tmp.get(), data.get(), data_format, i, 1, dim)) {
                        return expected<DataSetPtr>::Err(Status::invalid_args, "Unsupported data format");
                    }
                }
                return GenResultDataSet(rows, dim, std::move(data));
            } else if (data_format == DataFormatEnum::int8) {
                auto data = std::make_unique<knowhere::int8[]>(dim * rows);
                // faiss produces fp32 data format, we need some other format.
                // Let's create a temporary fp32 buffer for this.
                auto tmp = std::make_unique<float[]>(dim);
                for (int64_t i = 0; i < rows; i++) {
                    const int64_t id = ids[i];
                    assert(id >= 0 && id < Count());
                    if (!get_vector(id, tmp.get())) {
                        return expected<DataSetPtr>::Err(Status::invalid_index_error,
                                                         "index inner error, cannot proceed with GetVectorByIds");
                    }
                    if (!convert_rows_from_fp32(tmp.get(), data.get(), data_format, i, 1, dim)) {
                        return expected<DataSetPtr>::Err(Status::invalid_args, "Unsupported data format");
                    }
                }
                return GenResultDataSet(rows, dim, std::move(data));
            } else if (data_format == DataFormatEnum::bin1) {
                auto uint8_dim = (dim + 7) / 8;
                auto data = std::make_unique<knowhere::bin1[]>(uint8_dim * rows);
                // faiss produces fp32 data format, we need some other format.
                // Let's create a temporary fp32 buffer for this.
                auto tmp = std::make_unique<float[]>(uint8_dim);
                for (int64_t i = 0; i < rows; i++) {
                    const int64_t id = ids[i];
                    assert(id >= 0 && id < Count());
                    if (!get_vector(id, tmp.get())) {
                        return expected<DataSetPtr>::Err(Status::invalid_index_error,
                                                         "index inner error, cannot proceed with GetVectorByIds");
                    }
                    if (!convert_rows_from_fp32(tmp.get(), data.get(), data_format, i, 1, dim)) {
                        return expected<DataSetPtr>::Err(Status::invalid_args, "Unsupported data format");
                    }
                }
                return GenResultDataSet(rows, dim, std::move(data));
            } else {
                return expected<DataSetPtr>::Err(Status::invalid_args, "Unsupported data format");
            }
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
            return expected<DataSetPtr>::Err(Status::faiss_inner_error, e.what());
        }
    }

    expected<DataSetPtr>
    Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset_,
           milvus::OpContext* op_context) const override {
        if (this->indexes.empty()) {
            return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
        }
        for (const auto& index : indexes) {
            if (index == nullptr) {
                return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
            }
            if (!index->is_trained) {
                return expected<DataSetPtr>::Err(Status::index_not_trained, "index not trained");
            }
        }

        const auto dim = dataset->GetDim();
        const auto rows = dataset->GetRows();
        const auto* data = dataset->GetTensor();

        const auto hnsw_cfg = static_cast<const FaissHnswConfig&>(*cfg);
        const auto k = hnsw_cfg.k.value();

        BitsetView bitset(bitset_);
        if (!internal_offset_to_most_external_id.empty()) {
            if (emb_list_offset_ != nullptr) {
                // if emb list, manually calculate the number of filtered out ids
                size_t num_filtered_out_ids = 0;
                for (size_t i = 0; i < bitset.size(); i++) {
                    if (bitset.test(i)) {
                        num_filtered_out_ids += emb_list_offset_->offset[i + 1] - emb_list_offset_->offset[i];
                    }
                }
                bitset.set_out_ids(internal_offset_to_most_external_id.data(),
                                   internal_offset_to_most_external_id.size(), num_filtered_out_ids);
            } else {
                bitset.set_out_ids(internal_offset_to_most_external_id.data(),
                                   internal_offset_to_most_external_id.size());
            }
        }
        auto index_id = getIndexToSearchByScalarInfo(bitset);
        if (index_id < 0) {
            return expected<DataSetPtr>::Err(Status::invalid_args, "partition key value not correctly set");
        }
        if (indexes.size() > 1) {
            // calculate more accurate filter statistics for the single mv-index.
            size_t num_mv_ids = labels[index_id].get()->size();
            size_t num_mv_filtered_out_ids = num_mv_ids - (bitset.size() - bitset.count());
            if (!bitset.has_out_ids()) {
                bitset.set_out_ids(labels[index_id].get()->data(), num_mv_ids, num_mv_filtered_out_ids);
            } else {
                bitset.set_out_ids(internal_offset_to_most_external_id.data(), num_mv_ids, num_mv_filtered_out_ids);
                bitset.set_id_offset(index_rows_sum[index_id]);
            }
        }

        feder::hnsw::FederResultUniq feder_result;
        if (hnsw_cfg.trace_visit.value()) {
            if (rows != 1) {
                return expected<DataSetPtr>::Err(Status::invalid_args, "a single query vector is required");
            }
            feder_result = std::make_unique<feder::hnsw::FederResult>();
        }

        // check for brute-force search
        auto whether_bf_search = WhetherPerformBruteForceSearch(indexes[index_id].get(), hnsw_cfg, bitset);

        if (!whether_bf_search.has_value()) {
            return expected<DataSetPtr>::Err(Status::invalid_args, "k parameter is missing");
        }

        // whether a user wants a refine
        const bool whether_to_enable_refine = hnsw_cfg.refine_k.has_value();

        // set up an index wrapper
        auto [index_wrapper, is_refined] = create_conditional_hnsw_wrapper(
            indexes[index_id].get(), hnsw_cfg, whether_bf_search.value_or(false), whether_to_enable_refine);

        if (index_wrapper == nullptr) {
            return expected<DataSetPtr>::Err(Status::invalid_args, "an input index seems to be unrelated to HNSW");
        }

        // set up a bf wrapper as fallback
        std::unique_ptr<faiss::Index> bf_index_wrapper = nullptr;
        faiss::Index* bf_index_wrapper_ptr = nullptr;
        if (!whether_bf_search.value_or(false)) {
            std::tie(bf_index_wrapper, is_refined) =
                create_conditional_hnsw_wrapper(indexes[index_id].get(), hnsw_cfg, true, whether_to_enable_refine);
            if (bf_index_wrapper == nullptr) {
                return expected<DataSetPtr>::Err(Status::invalid_args, "an input index seems to be unrelated to HNSW");
            }
            bf_index_wrapper_ptr = bf_index_wrapper.get();
        }

        faiss::Index* index_wrapper_ptr = index_wrapper.get();

        // set up faiss search parameters
        knowhere::SearchParametersHNSWWrapper hnsw_search_params;
        if (hnsw_cfg.ef.has_value()) {
            hnsw_search_params.efSearch = hnsw_cfg.ef.value();
        }

        // do not collect HNSW stats
        hnsw_search_params.hnsw_stats = nullptr;
        // set up feder
        hnsw_search_params.feder = feder_result.get();
        // set up kAlpha
        hnsw_search_params.kAlpha = bitset.filter_ratio() * 0.7f;

        // set up a selector
        BitsetViewIDSelector bw_idselector(bitset);
        faiss::IDSelector* id_selector = &bw_idselector;
        hnsw_search_params.sel = id_selector;

        // run
        auto ids = std::make_unique<faiss::idx_t[]>(rows * k);
        auto distances = std::make_unique<float[]>(rows * k);

        try {
            std::vector<folly::Future<folly::Unit>> futs;
            futs.reserve(rows);

            for (int64_t i = 0; i < rows; ++i) {
                futs.emplace_back(search_pool->push([&, idx = i, is_refined = is_refined,
                                                     index_wrapper_ptr = index_wrapper_ptr,
                                                     bf_index_wrapper_ptr = bf_index_wrapper_ptr]() {
                    knowhere::checkCancellation(op_context);
                    // 1 thread per element
                    ThreadPool::ScopedSearchOmpSetter setter(1);

                    // set up a query
                    const float* cur_query = nullptr;

                    std::vector<float> cur_query_tmp(dim);
                    if (data_format == DataFormatEnum::fp32) {
                        cur_query = static_cast<const float*>(data) + idx * dim;
                    } else {
                        convert_rows_to_fp32(data, cur_query_tmp.data(), data_format, idx, 1, dim);
                        cur_query = cur_query_tmp.data();
                    }

                    // set up local results
                    faiss::idx_t* const __restrict local_ids = ids.get() + k * idx;
                    float* const __restrict local_distances = distances.get() + k * idx;

                    // check if we need to perform a brute-force search bcz of the lack of results
                    auto bf_search_needed = [&]() -> bool {
                        size_t real_topk = 0;
                        for (int j = 0; j < k; ++j) {
                            if (local_ids[j] < 0) {
                                continue;
                            }
                            real_topk++;
                        }
                        if (std::cmp_less(real_topk, k) && real_topk < bitset.size() - bitset.count() &&
                            bf_index_wrapper_ptr != nullptr && !hnsw_cfg.disable_fallback_brute_force.value()) {
                            LOG_KNOWHERE_WARNING_ << "required topk: " << k
                                                  << ", but the actual num of results got from hnsw: " << real_topk
                                                  << ", trigger brute force search as fallback for hnsw search";
                            return true;
                        }
                        return false;
                    };

                    // perform the search
                    if (is_refined) {
                        faiss::IndexRefineSearchParameters refine_params;
                        refine_params.k_factor = hnsw_cfg.refine_k.value_or(1);
                        // a refine procedure itself does not need to care about filtering
                        refine_params.sel = nullptr;
                        refine_params.base_index_params = &hnsw_search_params;

                        index_wrapper_ptr->search(1, cur_query, k, local_distances, local_ids, &refine_params);
                        if (bf_search_needed()) {
                            bf_index_wrapper_ptr->search(1, cur_query, k, local_distances, local_ids, &refine_params);
                        }
                    } else {
                        index_wrapper_ptr->search(1, cur_query, k, local_distances, local_ids, &hnsw_search_params);
                        if (bf_search_needed()) {
                            bf_index_wrapper_ptr->search(1, cur_query, k, local_distances, local_ids,
                                                         &hnsw_search_params);
                        }
                    }

                    if (!labels.empty()) {
                        for (int j = 0; j < k; ++j) {
                            local_ids[j] = local_ids[j] < 0 ? local_ids[j] : labels[index_id]->operator[](local_ids[j]);
                        }
                    }
                }));
            }

            // wait for the completion
            WaitAllSuccess(futs);
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
            return expected<DataSetPtr>::Err(Status::faiss_inner_error, e.what());
        }

        auto res = GenResultDataSet(rows, k, std::move(ids), std::move(distances));

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

    std::optional<size_t>
    GetQueryCodeSize(const DataSetPtr dataset) const override {
        auto dim = dataset->GetDim();
        switch (data_format) {
            case DataFormatEnum::fp32:
                return dim * 4;
            case DataFormatEnum::fp16:
            case DataFormatEnum::bf16:
                return dim * 2;
            case DataFormatEnum::int8:
                return dim;
            case DataFormatEnum::bin1:
                return (dim + 7) / 8;
            default:
                LOG_KNOWHERE_ERROR_ << "Unsupported data format";
                return std::nullopt;
        }
    }

    expected<DataSetPtr>
    CalcDistByIDs(const DataSetPtr dataset, const BitsetView& bitset_, const int64_t* labels, const size_t labels_len,
                  const bool is_cosine, milvus::OpContext* op_context) const override {
        // When emb_list_raw_index_ exists (MUVERA/LEMUR), use it for exact distance computation
        if (emb_list_raw_index_) {
            return CalcDistByRawIndex(dataset, labels, labels_len, is_cosine, search_pool, op_context);
        }

        if (this->indexes.empty()) {
            return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
        }
        for (const auto& index : indexes) {
            if (index == nullptr) {
                return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
            }
            if (!index->is_trained) {
                return expected<DataSetPtr>::Err(Status::index_not_trained, "index not trained");
            }
        }
        const auto dim = dataset->GetDim();
        const auto rows = dataset->GetRows();
        const float* data = static_cast<const float*>(dataset->GetTensor());
        auto distances = std::make_unique<float[]>(rows * labels_len);

        BitsetView bitset(bitset_);
        if (!internal_offset_to_most_external_id.empty()) {
            // Note: We do not need to calculate the actual number of filtered ids here;
            // we only need the bitset to determine the index_id.
            int32_t num_filtered_out_ids = 0;
            bitset.set_out_ids(internal_offset_to_most_external_id.data(), internal_offset_to_most_external_id.size(),
                               num_filtered_out_ids);
        }
        auto index_id = getIndexToSearchByScalarInfo(bitset);
        if (index_id < 0) {
            return expected<DataSetPtr>::Err(Status::invalid_args, "partition key value not correctly set");
        }

        try {
            std::vector<folly::Future<folly::Unit>> futs;
            futs.reserve(rows);
            for (auto i = 0; i < rows; i++) {
                futs.emplace_back(search_pool->push([&, idx = i, index_id = index_id]() {
                    knowhere::checkCancellation(op_context);
                    // set up a distance computer
                    std::unique_ptr<faiss::DistanceComputer> dist_computer;
                    const faiss::cppcontrib::knowhere::IndexRefine* index_refine =
                        dynamic_cast<const faiss::cppcontrib::knowhere::IndexRefine*>(indexes[index_id].get());
                    if (index_refine != nullptr) {
                        dist_computer.reset(index_refine->refine_index->get_distance_computer());
                    } else {
                        dist_computer.reset(indexes[index_id]->get_distance_computer());
                    }

                    // set up a query
                    const float* cur_query = nullptr;
                    std::vector<float> cur_query_tmp(dim);
                    if (data_format == DataFormatEnum::fp32) {
                        cur_query = data + idx * dim;
                    } else {
                        convert_rows_to_fp32(data, cur_query_tmp.data(), data_format, idx, 1, dim);
                        cur_query = cur_query_tmp.data();
                    }

                    dist_computer->set_query(cur_query);
                    for (size_t j = 0; j < labels_len; j++) {
                        auto id = labels[j];
                        if (indexes.size() > 1) {
                            id = label_to_internal_offset[labels[j]] - index_rows_sum[index_id];
                        }
                        distances[idx * labels_len + j] = (*dist_computer)(id);
                    }
                }));
            }
            // wait for the completion
            WaitAllSuccess(futs);
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
            return expected<DataSetPtr>::Err(Status::faiss_inner_error, e.what());
        }

        // ids is not used in this context, so we can directly initialize it as an empty unique_ptr
        std::unique_ptr<faiss::idx_t[]> ids = nullptr;
        return GenResultDataSet(rows, labels_len, std::move(ids), std::move(distances));
    };

    expected<DataSetPtr>
    RangeSearch(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset_,
                milvus::OpContext* op_context) const override {
        // Check brute-force threshold BEFORE iterator path.
        // At high filter ratios (>=97%), brute force is much faster than graph traversal
        // because the iterator visits too many filtered-out nodes.
        if (is_ann_iterator_supported() && !this->indexes.empty() && indexes[0] != nullptr) {
            const auto& hnsw_cfg_check = static_cast<const FaissHnswConfig&>(*cfg);
            BitsetView bitset_check(bitset_);
            auto whether_bf = WhetherPerformBruteForceRangeSearch(indexes[0].get(), hnsw_cfg_check, bitset_check);
            if (!whether_bf.has_value() || !whether_bf.value()) {
                // Not brute-force worthy: use iterator path
                return IndexNode::RangeSearch(dataset, std::move(cfg), bitset_, op_context);
            }
            // Fall through to brute-force range search path below
        } else if (is_ann_iterator_supported()) {
            return IndexNode::RangeSearch(dataset, std::move(cfg), bitset_, op_context);
        }
        if (this->indexes.empty()) {
            return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
        }
        for (const auto& index : indexes) {
            if (index == nullptr) {
                return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
            }
            if (!index->is_trained) {
                return expected<DataSetPtr>::Err(Status::index_not_trained, "index not trained");
            }
        }

        const auto dim = dataset->GetDim();
        const auto rows = dataset->GetRows();
        const auto* data = dataset->GetTensor();

        const auto hnsw_cfg = static_cast<const FaissHnswConfig&>(*cfg);
        BitsetView bitset(bitset_);
        if (!internal_offset_to_most_external_id.empty()) {
            bitset.set_out_ids(internal_offset_to_most_external_id.data(), internal_offset_to_most_external_id.size());
        }
        auto index_id = getIndexToSearchByScalarInfo(bitset);
        if (index_id < 0) {
            return expected<DataSetPtr>::Err(Status::invalid_args, "partition key value not correctly set");
        }
        if (indexes.size() > 1) {
            size_t num_mv_ids = labels[index_id].get()->size();
            size_t num_mv_filtered_out_ids = num_mv_ids - (bitset.size() - bitset.count());
            if (!bitset.has_out_ids()) {
                bitset.set_out_ids(labels[index_id].get()->data(), num_mv_ids, num_mv_filtered_out_ids);
            } else {
                bitset.set_out_ids(internal_offset_to_most_external_id.data(), num_mv_ids, num_mv_filtered_out_ids);
                bitset.set_id_offset(index_rows_sum[index_id]);
            }
        }

        const bool is_similarity_metric =
            faiss::cppcontrib::knowhere::is_similarity_metric(indexes[index_id]->metric_type);

        const float radius = hnsw_cfg.radius.value();
        const float range_filter = hnsw_cfg.range_filter.value();

        feder::hnsw::FederResultUniq feder_result;
        if (hnsw_cfg.trace_visit.value()) {
            if (rows != 1) {
                return expected<DataSetPtr>::Err(Status::invalid_args, "a single query vector is required");
            }
            feder_result = std::make_unique<feder::hnsw::FederResult>();
        }

        // check for brute-force search
        auto whether_bf_search = WhetherPerformBruteForceRangeSearch(indexes[index_id].get(), hnsw_cfg, bitset);

        if (!whether_bf_search.has_value()) {
            return expected<DataSetPtr>::Err(Status::invalid_args, "ef parameter is missing");
        }

        // whether a user wants a refine
        const bool whether_to_enable_refine = true;

        // set up an index wrapper
        auto [index_wrapper, is_refined] = create_conditional_hnsw_wrapper(
            indexes[index_id].get(), hnsw_cfg, whether_bf_search.value_or(false), whether_to_enable_refine);

        if (index_wrapper == nullptr) {
            return expected<DataSetPtr>::Err(Status::invalid_args, "an input index seems to be unrelated to HNSW");
        }

        faiss::Index* index_wrapper_ptr = index_wrapper.get();

        // set up faiss search parameters
        knowhere::SearchParametersHNSWWrapper hnsw_search_params;

        if (hnsw_cfg.ef.has_value()) {
            hnsw_search_params.efSearch = hnsw_cfg.ef.value();
        }

        // do not collect HNSW stats
        hnsw_search_params.hnsw_stats = nullptr;
        // set up feder
        hnsw_search_params.feder = feder_result.get();
        // set up kAlpha
        hnsw_search_params.kAlpha = bitset.filter_ratio() * 0.7f;

        // set up a selector
        BitsetViewIDSelector bw_idselector(bitset);
        faiss::IDSelector* id_selector = &bw_idselector;
        hnsw_search_params.sel = id_selector;

        ////////////////////////////////////////////////////////////////
        // run
        std::vector<std::vector<int64_t>> result_id_array(rows);
        std::vector<std::vector<float>> result_dist_array(rows);

        try {
            std::vector<folly::Future<folly::Unit>> futs;
            futs.reserve(rows);

            // a sequential version
            for (int64_t i = 0; i < rows; ++i) {
                // const int64_t idx = i;
                // {

                futs.emplace_back(
                    search_pool->push([&, idx = i, is_refined = is_refined, index_wrapper_ptr = index_wrapper_ptr] {
                        knowhere::checkCancellation(op_context);
                        // 1 thread per element
                        ThreadPool::ScopedSearchOmpSetter setter(1);

                        // set up a query
                        const float* cur_query = nullptr;

                        std::vector<float> cur_query_tmp(dim);
                        if (data_format == DataFormatEnum::fp32) {
                            cur_query = static_cast<const float*>(data) + idx * dim;
                        } else {
                            convert_rows_to_fp32(data, cur_query_tmp.data(), data_format, idx, 1, dim);
                            cur_query = cur_query_tmp.data();
                        }

                        // initialize a buffer
                        faiss::RangeSearchResult res(1);

                        // perform the search
                        if (is_refined) {
                            faiss::IndexRefineSearchParameters refine_params;
                            refine_params.k_factor = hnsw_cfg.refine_k.value_or(1);
                            // a refine procedure itself does not need to care about filtering
                            refine_params.sel = nullptr;
                            refine_params.base_index_params = &hnsw_search_params;

                            index_wrapper_ptr->range_search(1, cur_query, radius, &res, &refine_params);
                        } else {
                            index_wrapper_ptr->range_search(1, cur_query, radius, &res, &hnsw_search_params);
                        }

                        // post-process
                        const size_t elem_cnt = res.lims[1];
                        result_dist_array[idx].resize(elem_cnt);
                        result_id_array[idx].resize(elem_cnt);

                        if (labels.empty()) {
                            for (size_t j = 0; j < elem_cnt; j++) {
                                result_dist_array[idx][j] = res.distances[j];
                                result_id_array[idx][j] = res.labels[j];
                            }
                        } else {
                            for (size_t j = 0; j < elem_cnt; j++) {
                                result_dist_array[idx][j] = res.distances[j];
                                result_id_array[idx][j] =
                                    res.labels[j] < 0 ? res.labels[j] : labels[index_id]->operator[](res.labels[j]);
                            }
                        }

                        if (hnsw_cfg.range_filter.value() != defaultRangeFilter) {
                            FilterRangeSearchResultForOneNq(result_dist_array[idx], result_id_array[idx],
                                                            is_similarity_metric, radius, range_filter);
                        }
                    }));
            }

            // wait for the completion
            WaitAllSuccess(futs);
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
            return expected<DataSetPtr>::Err(Status::faiss_inner_error, e.what());
        }

        //
        RangeSearchResult range_search_result =
            GetRangeSearchResult(result_dist_array, result_id_array, is_similarity_metric, rows, radius, range_filter);

        return GenResultDataSet(rows, std::move(range_search_result));
    }

 protected:
    DataFormatEnum data_format;

    std::vector<std::vector<int>> tmp_combined_scalar_ids;

    Status
    AddInternal(const DataSetPtr dataset, const Config&) override {
        if (isIndexEmpty()) {
            LOG_KNOWHERE_ERROR_ << "Can not add data to an empty index.";
            return Status::empty_index;
        }

        auto rows = dataset->GetRows();

        const std::unordered_map<int64_t, std::vector<std::vector<uint32_t>>>& scalar_info_map =
            dataset->Get<std::unordered_map<int64_t, std::vector<std::vector<uint32_t>>>>(meta::SCALAR_INFO);
        if (scalar_info_map.empty() || tmp_combined_scalar_ids.size() <= 1) {
            try {
                LOG_KNOWHERE_INFO_ << "Adding " << rows << " rows to HNSW Index";

                auto status = add_to_index(indexes[0].get(), dataset, data_format);
                return status;
            } catch (const std::exception& e) {
                LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
                return Status::faiss_inner_error;
            }
        }

        if (scalar_info_map.size() > 1) {
            LOG_KNOWHERE_WARNING_ << "vector index build with multiple scalar info is not supported";
            return Status::invalid_args;
        }
        LOG_KNOWHERE_INFO_ << "Add data to Index with Scalar Info";

        try {
            for (const auto& [field_id, scalar_info] : scalar_info_map) {
                for (auto i = 0; i < tmp_combined_scalar_ids.size(); ++i) {
                    for (auto j = 0; j < tmp_combined_scalar_ids[i].size(); ++j) {
                        auto id = tmp_combined_scalar_ids[i][j];
                        // hnsw
                        LOG_KNOWHERE_INFO_ << "Adding " << scalar_info[id].size() << " to HNSW Index";

                        auto status =
                            add_partial_dataset_to_index(indexes[i].get(), dataset, data_format, scalar_info[id]);
                        if (status != Status::success) {
                            return status;
                        }
                    }
                }
            }
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
            return Status::faiss_inner_error;
        }

        return Status::success;
    }

    const faiss::Index*
    GetIndexToReconstructRawDataFrom(int i) const {
        if (indexes.size() <= i) {
            return nullptr;
        }
        if (indexes[i] == nullptr) {
            return nullptr;
        }

        // an index that is used for reconstruction
        const faiss::Index* index_to_reconstruct_from = nullptr;

        // check whether our index uses refine
        auto index_refine = dynamic_cast<const faiss::cppcontrib::knowhere::IndexRefine*>(indexes[i].get());
        if (index_refine == nullptr) {
            // non-refined index

            // cast as IndexHNSW
            auto index_hnsw = dynamic_cast<const faiss::cppcontrib::knowhere::IndexHNSW*>(indexes[i].get());
            if (index_hnsw == nullptr) {
                // this is unexpected, we expect IndexHNSW
                return nullptr;
            }

            // storage index is the one that holds the raw data
            auto index_data_format = get_index_data_format(index_hnsw->storage);

            // make sure that its data format matches our input format
            if (index_data_format.has_value() && index_data_format.value() == data_format) {
                index_to_reconstruct_from = index_hnsw->storage;
            }
        } else {
            // refined index

            // refine index holds the raw data
            auto index_data_format = get_index_data_format(index_refine->refine_index);

            // make sure that its data format matches our input format
            if (index_data_format.has_value() && index_data_format.value() == data_format) {
                index_to_reconstruct_from = index_refine->refine_index;
            }
        }

        // done
        return index_to_reconstruct_from;
    }

    Status
    TrainIndexByScalarInfo(std::function<Status(const float* data, const int i, const int64_t rows)> train_index,
                           const std::vector<std::vector<uint32_t>>& scalar_info, const void* data, const int64_t rows,
                           const int64_t dim) {
        label_to_internal_offset.resize(rows);
        index_rows_sum.resize(tmp_combined_scalar_ids.size() + 1);
        labels.resize(tmp_combined_scalar_ids.size());
        indexes.resize(tmp_combined_scalar_ids.size());
        for (auto i = 0; i < tmp_combined_scalar_ids.size(); ++i) {
            size_t partition_size = 0;
            for (int j : tmp_combined_scalar_ids[i]) {
                partition_size += scalar_info[j].size();
            }
            std::unique_ptr<float[]> tmp_data = std::make_unique<float[]>(dim * partition_size);
            labels[i] = std::make_shared<std::vector<uint32_t>>(partition_size);
            index_rows_sum[i + 1] = index_rows_sum[i] + partition_size;
            size_t cur_size = 0;

            for (size_t j = 0; j < tmp_combined_scalar_ids[i].size(); ++j) {
                auto scalar_id = tmp_combined_scalar_ids[i][j];
                if (!convert_rows_to_fp32(data, tmp_data.get() + dim * cur_size, data_format,
                                          scalar_info[scalar_id].data(), scalar_info[scalar_id].size(), dim)) {
                    LOG_KNOWHERE_ERROR_ << "Unsupported data format";
                    return Status::invalid_args;
                }
                for (size_t m = 0; m < scalar_info[scalar_id].size(); ++m) {
                    labels[i]->operator[](cur_size + m) = scalar_info[scalar_id][m];
                    label_to_internal_offset[scalar_info[scalar_id][m]] = index_rows_sum[i] + cur_size + m;
                }
                cur_size += scalar_info[scalar_id].size();
            }

            Status s = train_index(tmp_data.get(), i, partition_size);
            if (s != Status::success) {
                return s;
            }
        }
        return Status::success;
    }

 public:
    bool
    is_ann_iterator_supported() const {
        if (data_format != DataFormatEnum::fp32 && data_format != DataFormatEnum::fp16 &&
            data_format != DataFormatEnum::bf16 && data_format != DataFormatEnum::int8 &&
            data_format != DataFormatEnum::bin1) {
            return false;
        }
        return true;
    }

    expected<std::vector<IndexNode::IteratorPtr>>
    AnnIterator(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset_,
                bool use_knowhere_search_pool, milvus::OpContext* op_context) const override {
        if (isIndexEmpty()) {
            LOG_KNOWHERE_ERROR_ << "creating iterator on empty index";
            return expected<std::vector<IndexNode::IteratorPtr>>::Err(Status::empty_index, "index not loaded");
        }

        if (!is_ann_iterator_supported()) {
            LOG_KNOWHERE_ERROR_ << "Unsupported data format";
            return expected<std::vector<IndexNode::IteratorPtr>>::Err(Status::invalid_args, "unsupported data format");
        }

        // parse parameters
        const auto dim = dataset->GetDim();
        const auto n_queries = dataset->GetRows();
        const auto data = dataset->GetTensor();

        auto vec = std::vector<IndexNode::IteratorPtr>(n_queries, nullptr);

        const FaissHnswConfig& hnsw_cfg = static_cast<const FaissHnswConfig&>(*cfg);
        BitsetView bitset(bitset_);
        if (!internal_offset_to_most_external_id.empty()) {
            bitset.set_out_ids(internal_offset_to_most_external_id.data(), internal_offset_to_most_external_id.size());
        }
        int index_id = getIndexToSearchByScalarInfo(bitset);
        if (index_id < 0) {
            return expected<std::vector<IndexNode::IteratorPtr>>::Err(Status::invalid_args,
                                                                      "partition key value not correctly set");
        }
        if (indexes.size() > 1) {
            size_t num_mv_ids = labels[index_id].get()->size();
            size_t num_mv_filtered_out_ids = num_mv_ids - (bitset.size() - bitset.count());
            if (!bitset.has_out_ids()) {
                bitset.set_out_ids(labels[index_id].get()->data(), num_mv_ids, num_mv_filtered_out_ids);
            } else {
                bitset.set_out_ids(internal_offset_to_most_external_id.data(), num_mv_ids, num_mv_filtered_out_ids);
                bitset.set_id_offset(index_rows_sum[index_id]);
            }
        }
        const bool is_cosine = IsMetricType(hnsw_cfg.metric_type.value(), knowhere::metric::COSINE);
        const bool larger_is_closer = (IsMetricType(hnsw_cfg.metric_type.value(), knowhere::metric::IP) || is_cosine);

        const auto ef = hnsw_cfg.ef.value_or(kIteratorSeedEf);

        try {
            for (int64_t i = 0; i < n_queries; i++) {
                // The query data is always cloned
                std::unique_ptr<float[]> cur_query = std::make_unique<float[]>(dim);

                switch (data_format) {
                    case DataFormatEnum::fp32:
                        std::copy_n(reinterpret_cast<const float*>(data) + i * dim, dim, cur_query.get());
                        break;
                    case DataFormatEnum::fp16:
                    case DataFormatEnum::bf16:
                    case DataFormatEnum::int8:
                    case DataFormatEnum::bin1:
                        convert_rows_to_fp32(data, cur_query.get(), data_format, i, 1, dim);
                        break;
                    default:
                        // invalid one. Should not be triggered, bcz input parameters are validated.
                        LOG_KNOWHERE_ERROR_ << "unsupported data format: " << static_cast<int>(data_format);
                        return expected<std::vector<IndexNode::IteratorPtr>>::Err(
                            Status::invalid_args,
                            "unsupported data format: " + std::to_string(static_cast<int>(data_format)));
                }

                const bool should_use_refine =
                    (dynamic_cast<const faiss::cppcontrib::knowhere::IndexRefine*>(indexes[index_id].get()) != nullptr);

                const float iterator_refine_ratio =
                    should_use_refine ? hnsw_cfg.iterator_refine_ratio.value_or(0.5) : 0;

                // create an iterator and initialize it
                //   refine is not needed for flat
                //   hnsw_cfg.iterator_refine_ratio.value_or(0.5f)

                uint32_t mv_base_offset = index_rows_sum.size() > index_id ? index_rows_sum[index_id] : 0;

                auto it = std::make_shared<FaissHnswIterator>(
                    indexes[index_id], labels.empty() ? nullptr : labels[index_id], std::move(cur_query), bitset, ef,
                    larger_is_closer, iterator_refine_ratio, label_to_internal_offset, mv_base_offset,
                    use_knowhere_search_pool);
                // store
                vec[i] = it;
            }
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
            return expected<std::vector<IndexNode::IteratorPtr>>::Err(Status::faiss_inner_error, e.what());
        }
        return vec;
    }
};

//
class BaseFaissRegularIndexHNSWFlatNode : public BaseFaissRegularIndexHNSWNode {
 public:
    BaseFaissRegularIndexHNSWFlatNode(const int32_t& version, const Object& object, DataFormatEnum data_format)
        : BaseFaissRegularIndexHNSWNode(version, object, data_format) {
    }

    static std::unique_ptr<BaseConfig>
    StaticCreateConfig() {
        return std::make_unique<FaissHnswFlatConfig>();
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return StaticCreateConfig();
    }

    std::string
    Type() const override {
        return knowhere::IndexEnum::INDEX_HNSW;
    }

 protected:
    Status
    TrainInternal(const DataSetPtr dataset, const Config& cfg) override {
        // number of rows
        auto rows = dataset->GetRows();
        // dimensionality of the data
        auto dim = dataset->GetDim();
        // data
        auto data = dataset->GetTensor();

        // config
        auto hnsw_cfg = static_cast<const FaissHnswFlatConfig&>(cfg);

        auto metric = Str2FaissMetricType(hnsw_cfg.metric_type.value());
        if (!metric.has_value()) {
            LOG_KNOWHERE_ERROR_ << "Invalid metric type: " << hnsw_cfg.metric_type.value();
            return Status::invalid_metric_type;
        }

        // create an index
        const bool is_cosine = IsMetricType(hnsw_cfg.metric_type.value(), metric::COSINE);
        const bool is_binary = data_format == DataFormatEnum::bin1;

        std::unique_ptr<faiss::cppcontrib::knowhere::IndexHNSW> hnsw_index;
        auto train_index = [&](const float* data, const int i, const int64_t rows) {
            if (is_binary) {
                if (metric.value() == faiss::MetricType::METRIC_Hamming ||
                    metric.value() == faiss::MetricType::METRIC_Jaccard) {
                    hnsw_index = std::make_unique<faiss::cppcontrib::knowhere::IndexHNSWBinary>(dim, hnsw_cfg.M.value(),
                                                                                                metric.value());
                } else {
                    LOG_KNOWHERE_ERROR_ << "Unsupported metric for binary data: " << hnsw_cfg.metric_type.value();
                    return Status::invalid_metric_type;
                }
            } else {
                if (is_cosine) {
                    if (data_format == DataFormatEnum::fp32) {
                        hnsw_index =
                            std::make_unique<faiss::cppcontrib::knowhere::IndexHNSWFlatCosine>(dim, hnsw_cfg.M.value());
                    } else if (data_format == DataFormatEnum::fp16) {
                        hnsw_index = std::make_unique<faiss::cppcontrib::knowhere::IndexHNSWSQCosine>(
                            dim, faiss::ScalarQuantizer::QT_fp16, hnsw_cfg.M.value());
                    } else if (data_format == DataFormatEnum::bf16) {
                        hnsw_index = std::make_unique<faiss::cppcontrib::knowhere::IndexHNSWSQCosine>(
                            dim, faiss::ScalarQuantizer::QT_bf16, hnsw_cfg.M.value());
                    } else if (data_format == DataFormatEnum::int8) {
                        hnsw_index = std::make_unique<faiss::cppcontrib::knowhere::IndexHNSWSQCosine>(
                            dim, faiss::ScalarQuantizer::QT_8bit_direct_signed, hnsw_cfg.M.value());
                    } else {
                        LOG_KNOWHERE_ERROR_ << "Unsupported metric type: " << hnsw_cfg.metric_type.value();
                        return Status::invalid_metric_type;
                    }
                } else {
                    if (data_format == DataFormatEnum::fp32) {
                        hnsw_index = std::make_unique<faiss::cppcontrib::knowhere::IndexHNSWFlat>(
                            dim, hnsw_cfg.M.value(), metric.value());
                    } else if (data_format == DataFormatEnum::fp16) {
                        hnsw_index = std::make_unique<faiss::cppcontrib::knowhere::IndexHNSWSQ>(
                            dim, faiss::ScalarQuantizer::QT_fp16, hnsw_cfg.M.value(), metric.value());
                    } else if (data_format == DataFormatEnum::bf16) {
                        hnsw_index = std::make_unique<faiss::cppcontrib::knowhere::IndexHNSWSQ>(
                            dim, faiss::ScalarQuantizer::QT_bf16, hnsw_cfg.M.value(), metric.value());
                    } else if (data_format == DataFormatEnum::int8) {
                        hnsw_index = std::make_unique<faiss::cppcontrib::knowhere::IndexHNSWSQ>(
                            dim, faiss::ScalarQuantizer::QT_8bit_direct_signed, hnsw_cfg.M.value(), metric.value());
                    } else {
                        LOG_KNOWHERE_ERROR_ << "Unsupported metric type: " << hnsw_cfg.metric_type.value();
                        return Status::invalid_metric_type;
                    }
                }
            }

            hnsw_index->hnsw.efConstruction = hnsw_cfg.efConstruction.value();
            // train
            LOG_KNOWHERE_INFO_ << "Training HNSW Index";
            // this function does nothing for the given parameters and indices.
            //   as a result, I'm just keeping it to have is_trained set to true.
            // WARNING: this may cause problems if ->train() performs some action
            //   based on the data in the future. Otherwise, data needs to be
            //   converted into float*.
            hnsw_index->train(rows, data);

            // done
            indexes[i] = std::move(hnsw_index);
            return Status::success;
        };

        const std::unordered_map<int64_t, std::vector<std::vector<uint32_t>>>& scalar_info_map =
            dataset->Get<std::unordered_map<int64_t, std::vector<std::vector<uint32_t>>>>(meta::SCALAR_INFO);
        if (scalar_info_map.size() > 1) {
            LOG_KNOWHERE_WARNING_ << "vector index build with multiple scalar info is not supported";
            return Status::invalid_args;
        }
        for (const auto& [field_id, scalar_info] : scalar_info_map) {
            tmp_combined_scalar_ids =
                scalar_info.size() > 1 ? combine_partitions(scalar_info, 128) : std::vector<std::vector<int>>();
        }

        // no scalar info or just one partition(after possible combination), build index on whole data
        if (scalar_info_map.empty() || tmp_combined_scalar_ids.size() <= 1) {
            return train_index(static_cast<const float*>(data), 0, rows);
        }

        LOG_KNOWHERE_INFO_ << "Train HNSW index with Scalar Info";
        for (const auto& [field_id, scalar_info] : scalar_info_map) {
            return TrainIndexByScalarInfo(train_index, scalar_info, data, rows, dim);
        }
        return Status::success;
    }
};

template <typename DataType>
class BaseFaissRegularIndexHNSWFlatNodeTemplate : public BaseFaissRegularIndexHNSWFlatNode {
 public:
    BaseFaissRegularIndexHNSWFlatNodeTemplate(const int32_t& version, const Object& object)
        : BaseFaissRegularIndexHNSWFlatNode(version, object, datatype_v<DataType>) {
    }

    static bool
    StaticHasRawData(const knowhere::BaseConfig& config, const IndexVersion& version) {
        return true;
    }

    static Status
    StaticConfigCheck(const Config& cfg, PARAM_TYPE paramType, std::string& msg) {
        return HnswIndexNode<DataType, hnswlib::QuantType::None>::StaticConfigCheck(cfg, paramType, msg);
    }
};

// this is a regular node that can be initialized as some existing index type,
//   but a deserialization may override its search behavior.
// It is a concrete implementation's responsibility to initialize BaseIndex and
//   FallbackSearchIndex properly.
class HNSWIndexNodeWithFallback : public IndexNode {
 public:
    HNSWIndexNodeWithFallback(const int32_t& version, const Object& object) {
        constexpr int faiss_hnsw_support_version = 6;
        if (version >= faiss_hnsw_support_version) {
            use_base_index = true;
        } else {
            use_base_index = false;
        }
    }

    bool
    IsAdditionalScalarSupported(bool is_mv_only) const override {
        if (use_base_index) {
            return base_index->IsAdditionalScalarSupported(is_mv_only);
        } else {
            return fallback_search_index->IsAdditionalScalarSupported(is_mv_only);
        }
    }

    Status
    Train(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override {
        if (use_base_index) {
            return base_index->Train(dataset, cfg, use_knowhere_build_pool);
        } else {
            return fallback_search_index->Train(dataset, cfg, use_knowhere_build_pool);
        }
    }

    Status
    Add(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override {
        if (use_base_index) {
            return base_index->Add(dataset, cfg, use_knowhere_build_pool);
        } else {
            return fallback_search_index->Add(dataset, cfg, use_knowhere_build_pool);
        }
    }

    expected<DataSetPtr>
    GetIndexMeta(std::unique_ptr<Config> cfg) const override {
        if (use_base_index) {
            return base_index->GetIndexMeta(std::move(cfg));
        } else {
            return fallback_search_index->GetIndexMeta(std::move(cfg));
        }
    }

    Status
    Serialize(BinarySet& binset) const override {
        if (use_base_index) {
            return base_index->Serialize(binset);
        } else {
            return fallback_search_index->Serialize(binset);
        }
    }

    Status
    Deserialize(const BinarySet& binset, std::shared_ptr<Config> config) override {
        if (use_base_index) {
            return base_index->Deserialize(binset, config);
        } else {
            return fallback_search_index->Deserialize(binset, config);
        }
    }

    Status
    DeserializeFromFile(const std::string& filename, std::shared_ptr<Config> config) override {
        if (use_base_index) {
            return base_index->DeserializeFromFile(filename, config);
        } else {
            return fallback_search_index->DeserializeFromFile(filename, config);
        }
    }

    int64_t
    Dim() const override {
        if (use_base_index) {
            return base_index->Dim();
        } else {
            return fallback_search_index->Dim();
        }
    }

    int64_t
    Count() const override {
        if (use_base_index) {
            return base_index->Count();
        } else {
            return fallback_search_index->Count();
        }
    }

    int64_t
    Size() const override {
        if (use_base_index) {
            return base_index->Size();
        } else {
            return fallback_search_index->Size();
        }
    }

    std::string
    Type() const override {
        if (use_base_index) {
            return base_index->Type();
        } else {
            return fallback_search_index->Type();
        }
    }

    bool
    HasRawData(const std::string& metric_type) const override {
        if (use_base_index) {
            return base_index->HasRawData(metric_type);
        } else {
            return fallback_search_index->HasRawData(metric_type);
        }
    }

    expected<DataSetPtr>
    GetVectorByIds(const DataSetPtr dataset, milvus::OpContext* op_context) const override {
        if (use_base_index) {
            return base_index->GetVectorByIds(dataset, op_context);
        } else {
            return fallback_search_index->GetVectorByIds(dataset, op_context);
        }
    }

    expected<DataSetPtr>
    Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
           milvus::OpContext* op_context) const override {
        if (use_base_index) {
            return base_index->Search(dataset, std::move(cfg), bitset, op_context);
        } else {
            return fallback_search_index->Search(dataset, std::move(cfg), bitset, op_context);
        }
    }

    expected<DataSetPtr>
    RangeSearch(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
                milvus::OpContext* op_context) const override {
        if (use_base_index) {
            return base_index->RangeSearch(dataset, std::move(cfg), bitset, op_context);
        } else {
            return fallback_search_index->RangeSearch(dataset, std::move(cfg), bitset, op_context);
        }
    }

    expected<std::vector<IndexNode::IteratorPtr>>
    AnnIterator(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
                bool use_knowhere_search_pool, milvus::OpContext* op_context) const override {
        if (use_base_index) {
            return base_index->AnnIterator(dataset, std::move(cfg), bitset, use_knowhere_search_pool, op_context);
        } else {
            return fallback_search_index->AnnIterator(dataset, std::move(cfg), bitset, use_knowhere_search_pool,
                                                      op_context);
        }
    }

    std::shared_ptr<std::vector<uint32_t>>
    GetInternalIdToExternalIdMap() const override {
        if (use_base_index) {
            return base_index->GetInternalIdToExternalIdMap();
        } else {
            return fallback_search_index->GetInternalIdToExternalIdMap();
        }
    }

    Status
    SetInternalIdToMostExternalIdMap(std::vector<uint32_t>&& map) override {
        if (use_base_index) {
            return base_index->SetInternalIdToMostExternalIdMap(std::move(map));
        } else {
            return fallback_search_index->SetInternalIdToMostExternalIdMap(std::move(map));
        }
    }

    std::optional<size_t>
    GetQueryCodeSize(const DataSetPtr dataset) const override {
        if (use_base_index) {
            return base_index->GetQueryCodeSize(dataset);
        } else {
            return fallback_search_index->GetQueryCodeSize(dataset);
        }
    }

    expected<DataSetPtr>
    CalcDistByIDs(const DataSetPtr dataset, const BitsetView& bitset, const int64_t* labels, const size_t labels_len,
                  const bool is_cosine, milvus::OpContext* op_context) const override {
        if (use_base_index) {
            return base_index->CalcDistByIDs(dataset, bitset, labels, labels_len, is_cosine, op_context);
        } else {
            return fallback_search_index->CalcDistByIDs(dataset, bitset, labels, labels_len, is_cosine, op_context);
        }
    }

    Status
    BuildEmbListIfNeed(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override {
        if (use_base_index) {
            return base_index->BuildEmbListIfNeed(dataset, std::move(cfg), use_knowhere_build_pool);
        } else {
            return fallback_search_index->BuildEmbListIfNeed(dataset, std::move(cfg), use_knowhere_build_pool);
        }
    }

    Status
    AddEmbListIfNeed(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override {
        if (use_base_index) {
            return base_index->AddEmbListIfNeed(dataset, std::move(cfg), use_knowhere_build_pool);
        } else {
            return fallback_search_index->AddEmbListIfNeed(dataset, std::move(cfg), use_knowhere_build_pool);
        }
    }

    Status
    BulidAsyncEmbListIfNeed(const DataSetPtr dataset, std::shared_ptr<Config> cfg,
                            const Interrupt* interrupt) override {
        if (use_base_index) {
            return base_index->BulidAsyncEmbListIfNeed(dataset, std::move(cfg), interrupt);
        } else {
            return fallback_search_index->BulidAsyncEmbListIfNeed(dataset, std::move(cfg), interrupt);
        }
    }

    Status
    SerializeEmbListIfNeed(BinarySet& binset) const override {
        if (use_base_index) {
            return base_index->SerializeEmbListIfNeed(binset);
        } else {
            return fallback_search_index->SerializeEmbListIfNeed(binset);
        }
    }

    Status
    DeserializeEmbListIfNeed(const BinarySet& binset, std::shared_ptr<Config> config) override {
        if (use_base_index) {
            return base_index->DeserializeEmbListIfNeed(binset, std::move(config));
        } else {
            return fallback_search_index->DeserializeEmbListIfNeed(binset, std::move(config));
        }
    }

    Status
    DeserializeFromFileIfNeed(const std::string& filename, std::shared_ptr<Config> config) override {
        if (use_base_index) {
            return base_index->DeserializeFromFileIfNeed(filename, std::move(config));
        } else {
            return fallback_search_index->DeserializeFromFileIfNeed(filename, std::move(config));
        }
    }

    expected<DataSetPtr>
    SearchEmbListIfNeed(const DataSetPtr dataset, std::unique_ptr<Config> config, const BitsetView& bitset,
                        milvus::OpContext* op_context) const override {
        if (use_base_index) {
            return base_index->SearchEmbListIfNeed(dataset, std::move(config), bitset, op_context);
        } else {
            return fallback_search_index->SearchEmbListIfNeed(dataset, std::move(config), bitset, op_context);
        }
    }

    expected<DataSetPtr>
    RangeSearchEmbListIfNeed(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
                             milvus::OpContext* op_context) const override {
        if (use_base_index) {
            return base_index->RangeSearchEmbListIfNeed(dataset, std::move(cfg), bitset, op_context);
        } else {
            return fallback_search_index->RangeSearchEmbListIfNeed(dataset, std::move(cfg), bitset, op_context);
        }
    }

    expected<std::vector<IteratorPtr>>
    AnnIteratorEmbListIfNeed(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
                             bool use_knowhere_search_pool, milvus::OpContext* op_context) const override {
        if (use_base_index) {
            return base_index->AnnIteratorEmbListIfNeed(dataset, std::move(cfg), bitset, use_knowhere_search_pool,
                                                        op_context);
        } else {
            return fallback_search_index->AnnIteratorEmbListIfNeed(dataset, std::move(cfg), bitset,
                                                                   use_knowhere_search_pool, op_context);
        }
    }

    expected<DataSetPtr>
    GetEmbListByIds(const DataSetPtr dataset, const std::string& metric_type,
                    milvus::OpContext* op_context) const override {
        if (use_base_index) {
            return base_index->GetEmbListByIds(dataset, metric_type, op_context);
        } else {
            return fallback_search_index->GetEmbListByIds(dataset, metric_type, op_context);
        }
    }

 protected:
    bool use_base_index = true;
    std::unique_ptr<IndexNode> base_index;
    std::unique_ptr<IndexNode> fallback_search_index;
};

template <typename DataType>
class BaseFaissRegularIndexHNSWFlatNodeTemplateWithSearchFallback : public HNSWIndexNodeWithFallback {
 public:
    BaseFaissRegularIndexHNSWFlatNodeTemplateWithSearchFallback(const int32_t& version, const Object& object)
        : HNSWIndexNodeWithFallback(version, object) {
        // initialize underlying nodes
        base_index = std::make_unique<BaseFaissRegularIndexHNSWFlatNodeTemplate<DataType>>(version, object);
        fallback_search_index = std::make_unique<HnswIndexNode<DataType, hnswlib::QuantType::None>>(version, object);
    }

    static bool
    StaticHasRawData(const knowhere::BaseConfig& config, const IndexVersion& version) {
        return true;
    }

    static Status
    StaticConfigCheck(const Config& cfg, PARAM_TYPE paramType, std::string& msg) {
        return BaseFaissRegularIndexHNSWFlatNodeTemplate<DataType>::StaticConfigCheck(cfg, paramType, msg);
    }

    static std::unique_ptr<BaseConfig>
    StaticCreateConfig() {
        return std::make_unique<FaissHnswFlatConfig>();
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return StaticCreateConfig();
    }
};

//
class BaseFaissRegularIndexHNSWSQNode : public BaseFaissRegularIndexHNSWNode {
 public:
    BaseFaissRegularIndexHNSWSQNode(const int32_t& version, const Object& object, DataFormatEnum data_format)
        : BaseFaissRegularIndexHNSWNode(version, object, data_format) {
    }

    static std::unique_ptr<BaseConfig>
    StaticCreateConfig() {
        return std::make_unique<FaissHnswSqConfig>();
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return StaticCreateConfig();
    }

    std::string
    Type() const override {
        return knowhere::IndexEnum::INDEX_HNSW_SQ;
    }

 protected:
    Status
    TrainInternal(const DataSetPtr dataset, const Config& cfg) override {
        // number of rows
        auto rows = dataset->GetRows();
        // dimensionality of the data
        auto dim = dataset->GetDim();
        // data
        const void* data = dataset->GetTensor();

        // config
        auto hnsw_cfg = static_cast<const FaissHnswSqConfig&>(cfg);

        auto metric = Str2FaissMetricType(hnsw_cfg.metric_type.value());
        if (!metric.has_value()) {
            LOG_KNOWHERE_ERROR_ << "Invalid metric type: " << hnsw_cfg.metric_type.value();
            return Status::invalid_metric_type;
        }

        // parse a ScalarQuantizer type
        auto sq_type = get_sq_quantizer_type(hnsw_cfg.sq_type.value());
        if (!sq_type.has_value()) {
            LOG_KNOWHERE_ERROR_ << "Invalid scalar quantizer type: " << hnsw_cfg.sq_type.value();
            return Status::invalid_args;
        }

        // create an index
        const bool is_cosine = IsMetricType(hnsw_cfg.metric_type.value(), metric::COSINE);
        const bool is_sq4u = sq_type.value() == faiss::ScalarQuantizer::QT_4bit_uniform;

        // should refine be used?
        std::unique_ptr<faiss::Index> final_index;

        auto train_index = [&](const float* data, const int i, const int64_t rows) {
            std::unique_ptr<faiss::cppcontrib::knowhere::IndexHNSW> hnsw_index;
            if (is_sq4u && is_cosine) {
                // Create IndexHNSWSQ4UniformCosine for COSINE
                hnsw_index = std::make_unique<faiss::cppcontrib::knowhere::IndexHNSWSQ4UniformCosine>(
                    dim, sq_type.value(), hnsw_cfg.M.value());
            } else if (is_sq4u && metric.value() == faiss::METRIC_INNER_PRODUCT) {
                // Create IndexHNSWSQ4UniformIP for IP
                hnsw_index = std::make_unique<faiss::cppcontrib::knowhere::IndexHNSWSQ4UniformIP>(dim, sq_type.value(),
                                                                                                  hnsw_cfg.M.value());
            } else if (is_cosine) {
                // Other quantizers with COSINE
                hnsw_index = std::make_unique<faiss::cppcontrib::knowhere::IndexHNSWSQCosine>(dim, sq_type.value(),
                                                                                              hnsw_cfg.M.value());
            } else {
                hnsw_index = std::make_unique<faiss::cppcontrib::knowhere::IndexHNSWSQ>(
                    dim, sq_type.value(), hnsw_cfg.M.value(), metric.value());
                // QT_4bit_uniform + L2 benefits from quantile-based range
                // estimation. This used to be hard-coded inside the fork
                // IndexScalarQuantizer ctor; moved here so that ctor is
                // behaviorally equivalent to baseline.
                if (is_sq4u) {
                    auto* idx_sq = dynamic_cast<faiss::IndexScalarQuantizer*>(hnsw_index->storage);
                    if (idx_sq != nullptr) {
                        idx_sq->sq.rangestat = faiss::ScalarQuantizer::RS_quantiles;
                        idx_sq->sq.rangestat_arg = 0.01;
                    }
                }
            }

            hnsw_index->hnsw.efConstruction = hnsw_cfg.efConstruction.value();

            if (hnsw_cfg.refine.value_or(false) && hnsw_cfg.refine_type.has_value()) {
                // yes
                const auto hnsw_d = hnsw_index->storage->d;
                const auto hnsw_metric_type = hnsw_index->storage->metric_type;
                auto final_index_cnd = pick_refine_index(data_format, hnsw_cfg.refine_type, std::move(hnsw_index),
                                                         hnsw_d, hnsw_metric_type);
                if (!final_index_cnd.has_value()) {
                    return Status::invalid_args;
                }

                // assign
                final_index = std::move(final_index_cnd.value());
            } else {
                // no refine

                // assign
                final_index = std::move(hnsw_index);
            }

            // train
            LOG_KNOWHERE_INFO_ << "Training HNSW Index";

            final_index->train(rows, data);

            // done
            indexes[i] = std::move(final_index);
            return Status::success;
        };

        const std::unordered_map<int64_t, std::vector<std::vector<uint32_t>>>& scalar_info_map =
            dataset->Get<std::unordered_map<int64_t, std::vector<std::vector<uint32_t>>>>(meta::SCALAR_INFO);
        if (scalar_info_map.size() > 1) {
            LOG_KNOWHERE_WARNING_ << "vector index build with multiple scalar info is not supported";
            return Status::invalid_args;
        }
        for (const auto& [field_id, scalar_info] : scalar_info_map) {
            tmp_combined_scalar_ids =
                scalar_info.size() > 1 ? combine_partitions(scalar_info, 128) : std::vector<std::vector<int>>();
        }
        // no scalar info or just one partition(after possible combination), build index on whole data
        if (scalar_info_map.empty() || tmp_combined_scalar_ids.size() <= 1) {
            // we have to convert the data to float, unfortunately, which costs extra RAM
            auto float_ds_ptr = convert_ds_to_float(dataset, data_format);
            if (float_ds_ptr == nullptr) {
                LOG_KNOWHERE_ERROR_ << "Unsupported data format";
                return Status::invalid_args;
            }
            return train_index(reinterpret_cast<const float*>(float_ds_ptr->GetTensor()), 0, rows);
        }
        LOG_KNOWHERE_INFO_ << "Train HNSWSQ Index with Scalar Info";
        for (const auto& [field_id, scalar_info] : scalar_info_map) {
            return TrainIndexByScalarInfo(train_index, scalar_info, data, rows, dim);
        }
        return Status::success;
    }
};

template <typename DataType>
class BaseFaissRegularIndexHNSWSQNodeTemplate : public BaseFaissRegularIndexHNSWSQNode {
 public:
    BaseFaissRegularIndexHNSWSQNodeTemplate(const int32_t& version, const Object& object)
        : BaseFaissRegularIndexHNSWSQNode(version, object, datatype_v<DataType>) {
    }

    static bool
    StaticHasRawData(const knowhere::BaseConfig& config, const IndexVersion& version) {
        auto hnsw_sq_cfg = static_cast<const FaissHnswSqConfig&>(config);

        auto sq_type = get_sq_quantizer_type(hnsw_sq_cfg.sq_type.value());
        if (has_lossless_quant(sq_type, datatype_v<DataType>)) {
            return true;
        }

        return has_lossless_refine_index(hnsw_sq_cfg.refine, hnsw_sq_cfg.refine_type, datatype_v<DataType>);
    }
};

// this index trains PQ and HNSW+FLAT separately, then constructs HNSW+PQ
class BaseFaissRegularIndexHNSWPQNode : public BaseFaissRegularIndexHNSWNode {
 public:
    BaseFaissRegularIndexHNSWPQNode(const int32_t& version, const Object& object, DataFormatEnum data_format)
        : BaseFaissRegularIndexHNSWNode(version, object, data_format) {
    }

    static std::unique_ptr<BaseConfig>
    StaticCreateConfig() {
        return std::make_unique<FaissHnswPqConfig>();
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return StaticCreateConfig();
    }

    std::string
    Type() const override {
        return knowhere::IndexEnum::INDEX_HNSW_PQ;
    }

 protected:
    std::vector<std::unique_ptr<faiss::IndexPQ>> tmp_index_pq;

    Status
    TrainInternal(const DataSetPtr dataset, const Config& cfg) override {
        // number of rows
        auto rows = dataset->GetRows();
        // dimensionality of the data
        auto dim = dataset->GetDim();
        // data
        const void* data = dataset->GetTensor();

        // config
        auto hnsw_cfg = static_cast<const FaissHnswPqConfig&>(cfg);

        if (rows < (1 << hnsw_cfg.nbits.value())) {
            LOG_KNOWHERE_ERROR_ << rows << " rows not enough, needs at least " << (1 << hnsw_cfg.nbits.value())
                                << " rows";
            return Status::faiss_inner_error;
        }

        auto metric = Str2FaissMetricType(hnsw_cfg.metric_type.value());
        if (!metric.has_value()) {
            LOG_KNOWHERE_ERROR_ << "Invalid metric type: " << hnsw_cfg.metric_type.value();
            return Status::invalid_metric_type;
        }

        // create an index
        const bool is_cosine = IsMetricType(hnsw_cfg.metric_type.value(), metric::COSINE);

        // HNSW + PQ index yields BAD recall somewhy.
        // Let's build HNSW+FLAT index, then replace FLAT with PQ
        auto train_index = [&](const float* data, const int i, const int64_t rows) {
            std::unique_ptr<faiss::cppcontrib::knowhere::IndexHNSW> hnsw_index;
            if (is_cosine) {
                hnsw_index =
                    std::make_unique<faiss::cppcontrib::knowhere::IndexHNSWFlatCosine>(dim, hnsw_cfg.M.value());
            } else {
                hnsw_index = std::make_unique<faiss::cppcontrib::knowhere::IndexHNSWFlat>(dim, hnsw_cfg.M.value(),
                                                                                          metric.value());
            }

            hnsw_index->hnsw.efConstruction = hnsw_cfg.efConstruction.value();

            // pq
            std::unique_ptr<faiss::IndexPQ> pq_index;
            if (is_cosine) {
                pq_index = std::make_unique<faiss::cppcontrib::knowhere::IndexPQCosine>(dim, hnsw_cfg.m.value(),
                                                                                        hnsw_cfg.nbits.value());
            } else {
                pq_index =
                    std::make_unique<faiss::IndexPQ>(dim, hnsw_cfg.m.value(), hnsw_cfg.nbits.value(), metric.value());
            }

            // should refine be used?
            std::unique_ptr<faiss::Index> final_index;
            if (hnsw_cfg.refine.value_or(false) && hnsw_cfg.refine_type.has_value()) {
                // yes
                const auto hnsw_d = hnsw_index->storage->d;
                const auto hnsw_metric_type = hnsw_index->storage->metric_type;
                auto final_index_cnd = pick_refine_index(data_format, hnsw_cfg.refine_type, std::move(hnsw_index),
                                                         hnsw_d, hnsw_metric_type);
                if (!final_index_cnd.has_value()) {
                    return Status::invalid_args;
                }

                // assign
                final_index = std::move(final_index_cnd.value());
            } else {
                // no refine

                // assign
                final_index = std::move(hnsw_index);
            }

            // train hnswflat
            LOG_KNOWHERE_INFO_ << "Training HNSW Index";

            final_index->train(rows, data);

            // train pq
            LOG_KNOWHERE_INFO_ << "Training PQ Index";

            ApplyClusteringConfig(pq_index->pq.cp);
            pq_index->train(rows, data);
            pq_index->pq.compute_sdc_table();

            // done
            indexes[i] = std::move(final_index);
            tmp_index_pq[i] = std::move(pq_index);
            return Status::success;
        };

        const std::unordered_map<int64_t, std::vector<std::vector<uint32_t>>>& scalar_info_map =
            dataset->Get<std::unordered_map<int64_t, std::vector<std::vector<uint32_t>>>>(meta::SCALAR_INFO);
        if (scalar_info_map.size() > 1) {
            LOG_KNOWHERE_WARNING_ << "vector index build with multiple scalar info is not supported";
            return Status::invalid_args;
        }
        for (const auto& [field_id, scalar_info] : scalar_info_map) {
            tmp_combined_scalar_ids = scalar_info.size() > 1
                                          ? combine_partitions(scalar_info, (1 << hnsw_cfg.nbits.value()))
                                          : std::vector<std::vector<int>>();
        }

        // no scalar info or just one partition(after possible combination), build index on whole data
        if (scalar_info_map.empty() || tmp_combined_scalar_ids.size() <= 1) {
            tmp_index_pq.resize(1);
            // we have to convert the data to float, unfortunately, which costs extra RAM
            auto float_ds_ptr = convert_ds_to_float(dataset, data_format);
            if (float_ds_ptr == nullptr) {
                LOG_KNOWHERE_ERROR_ << "Unsupported data format";
                return Status::invalid_args;
            }
            return train_index(static_cast<const float*>(float_ds_ptr->GetTensor()), 0, rows);
        }

        LOG_KNOWHERE_INFO_ << "Train HNSWPQ Index with Scalar Info";
        tmp_index_pq.resize(tmp_combined_scalar_ids.size());
        for (const auto& [field_id, scalar_info] : scalar_info_map) {
            return TrainIndexByScalarInfo(train_index, scalar_info, data, rows, dim);
        }
        return Status::success;
    }

    Status
    AddInternal(const DataSetPtr dataset, const Config&) override {
        if (isIndexEmpty()) {
            LOG_KNOWHERE_ERROR_ << "Can not add data to an empty index.";
            return Status::empty_index;
        }

        auto rows = dataset->GetRows();

        auto finalize_index = [&](int i) {
            // we're done.
            // throw away flat and replace it with pq

            // check if we have a refine available.
            faiss::cppcontrib::knowhere::IndexHNSW* index_hnsw = nullptr;

            faiss::cppcontrib::knowhere::IndexRefine* const index_refine =
                dynamic_cast<faiss::cppcontrib::knowhere::IndexRefine*>(indexes[i].get());

            if (index_refine != nullptr) {
                index_hnsw = dynamic_cast<faiss::cppcontrib::knowhere::IndexHNSW*>(index_refine->base_index);
            } else {
                index_hnsw = dynamic_cast<faiss::cppcontrib::knowhere::IndexHNSW*>(indexes[i].get());
            }

            // recreate hnswpq
            std::unique_ptr<faiss::cppcontrib::knowhere::IndexHNSW> index_hnsw_pq;

            if (faiss::cppcontrib::knowhere::is_cosine_index(index_hnsw->storage)) {
                index_hnsw_pq = std::make_unique<faiss::cppcontrib::knowhere::IndexHNSWPQCosine>();
            } else {
                index_hnsw_pq = std::make_unique<faiss::cppcontrib::knowhere::IndexHNSWPQ>();
            }

            // C++ slicing.
            // we can't use move, because faiss::cppcontrib::knowhere::IndexHNSW overrides a destructor.
            static_cast<faiss::cppcontrib::knowhere::IndexHNSW&>(*index_hnsw_pq) =
                static_cast<faiss::cppcontrib::knowhere::IndexHNSW&>(*index_hnsw);

            // clear out the storage
            delete index_hnsw->storage;
            index_hnsw->storage = nullptr;
            index_hnsw_pq->storage = nullptr;

            // replace storage
            index_hnsw_pq->storage = tmp_index_pq[i].release();

            // replace if refine
            if (index_refine != nullptr) {
                delete index_refine->base_index;
                index_refine->base_index = index_hnsw_pq.release();
            } else {
                indexes[i] = std::move(index_hnsw_pq);
            }
            return Status::success;
        };
        try {
            const std::unordered_map<int64_t, std::vector<std::vector<uint32_t>>>& scalar_info_map =
                dataset->Get<std::unordered_map<int64_t, std::vector<std::vector<uint32_t>>>>(meta::SCALAR_INFO);

            if (scalar_info_map.empty() || tmp_combined_scalar_ids.size() <= 1) {
                // hnsw
                LOG_KNOWHERE_INFO_ << "Adding " << rows << " to HNSW Index";

                auto status_reg = add_to_index(indexes[0].get(), dataset, data_format);
                if (status_reg != Status::success) {
                    return status_reg;
                }

                // pq
                LOG_KNOWHERE_INFO_ << "Adding " << rows << " to PQ Index";

                auto status_pq = add_to_index(tmp_index_pq[0].get(), dataset, data_format);
                if (status_pq != Status::success) {
                    return status_pq;
                }
                return finalize_index(0);
            }
            if (scalar_info_map.size() > 1) {
                LOG_KNOWHERE_WARNING_ << "vector index build with multiple scalar info is not supported";
                return Status::invalid_args;
            }
            LOG_KNOWHERE_INFO_ << "Add data to Index with Scalar Info";

            for (const auto& [field_id, scalar_info] : scalar_info_map) {
                for (auto i = 0; i < tmp_combined_scalar_ids.size(); ++i) {
                    for (auto j = 0; j < tmp_combined_scalar_ids[i].size(); ++j) {
                        auto id = tmp_combined_scalar_ids[i][j];
                        // hnsw
                        LOG_KNOWHERE_INFO_ << "Adding " << scalar_info[id].size() << " to HNSW Index";

                        auto status_reg =
                            add_partial_dataset_to_index(indexes[i].get(), dataset, data_format, scalar_info[id]);
                        if (status_reg != Status::success) {
                            return status_reg;
                        }

                        // pq
                        LOG_KNOWHERE_INFO_ << "Adding " << scalar_info[id].size() << " to PQ Index";

                        auto status_pq =
                            add_partial_dataset_to_index(tmp_index_pq[i].get(), dataset, data_format, scalar_info[id]);

                        if (status_pq != Status::success) {
                            return status_pq;
                        }
                    }
                    finalize_index(i);
                }
            }

        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
            return Status::faiss_inner_error;
        }

        return Status::success;
    }
};

template <typename DataType>
class BaseFaissRegularIndexHNSWPQNodeTemplate : public BaseFaissRegularIndexHNSWPQNode {
 public:
    BaseFaissRegularIndexHNSWPQNodeTemplate(const int32_t& version, const Object& object)
        : BaseFaissRegularIndexHNSWPQNode(version, object, datatype_v<DataType>) {
    }

    static bool
    StaticHasRawData(const knowhere::BaseConfig& config, const IndexVersion& version) {
        auto hnsw_cfg = static_cast<const FaissHnswConfig&>(config);
        return has_lossless_refine_index(hnsw_cfg.refine, hnsw_cfg.refine_type, datatype_v<DataType>);
    }
};

// this index trains PRQ and HNSW+FLAT separately, then constructs HNSW+PRQ
class BaseFaissRegularIndexHNSWPRQNode : public BaseFaissRegularIndexHNSWNode {
 public:
    BaseFaissRegularIndexHNSWPRQNode(const int32_t& version, const Object& object, DataFormatEnum data_format)
        : BaseFaissRegularIndexHNSWNode(version, object, data_format) {
    }

    static std::unique_ptr<BaseConfig>
    StaticCreateConfig() {
        return std::make_unique<FaissHnswPrqConfig>();
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return StaticCreateConfig();
    }

    std::string
    Type() const override {
        return knowhere::IndexEnum::INDEX_HNSW_PRQ;
    }

 protected:
    std::vector<std::unique_ptr<faiss::IndexProductResidualQuantizer>> tmp_index_prq;

    Status
    TrainInternal(const DataSetPtr dataset, const Config& cfg) override {
        // number of rows
        auto rows = dataset->GetRows();
        // dimensionality of the data
        auto dim = dataset->GetDim();
        // data
        const void* data = dataset->GetTensor();

        // config
        auto hnsw_cfg = static_cast<const FaissHnswPrqConfig&>(cfg);

        if (rows < (1 << hnsw_cfg.nbits.value())) {
            LOG_KNOWHERE_ERROR_ << rows << " rows not enough, needs at least " << (1 << hnsw_cfg.nbits.value())
                                << " rows";
            return Status::faiss_inner_error;
        }

        auto metric = Str2FaissMetricType(hnsw_cfg.metric_type.value());
        if (!metric.has_value()) {
            LOG_KNOWHERE_ERROR_ << "Invalid metric type: " << hnsw_cfg.metric_type.value();
            return Status::invalid_metric_type;
        }

        // create an index
        const bool is_cosine = IsMetricType(hnsw_cfg.metric_type.value(), metric::COSINE);

        // HNSW + PRQ index yields BAD recall somewhy.
        // Let's build HNSW+FLAT index, then replace FLAT with PRQ
        auto train_index = [&](const float* data, const int i, const int64_t rows) {
            std::unique_ptr<faiss::cppcontrib::knowhere::IndexHNSW> hnsw_index;
            if (is_cosine) {
                hnsw_index =
                    std::make_unique<faiss::cppcontrib::knowhere::IndexHNSWFlatCosine>(dim, hnsw_cfg.M.value());
            } else {
                hnsw_index = std::make_unique<faiss::cppcontrib::knowhere::IndexHNSWFlat>(dim, hnsw_cfg.M.value(),
                                                                                          metric.value());
            }

            hnsw_index->hnsw.efConstruction = hnsw_cfg.efConstruction.value();

            // prq
            faiss::AdditiveQuantizer::Search_type_t prq_search_type =
                (metric.value() == faiss::MetricType::METRIC_INNER_PRODUCT)
                    ? faiss::AdditiveQuantizer::Search_type_t::ST_LUT_nonorm
                    : faiss::AdditiveQuantizer::Search_type_t::ST_norm_float;

            std::unique_ptr<faiss::IndexProductResidualQuantizer> prq_index;
            if (is_cosine) {
                prq_index = std::make_unique<faiss::cppcontrib::knowhere::IndexProductResidualQuantizerCosine>(
                    dim, hnsw_cfg.m.value(), hnsw_cfg.nrq.value(), hnsw_cfg.nbits.value(), prq_search_type);
            } else {
                prq_index = std::make_unique<faiss::IndexProductResidualQuantizer>(
                    dim, hnsw_cfg.m.value(), hnsw_cfg.nrq.value(), hnsw_cfg.nbits.value(), metric.value(),
                    prq_search_type);
            }

            // should refine be used?
            std::unique_ptr<faiss::Index> final_index;
            if (hnsw_cfg.refine.value_or(false) && hnsw_cfg.refine_type.has_value()) {
                // yes
                const auto hnsw_d = hnsw_index->storage->d;
                const auto hnsw_metric_type = hnsw_index->storage->metric_type;
                auto final_index_cnd = pick_refine_index(data_format, hnsw_cfg.refine_type, std::move(hnsw_index),
                                                         hnsw_d, hnsw_metric_type);
                if (!final_index_cnd.has_value()) {
                    return Status::invalid_args;
                }

                // assign
                final_index = std::move(final_index_cnd.value());
            } else {
                // no refine

                // assign
                final_index = std::move(hnsw_index);
            }

            // train hnswflat
            LOG_KNOWHERE_INFO_ << "Training HNSW Index";

            final_index->train(rows, data);

            // train prq
            LOG_KNOWHERE_INFO_ << "Training ProductResidualQuantizer Index";

            prq_index->train(rows, data);

            // done
            indexes[i] = std::move(final_index);
            tmp_index_prq[i] = std::move(prq_index);

            return Status::success;
        };
        const std::unordered_map<int64_t, std::vector<std::vector<uint32_t>>>& scalar_info_map =
            dataset->Get<std::unordered_map<int64_t, std::vector<std::vector<uint32_t>>>>(meta::SCALAR_INFO);
        if (scalar_info_map.size() > 1) {
            LOG_KNOWHERE_WARNING_ << "vector index build with multiple scalar info is not supported";
            return Status::invalid_args;
        }
        for (const auto& [field_id, scalar_info] : scalar_info_map) {
            tmp_combined_scalar_ids = scalar_info.size() > 1
                                          ? combine_partitions(scalar_info, (1 << hnsw_cfg.nbits.value()))
                                          : std::vector<std::vector<int>>();
        }

        // no scalar info or just one partition(after possible combination), build index on whole data
        if (scalar_info_map.empty() || tmp_combined_scalar_ids.size() <= 1) {
            tmp_index_prq.resize(1);
            // we have to convert the data to float, unfortunately, which costs extra RAM
            auto float_ds_ptr = convert_ds_to_float(dataset, data_format);

            if (float_ds_ptr == nullptr) {
                LOG_KNOWHERE_ERROR_ << "Unsupported data format";
                return Status::invalid_args;
            }
            return train_index(static_cast<const float*>(float_ds_ptr->GetTensor()), 0, rows);
        }

        LOG_KNOWHERE_INFO_ << "Train HNSWPRQ Index with Scalar Info";
        tmp_index_prq.resize(tmp_combined_scalar_ids.size());
        for (const auto& [field_id, scalar_info] : scalar_info_map) {
            return TrainIndexByScalarInfo(train_index, scalar_info, data, rows, dim);
        }
        return Status::success;
    }

    Status
    AddInternal(const DataSetPtr dataset, const Config&) override {
        if (isIndexEmpty()) {
            LOG_KNOWHERE_ERROR_ << "Can not add data to an empty index.";
            return Status::empty_index;
        }

        auto rows = dataset->GetRows();

        auto finalize_index = [&](int i) {
            // we're done.
            // throw away flat and replace it with prq

            // check if we have a refine available.
            faiss::cppcontrib::knowhere::IndexHNSW* index_hnsw = nullptr;

            faiss::cppcontrib::knowhere::IndexRefine* const index_refine =
                dynamic_cast<faiss::cppcontrib::knowhere::IndexRefine*>(indexes[i].get());

            if (index_refine != nullptr) {
                index_hnsw = dynamic_cast<faiss::cppcontrib::knowhere::IndexHNSW*>(index_refine->base_index);
            } else {
                index_hnsw = dynamic_cast<faiss::cppcontrib::knowhere::IndexHNSW*>(indexes[i].get());
            }

            // recreate hnswprq
            std::unique_ptr<faiss::cppcontrib::knowhere::IndexHNSW> index_hnsw_prq;

            if (faiss::cppcontrib::knowhere::is_cosine_index(index_hnsw->storage)) {
                index_hnsw_prq =
                    std::make_unique<faiss::cppcontrib::knowhere::IndexHNSWProductResidualQuantizerCosine>();
            } else {
                index_hnsw_prq = std::make_unique<faiss::cppcontrib::knowhere::IndexHNSWProductResidualQuantizer>();
            }

            // C++ slicing
            // we can't use move, because faiss::IndexHNSW overrides a destructor.
            static_cast<faiss::cppcontrib::knowhere::IndexHNSW&>(*index_hnsw_prq) =
                static_cast<faiss::cppcontrib::knowhere::IndexHNSW&>(*index_hnsw);

            // clear out the storage
            delete index_hnsw->storage;
            index_hnsw->storage = nullptr;
            index_hnsw_prq->storage = nullptr;

            // replace storage
            index_hnsw_prq->storage = tmp_index_prq[i].release();

            // replace if refine
            if (index_refine != nullptr) {
                delete index_refine->base_index;
                index_refine->base_index = index_hnsw_prq.release();
            } else {
                indexes[i] = std::move(index_hnsw_prq);
            }
            return Status::success;
        };
        try {
            const std::unordered_map<int64_t, std::vector<std::vector<uint32_t>>>& scalar_info_map =
                dataset->Get<std::unordered_map<int64_t, std::vector<std::vector<uint32_t>>>>(meta::SCALAR_INFO);

            if (scalar_info_map.empty() || tmp_combined_scalar_ids.size() <= 1) {
                // hnsw
                LOG_KNOWHERE_INFO_ << "Adding " << rows << " to HNSW Index";

                auto status_reg = add_to_index(indexes[0].get(), dataset, data_format);
                if (status_reg != Status::success) {
                    return status_reg;
                }

                // prq
                LOG_KNOWHERE_INFO_ << "Adding " << rows << " to ProductResidualQuantizer Index";

                auto status_prq = add_to_index(tmp_index_prq[0].get(), dataset, data_format);
                if (status_prq != Status::success) {
                    return status_prq;
                }
                return finalize_index(0);
            }

            if (scalar_info_map.size() > 1) {
                LOG_KNOWHERE_WARNING_ << "vector index build with multiple scalar info is not supported";
                return Status::invalid_args;
            }
            LOG_KNOWHERE_INFO_ << "Add data to Index with Scalar Info";

            for (const auto& [field_id, scalar_info] : scalar_info_map) {
                for (auto i = 0; i < tmp_combined_scalar_ids.size(); ++i) {
                    for (auto j = 0; j < tmp_combined_scalar_ids[i].size(); ++j) {
                        auto id = tmp_combined_scalar_ids[i][j];
                        // hnsw
                        LOG_KNOWHERE_INFO_ << "Adding " << scalar_info[id].size() << " to HNSW Index";

                        auto status_reg =
                            add_partial_dataset_to_index(indexes[i].get(), dataset, data_format, scalar_info[id]);
                        if (status_reg != Status::success) {
                            return status_reg;
                        }

                        // prq
                        LOG_KNOWHERE_INFO_ << "Adding " << scalar_info[id].size() << " to PQ Index";

                        auto status_prq =
                            add_partial_dataset_to_index(tmp_index_prq[i].get(), dataset, data_format, scalar_info[id]);

                        if (status_prq != Status::success) {
                            return status_prq;
                        }
                    }
                    finalize_index(i);
                }
            }

        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
            return Status::faiss_inner_error;
        }

        return Status::success;
    }
};

template <typename DataType>
class BaseFaissRegularIndexHNSWPRQNodeTemplate : public BaseFaissRegularIndexHNSWPRQNode {
 public:
    BaseFaissRegularIndexHNSWPRQNodeTemplate(const int32_t& version, const Object& object)
        : BaseFaissRegularIndexHNSWPRQNode(version, object, datatype_v<DataType>) {
    }

    static bool
    StaticHasRawData(const knowhere::BaseConfig& config, const IndexVersion& version) {
        auto hnsw_cfg = static_cast<const FaissHnswConfig&>(config);
        return has_lossless_refine_index(hnsw_cfg.refine, hnsw_cfg.refine_type, datatype_v<DataType>);
    }
};

#ifdef KNOWHERE_WITH_CARDINAL
KNOWHERE_SIMPLE_REGISTER_DENSE_FLOAT_ALL_GLOBAL(HNSW_DEPRECATED,
                                                BaseFaissRegularIndexHNSWFlatNodeTemplateWithSearchFallback,
                                                knowhere::feature::MMAP | knowhere::feature::MV |
                                                    knowhere::feature::EMB_LIST)
KNOWHERE_SIMPLE_REGISTER_DENSE_INT_GLOBAL(HNSW_DEPRECATED, BaseFaissRegularIndexHNSWFlatNodeTemplate,
                                          knowhere::feature::MMAP | knowhere::feature::MV | knowhere::feature::EMB_LIST)
KNOWHERE_SIMPLE_REGISTER_DENSE_BIN_GLOBAL(HNSW_DEPRECATED, BaseFaissRegularIndexHNSWFlatNodeTemplate,
                                          knowhere::feature::MMAP | knowhere::feature::MV | knowhere::feature::EMB_LIST)
#else
KNOWHERE_SIMPLE_REGISTER_DENSE_FLOAT_ALL_GLOBAL(HNSW, BaseFaissRegularIndexHNSWFlatNodeTemplateWithSearchFallback,
                                                knowhere::feature::MMAP | knowhere::feature::MV |
                                                    knowhere::feature::EMB_LIST)
KNOWHERE_SIMPLE_REGISTER_DENSE_INT_GLOBAL(HNSW, BaseFaissRegularIndexHNSWFlatNodeTemplate,
                                          knowhere::feature::MMAP | knowhere::feature::MV | knowhere::feature::EMB_LIST)
KNOWHERE_SIMPLE_REGISTER_DENSE_BIN_GLOBAL(HNSW, BaseFaissRegularIndexHNSWFlatNodeTemplate,
                                          knowhere::feature::MMAP | knowhere::feature::MV | knowhere::feature::EMB_LIST)
#endif

KNOWHERE_SIMPLE_REGISTER_DENSE_FLOAT_ALL_GLOBAL(HNSW_SQ, BaseFaissRegularIndexHNSWSQNodeTemplate,
                                                knowhere::feature::MMAP | knowhere::feature::MV |
                                                    knowhere::feature::EMB_LIST)
KNOWHERE_SIMPLE_REGISTER_DENSE_INT_GLOBAL(HNSW_SQ, BaseFaissRegularIndexHNSWSQNodeTemplate,
                                          knowhere::feature::MMAP | knowhere::feature::MV | knowhere::feature::EMB_LIST)
KNOWHERE_SIMPLE_REGISTER_DENSE_FLOAT_ALL_GLOBAL(HNSW_PQ, BaseFaissRegularIndexHNSWPQNodeTemplate,
                                                knowhere::feature::MMAP | knowhere::feature::MV |
                                                    knowhere::feature::EMB_LIST)
KNOWHERE_SIMPLE_REGISTER_DENSE_INT_GLOBAL(HNSW_PQ, BaseFaissRegularIndexHNSWPQNodeTemplate,
                                          knowhere::feature::MMAP | knowhere::feature::MV | knowhere::feature::EMB_LIST)
KNOWHERE_SIMPLE_REGISTER_DENSE_FLOAT_ALL_GLOBAL(HNSW_PRQ, BaseFaissRegularIndexHNSWPRQNodeTemplate,
                                                knowhere::feature::MMAP | knowhere::feature::MV |
                                                    knowhere::feature::EMB_LIST)
KNOWHERE_SIMPLE_REGISTER_DENSE_INT_GLOBAL(HNSW_PRQ, BaseFaissRegularIndexHNSWPRQNodeTemplate,
                                          knowhere::feature::MMAP | knowhere::feature::MV | knowhere::feature::EMB_LIST)

#ifdef KNOWHERE_WITH_CUVS
}  // namespace knowhere — temporarily close to include GPU headers at file scope
// ── GPU HNSW ─────────────────────────────────────────────────────────────────
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/gpu/GpuIndexHNSW.h>
#include <faiss/gpu/StandardGpuResources.h>
namespace knowhere {  // reopen namespace knowhere

// Process-global StandardGpuResources shared by all GpuHnswIndexNode instances.
// Avoids per-segment 256 MiB pinned memory, cuBLAS handle, and CUDA stream
// allocations that accumulate to tens of GiB with many segments.
//
// Device model: this assumes a single visible GPU per process, which is how
// Milvus GPU querynodes are deployed (one device pinned via CUDA_VISIBLE_DEVICES
// / MIG). GpuIndexHNSW below is constructed with the default device (0), so all
// segments in a process land on that one GPU. StandardGpuResources can manage
// multiple devices, but true multi-GPU-per-process would additionally require
// explicit per-segment device selection in the GpuIndexHNSW config; that is a
// follow-up and is intentionally not wired here (cannot be validated without
// multi-GPU capacity).
static std::shared_ptr<faiss::gpu::StandardGpuResources>&
GetSharedGpuResources() {
    static std::once_flag flag;
    static std::shared_ptr<faiss::gpu::StandardGpuResources> instance;
    std::call_once(flag, [] {
        instance = std::make_shared<faiss::gpu::StandardGpuResources>();
        instance->setTempMemory(0);
        instance->setPinnedMemory(0);
    });
    return instance;
}

// Serialize GpuIndexHNSW construction across segments.
// StandardGpuResourcesImpl::initializeForDevice is not thread-safe;
// concurrent constructors race on the allocs_ map assertion.
static std::mutex&
GetGpuConstructionMutex() {
    static std::mutex mtx;
    return mtx;
}

// L2-normalize each row of an n x d row-major matrix in place. Zero-norm rows
// are left unchanged (their inverse norm is treated as 1), matching the cosine
// convention used elsewhere in this file.
static void
NormalizeRowsInPlace(float* x, int64_t n, int64_t d) {
    for (int64_t i = 0; i < n; i++) {
        float* row = x + i * d;
        float sq = 0.0f;
        for (int64_t j = 0; j < d; j++) {
            sq += row[j] * row[j];
        }
        if (sq > 0.0f) {
            const float inv = 1.0f / std::sqrt(sq);
            for (int64_t j = 0; j < d; j++) {
                row[j] *= inv;
            }
        }
    }
}

// Convert a deserialized cppcontrib::knowhere::IndexHNSW into a vanilla
// faiss::IndexHNSW suitable for faiss::gpu::GpuIndexHNSW::copyFrom().
//
// The HNSW graph layout is identical between the two (same faiss::HNSW CSR
// fields backed by the same faiss::MaybeOwnedVector<int32_t>), so the graph is
// copied field-by-field. Storage handling:
//   * L2 / IP (non-cosine): the source storage already derives from vanilla
//     faiss::IndexFlat / faiss::IndexScalarQuantizer, so it is borrowed
//     directly (own_fields=false) — no re-encode, byte-identical codes. Safe
//     because the returned index is only used for the synchronous copyFrom()
//     upload and is destroyed before the source CPU index is freed.
//   * COSINE: faiss has no cosine metric / inverse-norm buffer on GPU, so the
//     vectors are reconstructed, L2-normalized, and re-encoded into a fresh
//     vanilla storage with METRIC_INNER_PRODUCT (normalize+IP convention). For
//     scalar-quantized storage this re-quantizes the normalized vectors, which
//     is not byte-identical to the legacy inverse-norm path (recall is gated in
//     tests; a collection may need re-ingest for quantized cosine).
//     int8 (QT_8bit_direct_signed) cosine is re-encoded as fp16, not int8:
//     direct_signed is a fixed code=x+128 map, so normalized vectors (whose
//     components shrink as ~1/sqrt(d)) would collapse onto a few of the 256
//     levels. fp16 preserves normalized components accurately at any dimension
//     and uses the validated fp16 GPU search path (2 B/dim for int8-cosine
//     segments; a 1x-VRAM trained-QT_8bit int8-cosine path is a follow-up).
static std::unique_ptr<faiss::IndexHNSW>
ToVanillaHnsw(const faiss::cppcontrib::knowhere::IndexHNSW* src, bool is_cosine) {
    const int64_t d = src->d;
    const int64_t n = src->ntotal;
    const faiss::MetricType out_metric =
        (is_cosine || src->metric_type == faiss::METRIC_INNER_PRODUCT) ? faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;

    auto dst = std::make_unique<faiss::IndexHNSW>();
    dst->d = static_cast<int>(d);
    dst->ntotal = n;
    dst->metric_type = out_metric;
    dst->is_trained = true;

    if (is_cosine) {
        // Reconstruct -> normalize -> re-encode into fresh vanilla storage.
        //
        // The reconstruct/normalize/add is streamed in bounded row chunks rather
        // than materializing one n*d*4 fp32 buffer. For large int8-cosine
        // segments that full buffer (e.g. ~17 GiB for a ~5 GB on-disk segment)
        // dominated the host-RAM load peak and caused the querynode admission
        // guard to under-reserve and OOM. Chunking caps the transient fp32
        // staging to kReconChunkBytes while the re-encoded storage grows in
        // place, so the peak is (src compact index + re-encoded storage + one
        // chunk) instead of (src + full fp32 + re-encoded storage).
        const auto* src_sq = dynamic_cast<const faiss::IndexScalarQuantizer*>(src->storage);

        // QT_8bit_direct_signed is a fixed code=x+128 map with no trained range:
        // L2-normalized vectors live in [-1,1] and (worse) their components
        // shrink as ~1/sqrt(d), so direct-signed encoding of normalized data
        // collapses onto a handful of the 256 levels and destroys int8-cosine
        // recall. Re-encode int8 cosine as fp16, which represents normalized
        // components accurately at any dimension and reuses the validated fp16
        // GPU search path. int8 L2/IP are unaffected (they keep direct_signed +
        // DP4A). A 1x-VRAM trained QT_8bit int8-cosine path is a follow-up; fp16
        // costs 2 B/dim here.
        faiss::Index* out_storage = nullptr;
        // Whether the output quantizer needs the full dataset to fit a range.
        // fp16/bf16/direct(_signed) carry no trained parameters, so they can be
        // streamed in chunks; the trained integer quantizers (QT_8bit/QT_4bit/
        // QT_6bit) need the whole matrix and fall back to the full-buffer path.
        // Those are not production int8-cosine paths.
        bool needs_full_train = false;
        if (src_sq != nullptr) {
            faiss::ScalarQuantizer::QuantizerType out_qtype = src_sq->sq.qtype;
            if (out_qtype == faiss::ScalarQuantizer::QT_8bit_direct_signed) {
                out_qtype = faiss::ScalarQuantizer::QT_fp16;
            }
            needs_full_train =
                (out_qtype == faiss::ScalarQuantizer::QT_8bit || out_qtype == faiss::ScalarQuantizer::QT_4bit ||
                 out_qtype == faiss::ScalarQuantizer::QT_6bit);
            out_storage = new faiss::IndexScalarQuantizer(static_cast<int>(d), out_qtype, out_metric);
        } else {
            out_storage = new faiss::IndexFlat(static_cast<int>(d), out_metric);
        }

        if (n > 0) {
            if (needs_full_train) {
                std::vector<float> recons(static_cast<size_t>(n) * d);
                src->storage->reconstruct_n(0, n, recons.data());
                NormalizeRowsInPlace(recons.data(), n, d);
                out_storage->train(n, recons.data());
                out_storage->add(n, recons.data());
            } else {
                constexpr size_t kReconChunkBytes = static_cast<size_t>(256) << 20;
                const int64_t chunk_rows = std::max<int64_t>(
                    1, static_cast<int64_t>(kReconChunkBytes / (static_cast<size_t>(d) * sizeof(float))));
                std::vector<float> buf(static_cast<size_t>(chunk_rows) * d);
                bool trained = false;
                for (int64_t start = 0; start < n; start += chunk_rows) {
                    const int64_t cur = std::min<int64_t>(chunk_rows, n - start);
                    src->storage->reconstruct_n(start, cur, buf.data());
                    NormalizeRowsInPlace(buf.data(), cur, d);
                    if (!trained) {
                        // No-op for fp16/bf16/direct, but sets is_trained so add()
                        // is legal; IndexFlat::train is a base no-op.
                        out_storage->train(cur, buf.data());
                        trained = true;
                    }
                    out_storage->add(cur, buf.data());
                }
            }
        }
        dst->storage = out_storage;
        dst->own_fields = true;
    } else {
        // Borrow the source storage (already a vanilla Flat/SQ subclass). The
        // returned index must not outlive src; it is consumed by copyFrom().
        dst->storage = src->storage;
        dst->own_fields = false;
    }

    // Copy the graph (identical CSR layout).
    dst->hnsw.assign_probas = src->hnsw.assign_probas;
    dst->hnsw.cum_nneighbor_per_level = src->hnsw.cum_nneighbor_per_level;
    dst->hnsw.levels = src->hnsw.levels;
    dst->hnsw.offsets = src->hnsw.offsets;
    dst->hnsw.neighbors = src->hnsw.neighbors;
    dst->hnsw.entry_point = src->hnsw.entry_point;
    dst->hnsw.max_level = src->hnsw.max_level;

    return dst;
}

// GPU HNSW index node. Registered for F32, FP16, BF16 and INT8 (see
// registrations at the bottom of this file); it loads CPU-serialized HNSW (F32
// Flat) or HNSW_SQ (SQ8/INT8, fp16, bf16) binaries at load time and uploads them
// to the GPU via faiss::gpu::GpuIndexHNSW. FP16/BF16/INT8 stay in their native
// low-precision layout on the device (2 bytes for fp16/bf16, 1 byte for int8)
// and are up-converted to fp32 per element inside the search kernel. The CPU
// copy is freed after upload, so this node exposes no CPU raw data.
class GpuHnswIndexNode : public BaseFaissRegularIndexHNSWNode {
 public:
    GpuHnswIndexNode(const int32_t& version, const Object& object)
        : BaseFaissRegularIndexHNSWNode(version, object, DataFormatEnum::fp32) {
    }

    // Constructor for native data-type variants (e.g. int8 DP4A path).
    GpuHnswIndexNode(const int32_t& version, const Object& object, DataFormatEnum query_format)
        : BaseFaissRegularIndexHNSWNode(version, object, query_format) {
    }

    static std::unique_ptr<BaseConfig>
    StaticCreateConfig() {
        return std::make_unique<FaissHnswConfig>();
    }

    static bool
    StaticHasRawData(const knowhere::BaseConfig& config, const IndexVersion& version) {
        // The CPU copy is freed after the GPU upload, so no CPU-reconstructable
        // raw data is retained. Keep this consistent with the instance
        // HasRawData() override below.
        return false;
    }

    static expected<Resource>
    StaticEstimateLoadResource(const uint64_t file_size_in_bytes, const int64_t num_rows, const int64_t dim,
                               const knowhere::BaseConfig& config, const IndexVersion& version) {
        // GPU HNSW stores the vectors and graph in VRAM; the CPU copy is freed
        // after upload in Deserialize()/DeserializeFromFile(). Phase-separated
        // accounting:
        //   memoryCost (retained after load): ~0 — the CPU index is released
        //     once the graph/vectors are on the GPU, so a loaded segment holds
        //     no steady-state host RAM (its data lives in VRAM).
        //   maxMemoryCost (transient peak during load): the host RAM that is
        //     briefly live before the GPU upload frees the CPU copy.
        //
        // The on-disk index for this collection is the compact FAISS int8-SQ
        // form (IndexHNSWSQCosine: signed-int8 codes + graph), the SAME form the
        // CPU HNSW cluster loads resident at ~file_size. Deserialize reads it
        // into host RAM at ~file_size. For native int8/fp16/bf16 upload paths
        // the codes are uploaded to the device directly with no fp32
        // reconstruct/decode staging. (Note: unsigned SQ8 via QT_8bit does call
        // sa_decode() into a float buffer, so that path would incur fp32
        // staging — but it is not used in production GPU HNSW loads here.)
        // The transient peak is therefore dominated by the serialized read
        // buffer plus the deserialized compact index, ~2x the on-disk file
        // size — not the fp32 rows*dim*4 blowup that a Flat/hnswlib path would
        // incur. Basing the estimate on file_size also avoids relying on
        // num_rows/dim, which can arrive as 0 at estimate time.
        //
        // COSINE is the exception. int8-cosine (and any cosine) cannot upload
        // directly: normalized signed-int8 codes collapse onto a few levels, so
        // ToVanillaHnsw() reconstructs the segment, normalizes, and re-encodes
        // to fp16 before the upload (see the cosine branch there). Even with the
        // chunked reconstruct, the host transient then holds, concurrently:
        //   serialized read buffer (~1x file) + deserialized compact index
        //   (~1x file) + fp16 re-encoded storage (~2x the int8 codes, i.e. up to
        //   ~2x file) + one bounded reconstruct chunk (256 MiB).
        // The [gpu_hnsw_load_mem] unit test measures ~3.9x file above a baseline
        // that already holds the read buffer, i.e. ~4x file + one chunk from the
        // pre-load state. Charging only 2x here is what under-reserved the
        // querynode load admission and let concurrent cosine loads OOM the pod.
        // So the cosine path reserves 4x file + a fixed slack (chunk + CUDA/
        // allocator overhead). Native L2/IP int8/fp16/bf16 keep 2x (direct
        // upload, no fp32/fp16 re-encode).
        //
        // KNOWN LIMITATION: 2x still UNDER-counts two non-cosine fp32-staging
        // paths — VECTOR_FLOAT IndexHNSWFlat (reconstruct_n into a host float[])
        // and unsigned QT_8bit SQ8 (sa_decode into float) — whose true peak is
        // closer to file_size + rows*dim*4. Not corrected here because this
        // static entry point has no data-type discriminator (one
        // StaticEstimateLoadResource is registered for fp32/fp16/bf16/int8);
        // production GPU HNSW does not use those paths. Tracked as a follow-up.
        (void)num_rows;
        (void)dim;
        (void)version;

        // Fixed slack covering the bounded reconstruct chunk plus CUDA/allocator
        // overhead that does not scale with file_size.
        constexpr uint64_t kLoadFixedOverheadBytes = static_cast<uint64_t>(512) << 20;

        bool is_cosine = true;  // conservative default when metric is absent
        if (config.metric_type.has_value()) {
            is_cosine = IsMetricType(config.metric_type.value(), knowhere::metric::COSINE);
        }

        const uint64_t peak = is_cosine ? (file_size_in_bytes * 4 + kLoadFixedOverheadBytes) : (file_size_in_bytes * 2);

        // Device (VRAM) footprint retained after upload, independent of the host
        // transient peak above. The [gpu_hnsw_load_mem] test measures ~1.7x file
        // for int8-cosine (vectors re-encoded to fp16 on the device); fp16/bf16
        // native storage is ~2x the codes and int8/L2 native ~1x. 2x file covers
        // all of these with margin. Reported separately so querynode GPU
        // admission reserves the real VRAM growth rather than the host peak.
        const uint64_t gpu_peak = file_size_in_bytes * 2;

        LOG_KNOWHERE_INFO_ << "GPU_HNSW load estimate: file_size=" << file_size_in_bytes << " is_cosine=" << is_cosine
                           << " transient_peak=" << peak << " gpu_peak=" << gpu_peak << " retained=0";

        return Resource{.memoryCost = 0, .diskCost = 0, .maxMemoryCost = peak, .gpuMemoryCost = gpu_peak};
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return StaticCreateConfig();
    }

    std::string
    Type() const override {
        return IndexEnum::INDEX_GPU_HNSW;
    }

    // After the eager GPU upload the CPU index (indexes[0]) is freed, so the
    // inherited Count()/Dim()/HasRawData()/GetVectorByIds() would report an
    // empty index. Serve vector count and dimension from the metadata captured
    // before the CPU copy was released.
    int64_t
    Count() const override {
        if (gpu_ready_.load(std::memory_order_acquire)) {
            return gpu_ntotal_;
        }
        return BaseFaissRegularIndexHNSWNode::Count();
    }

    int64_t
    Dim() const override {
        if (gpu_ready_.load(std::memory_order_acquire)) {
            return gpu_dim_;
        }
        return BaseFaissRegularIndexHNSWNode::Dim();
    }

    bool
    HasRawData(const std::string& metric_type) const override {
        // The CPU copy is freed after the GPU upload; GPU_HNSW does not expose
        // CPU-reconstructable raw data (kept consistent with StaticHasRawData).
        if (gpu_ready_.load(std::memory_order_acquire)) {
            return false;
        }
        return BaseFaissRegularIndexHNSWNode::HasRawData(metric_type);
    }

    expected<DataSetPtr>
    GetVectorByIds(const DataSetPtr dataset, milvus::OpContext* op_context) const override {
        if (gpu_ready_.load(std::memory_order_acquire)) {
            return expected<DataSetPtr>::Err(Status::not_implemented,
                                             "GPU_HNSW keeps vectors on GPU; raw data retrieval is not supported");
        }
        return BaseFaissRegularIndexHNSWNode::GetVectorByIds(dataset, op_context);
    }

 protected:
    Status
    TrainInternal(const DataSetPtr /*dataset*/, const Config& /*cfg*/) override {
        return Status::not_implemented;
    }

 public:
    Status
    Deserialize(const BinarySet& binset, std::shared_ptr<Config> cfg) override {
        std::unique_lock lock(gpu_mutex_);
        UnpublishGpuIndexLocked();

        // Accept CPU-built HNSW (F32) or HNSW_SQ (quantized) binaries.
        // The base loader reads the payload via binset.GetByName(Type()), so
        // the incoming binary must be filed under *this* node's Type():
        // INDEX_GPU_HNSW for the base node, INDEX_GPU_HNSW_SQ for the SQ
        // subclass. Aliasing under a hardcoded INDEX_GPU_HNSW would make
        // GPU_HNSW_SQ loads fail — the SQ loader looks up INDEX_GPU_HNSW_SQ and
        // would find nothing.
        Status status;
        const std::string self_key = Type();
        if (!binset.Contains(self_key)) {
            BinarySet aliased = binset;
            for (const char* key : {IndexEnum::INDEX_GPU_HNSW, IndexEnum::INDEX_GPU_HNSW_SQ, IndexEnum::INDEX_HNSW_SQ,
                                    IndexEnum::INDEX_HNSW}) {
                if (binset.Contains(key)) {
                    aliased.Append(self_key, binset.GetByName(key));
                    break;
                }
            }
            status = BaseFaissRegularIndexHNSWNode::Deserialize(aliased, cfg);
        } else {
            status = BaseFaissRegularIndexHNSWNode::Deserialize(binset, cfg);
        }
        if (status != Status::success) {
            return status;
        }

        return UploadCpuIndexToGpuLocked();
    }

    // Milvus loads sealed segments through the mmap/file path
    // (VectorMemIndex::LoadFromFile -> DeserializeFromFile), NOT Deserialize().
    // The base only materializes the CPU index, so without this override the
    // eager GPU upload would never run on that path: the CPU-built HNSW would
    // stay resident in host RAM (fp32-expanded for quantized inputs) and the GPU
    // would sit unused. Load the CPU index via the base, then upload + free it.
    Status
    DeserializeFromFile(const std::string& filename, std::shared_ptr<Config> config) override {
        std::unique_lock lock(gpu_mutex_);
        UnpublishGpuIndexLocked();

        auto status = BaseFaissRegularIndexHNSWNode::DeserializeFromFile(filename, config);
        if (status != Status::success) {
            return status;
        }

        return UploadCpuIndexToGpuLocked();
    }

    expected<DataSetPtr>
    Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
           milvus::OpContext* op_context) const override {
        // Filtered search (deletes / TTL / partition / visibility bitset) is
        // applied inside the GPU kernel: filtered rows stay graph waypoints but
        // never enter the top-k, and a brute-force fallback guarantees k live
        // results under heavy deletes (see faiss GpuHnswSearch.cuh /
        // GpuHnswSearchKernel.cuh). The kernel tests the bitset by raw FAISS
        // node id, which equals the segment row offset == BitsetView index
        // (proven by the gpu_hnsw_id_mapping test). That direct mapping only
        // holds when the view has no id remapping and no id offset, so those
        // rarer shapes (never produced for a sealed GPU segment's delete
        // bitset) are still rejected rather than silently mis-filtered.
        const bool has_filter = !bitset.empty() && bitset.data() != nullptr;
        if (has_filter && (bitset.has_out_ids() || bitset.id_offset() != 0)) {
            return expected<DataSetPtr>::Err(Status::invalid_args,
                                             "GPU_HNSW filtered search does not support id-remapped or "
                                             "id-offset bitsets");
        }

        // Take a shared-ownership snapshot of the GPU index under gpu_mutex_. The
        // snapshot keeps the index alive for the whole search even if a concurrent
        // Deserialize() reload resets/rebuilds gpu_index_; the lock is held only
        // for the pointer copy, not the search itself. (std::atomic<shared_ptr>
        // is C++20/libstdc++-12 only, so guard the plain shared_ptr with the
        // existing mutex instead.)
        std::shared_ptr<faiss::gpu::GpuIndexHNSW> gpu_snapshot;
        {
            std::lock_guard<std::mutex> lock(gpu_mutex_);
            gpu_snapshot = gpu_index_;
        }
        if (!gpu_snapshot) {
            std::unique_lock<std::mutex> lock(gpu_mutex_);
            gpu_snapshot = gpu_index_;
            if (!gpu_snapshot) {
                if (GetFaissHnswIndex() == nullptr) {
                    return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
                }
                // Reuse the eager upload path (it holds no lock itself and we
                // already hold gpu_mutex_). This keeps metric detection
                // identical to Deserialize()/DeserializeFromFile(): the metric
                // is derived from the FAISS index type (dynamic_cast for cosine,
                // metric_type for IP), NOT from the query config. Detecting from
                // config here was a latent bug — a search whose config omits
                // metric_type would default to L2 and silently run a cosine/IP
                // index with the wrong metric.
                Status upload_status = UploadCpuIndexToGpuLocked();
                if (upload_status != Status::success) {
                    return expected<DataSetPtr>::Err(Status::cuvs_inner_error, "failed to build GPU HNSW index");
                }
                gpu_snapshot = gpu_index_;
                if (!gpu_snapshot) {
                    return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
                }
            }
        }

        const auto& hnsw_cfg = static_cast<const FaissHnswConfig&>(*cfg);
        auto k = hnsw_cfg.k.value();
        auto nq = dataset->GetRows();
        auto dim = dataset->GetDim();
        auto ef = hnsw_cfg.ef.value_or(200);

        // Derive the metric from the built GPU index (captured at upload time
        // from the FAISS index type), NOT from the search config. A config that
        // omits metric_type defaults to L2, which would skip cosine query
        // normalization and silently return wrong scores for a cosine index.
        const bool is_cosine = gpu_is_cosine_;

        // The kernel indexes the bitset by node id in [0, ntotal); a bitset
        // shorter than ntotal would read out of bounds. Milvus sizes the delete
        // bitset to the segment row count (== ntotal), so this only guards
        // against a malformed caller.
        if (has_filter && static_cast<int64_t>(bitset.size()) < gpu_snapshot->ntotal) {
            return expected<DataSetPtr>::Err(Status::invalid_args,
                                             "GPU_HNSW filtered search: bitset shorter than index size");
        }

        // Fill the filtered-search fields of a search-params struct from the
        // incoming bitset. A null/empty bitset leaves them defaulted, taking
        // the unfiltered fast path.
        auto set_filter_params = [&](faiss::gpu::GpuHnswSearchParams& gsp) {
            if (!has_filter) {
                return;
            }
            gsp.bitset_data = bitset.data();
            gsp.bitset_nbits = static_cast<int64_t>(bitset.size());
            gsp.bitset_filtered_count = static_cast<int64_t>(bitset.count());
            gsp.disable_fallback_brute_force = hnsw_cfg.disable_fallback_brute_force.value_or(false);
        };

        auto h_ids = std::make_unique<int64_t[]>(nq * k);
        auto h_dist = std::make_unique<float[]>(nq * k);

        // Native int8 DP4A path: L2/IP queries arrive as raw int8 (no
        // MockWrapper upcast) and match the direct_signed int8 DB via DP4A.
        // int8 COSINE is deliberately excluded: its DB was re-encoded as fp16
        // in ToVanillaHnsw() (direct_signed collapses normalized vectors), so
        // it must go through the fp32/normalized searchHost() path below, not
        // DP4A against fp16 codes.
        if (data_format == DataFormatEnum::int8 && !is_cosine) {
            const auto* h_queries_i8 = reinterpret_cast<const int8_t*>(dataset->GetTensor());
            try {
                faiss::gpu::GpuHnswSearchParams gsp;
                gsp.ef = ef;
                set_filter_params(gsp);
                gpu_snapshot->searchHostInt8(nq, h_queries_i8, k, h_dist.get(), h_ids.get(), gsp);
            } catch (const std::exception& e) {
                LOG_KNOWHERE_ERROR_ << "GPU_HNSW int8 search failed: " << e.what();
                return expected<DataSetPtr>::Err(Status::cuvs_inner_error,
                                                 std::string("GPU HNSW int8 search failed: ") + e.what());
            }

            // No sign flip: faiss now returns the true similarity for IP
            // (larger == closer), negating its internal min-first score on
            // copy-out. Previously the kernel returned the negated score and
            // this path flipped it back.
            return GenResultDataSet(nq, k, h_ids.release(), h_dist.release());
        }

        // fp32 path (fp16/bf16 have already been upcast by MockWrapper). int8
        // COSINE also lands here: the raw int8 query is widened to fp32 so it
        // can be normalized and searched against the fp16-re-encoded DB.
        std::unique_ptr<float[]> int8_queries_f32;
        const float* h_queries_raw;
        if (data_format == DataFormatEnum::int8) {
            const auto* h_queries_i8 = reinterpret_cast<const int8_t*>(dataset->GetTensor());
            int8_queries_f32 = std::make_unique<float[]>(static_cast<size_t>(nq) * dim);
            for (int64_t i = 0; i < static_cast<int64_t>(nq) * dim; i++) {
                int8_queries_f32[i] = static_cast<float>(h_queries_i8[i]);
            }
            h_queries_raw = int8_queries_f32.get();
        } else {
            h_queries_raw = reinterpret_cast<const float*>(dataset->GetTensor());
        }

        // For COSINE metric, normalize queries to unit length.
        const float* h_queries = h_queries_raw;
        std::unique_ptr<float[]> normalized_queries;
        if (is_cosine) {
            normalized_queries = std::make_unique<float[]>(nq * dim);
            for (int64_t i = 0; i < nq; i++) {
                const float* src = h_queries_raw + i * dim;
                float* dst = normalized_queries.get() + i * dim;
                float sq_norm = 0.0f;
                for (int64_t d = 0; d < dim; d++) sq_norm += src[d] * src[d];
                float inv = (sq_norm > 0.0f) ? (1.0f / std::sqrt(sq_norm)) : 1.0f;
                for (int64_t d = 0; d < dim; d++) dst[d] = src[d] * inv;
            }
            h_queries = normalized_queries.get();
        }

        try {
            faiss::gpu::GpuHnswSearchParams gsp;
            gsp.ef = ef;
            set_filter_params(gsp);
            gpu_snapshot->searchHost(nq, h_queries, k, h_dist.get(), h_ids.get(), gsp);
        } catch (const std::exception& e) {
            LOG_KNOWHERE_ERROR_ << "GPU_HNSW search failed: " << e.what();
            return expected<DataSetPtr>::Err(Status::cuvs_inner_error,
                                             std::string("GPU HNSW search failed: ") + e.what());
        }

        // No sign flip: faiss now returns the true similarity for IP/COSINE
        // (larger == closer), negating its internal min-first score on copy-out.
        return GenResultDataSet(nq, k, h_ids.release(), h_dist.release());
    }

    ~GpuHnswIndexNode() override = default;

 private:
    // Requires gpu_mutex_ held. Unpublish before tearing down: on a reload a
    // stale gpu_ready_==true combined with a reset (or failed re-upload of)
    // gpu_index_ would let the lock-free reader in Search() dereference a null
    // index. Clearing it here leaves the node honestly "not ready" if the
    // subsequent (re-)upload fails.
    void
    UnpublishGpuIndexLocked() {
        gpu_ready_.store(false, std::memory_order_release);
        gpu_index_ = nullptr;
    }

    // Requires gpu_mutex_ held. Uploads the just-deserialized CPU HNSW to the GPU
    // via faiss::gpu::GpuIndexHNSW, releases the host copy, and publishes the
    // device index. Shared by Deserialize() (BinarySet) and DeserializeFromFile()
    // (Milvus mmap/file load path) so both entrypoints upload identically. A
    // no-op success when there is no CPU index to upload (empty segment).
    Status
    UploadCpuIndexToGpuLocked() const {
        const auto* faiss_idx = GetFaissHnswIndex();
        if (faiss_idx == nullptr) {
            return Status::success;
        }
        try {
            // Detect metric from the FAISS index type rather than config, because
            // deserialize may be called without metric_type in the config (e.g.
            // an empty json defaults metric_type to L2).
            bool is_cosine =
                dynamic_cast<const ::faiss::cppcontrib::knowhere::HasInverseL2Norms*>(faiss_idx) != nullptr;

            // Convert the deserialized cppcontrib index to a vanilla
            // faiss::IndexHNSW and upload via the vanilla copyFrom(). Cosine is
            // folded into normalize+IP inside ToVanillaHnsw (out metric == IP),
            // so the GPU index needs no cosine/inverse-norm handling.
            std::unique_ptr<faiss::IndexHNSW> vanilla = ToVanillaHnsw(faiss_idx, is_cosine);

            std::shared_ptr<faiss::gpu::GpuIndexHNSW> local;
            {
                std::lock_guard<std::mutex> gpu_ctor_lock(GetGpuConstructionMutex());
                gpu_resources_ = GetSharedGpuResources();
                local =
                    std::make_shared<faiss::gpu::GpuIndexHNSW>(gpu_resources_.get(), vanilla->d, vanilla->metric_type);
            }
            local->copyFrom(vanilla.get());
            // Capture count/dim before releasing the CPU copy so Count()/Dim()
            // keep working once indexes[0] is freed.
            gpu_ntotal_ = faiss_idx->ntotal;
            gpu_dim_ = faiss_idx->d;
            // Remember cosine-ness for the query-side normalization in Search()
            // (the GPU index only knows IP, not cosine).
            gpu_is_cosine_ = is_cosine;
            // Release CPU copy — vectors and graph are now on GPU. This is what
            // drops the transient host-RAM peak back to ~0. const_cast because
            // this helper is const (reused from the const Search() lazy path);
            // it only frees the now-uploaded host copy.
            const_cast<std::shared_ptr<faiss::Index>&>(indexes[0]).reset();
            // Publish last: Search() takes a locked snapshot of gpu_index_, and
            // gpu_ready_ is released for the lock-free Count()/Dim() readers.
            gpu_index_ = local;
            gpu_ready_.store(true, std::memory_order_release);
        } catch (const std::exception& e) {
            // Fail the load rather than reporting success with no GPU index: a
            // silently "loaded" segment would only surface the error on the first
            // search. Leave the member null so gpu_ready_ stays false and return
            // an error so the caller treats the load as failed.
            LOG_KNOWHERE_ERROR_ << "eager GPU upload failed: " << e.what();
            gpu_index_ = nullptr;
            return Status::cuda_runtime_error;
        }
        return Status::success;
    }

    const ::faiss::cppcontrib::knowhere::IndexHNSW*
    GetFaissHnswIndex() const {
        if (indexes.empty() || !indexes[0])
            return nullptr;
        return dynamic_cast<const ::faiss::cppcontrib::knowhere::IndexHNSW*>(indexes[0].get());
    }

    mutable std::mutex gpu_mutex_;
    mutable std::shared_ptr<faiss::gpu::StandardGpuResources> gpu_resources_;
    // Guarded by gpu_mutex_. Search() copies this into a local shared_ptr under
    // the lock so the GPU index stays alive for the entire (unlocked) search even
    // if a concurrent Deserialize() reload resets/rebuilds the member. A bare
    // pointer read on the search fast path would be a use-after-free, and
    // std::atomic<std::shared_ptr> is C++20/libstdc++-12 only (the build
    // toolchain is GCC 11), so the mutex-guarded snapshot is used instead.
    mutable std::shared_ptr<faiss::gpu::GpuIndexHNSW> gpu_index_;
    // Published (release) after gpu_index_ is fully built; read lock-free
    // (acquire) by Count()/Dim()/HasRawData()/GetVectorByIds().
    mutable std::atomic<bool> gpu_ready_{false};
    // Vector count/dim captured before the CPU copy is freed (written from the
    // const lazy-upload path in Search(), hence mutable).
    mutable int64_t gpu_ntotal_ = 0;
    mutable int64_t gpu_dim_ = 0;
    // Whether the loaded index used the COSINE metric. The GPU index only
    // distinguishes L2 vs IP (cosine is folded into normalize+IP at upload),
    // so Search() consults this to decide whether to normalize queries.
    // Written under the construction path before gpu_ready_ is published.
    mutable bool gpu_is_cosine_ = false;
};

// GPU_HNSW_SQ is an alias of GPU_HNSW: the GPU search path is identical (a
// CPU-built HNSW/HNSW_SQ index uploaded to GpuIndexHNSW), so it reuses
// GpuHnswIndexNode and only reports a distinct Type() so Milvus's configured
// index_type matches the loaded node.
class GpuHnswSQIndexNode : public GpuHnswIndexNode {
 public:
    GpuHnswSQIndexNode(const int32_t& version, const Object& object) : GpuHnswIndexNode(version, object) {
    }

    GpuHnswSQIndexNode(const int32_t& version, const Object& object, DataFormatEnum query_format)
        : GpuHnswIndexNode(version, object, query_format) {
    }

    std::string
    Type() const override {
        return IndexEnum::INDEX_GPU_HNSW_SQ;
    }
};

// Register GPU_HNSW / GPU_HNSW_SQ in the static config map at process startup.
__attribute__((constructor)) static void
register_gpu_hnsw_static_config() {
    IndexStaticFaced<fp32>::Instance().RegisterStaticFunc<GpuHnswIndexNode>(IndexEnum::INDEX_GPU_HNSW);
    IndexStaticFaced<fp16>::Instance().RegisterStaticFunc<GpuHnswIndexNode>(IndexEnum::INDEX_GPU_HNSW);
    IndexStaticFaced<bf16>::Instance().RegisterStaticFunc<GpuHnswIndexNode>(IndexEnum::INDEX_GPU_HNSW);
    IndexStaticFaced<int8>::Instance().RegisterStaticFunc<GpuHnswIndexNode>(IndexEnum::INDEX_GPU_HNSW);
    IndexStaticFaced<fp32>::Instance().RegisterStaticFunc<GpuHnswSQIndexNode>(IndexEnum::INDEX_GPU_HNSW_SQ);
    IndexStaticFaced<fp16>::Instance().RegisterStaticFunc<GpuHnswSQIndexNode>(IndexEnum::INDEX_GPU_HNSW_SQ);
    IndexStaticFaced<bf16>::Instance().RegisterStaticFunc<GpuHnswSQIndexNode>(IndexEnum::INDEX_GPU_HNSW_SQ);
    IndexStaticFaced<int8>::Instance().RegisterStaticFunc<GpuHnswSQIndexNode>(IndexEnum::INDEX_GPU_HNSW_SQ);
}

// GPU_HNSW is not registered as MMAP-capable: the CPU index is deserialized
// into host RAM, uploaded to VRAM, then the CPU copy is freed. For a native
// int8 index the transient CPU copy is compact (~1 byte/dim), matching how the
// CPU HNSW int8 node loads, so no memory-mapped-file path is required.
KNOWHERE_REGISTER_GLOBAL(
    GPU_HNSW,
    [](const int32_t& version, const Object& object) { return Index<GpuHnswIndexNode>::Create(version, object); }, fp32,
    true, (feature::GPU_ANN_FLOAT_INDEX));

KNOWHERE_REGISTER_GLOBAL(
    GPU_HNSW,
    [](const int32_t& version, const Object& object) {
        return Index<IndexNodeDataMockWrapper<fp16>>::Create(std::make_unique<GpuHnswIndexNode>(version, object));
    },
    fp16, true, (feature::FP16 | feature::GPU));

KNOWHERE_REGISTER_GLOBAL(
    GPU_HNSW,
    [](const int32_t& version, const Object& object) {
        return Index<IndexNodeDataMockWrapper<bf16>>::Create(std::make_unique<GpuHnswIndexNode>(version, object));
    },
    bf16, true, (feature::BF16 | feature::GPU));

KNOWHERE_REGISTER_GLOBAL(
    GPU_HNSW,
    [](const int32_t& version, const Object& object) {
        // Native int8 path: bypass MockWrapper upcast, pass int8 queries directly
        // to searchHostInt8(), which uploads the signed int8 queries verbatim (no
        // +128 shift; the dataset upload already reverses FAISS's +128 SQ bias)
        // and uses the DP4A kernel.
        return Index<GpuHnswIndexNode>::Create(version, object, DataFormatEnum::int8);
    },
    int8, true, (feature::INT8 | feature::GPU));

KNOWHERE_REGISTER_GLOBAL(
    GPU_HNSW_SQ,
    [](const int32_t& version, const Object& object) { return Index<GpuHnswSQIndexNode>::Create(version, object); },
    fp32, true, (feature::GPU_ANN_FLOAT_INDEX));

KNOWHERE_REGISTER_GLOBAL(
    GPU_HNSW_SQ,
    [](const int32_t& version, const Object& object) {
        return Index<IndexNodeDataMockWrapper<fp16>>::Create(std::make_unique<GpuHnswSQIndexNode>(version, object));
    },
    fp16, true, (feature::FP16 | feature::GPU));

KNOWHERE_REGISTER_GLOBAL(
    GPU_HNSW_SQ,
    [](const int32_t& version, const Object& object) {
        return Index<IndexNodeDataMockWrapper<bf16>>::Create(std::make_unique<GpuHnswSQIndexNode>(version, object));
    },
    bf16, true, (feature::BF16 | feature::GPU));

KNOWHERE_REGISTER_GLOBAL(
    GPU_HNSW_SQ,
    [](const int32_t& version, const Object& object) {
        // Native int8 path: bypass MockWrapper upcast, pass int8 queries directly
        // to searchHostInt8(), which uploads the signed int8 queries verbatim (no
        // +128 shift; the dataset upload already reverses FAISS's +128 SQ bias)
        // and uses the DP4A kernel.
        return Index<GpuHnswSQIndexNode>::Create(version, object, DataFormatEnum::int8);
    },
    int8, true, (feature::INT8 | feature::GPU));
#endif  // KNOWHERE_WITH_CUVS

}  // namespace knowhere
