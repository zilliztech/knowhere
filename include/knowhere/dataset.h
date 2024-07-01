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

#ifndef DATASET_H
#define DATASET_H

#include <any>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <utility>
#include <variant>

#include "comp/index_param.h"
#include "knowhere/range_util.h"
#include "knowhere/sparse_utils.h"

namespace knowhere {

class DataSet : public std::enable_shared_from_this<const DataSet> {
 public:
    typedef std::variant<const float*, const size_t*, const int64_t*, const void*, int64_t, std::string, std::any> Var;
    DataSet() = default;
    ~DataSet() {
        if (!is_owner) {
            return;
        }
        for (auto&& x : this->data_) {
            {
                auto ptr = std::get_if<0>(&x.second);
                if (ptr != nullptr) {
                    delete[] * ptr;
                }
            }
            {
                auto ptr = std::get_if<1>(&x.second);
                if (ptr != nullptr) {
                    delete[] * ptr;
                }
            }
            {
                auto ptr = std::get_if<2>(&x.second);
                if (ptr != nullptr) {
                    delete[] * ptr;
                }
            }
            {
                auto ptr = std::get_if<3>(&x.second);
                if (ptr != nullptr) {
                    if (is_sparse) {
                        delete[](sparse::SparseRow<float>*)(*ptr);
                    } else {
                        delete[](char*)(*ptr);
                    }
                }
            }
        }
    }

    void
    SetDistance(const float* dis) {
        std::unique_lock lock(mutex_);
        this->data_[meta::DISTANCE] = Var(std::in_place_index<0>, dis);
    }

    void
    SetDistance(std::unique_ptr<float[]>&& dis) {
        std::unique_lock lock(mutex_);
        this->data_[meta::DISTANCE] = Var(std::in_place_index<0>, dis.release());
    }

    void
    SetLims(const size_t* lims) {
        std::unique_lock lock(mutex_);
        this->data_[meta::LIMS] = Var(std::in_place_index<1>, lims);
    }

    void
    SetLims(std::unique_ptr<size_t[]>&& lims) {
        std::unique_lock lock(mutex_);
        this->data_[meta::LIMS] = Var(std::in_place_index<1>, lims.release());
    }

    void
    SetIds(const int64_t* ids) {
        std::unique_lock lock(mutex_);
        this->data_[meta::IDS] = Var(std::in_place_index<2>, ids);
    }

    void
    SetIds(std::unique_ptr<long int[]>&& ids) {
        static_assert(sizeof(long int) == sizeof(int64_t));

        std::unique_lock lock(mutex_);
        this->data_[meta::IDS] = Var(std::in_place_index<2>, reinterpret_cast<int64_t*>(ids.release()));
    }

    void
    SetIds(std::unique_ptr<long long int[]>&& ids) {
        static_assert(sizeof(long long int) == sizeof(int64_t));

        std::unique_lock lock(mutex_);
        this->data_[meta::IDS] = Var(std::in_place_index<2>, reinterpret_cast<int64_t*>(ids.release()));
    }

    /**
     * For dense float vector, tensor is a rows * dim float array
     * For sparse float vector, tensor is pointer to sparse::Sparse<float>*
     * and values in each row should be sorted by column id.
     */
    void
    SetTensor(const void* tensor) {
        std::unique_lock lock(mutex_);
        this->data_[meta::TENSOR] = Var(std::in_place_index<3>, tensor);
    }

    void
    SetTensor(std::unique_ptr<uint8_t[]>&& tensor) {
        std::unique_lock lock(mutex_);
        this->data_[meta::TENSOR] = Var(std::in_place_index<3>, tensor.release());
    }

    void
    SetTensor(std::unique_ptr<float[]>&& tensor) {
        std::unique_lock lock(mutex_);
        this->data_[meta::TENSOR] = Var(std::in_place_index<3>, tensor.release());
    }

    void
    SetRows(const int64_t rows) {
        std::unique_lock lock(mutex_);
        this->data_[meta::ROWS] = Var(std::in_place_index<4>, rows);
    }

    void
    SetDim(const int64_t dim) {
        std::unique_lock lock(mutex_);
        this->data_[meta::DIM] = Var(std::in_place_index<4>, dim);
    }

    void
    SetJsonInfo(const std::string& info) {
        std::unique_lock lock(mutex_);
        this->data_[meta::JSON_INFO] = Var(std::in_place_index<5>, info);
    }

    void
    SetJsonIdSet(const std::string& idset) {
        std::unique_lock lock(mutex_);
        this->data_[meta::JSON_ID_SET] = Var(std::in_place_index<5>, idset);
    }

    const float*
    GetDistance() const {
        std::shared_lock lock(mutex_);
        auto it = this->data_.find(meta::DISTANCE);
        if (it != this->data_.end()) {
            const float* res = *std::get_if<0>(&it->second);
            return res;
        }
        return nullptr;
    }

    const size_t*
    GetLims() const {
        std::shared_lock lock(mutex_);
        auto it = this->data_.find(meta::LIMS);
        if (it != this->data_.end()) {
            const size_t* res = *std::get_if<1>(&it->second);
            return res;
        }
        return nullptr;
    }

    const int64_t*
    GetIds() const {
        std::shared_lock lock(mutex_);
        auto it = this->data_.find(meta::IDS);
        if (it != this->data_.end()) {
            const int64_t* res = *std::get_if<2>(&it->second);
            return res;
        }
        return nullptr;
    }

    const void*
    GetTensor() const {
        std::shared_lock lock(mutex_);
        auto it = this->data_.find(meta::TENSOR);
        if (it != this->data_.end()) {
            const void* res = *std::get_if<3>(&it->second);
            return res;
        }
        return nullptr;
    }

    int64_t
    GetRows() const {
        std::shared_lock lock(mutex_);
        auto it = this->data_.find(meta::ROWS);
        if (it != this->data_.end()) {
            int64_t res = *std::get_if<4>(&it->second);
            return res;
        }
        return 0;
    }

    int64_t
    GetDim() const {
        std::shared_lock lock(mutex_);
        auto it = this->data_.find(meta::DIM);
        if (it != this->data_.end()) {
            int64_t res = *std::get_if<4>(&it->second);
            return res;
        }
        return 0;
    }

    std::string
    GetJsonInfo() const {
        std::shared_lock lock(mutex_);
        auto it = this->data_.find(meta::JSON_INFO);
        if (it != this->data_.end()) {
            std::string res = *std::get_if<5>(&it->second);
            return res;
        }
        return "";
    }

    std::string
    GetJsonIdSet() const {
        std::shared_lock lock(mutex_);
        auto it = this->data_.find(meta::JSON_ID_SET);
        if (it != this->data_.end()) {
            std::string res = *std::get_if<5>(&it->second);
            return res;
        }
        return "";
    }

    void
    SetIsOwner(bool is_owner) {
        std::unique_lock lock(mutex_);
        this->is_owner = is_owner;
    }

    bool
    GetIsSparse() const {
        std::unique_lock lock(mutex_);
        return this->is_sparse;
    }

    void
    SetIsSparse(bool is_sparse) {
        std::unique_lock lock(mutex_);
        this->is_sparse = is_sparse;
    }

    // deprecated API
    template <typename T>
    void
    Set(const std::string& k, T&& v) {
        std::unique_lock lock(mutex_);
        data_[k] = Var(std::in_place_type<std::any>, std::forward<T>(v));
    }

    template <typename T>
    T
    Get(const std::string& k) {
        std::shared_lock lock(mutex_);
        auto it = this->data_.find(k);
        if (it != this->data_.end()) {
            return *std::any_cast<T>(std::get_if<std::any>(&it->second));
        }
        return T();
    }

 private:
    mutable std::shared_mutex mutex_;
    std::map<std::string, Var> data_;
    bool is_owner = true;
    bool is_sparse = false;
};
using DataSetPtr = std::shared_ptr<DataSet>;

inline DataSetPtr
GenDataSet(const int64_t nb, const int64_t dim, const void* xb) {
    auto ret_ds = std::make_shared<DataSet>();
    ret_ds->SetRows(nb);
    ret_ds->SetDim(dim);
    ret_ds->SetTensor(xb);
    ret_ds->SetIsOwner(false);
    return ret_ds;
}

// swig won't compile when using int64_t* or size_t* as parameter
inline DataSetPtr
#ifdef NOT_COMPILE_FOR_SWIG
GenIdsDataSet(const int64_t rows, const int64_t* ids) {
#else
GenIdsDataSet(const int64_t rows, const void* ids) {
#endif
    auto ret_ds = std::make_shared<DataSet>();
    ret_ds->SetRows(rows);
    ret_ds->SetIds((const int64_t*)ids);
    ret_ds->SetIsOwner(false);
    return ret_ds;
}

inline DataSetPtr
GenResultDataSet(const int64_t rows, const int64_t dim, const void* tensor) {
    auto ret_ds = std::make_shared<DataSet>();
    ret_ds->SetRows(rows);
    ret_ds->SetDim(dim);
    ret_ds->SetTensor(tensor);
    ret_ds->SetIsOwner(true);
    return ret_ds;
}

inline DataSetPtr
GenResultDataSet(const int64_t rows, const int64_t dim, std::unique_ptr<uint8_t[]>&& tensor) {
    auto ret_ds = std::make_shared<DataSet>();
    ret_ds->SetRows(rows);
    ret_ds->SetDim(dim);
    ret_ds->SetTensor(std::move(tensor));
    ret_ds->SetIsOwner(true);
    return ret_ds;
}

inline DataSetPtr
GenResultDataSet(const int64_t rows, const int64_t dim, std::unique_ptr<float[]>&& tensor) {
    auto ret_ds = std::make_shared<DataSet>();
    ret_ds->SetRows(rows);
    ret_ds->SetDim(dim);
    ret_ds->SetTensor(std::move(tensor));
    ret_ds->SetIsOwner(true);
    return ret_ds;
}

inline DataSetPtr
#ifdef NOT_COMPILE_FOR_SWIG
GenResultDataSet(const int64_t nq, const int64_t topk, const int64_t* ids, const float* distance) {
#else
GenResultDataSet(const int64_t nq, const int64_t topk, const void* ids, const float* distance) {
#endif
    static_assert(sizeof(int64_t) == sizeof(long long int));

    auto ret_ds = std::make_shared<DataSet>();
    ret_ds->SetRows(nq);
    ret_ds->SetDim(topk);
    ret_ds->SetIds((const int64_t*)ids);
    ret_ds->SetDistance(distance);
    ret_ds->SetIsOwner(true);
    return ret_ds;
}

inline DataSetPtr
GenResultDataSet(const int64_t nq, const int64_t topk, std::unique_ptr<long int[]>&& ids,
                 std::unique_ptr<float[]>&& distance) {
    static_assert(sizeof(int64_t) == sizeof(long int));

    auto ret_ds = std::make_shared<DataSet>();
    ret_ds->SetRows(nq);
    ret_ds->SetDim(topk);
    ret_ds->SetIds(std::move(ids));
    ret_ds->SetDistance(std::move(distance));
    ret_ds->SetIsOwner(true);
    return ret_ds;
}

inline DataSetPtr
GenResultDataSet(const int64_t nq, const int64_t topk, std::unique_ptr<long long int[]>&& ids,
                 std::unique_ptr<float[]>&& distance) {
    static_assert(sizeof(int64_t) == sizeof(long long int));

    auto ret_ds = std::make_shared<DataSet>();
    ret_ds->SetRows(nq);
    ret_ds->SetDim(topk);
    ret_ds->SetIds(std::move(ids));
    ret_ds->SetDistance(std::move(distance));
    ret_ds->SetIsOwner(true);
    return ret_ds;
}

inline DataSetPtr
#ifdef NOT_COMPILE_FOR_SWIG
GenResultDataSet(const int64_t nq, const int64_t* ids, const float* distance, const size_t* lims) {
#else
GenResultDataSet(const int64_t nq, const void* ids, const float* distance, const void* lims) {
#endif
    auto ret_ds = std::make_shared<DataSet>();
    ret_ds->SetRows(nq);
    ret_ds->SetIds((const int64_t*)ids);
    ret_ds->SetDistance(distance);
    ret_ds->SetLims((const size_t*)lims);
    ret_ds->SetIsOwner(true);
    return ret_ds;
}

inline DataSetPtr
GenResultDataSet(const int64_t nq, RangeSearchResult&& range_search_result) {
    auto ret_ds = std::make_shared<DataSet>();
    ret_ds->SetRows(nq);
    ret_ds->SetIds(std::move(range_search_result.labels));
    ret_ds->SetDistance(std::move(range_search_result.distances));
    ret_ds->SetLims(std::move(range_search_result.lims));
    ret_ds->SetIsOwner(true);
    return ret_ds;
}

inline DataSetPtr
GenResultDataSet(const std::string& json_info, const std::string& json_id_set) {
    auto ret_ds = std::make_shared<DataSet>();
    ret_ds->SetJsonInfo(json_info);
    ret_ds->SetJsonIdSet(json_id_set);
    ret_ds->SetIsOwner(true);
    return ret_ds;
}

}  // namespace knowhere
#endif /* DATASET_H */
