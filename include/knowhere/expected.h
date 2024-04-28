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

#ifndef EXPECTED_H
#define EXPECTED_H

#include <cassert>
#include <iostream>
#include <optional>
#include <string>

namespace knowhere {

enum class Status {
    success = 0,
    invalid_args = 1,
    invalid_param_in_json = 2,
    out_of_range_in_json = 3,
    type_conflict_in_json = 4,
    invalid_metric_type = 5,
    empty_index = 6,
    not_implemented = 7,
    index_not_trained = 8,
    index_already_trained = 9,
    faiss_inner_error = 10,
    hnsw_inner_error = 12,
    malloc_error = 13,
    diskann_inner_error = 14,
    disk_file_error = 15,
    invalid_value_in_json = 16,
    arithmetic_overflow = 17,
    raft_inner_error = 18,
    invalid_binary_set = 19,
    invalid_instruction_set = 20,
    cardinal_inner_error = 21,
    cuda_runtime_error = 22,
    invalid_index_error = 23,
    invalid_cluster_error = 24,
    cluster_inner_error = 25,
};

inline std::string
Status2String(knowhere::Status status) {
    switch (status) {
        case knowhere::Status::invalid_args:
            return "invalid args";
        case knowhere::Status::invalid_param_in_json:
            return "invalid param in json";
        case knowhere::Status::out_of_range_in_json:
            return "out of range in json";
        case knowhere::Status::type_conflict_in_json:
            return "type conflict in json";
        case knowhere::Status::invalid_metric_type:
            return "invalid metric type";
        case knowhere::Status::empty_index:
            return "empty index";
        case knowhere::Status::not_implemented:
            return "not implemented";
        case knowhere::Status::index_not_trained:
            return "index not trained";
        case knowhere::Status::index_already_trained:
            return "index already trained";
        case knowhere::Status::faiss_inner_error:
            return "faiss inner error";
        case knowhere::Status::hnsw_inner_error:
            return "hnsw inner error";
        case knowhere::Status::malloc_error:
            return "malloc error";
        case knowhere::Status::diskann_inner_error:
            return "diskann inner error";
        case knowhere::Status::disk_file_error:
            return "disk file error";
        case knowhere::Status::invalid_value_in_json:
            return "invalid value in json";
        case knowhere::Status::arithmetic_overflow:
            return "arithmetic overflow";
        case knowhere::Status::raft_inner_error:
            return "raft inner error";
        case knowhere::Status::invalid_binary_set:
            return "invalid binary set";
        case knowhere::Status::invalid_instruction_set:
            return "the current index is not supported on the current CPU model";
        case knowhere::Status::cardinal_inner_error:
            return "cardinal inner error";
        case knowhere::Status::invalid_cluster_error:
            return "invalid cluster type";
        case knowhere::Status::cluster_inner_error:
            return "cluster inner error";
        default:
            return "unexpected status";
    }
}

template <typename T>
class expected {
 public:
    template <typename... Args>
    expected(Args&&... args) : val(std::make_optional<T>(std::forward<Args>(args)...)), err(Status::success) {
    }

    expected(const expected<T>&) = default;

    expected(expected<T>&&) = default;

    expected&
    operator=(const expected<T>&) = default;

    expected&
    operator=(expected<T>&&) = default;

    bool
    has_value() const {
        return val.has_value();
    }

    Status
    error() const {
        return err;
    }

    const T&
    value() const {
        assert(val.has_value() == true);
        return val.value();
    }
    T&
    value() {
        assert(val.has_value() == true);
        return val.value();
    }

    const std::string&
    what() const {
        return msg;
    }

    void
    operator<<(const std::string& msg) {
        this->msg += msg;
    }

    expected<T>&
    operator=(const Status& err) {
        assert(err != Status::success);
        this->err = err;
        return *this;
    }

    static expected<T>
    OK() {
        return expected(Status::success);
    }

    static expected<T>
    Err(const Status err, std::string msg) {
        return expected(err, std::move(msg));
    }

 private:
    // keep these private to avoid creating directly
    expected(const Status err) : err(err) {
    }

    expected(const Status err, std::string msg) : err(err), msg(std::move(msg)) {
        assert(err != Status::success);
    }

    std::optional<T> val = std::nullopt;
    Status err;
    std::string msg;
};

// Evaluates expr that returns a Status. Does nothing if the returned Status is
// a Status::success, otherwise returns the Status from the current function.
#define RETURN_IF_ERROR(expr)            \
    do {                                 \
        auto status = (expr);            \
        if (status != Status::success) { \
            return status;               \
        }                                \
    } while (0)

template <typename T>
expected<T>
DoAssignOrReturn(T& lhs, const expected<T>& exp) {
    if (exp.has_value()) {
        lhs = exp.value();
    }
    return exp;
}

}  // namespace knowhere

#endif /* EXPECTED_H */
