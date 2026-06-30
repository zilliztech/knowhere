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
#include <functional>
#include <iostream>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#if defined(SWIG)
#define KNOWHERE_NODISCARD
#else
#define KNOWHERE_NODISCARD [[nodiscard]]
#endif

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
    cuvs_inner_error = 18,
    invalid_binary_set = 19,
    invalid_instruction_set = 20,
    cardinal_inner_error = 21,
    cuda_runtime_error = 22,
    invalid_index_error = 23,
    invalid_cluster_error = 24,
    cluster_inner_error = 25,
    timeout = 26,
    internal_error = 27,
    invalid_serialized_index_type = 28,
    sparse_inner_error = 29,
    brute_force_inner_error = 30,
    emb_list_inner_error = 31,
    aisaq_error = 32,
    knowhere_inner_error = 33,
};

enum class StatusCategory {
    success = 0,
    input_error = 1,
    inner_error = 2,
};

// Classify every knowhere::Status into a closed 3-value category. This is a
// switch with NO `default:` plus a post-switch fallback on purpose:
//   * a `default:` inside the switch suppresses -Wswitch, so a newly added
//     Status would silently fall into `inner_error` instead of being
//     deliberately classified;
//   * the post-switch `return` keeps the function total (and satisfies
//     -Wreturn-type) without suppressing the exhaustiveness warning.
// The surrounding pragma promotes -Wswitch to an error, so adding a
// knowhere::Status without categorizing it here breaks the build -- forcing the
// author to decide input vs inner. retry/ownership decisions downstream are
// derived from this category, so a missed status must never default silently.
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
inline constexpr StatusCategory
StatusCategoryOf(knowhere::Status status) {
    switch (status) {
        case knowhere::Status::success:
            return StatusCategory::success;
        case knowhere::Status::invalid_args:
        case knowhere::Status::invalid_param_in_json:
        case knowhere::Status::out_of_range_in_json:
        case knowhere::Status::type_conflict_in_json:
        case knowhere::Status::invalid_metric_type:
        case knowhere::Status::empty_index:
        case knowhere::Status::not_implemented:
        case knowhere::Status::index_not_trained:
        case knowhere::Status::index_already_trained:
        case knowhere::Status::invalid_value_in_json:
        case knowhere::Status::arithmetic_overflow:
        case knowhere::Status::invalid_binary_set:
        case knowhere::Status::invalid_instruction_set:
        case knowhere::Status::invalid_index_error:
        case knowhere::Status::invalid_cluster_error:
        case knowhere::Status::invalid_serialized_index_type:
            return StatusCategory::input_error;
        case knowhere::Status::faiss_inner_error:
        case knowhere::Status::hnsw_inner_error:
        case knowhere::Status::malloc_error:
        case knowhere::Status::diskann_inner_error:
        case knowhere::Status::disk_file_error:
        case knowhere::Status::cuvs_inner_error:
        case knowhere::Status::cardinal_inner_error:
        case knowhere::Status::cuda_runtime_error:
        case knowhere::Status::cluster_inner_error:
        case knowhere::Status::timeout:
        case knowhere::Status::internal_error:
        case knowhere::Status::sparse_inner_error:
        case knowhere::Status::brute_force_inner_error:
        case knowhere::Status::emb_list_inner_error:
        case knowhere::Status::aisaq_error:
        case knowhere::Status::knowhere_inner_error:
            return StatusCategory::inner_error;
    }
    return StatusCategory::inner_error;
}
#pragma GCC diagnostic pop

inline constexpr bool
IsInputError(knowhere::Status status) {
    return StatusCategoryOf(status) == StatusCategory::input_error;
}

inline constexpr bool
IsInnerError(knowhere::Status status) {
    return StatusCategoryOf(status) == StatusCategory::inner_error;
}

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
        case knowhere::Status::cuvs_inner_error:
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
        case knowhere::Status::internal_error:
            return "internal error (something that must not have happened at all)";
        case knowhere::Status::invalid_serialized_index_type:
            return "the serialized index type is not recognized";
        case knowhere::Status::sparse_inner_error:
            return "sparse index inner error";
        case knowhere::Status::brute_force_inner_error:
            return "brute_force inner error";
        case knowhere::Status::emb_list_inner_error:
            return "emb_list inner error";
        case knowhere::Status::aisaq_error:
            return "internal AiSAQ error";
        case knowhere::Status::knowhere_inner_error:
            return "knowhere inner error";
        default:
            return "unexpected status";
    }
}

template <typename T>
class KNOWHERE_NODISCARD expected {
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

    bool
    unexpected() {
        return err != knowhere::Status::success;
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

#if !defined(SWIG)

namespace detail {

template <typename T>
struct is_expected : std::false_type {};

template <typename T>
struct is_expected<expected<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_expected_v = is_expected<std::decay_t<T>>::value;

template <typename T>
struct expected_value;

template <typename T>
struct expected_value<expected<T>> {
    using type = T;
};

inline std::string
ExceptionMessage(const char* prefix, const std::string& what) {
    if (what.empty()) {
        return prefix;
    }
    return std::string(prefix) + ": " + what;
}

template <typename R>
std::decay_t<R>
GuardedFailure(Status status, std::string msg) noexcept {
    using Result = std::decay_t<R>;
    if constexpr (std::is_same_v<Result, Status>) {
        return status;
    } else if constexpr (is_expected_v<Result>) {
        using Value = typename expected_value<Result>::type;
        return expected<Value>::Err(status, std::move(msg));
    } else if constexpr (std::is_same_v<Result, bool>) {
        return false;
    } else if constexpr (std::is_integral_v<Result> || std::is_floating_point_v<Result>) {
        return Result{};
    } else if constexpr (std::is_same_v<Result, std::string>) {
        return {};
    } else if constexpr (std::is_default_constructible_v<Result>) {
        return Result{};
    } else {
        static_assert(std::is_default_constructible_v<Result>,
                      "GuardedCall requires Status, expected<T>, or a default-constructible return type");
    }
}

template <typename R>
std::decay_t<R>
GuardedCallFailure(Status status, std::string msg) noexcept {
    if constexpr (std::is_void_v<R>) {
        return;
    } else {
        return GuardedFailure<R>(status, std::move(msg));
    }
}

}  // namespace detail

template <typename Func, typename... Args>
std::decay_t<std::invoke_result_t<Func, Args...>>
GuardedCall(Func&& func, Args&&... args) noexcept {
    using Result = std::invoke_result_t<Func, Args...>;
    try {
        if constexpr (std::is_void_v<Result>) {
            std::invoke(std::forward<Func>(func), std::forward<Args>(args)...);
            return;
        } else {
            return std::invoke(std::forward<Func>(func), std::forward<Args>(args)...);
        }
    } catch (const std::bad_alloc& e) {
        return detail::GuardedCallFailure<Result>(Status::malloc_error,
                                                  detail::ExceptionMessage("bad alloc", e.what()));
    } catch (const std::exception& e) {
        return detail::GuardedCallFailure<Result>(Status::knowhere_inner_error,
                                                  detail::ExceptionMessage("unhandled exception", e.what()));
    } catch (...) {
        return detail::GuardedCallFailure<Result>(Status::knowhere_inner_error, "unknown exception");
    }
}

#endif

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
