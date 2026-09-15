// Copyright (C) 2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include "knowhere/c_api.h"

#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#include "filemanager/impl/LocalFileManager.h"
#include "knowhere/comp/brute_force.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/version.h"

namespace {
thread_local char last_error[4096] = {};

class Failure : public std::runtime_error {
 public:
    Failure(int32_t code, const std::string& message) : std::runtime_error(message), code(code) {
    }
    int32_t code;
};

void
Require(bool valid, const char* message) {
    if (!valid) {
        throw Failure(KNOWHERE_INVALID_ARGUMENT, message);
    }
}

int32_t
StatusCode(knowhere::Status status) {
    if (status == knowhere::Status::malloc_error) {
        return KNOWHERE_OUT_OF_MEMORY;
    }
    if (status == knowhere::Status::not_implemented || status == knowhere::Status::invalid_instruction_set) {
        return KNOWHERE_UNSUPPORTED;
    }
    return knowhere::IsInputError(status) ? KNOWHERE_INVALID_ARGUMENT : KNOWHERE_ERROR;
}

void
CheckStatus(knowhere::Status status, const std::string& detail = {}) {
    if (status != knowhere::Status::success) {
        throw Failure(StatusCode(status), knowhere::Status2String(status) + (detail.empty() ? "" : ": " + detail));
    }
}

template <typename F>
int32_t
Boundary(F&& action) noexcept {
    last_error[0] = '\0';
    try {
        action();
        return KNOWHERE_SUCCESS;
    } catch (const Failure& e) {
        std::snprintf(last_error, sizeof(last_error), "%s", e.what());
        return e.code;
    } catch (const knowhere::Json::exception& e) {
        std::snprintf(last_error, sizeof(last_error), "invalid JSON parameters: %s", e.what());
        return KNOWHERE_INVALID_ARGUMENT;
    } catch (const knowhere::StatusException& e) {
        std::snprintf(last_error, sizeof(last_error), "Knowhere status %d: %s", static_cast<int>(e.status()), e.what());
        return StatusCode(e.status());
    } catch (const std::bad_alloc& e) {
        std::snprintf(last_error, sizeof(last_error), "%s", e.what());
        return KNOWHERE_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        std::snprintf(last_error, sizeof(last_error), "%s", e.what());
        return KNOWHERE_ERROR;
    } catch (...) {
        std::snprintf(last_error, sizeof(last_error), "unknown C++ exception");
        return KNOWHERE_ERROR;
    }
}

struct IndexResource {
    enum class State { Empty, Failed, Ready };
    std::mutex mutex;
    int32_t dtype;
    bool file_build;
    State state = State::Empty;
    // Keep retained storage alive until after the index destructor runs.
    std::shared_ptr<uint8_t[]> build_data;
    knowhere::BinarySet deserialized_data;
    knowhere::Index<knowhere::IndexNode> index;

    IndexResource(int32_t dtype, bool file_build, knowhere::Index<knowhere::IndexNode>&& index)
        : dtype(dtype), file_build(file_build), index(std::move(index)) {
    }
};

struct BinaryResource {
    std::mutex mutex;
    knowhere::BinarySet data;
};

std::mutex registry_mutex;
uint64_t next_sequence = 1;
std::unordered_map<uint64_t, std::shared_ptr<IndexResource>> indexes;
std::unordered_map<uint64_t, std::shared_ptr<BinaryResource>> binary_sets;

// The low bit identifies the resource kind. Sequence numbers are never recycled.
template <typename T>
uint64_t
Register(std::unordered_map<uint64_t, std::shared_ptr<T>>& registry, std::shared_ptr<T> resource, uint64_t kind) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    if (next_sequence > (std::numeric_limits<uint64_t>::max() >> 1)) {
        throw Failure(KNOWHERE_ERROR, "resource identifier space exhausted");
    }
    const uint64_t handle = (next_sequence++ << 1) | kind;
    registry.emplace(handle, std::move(resource));
    return handle;
}

template <typename T>
std::shared_ptr<T>
Get(std::unordered_map<uint64_t, std::shared_ptr<T>>& registry, uint64_t handle, uint64_t kind) {
    Require(handle == 0 || (handle & 1) == kind, "handle has the wrong resource type");
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto it = registry.find(handle);
    if (it == registry.end()) {
        throw Failure(KNOWHERE_CLOSED, "resource handle is closed or unknown");
    }
    return it->second;
}

template <typename T>
void
Close(std::unordered_map<uint64_t, std::shared_ptr<T>>& registry, uint64_t handle, uint64_t kind) {
    Require(handle == 0 || (handle & 1) == kind, "handle has the wrong resource type");
    std::shared_ptr<T> released;
    {
        std::lock_guard<std::mutex> lock(registry_mutex);
        auto it = registry.find(handle);
        if (it != registry.end()) {
            released = std::move(it->second);
            registry.erase(it);
        }
    }
    // Destruction is outside the registry lock. An in-flight call owns another reference.
}

uint64_t
Multiply(uint64_t left, uint64_t right) {
    Require(right == 0 || left <= std::numeric_limits<uint64_t>::max() / right, "buffer size multiplication overflow");
    const uint64_t size = left * right;
    Require(size <= std::numeric_limits<size_t>::max() && size <= INT64_MAX, "buffer size exceeds addressable range");
    return size;
}

uint64_t
ElementBytes(int32_t dtype) {
    switch (dtype) {
        case KNOWHERE_FP32:
            return 4;
        case KNOWHERE_FP16:
        case KNOWHERE_BF16:
            return 2;
        case KNOWHERE_BIN1:
        case KNOWHERE_INT8:
            return 1;
        default:
            throw Failure(KNOWHERE_UNSUPPORTED, "unsupported vector dtype");
    }
}

void
CheckBuffer(const void* data, uint64_t bytes, uint64_t needed, uint64_t alignment) {
    Require(bytes >= needed, "buffer capacity is too small");
    Require(needed == 0 || data != nullptr, "nonempty buffer has a null address");
    Require(data == nullptr || reinterpret_cast<uintptr_t>(data) % alignment == 0, "buffer address is misaligned");
}

uint64_t
CheckVectors(const knowhere_vectors* vectors) {
    Require(vectors != nullptr, "vectors must not be null");
    Require(vectors->rows >= 0 && vectors->dimensions > 0, "rows must be nonnegative and dimensions must be positive");
    const uint64_t element = ElementBytes(vectors->dtype);
    Require(vectors->dtype != KNOWHERE_BIN1 || vectors->dimensions % 8 == 0,
            "binary vector dimensions must be divisible by eight");
    const uint64_t row_bytes =
        vectors->dtype == KNOWHERE_BIN1 ? vectors->dimensions / 8 : Multiply(vectors->dimensions, element);
    const uint64_t bytes = Multiply(vectors->rows, row_bytes);
    CheckBuffer(vectors->data, vectors->bytes, bytes, element);
    return bytes;
}

knowhere::Json
Parameters(const char* parameters) {
    auto json = parameters == nullptr ? knowhere::Json::object() : knowhere::Json::parse(parameters);
    Require(json.is_object(), "parameters must be a JSON object");
    return json;
}

uint64_t
SearchParameters(knowhere::Json& json, int64_t rows, const knowhere_search_result* result) {
    Require(result != nullptr && result->top_k > 0, "search result must specify a positive top_k");
    if (json.contains(knowhere::meta::TOPK)) {
        const auto& k = json.at(knowhere::meta::TOPK);
        Require(k.is_number_integer() && k == result->top_k, "JSON k conflicts with result.top_k");
    }
    json[knowhere::meta::TOPK] = result->top_k;
    const uint64_t count = Multiply(rows, result->top_k);
    CheckBuffer(result->ids, result->ids_bytes, Multiply(count, sizeof(int64_t)), alignof(int64_t));
    CheckBuffer(result->distances, result->distances_bytes, Multiply(count, sizeof(float)), alignof(float));
    return count;
}

knowhere::BitsetView
Filter(const knowhere_bitset* excluded, int64_t rows) {
    if (excluded == nullptr) {
        return knowhere::BitsetView();
    }
    Require(excluded->bits >= 0, "filter bit count must be nonnegative");
    if (excluded->bits == 0) {
        return knowhere::BitsetView();
    }
    Require(excluded->bits >= rows, "filter must cover every base row");
    const uint64_t bytes = (static_cast<uint64_t>(excluded->bits) + 7) / 8;
    CheckBuffer(excluded->data, excluded->bytes, bytes, 1);
    // Limit the view to base rows so trailing bits never affect filtered counts.
    return knowhere::BitsetView(excluded->data, static_cast<size_t>(rows));
}

template <typename T>
knowhere::expected<knowhere::Index<knowhere::IndexNode>>
Create(const char* type, int32_t version) {
    if (std::strcmp(type, knowhere::IndexEnum::INDEX_DISKANN) == 0 ||
        std::strcmp(type, knowhere::IndexEnum::INDEX_CARDINAL_TIERED) == 0) {
        std::shared_ptr<milvus::FileManager> manager = std::make_shared<milvus::LocalFileManager>();
        auto pack = knowhere::Pack(manager);
        return knowhere::IndexFactory::Instance().Create<T>(type, version, pack);
    }
    return knowhere::IndexFactory::Instance().Create<T>(type, version);
}

knowhere::expected<knowhere::Index<knowhere::IndexNode>>
CreateTyped(const char* type, int32_t dtype, int32_t version) {
    switch (dtype) {
        case KNOWHERE_FP32:
            return Create<knowhere::fp32>(type, version);
        case KNOWHERE_BIN1:
            return Create<knowhere::bin1>(type, version);
        case KNOWHERE_FP16:
            return Create<knowhere::fp16>(type, version);
        case KNOWHERE_BF16:
            return Create<knowhere::bf16>(type, version);
        case KNOWHERE_INT8:
            return Create<knowhere::int8>(type, version);
        default:
            throw Failure(KNOWHERE_UNSUPPORTED, "unsupported vector dtype");
    }
}

void
Ready(const IndexResource& resource) {
    Require(resource.state == IndexResource::State::Ready, "index is not initialized; build or deserialize first");
}

knowhere::BinaryPtr
Blob(BinaryResource& resource, const char* name) {
    Require(name != nullptr && name[0] != '\0', "binary entry name must not be empty");
    auto binary = resource.data.GetByName(name);
    Require(binary != nullptr, "binary entry does not exist");
    Require(binary->size >= 0, "binary entry has a negative length");
    return binary;
}
}  // namespace

extern "C" {
const char*
knowhere_last_error(void) {
    return last_error;
}

int32_t
knowhere_c_abi_version(void) {
    return 1;
}

int32_t
knowhere_index_version_minimum(void) {
    return knowhere::Version::GetMinimalVersion().VersionNumber();
}

int32_t
knowhere_index_version_current(void) {
    return knowhere::Version::GetCurrentVersion().VersionNumber();
}

int32_t
knowhere_index_version_maximum(void) {
    return knowhere::Version::GetMaximumVersion().VersionNumber();
}

int32_t
knowhere_index_create(const char* type, int32_t dtype, int32_t version, knowhere_index_handle* output) {
    if (output != nullptr) {
        *output = 0;
    }
    return Boundary([&] {
        Require(output != nullptr, "output handle must not be null");
        Require(type != nullptr && type[0] != '\0', "index type must not be empty");
        ElementBytes(dtype);
        if (!knowhere::Version::VersionSupport(knowhere::Version(version))) {
            throw Failure(KNOWHERE_UNSUPPORTED, "unsupported index version");
        }
        auto created = CreateTyped(type, dtype, version);
        if (!created.has_value()) {
            throw Failure(created.error() == knowhere::Status::invalid_index_error ? KNOWHERE_UNSUPPORTED
                                                                                   : StatusCode(created.error()),
                          knowhere::Status2String(created.error()) + ": " + created.what());
        }
        Require(created.value().Node() != nullptr, "index factory returned a null index");
        const bool file_build = std::strcmp(type, knowhere::IndexEnum::INDEX_DISKANN) == 0;
        *output = Register(indexes, std::make_shared<IndexResource>(dtype, file_build, std::move(created.value())), 1);
    });
}

int32_t
knowhere_index_destroy(knowhere_index_handle handle) {
    return Boundary([&] { Close(indexes, handle, 1); });
}

int32_t
knowhere_index_build(knowhere_index_handle handle, const knowhere_vectors* vectors, const char* parameters) {
    return Boundary([&] {
        auto resource = Get(indexes, handle, 1);
        std::lock_guard<std::mutex> lock(resource->mutex);
        Require(resource->state == IndexResource::State::Empty, "index initialization was already attempted");
        if (resource->file_build) {
            Require(vectors == nullptr, "DISKANN builds from JSON data_path; vectors must be null");
            auto json = Parameters(parameters);
            resource->state = IndexResource::State::Failed;
            CheckStatus(resource->index.Build(nullptr, json));
            resource->state = IndexResource::State::Ready;
            return;
        }
        const uint64_t bytes = CheckVectors(vectors);
        Require(vectors->dtype == resource->dtype, "build dtype differs from index dtype");
        auto json = Parameters(parameters);
        resource->build_data.reset(new uint8_t[bytes]);
        if (bytes != 0) {
            std::memcpy(resource->build_data.get(), vectors->data, bytes);
        }
        auto dataset = knowhere::GenDataSet(vectors->rows, vectors->dimensions, resource->build_data.get());
        resource->state = IndexResource::State::Failed;
        CheckStatus(resource->index.Build(dataset, json));
        resource->state = IndexResource::State::Ready;
    });
}

int32_t
knowhere_index_search(knowhere_index_handle handle, const knowhere_vectors* queries, const knowhere_bitset* excluded,
                      knowhere_search_result* result, const char* parameters) {
    return Boundary([&] {
        auto resource = Get(indexes, handle, 1);
        std::lock_guard<std::mutex> lock(resource->mutex);
        Ready(*resource);
        CheckVectors(queries);
        Require(queries->dtype == resource->dtype, "query dtype differs from index dtype");
        Require(queries->dimensions == resource->index.Dim(), "query dimensions differ from index dimensions");
        auto json = Parameters(parameters);
        const uint64_t count = SearchParameters(json, queries->rows, result);
        auto filter = Filter(excluded, resource->index.Count());
        if (queries->rows == 0) {
            return;
        }
        auto found = resource->index.Search(knowhere::GenDataSet(queries->rows, queries->dimensions, queries->data),
                                            json, filter);
        if (!found.has_value()) {
            CheckStatus(found.error(), found.what());
        }
        Require(found.value() != nullptr && found.value()->GetRows() == queries->rows &&
                    found.value()->GetDim() == result->top_k,
                "index returned an unexpected result shape");
        Require(found.value()->GetIds() != nullptr && found.value()->GetDistance() != nullptr,
                "index returned null search buffers");
        std::memcpy(result->ids, found.value()->GetIds(), count * sizeof(int64_t));
        std::memcpy(result->distances, found.value()->GetDistance(), count * sizeof(float));
    });
}

int32_t
knowhere_index_info(knowhere_index_handle handle, int64_t* rows, int64_t* dimensions) {
    return Boundary([&] {
        auto resource = Get(indexes, handle, 1);
        std::lock_guard<std::mutex> lock(resource->mutex);
        Require(rows != nullptr && dimensions != nullptr, "info outputs must not be null");
        Ready(*resource);
        *rows = resource->index.Count();
        *dimensions = resource->index.Dim();
    });
}

int32_t
knowhere_index_serialize(knowhere_index_handle handle, knowhere_binary_set_handle* output) {
    if (output != nullptr) {
        *output = 0;
    }
    return Boundary([&] {
        Require(output != nullptr, "output handle must not be null");
        auto resource = Get(indexes, handle, 1);
        std::lock_guard<std::mutex> lock(resource->mutex);
        Ready(*resource);
        auto serialized = std::make_shared<BinaryResource>();
        CheckStatus(resource->index.Serialize(serialized->data));
        *output = Register(binary_sets, serialized, 0);
    });
}

int32_t
knowhere_index_deserialize(knowhere_index_handle handle, knowhere_binary_set_handle data, const char* parameters) {
    return Boundary([&] {
        auto resource = Get(indexes, handle, 1);
        auto binary = Get(binary_sets, data, 0);
        std::lock_guard<std::mutex> index_lock(resource->mutex);
        Require(resource->state == IndexResource::State::Empty, "index initialization was already attempted");
        auto json = Parameters(parameters);
        knowhere::BinarySet copy;
        {
            std::lock_guard<std::mutex> binary_lock(binary->mutex);
            for (const auto& entry : binary->data.binary_map_) {
                const auto& blob = entry.second;
                Require(blob != nullptr && blob->size >= 0, "invalid serialized binary entry");
                const uint64_t size = Multiply(blob->size, 1);
                Require(size == 0 || blob->data != nullptr, "serialized binary entry has no data");
                std::shared_ptr<uint8_t[]> bytes(new uint8_t[size]);
                if (size != 0) {
                    std::memcpy(bytes.get(), blob->data.get(), size);
                }
                copy.Append(entry.first, bytes, size);
            }
        }
        resource->deserialized_data = std::move(copy);
        resource->state = IndexResource::State::Failed;
        CheckStatus(resource->index.Deserialize(resource->deserialized_data, json));
        Require(resource->index.Dim() > 0 && resource->index.Count() >= 0, "deserialized index has invalid shape");
        resource->state = IndexResource::State::Ready;
    });
}

int32_t
knowhere_bruteforce(const knowhere_vectors* base, const knowhere_vectors* queries, const knowhere_bitset* excluded,
                    knowhere_search_result* result, const char* parameters) {
    return Boundary([&] {
        CheckVectors(base);
        CheckVectors(queries);
        Require(base->dtype == queries->dtype, "query dtype differs from base dtype");
        Require(base->dimensions == queries->dimensions, "query dimensions differ from base dimensions");
        auto json = Parameters(parameters);
        SearchParameters(json, queries->rows, result);
        auto filter = Filter(excluded, base->rows);
        if (queries->rows == 0) {
            return;
        }
        auto base_dataset = knowhere::GenDataSet(base->rows, base->dimensions, base->data);
        auto query_dataset = knowhere::GenDataSet(queries->rows, queries->dimensions, queries->data);
#define KNOWHERE_BF_CASE(dtype, type)                                                                   \
    case dtype:                                                                                         \
        CheckStatus(knowhere::BruteForce::SearchWithBuf<type>(base_dataset, query_dataset, result->ids, \
                                                              result->distances, json, filter));        \
        break
        switch (base->dtype) {
            KNOWHERE_BF_CASE(KNOWHERE_FP32, knowhere::fp32);
            KNOWHERE_BF_CASE(KNOWHERE_BIN1, knowhere::bin1);
            KNOWHERE_BF_CASE(KNOWHERE_FP16, knowhere::fp16);
            KNOWHERE_BF_CASE(KNOWHERE_BF16, knowhere::bf16);
            KNOWHERE_BF_CASE(KNOWHERE_INT8, knowhere::int8);
        }
#undef KNOWHERE_BF_CASE
    });
}

int32_t
knowhere_binary_set_create(knowhere_binary_set_handle* output) {
    if (output != nullptr) {
        *output = 0;
    }
    return Boundary([&] {
        Require(output != nullptr, "output handle must not be null");
        *output = Register(binary_sets, std::make_shared<BinaryResource>(), 0);
    });
}

int32_t
knowhere_binary_set_destroy(knowhere_binary_set_handle handle) {
    return Boundary([&] { Close(binary_sets, handle, 0); });
}

int32_t
knowhere_binary_set_count(knowhere_binary_set_handle handle, uint64_t* output) {
    return Boundary([&] {
        auto resource = Get(binary_sets, handle, 0);
        std::lock_guard<std::mutex> lock(resource->mutex);
        Require(output != nullptr, "count output must not be null");
        *output = resource->data.binary_map_.size();
    });
}

int32_t
knowhere_binary_set_name(knowhere_binary_set_handle handle, uint64_t index, char* output, uint64_t capacity,
                         uint64_t* required) {
    return Boundary([&] {
        auto resource = Get(binary_sets, handle, 0);
        std::lock_guard<std::mutex> lock(resource->mutex);
        Require(required != nullptr, "required output must not be null");
        Require(index < resource->data.binary_map_.size(), "binary entry index is out of range");
        auto it = resource->data.binary_map_.begin();
        std::advance(it, index);
        *required = it->first.size() + 1;
        if (output == nullptr && capacity == 0) {
            return;
        }
        CheckBuffer(output, capacity, *required, 1);
        std::memcpy(output, it->first.c_str(), *required);
    });
}

int32_t
knowhere_binary_set_length(knowhere_binary_set_handle handle, const char* name, uint64_t* output) {
    return Boundary([&] {
        auto resource = Get(binary_sets, handle, 0);
        std::lock_guard<std::mutex> lock(resource->mutex);
        Require(output != nullptr, "length output must not be null");
        *output = Blob(*resource, name)->size;
    });
}

int32_t
knowhere_binary_set_allocate(knowhere_binary_set_handle handle, const char* name, uint64_t length) {
    return Boundary([&] {
        auto resource = Get(binary_sets, handle, 0);
        std::lock_guard<std::mutex> lock(resource->mutex);
        Require(name != nullptr && name[0] != '\0', "binary entry name must not be empty");
        Multiply(length, 1);
        std::shared_ptr<uint8_t[]> data(new uint8_t[length]());
        resource->data.Append(name, data, length);
    });
}

int32_t
knowhere_binary_set_write(knowhere_binary_set_handle handle, const char* name, uint64_t offset, const void* data,
                          uint64_t length) {
    return Boundary([&] {
        auto resource = Get(binary_sets, handle, 0);
        std::lock_guard<std::mutex> lock(resource->mutex);
        auto blob = Blob(*resource, name);
        Require(offset <= static_cast<uint64_t>(blob->size) && length <= static_cast<uint64_t>(blob->size) - offset,
                "binary write range exceeds entry length");
        CheckBuffer(data, length, length, 1);
        if (length != 0) {
            std::memmove(blob->data.get() + offset, data, length);
        }
    });
}

int32_t
knowhere_binary_set_read(knowhere_binary_set_handle handle, const char* name, uint64_t offset, void* data,
                         uint64_t length) {
    return Boundary([&] {
        auto resource = Get(binary_sets, handle, 0);
        std::lock_guard<std::mutex> lock(resource->mutex);
        auto blob = Blob(*resource, name);
        Require(offset <= static_cast<uint64_t>(blob->size) && length <= static_cast<uint64_t>(blob->size) - offset,
                "binary read range exceeds entry length");
        CheckBuffer(data, length, length, 1);
        if (length != 0) {
            std::memmove(data, blob->data.get() + offset, length);
        }
    });
}
}  // extern "C"
