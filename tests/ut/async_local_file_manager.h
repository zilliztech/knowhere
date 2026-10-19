// Copyright (C) 2019-2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and limitations under the License.

#pragma once

#include <memory>
#include <string>

#include "filemanager/impl/LocalFileManager.h"
#include "filemanager/impl/LocalInputStream.h"

namespace knowhere::test {

// File-backed UT streams implement the async contract with inline completion.
class AsyncLocalInputStream final : public milvus::LocalInputStream {
 public:
    using milvus::LocalInputStream::LocalInputStream;

    [[nodiscard]] folly::SemiFuture<size_t>
    ReadAtAsync(void* buffer, size_t offset, size_t size) override {
        return folly::makeSemiFutureWith([this, buffer, offset, size] { return ReadAt(buffer, offset, size); });
    }
};

class AsyncLocalFileManager final : public milvus::LocalFileManager {
 public:
    std::shared_ptr<milvus::InputStream>
    OpenInputStream(const std::string& filename) override {
        return std::make_shared<AsyncLocalInputStream>(filename);
    }
};

}  // namespace knowhere::test
