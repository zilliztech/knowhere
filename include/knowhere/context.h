// Copyright (C) 2019-2025 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#ifndef CONTEXT_H
#define CONTEXT_H

#include <folly/CancellationToken.h>
#include <folly/futures/Future.h>

#include "common/OpContext.h"

namespace knowhere {
inline void
checkCancellation(const milvus::OpContext* op_context) {
    if (op_context == nullptr) {
        return;
    }
    if (op_context->cancellation_token.isCancellationRequested()) {
        throw folly::FutureCancellation();
    }
}
}  // namespace knowhere

#endif /* CONTEXT_H */
