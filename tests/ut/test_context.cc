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

#include <folly/CancellationToken.h>
#include <folly/futures/Future.h>

#include "catch2/catch_test_macros.hpp"
#include "knowhere/context.h"

TEST_CASE("Test checkCancellation", "[context]") {
    SECTION("checkCancellation with nullptr should not throw") {
        REQUIRE_NOTHROW(knowhere::checkCancellation(nullptr));
    }

    SECTION("checkCancellation with non-cancelled context should not throw") {
        folly::CancellationSource cs;
        milvus::OpContext op_context(cs.getToken());
        REQUIRE_NOTHROW(knowhere::checkCancellation(&op_context));
    }

    SECTION("checkCancellation with cancelled context should throw FutureCancellation") {
        folly::CancellationSource cs;
        milvus::OpContext op_context(cs.getToken());

        // Request cancellation
        cs.requestCancellation();

        // Verify cancellation is requested
        REQUIRE((op_context.cancellation_token.isCancellationRequested()));

        // checkCancellation should throw folly::FutureCancellation
        REQUIRE_THROWS_AS(knowhere::checkCancellation(&op_context), folly::FutureCancellation);
    }

    SECTION("checkCancellation before and after cancellation request") {
        folly::CancellationSource cs;
        milvus::OpContext op_context(cs.getToken());

        // Before cancellation - should not throw
        REQUIRE_NOTHROW(knowhere::checkCancellation(&op_context));
        REQUIRE((!op_context.cancellation_token.isCancellationRequested()));

        // Request cancellation
        cs.requestCancellation();

        // After cancellation - should throw
        REQUIRE((op_context.cancellation_token.isCancellationRequested()));
        REQUIRE_THROWS_AS(knowhere::checkCancellation(&op_context), folly::FutureCancellation);
    }
}
