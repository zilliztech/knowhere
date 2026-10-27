#include "catch2/catch_test_macros.hpp"
#include "common/EasyAssert.h"
#include "folly/futures/Future.h"
#include "knowhere/expected.h"
#include "knowhere/segcore_error_code.h"

TEST_CASE("GuardedCall preserves typed cancellation", "[cancellation]") {
    auto check = [](const auto& error, knowhere::Status expected_status) {
        auto result = knowhere::GuardedCall([&]() -> knowhere::expected<int> { throw error; });
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error() == expected_status);
        auto status = knowhere::GuardedCall([&]() -> knowhere::Status { throw error; });
        REQUIRE(status == expected_status);
    };
    check(folly::FutureCancellation{}, knowhere::Status::cancelled);
    check(milvus::SegcoreError(milvus::ErrorCode::FollyCancel, "storage stopped"), knowhere::Status::cancelled);
    check(milvus::SegcoreError(milvus::ErrorCode::UnexpectedError, "cancelled"),
          knowhere::Status::knowhere_inner_error);
    check(std::runtime_error("cancelled"), knowhere::Status::knowhere_inner_error);
    REQUIRE(knowhere::ToSegcoreErrorCode(knowhere::Status::cancelled) == milvus::ErrorCode::FollyCancel);
}
