#include "catch2/catch_test_macros.hpp"
#include "knowhere/expected.h"

TEST_CASE("Cancellation status preserves the cross-branch ABI", "[cancellation]") {
    REQUIRE(static_cast<int>(knowhere::Status::cancelled) == 34);
    REQUIRE(static_cast<int>(knowhere::Status::aisaq_error) == 32);
    REQUIRE(knowhere::Status2String(knowhere::Status::cancelled) == "cancelled");
    auto result = knowhere::expected<int>::Err(knowhere::Status::cancelled, "storage stopped");
    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error() == knowhere::Status::cancelled);
}
