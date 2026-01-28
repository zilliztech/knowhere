---
name: testing
description: Use when running unit tests, debugging test failures, or adding new test cases using Catch2 framework
---

# Testing Knowhere

## Running Tests

```bash
# Run all tests
./Release/tests/ut/knowhere_tests
./Debug/tests/ut/knowhere_tests

# Run specific test by name pattern
./Release/tests/ut/knowhere_tests "[float metrics]"
./Release/tests/ut/knowhere_tests "Test Mem Index*"

# List all test names
./Release/tests/ut/knowhere_tests --list-tests
```

## Catch2 Quick Reference

| Command | Description |
|---------|-------------|
| `[tag]` | Run tests with specific tag |
| `"Test Name*"` | Run tests matching pattern |
| `--list-tests` | List all available tests |
| `--list-tags` | List all available tags |
| `-s` | Show successful test output |
| `-d yes` | Show test durations |

## Test Location

Tests are located in `tests/ut/`. Key test files:

| File | Tests |
|------|-------|
| `test_index.cc` | Index build/search operations |
| `test_float_metrics.cc` | Float vector metrics (L2, IP, COSINE) |
| `test_binary_metrics.cc` | Binary vector metrics (Hamming, Jaccard) |
| `test_sparse.cc` | Sparse vector index |

## Writing New Tests

```cpp
#include "catch2/catch_test_macros.hpp"
#include "knowhere/index/index_factory.h"

TEST_CASE("Test Description", "[tag1][tag2]") {
    // Setup
    auto index = knowhere::IndexFactory::Instance().Create(...);

    // Test
    REQUIRE(index.has_value());

    // Sections for sub-tests
    SECTION("sub-test 1") {
        // ...
    }
}
```

---

> If adding tests for new features, consider updating documentation. See `documenting` skill.
