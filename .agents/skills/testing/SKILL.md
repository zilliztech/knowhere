---
name: testing
description: Use when running unit tests, debugging test failures, or adding new test cases using Catch2 framework
---

# Testing Knowhere

## Running Tests

Run commands from the repository root after building with `WITH_UT=True`; see the
[building skill](../building/SKILL.md). The examples use the default `BUILD_DIR=build`.

```bash
# Run all tests for the configuration that was built
make test
make test WITH_DEBUG=True

# Run specific test by name pattern
./build/Release/tests/ut/knowhere_tests "[float metrics]"
./build/Release/tests/ut/knowhere_tests "Test Mem Index*"

# List all test names
./build/Release/tests/ut/knowhere_tests --list-tests
```

For Debug builds, use `build/Debug/tests/ut/knowhere_tests`. For a custom output root, pass the same
`BUILD_DIR` to `make test` or adjust the executable path. Available tests depend on the build options;
check `--list-tests` before selecting a filter.

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
| `test_search.cc` | Index build/search with float and binary vectors |
| `test_distances.cc` | Distance calculations |
| `test_sparse.cc` | Sparse vector index |
| `test_diskann.cc` | DiskANN (requires a build with DiskANN enabled) |

## Writing New Tests

```cpp
#include "catch2/catch_test_macros.hpp"
#include "knowhere/index/index_factory.h"
#include "knowhere/version.h"

TEST_CASE("Create a flat index", "[flat]") {
    const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto index = knowhere::IndexFactory::Instance().Create<knowhere::fp32>("FLAT", version);
    REQUIRE(index.has_value());
}
```

Follow nearby tests for setup and use `SECTION` for related cases. New regression tests should assert the
behavior being changed, including relevant error cases, rather than only successful index creation.

---

If tests introduce a new feature or interface, update the corresponding repository documentation.
