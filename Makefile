# Copyright (C) 2019-2023 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under the License.

PWD := $(shell pwd)
BUILD_DIR := $(PWD)/build
SHELL := /bin/bash

# Conan base flags (single source of truth for all pipelines)
CONAN_BASE_FLAGS := --update --build=missing -o with_diskann=True

# ---------- User-facing build flags ----------
# Usage: make WITH_GPU=True WITH_UT=True WITH_ASAN=True WITH_DEBUG=True
WITH_GPU ?=
WITH_UT ?=
WITH_ASAN ?=
WITH_CARDINAL ?=
WITH_DEBUG ?=
WITH_MACOS ?=

# ---------- Derived settings ----------
ifdef WITH_DEBUG
    BUILD_TYPE := Debug
else
    BUILD_TYPE := Release
endif

ifdef WITH_MACOS
    LIBCXX := libc++
else
    LIBCXX := libstdc++11
endif

# ---------- Compose conan flags from user flags ----------
CONAN_FLAGS := $(CONAN_BASE_FLAGS) -s compiler.libcxx=$(LIBCXX) -s build_type=$(BUILD_TYPE)

# GPU builds use cuVS; CPU builds need liburing built from source.
ifdef WITH_GPU
    CONAN_FLAGS += -o with_cuvs=True
else
    CONAN_FLAGS += --build=liburing
endif

CONAN_FLAGS += -s compiler.cppstd=17

ifdef WITH_UT
    CONAN_FLAGS += -o with_ut=True
endif

ifdef WITH_ASAN
    CONAN_FLAGS += -o with_asan=True
endif

ifdef WITH_CARDINAL
    CONAN_FLAGS += -o with_cardinal=True
endif

.PHONY: build test \
	lint format pre-commit \
	wheel codecov \
	clean help

all: build ## Default: CPU release build

# ---------- Build ----------

build: ## Build knowhere (use WITH_GPU=True, WITH_UT=True, WITH_ASAN=True)
ifdef WITH_GPU
	@$(PWD)/scripts/prepare_gpu_build.sh
endif
	@mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && \
		conan install .. $(CONAN_FLAGS) && \
		conan build ..

# ---------- Test ----------

test: ## Run unit tests (requires prior build with WITH_UT=True)
	@$(BUILD_DIR)/$(BUILD_TYPE)/tests/ut/knowhere_tests

# ---------- Code quality ----------

lint: ## Run clang-tidy (requires a prior build)
	@$(PWD)/scripts/prepare_clang_tidy.sh
	@find src -type f -name '*.cc' | xargs run-clang-tidy -quiet -p=$(BUILD_DIR)/$(BUILD_TYPE)

format: ## Run clang-format via pre-commit
	@pre-commit run clang-format --all-files

pre-commit: ## Run all pre-commit hooks
	@pre-commit run --all-files

# ---------- Python ----------

wheel: ## Build portable Python wheel (requires a prior build)
	@cd $(PWD)/python && ./build_portable_wheel.sh

# ---------- Coverage ----------

codecov: ## Generate code coverage report (requires a coverage build)
	@$(PWD)/scripts/run_codecov.sh

# ---------- Housekeeping ----------

clean: ## Remove build directory
	@rm -rf $(BUILD_DIR)

# ---------- Help ----------

help: ## Show available targets
	@echo "Knowhere build targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Build flags:"
	@echo "  WITH_GPU=True      Enable GPU (cuVS) build"
	@echo "  WITH_UT=True       Enable unit tests"
	@echo "  WITH_ASAN=True     Enable AddressSanitizer"
	@echo "  WITH_CARDINAL=True Enable Cardinal build"
	@echo "  WITH_DEBUG=True    Debug build (default: Release)"
	@echo "  WITH_MACOS=True    Use macOS conventions (libc++)"
	@echo ""
	@echo "Examples:"
	@echo "  make                              # CPU release"
	@echo "  make WITH_GPU=True                # GPU release"
	@echo "  make WITH_UT=True WITH_ASAN=True  # CPU UT + ASAN"
	@echo "  make WITH_GPU=True WITH_UT=True   # GPU UT"
	@echo "  make WITH_DEBUG=True WITH_UT=True # CPU debug + UT"
	@echo "  make WITH_MACOS=True              # macOS CPU release"
	@echo ""
