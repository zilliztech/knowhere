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
CONAN_BASE_FLAGS := --update --build=missing -s compiler.libcxx=libstdc++11 -s build_type=Release -o with_diskann=True

# ---------- User-facing build flags ----------
# Usage: make WITH_GPU=True WITH_UT=True WITH_ASAN=True
WITH_GPU ?=
WITH_UT ?=
WITH_ASAN ?=
# Set False to disable compiler.cppstd=17 (legacy release pipelines)
WITH_CPPSTD ?= True

# ---------- Compose conan flags from user flags ----------
CONAN_FLAGS := $(CONAN_BASE_FLAGS) --build=liburing

ifdef WITH_GPU
    # GPU builds don't need liburing; add cuVS instead
    CONAN_FLAGS := $(CONAN_BASE_FLAGS) -o with_cuvs=True
endif

ifeq ($(WITH_CPPSTD),True)
    CONAN_FLAGS += -s compiler.cppstd=17
endif

ifdef WITH_UT
    CONAN_FLAGS += -o with_ut=True
endif

ifdef WITH_ASAN
    CONAN_FLAGS += -o with_asan=True
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
	@$(BUILD_DIR)/Release/tests/ut/knowhere_tests

# ---------- Code quality ----------

lint: ## Run clang-tidy (requires a prior build)
	@$(PWD)/scripts/prepare_clang_tidy.sh
	@find src -type f -name '*.cc' | xargs run-clang-tidy -quiet -p=$(BUILD_DIR)/Release

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
	@echo "  WITH_CPPSTD=False  Disable compiler.cppstd=17 (legacy release)"
	@echo ""
	@echo "Examples:"
	@echo "  make                              # CPU release"
	@echo "  make WITH_GPU=True                # GPU release"
	@echo "  make WITH_UT=True WITH_ASAN=True  # CPU UT + ASAN"
	@echo "  make WITH_GPU=True WITH_UT=True   # GPU UT"
	@echo ""
