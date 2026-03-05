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
CONAN_BASE_FLAGS := --update --build=missing -s compiler.libcxx=libstdc++11 -o with_diskann=True
# Override with CONAN_CPPSTD= (empty) for legacy release pipelines that omit it
CONAN_CPPSTD ?= -s compiler.cppstd=17

# CPU builds need liburing built from source
CONAN_CPU_FLAGS := $(CONAN_BASE_FLAGS) $(CONAN_CPPSTD) --build=liburing
# GPU builds don't need liburing
CONAN_GPU_FLAGS := $(CONAN_BASE_FLAGS) $(CONAN_CPPSTD)

.PHONY: build build-release build-gpu build-ut \
	ut ut-gpu \
	lint format pre-commit \
	wheel codecov \
	clean help

all: build ## Default target: CPU debug build

# ---------- Build targets ----------

build: ## Build CPU (debug)
	@echo "Building Knowhere (CPU, debug)..."
	@mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && \
		conan install .. $(CONAN_CPU_FLAGS) && \
		conan build ..

build-release: ## Build CPU (release)
	@echo "Building Knowhere (CPU, release)..."
	@mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && \
		conan install .. $(CONAN_CPU_FLAGS) -s build_type=Release && \
		conan build ..

build-ut: ## Build CPU with unit tests enabled (no run)
	@echo "Building Knowhere (CPU, with unit tests)..."
	@mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && \
		conan install .. $(CONAN_CPU_FLAGS) -o with_ut=True && \
		conan build ..

build-gpu: ## Build GPU (release)
	@echo "Building Knowhere (GPU, release)..."
	@$(PWD)/scripts/prepare_gpu_build.sh
	@mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && \
		conan install .. $(CONAN_GPU_FLAGS) -o with_cuvs=True -s build_type=Release && \
		conan build ..

# ---------- Test targets ----------

ut: ## Build and run CPU unit tests (with ASAN)
	@echo "Building and running CPU unit tests..."
	@mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && \
		conan install .. $(CONAN_CPU_FLAGS) -s build_type=Release -o with_ut=True -o with_asan=True && \
		conan build .. && \
		./Release/tests/ut/knowhere_tests

ut-gpu: ## Build and run GPU unit tests
	@echo "Building and running GPU unit tests..."
	@$(PWD)/scripts/prepare_gpu_build.sh
	@mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && \
		conan install .. $(CONAN_GPU_FLAGS) -o with_cuvs=True -s build_type=Release -o with_ut=True && \
		conan build .. && \
		./Release/tests/ut/knowhere_tests

# ---------- Code quality ----------

lint: ## Run clang-tidy (requires a prior build)
	@echo "Running clang-tidy..."
	@$(PWD)/scripts/prepare_clang_tidy.sh
	@find src -type f -name '*.cc' | xargs run-clang-tidy -quiet -p=$(BUILD_DIR)/Release

format: ## Run clang-format via pre-commit
	@echo "Running clang-format..."
	@pre-commit run clang-format --all-files

pre-commit: ## Run all pre-commit hooks
	@echo "Running pre-commit hooks..."
	@pre-commit run --all-files

# ---------- Python ----------

wheel: ## Build portable Python wheel (requires a prior build)
	@echo "Building Python wheel..."
	@cd $(PWD)/python && ./build_portable_wheel.sh

# ---------- Coverage ----------

codecov: ## Generate code coverage report (requires a coverage build)
	@echo "Generating code coverage report..."
	@$(PWD)/scripts/run_codecov.sh

# ---------- Housekeeping ----------

clean: ## Remove build directory
	@echo "Cleaning up all generated files..."
	@rm -rf $(BUILD_DIR)

# ---------- Help ----------

help: ## Show available targets
	@echo "Knowhere build targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
