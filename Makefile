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
# BUILD_DIR is where conan + cmake put build outputs. Overridable from the
# command line, e.g. `make BUILD_DIR=$(pwd)/build_asan WITH_ASAN=True`.
# Binaries end up at $(BUILD_DIR)/$(BUILD_TYPE)/... (e.g. build/Release/...).
BUILD_DIR ?= $(PWD)/build
SHELL := /bin/bash

# Conan install-only flags (not valid for conan build)
CONAN_INSTALL_FLAGS := --update --build=missing

# ---------- User-facing build flags ----------
# Usage: make WITH_GPU=True WITH_UT=True WITH_ASAN=True WITH_DEBUG=True
WITH_GPU ?=
WITH_UT ?=
WITH_BENCHMARK ?=
WITH_ASAN ?=
WITH_SVS ?=
WITH_CARDINAL ?=
CARDINAL_VERSION_FORCE_CHECKOUT ?=
WITH_DEBUG ?=
CONAN_PROFILE ?=

# Prevent build-flag variables from leaking into the environment of child
# processes (conan / cmake).  Without this, GNU Make exports command-line
# variables such as WITH_ASAN to every sub-process, which causes the custom
# folly recipe to pick up $ENV{WITH_ASAN} and compile folly itself with
# -fsanitize=address — breaking the build on GCC.
unexport WITH_GPU WITH_UT WITH_BENCHMARK WITH_ASAN WITH_CARDINAL CARDINAL_VERSION_FORCE_CHECKOUT WITH_DEBUG

# ---------- Derived settings ----------
ifdef WITH_DEBUG
    BUILD_TYPE := Debug
else
    BUILD_TYPE := Release
endif

# Auto-detect OS for compiler.libcxx; override with LIBCXX=<value>.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LIBCXX ?= libc++
else
    LIBCXX ?= libstdc++11
endif

# ---------- Compose conan flags from user flags ----------
# Settings/options flags shared by conan install and conan build
# -s:b compiler.cppstd=20: build-context tools (e.g. grpc_cpp_plugin) must also
# use C++20, because abseil's installed headers hardcode ABSL_OPTION_USE_STD_ORDERING=1
# which requires std::partial_ordering from <compare> (a C++20 feature).
CONAN_SETTINGS := -s compiler.libcxx=$(LIBCXX) -s build_type=$(BUILD_TYPE) -s compiler.cppstd=20 -s:b compiler.cppstd=20

# DiskANN and liburing require libaio (Linux-only).
ifneq ($(UNAME_S),Darwin)
    CONAN_SETTINGS += -o \&:with_diskann=True
    ifndef WITH_GPU
        CONAN_INSTALL_FLAGS += --build=liburing
    endif
endif

# GPU builds use cuVS.
ifdef WITH_GPU
    CONAN_SETTINGS += -o \&:with_cuvs=True
endif

ifdef WITH_UT
    CONAN_SETTINGS += -o \&:with_ut=True
endif

ifdef WITH_BENCHMARK
    CONAN_SETTINGS += -o \&:with_benchmark=True
endif

ifdef WITH_ASAN
    CONAN_SETTINGS += -o \&:with_asan=True
endif

ifdef WITH_SVS
    CONAN_FLAGS += -o \&:with_svs=True
endif

ifdef WITH_CARDINAL
    CONAN_SETTINGS += -o \&:with_cardinal=True
endif

ifneq ($(CARDINAL_VERSION_FORCE_CHECKOUT),)
    ifneq ($(filter True true ON on 1,$(CARDINAL_VERSION_FORCE_CHECKOUT)),)
        CONAN_SETTINGS += -o \&:cardinal_version_force_checkout=True
    else ifneq ($(filter False false OFF off 0,$(CARDINAL_VERSION_FORCE_CHECKOUT)),)
        CONAN_SETTINGS += -o \&:cardinal_version_force_checkout=False
    else
        $(error CARDINAL_VERSION_FORCE_CHECKOUT must be True/False, ON/OFF, or 1/0)
    endif
endif

ifdef CONAN_PROFILE
    CONAN_SETTINGS += -pr $(CONAN_PROFILE)
endif

.PHONY: build test \
	lint format pre-commit \
	wheel codecov \
	clean help

all: build ## Default: CPU release build

# ---------- Build ----------

build: ## Build knowhere (use WITH_GPU=True, WITH_UT=True, WITH_BENCHMARK=True, WITH_ASAN=True)
ifdef WITH_GPU
	@$(PWD)/scripts/prepare_gpu_build.sh
endif
	@mkdir -p $(BUILD_DIR) && \
		conan install . -of $(BUILD_DIR) $(CONAN_INSTALL_FLAGS) $(CONAN_SETTINGS) && \
		conan build . -of $(BUILD_DIR) $(CONAN_SETTINGS)

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
	@echo "  WITH_BENCHMARK=True Enable benchmarks build"
	@echo "  WITH_ASAN=True     Enable AddressSanitizer"
	@echo "  WITH_SVS=True      Enable SVS (Intel Scalable Vector Search, x86 only)"
	@echo "  WITH_CARDINAL=True Enable Cardinal build"
	@echo "  CARDINAL_VERSION_FORCE_CHECKOUT=True Force Cardinal checkout to configured version"
	@echo "  WITH_DEBUG=True    Debug build (default: Release)"
	@echo "  CONAN_PROFILE=<p>  Use a custom Conan profile (e.g. clang, gcc-15)"
	@echo "  LIBCXX=<lib>       Override compiler.libcxx (auto-detected from OS)"
	@echo ""
	@echo "Examples:"
	@echo "  make                              # CPU release"
	@echo "  make WITH_GPU=True                # GPU release"
	@echo "  make WITH_UT=True WITH_ASAN=True  # CPU UT + ASAN"
	@echo "  make WITH_GPU=True WITH_UT=True   # GPU UT"
	@echo "  make WITH_DEBUG=True WITH_UT=True # CPU debug + UT"
	@echo "  make LIBCXX=libc++                # override compiler.libcxx"
	@echo "  make CONAN_PROFILE=gcc15          # CPU with custom profile"
	@echo ""
