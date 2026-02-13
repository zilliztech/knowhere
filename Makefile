.PHONY: conan-install build clean rebuild test

BUILD_TYPE ?= Release
BUILD_DIR  ?= build/$(BUILD_TYPE)
CPPSTD     ?= 17
JOBS       ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
CONAN_OPTS ?=

conan-install:
	conan install . --build=missing \
		-s build_type=$(BUILD_TYPE) \
		-s compiler.cppstd=$(CPPSTD) \
		$(CONAN_OPTS)

build: conan-install
	cmake --preset $(shell echo $(BUILD_TYPE) | tr A-Z a-z)
	cmake --build $(BUILD_DIR) --parallel $(JOBS)

test: build
	cd $(BUILD_DIR) && ctest --output-on-failure

clean:
	rm -rf build

rebuild: clean build
