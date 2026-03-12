<p>
    <img src="static/knowhere-logo.png" alt="Knowhere Logo"/>
</p>

This document will help you to build the Knowhere repository from source code and to run unit tests. Please [file an issue](https://github.com/zilliztech/knowhere/issues/new) if there's a problem.

## Introduction

Knowhere is written in C++. It is an independent project that act as Milvus's internal core.

## Building Knowhere Within Milvus

If you wish to only use Knowhere within Milvus without changing any of the Knowhere source code, we suggest that you move to the [Milvus main project](https://github.com/milvus-io/milvus) and build Milvus directly, where Knowhere is then built implicitly during Milvus build.

## System Requirements

All Linux distributions are available for Knowhere development. However, a majority of our contributor worked with Ubuntu or CentOS systems, with a small portion of Mac (both x86_64 and Apple Silicon) contributors. If you would like Knowhere to build and run on other distributions, you are more than welcome to file an issue and contribute!

Here's a list of verified OS types where Knowhere can successfully build and run:

- Ubuntu 22.04 x86_64
- Ubuntu 22.04 Aarch64
- Ubuntu 20.04 x86_64 / Aarch64 (EOL by April 2025 and kept for legacy reasons; see `scripts/install_deps.sh` for details)
- MacOS (x86_64)
- MacOS (Apple Silicon)

## Building Knowhere From Source Code

#### Install Dependencies

`scripts/install_deps.sh` install all dependencies for building, testing, and shipping the knowhere library. If you don't need the full pipeline, please refer to the file and modify it for a more fine-grained dependency control.

```bash
$ bash scripts/install_deps.sh
```

#### Build From Source Code

A top-level `Makefile` provides a unified build interface. Run `make help` to see all targets and flags.

```bash
# CPU release (default)
$ make

# GPU release (cuVS)
$ make WITH_GPU=True

# CPU with unit tests
$ make WITH_UT=True

# CPU UT + AddressSanitizer
$ make WITH_UT=True WITH_ASAN=True

# GPU with unit tests
$ make WITH_GPU=True WITH_UT=True

# Debug build
$ make WITH_DEBUG=True

# Custom compiler via Conan profile (e.g. clang, gcc-15)
$ make CONAN_PROFILE=clang14
```

#### Running Unit Tests

```bash
# requires a prior build with WITH_UT=True
$ make test
```

#### Clean up

```bash
$ make clean
```

## Python Wheel

Building the Python wheel requires `swig` and Python development headers (`python3-dev` on Ubuntu). These are installed automatically by `scripts/install_deps.sh`.

After building Knowhere with a Release configuration:

```bash
# Build portable manylinux wheel
$ make wheel

# Install
$ pip3 install python/dist/pyknowhere-*-manylinux*.whl
```

For more options (clean build, verbose, custom Python binary), see `python/build_portable_wheel.sh -h`.

## Contributing

### Pre-Commit

Before submitting a pull request, run pre-commit checks locally:

```bash
pip3 install pre-commit
pre-commit install --hook-type pre-commit --hook-type pre-push

# Run all pre-commit hooks
$ make pre-commit

# Or run individually:
$ make format   # clang-format
$ make lint     # clang-tidy (requires a prior build)
```

If clang-format and clang-tidy are not already installed:

```bash
# linux
apt install clang-format clang-tidy
# mac
brew install llvm
ln -s "$(brew --prefix llvm)/bin/clang-format" "/usr/local/bin/clang-format"
ln -s "$(brew --prefix llvm)/bin/clang-tidy" "/usr/local/bin/clang-tidy"
```
