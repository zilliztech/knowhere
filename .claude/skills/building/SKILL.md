---
name: building
description: Use when building knowhere from source, configuring build options (CPU/GPU/DISKANN/ASAN), or troubleshooting compilation errors
---

# Building Knowhere

## Prerequisites

```bash
# Ubuntu/Debian
sudo apt install build-essential libopenblas-openmp-dev libaio-dev python3-dev python3-pip
pip3 install conan==2.25.1 --user
conan profile detect --force
conan remote add default-conan-local2 https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local2
export PATH=$PATH:$HOME/.local/bin
```

## Build Commands

| Build Type | Command |
|------------|---------|
| CPU Release | `make` |
| CPU Debug | `make WITH_DEBUG=True` |
| CPU + UT | `make WITH_UT=True` |
| CPU + UT + ASAN | `make WITH_UT=True WITH_ASAN=True` |
| GPU (CUVS) | `make WITH_GPU=True` |
| GPU + UT | `make WITH_GPU=True WITH_UT=True` |
| macOS | `make` (auto-detects libc++) |

Run tests after building with `WITH_UT=True`:

```bash
make test
```

## Build Flags

| Flag | Description |
|------|-------------|
| `WITH_GPU=True` | Enable GPU (cuVS) build |
| `WITH_UT=True` | Enable unit tests |
| `WITH_ASAN=True` | Enable AddressSanitizer |
| `WITH_CARDINAL=True` | Enable Cardinal build |
| `WITH_DEBUG=True` | Debug build (default: Release) |
| `CONAN_PROFILE=<p>` | Use a custom Conan profile |
| `LIBCXX=<lib>` | Override compiler.libcxx (auto-detected from OS) |

## Common Issues

- **libstdc++ errors**: Override with `make LIBCXX=libstdc++11`
- **macOS**: `libc++` is auto-detected; OpenMP uses Homebrew's libomp
- **CMake 4.x**: Set `export CMAKE_POLICY_VERSION_MINIMUM=3.5` for old dependency recipes
- **ASAN failures**: Check for memory leaks, use-after-free, buffer overflows

---

> If this build relates to new features or interface changes, consider updating documentation. See `documenting` skill.
