---
name: building
description: Use when building knowhere from source, configuring build options (CPU/GPU/DISKANN/ASAN), or troubleshooting compilation errors
---

# Building Knowhere

## Prerequisites

Run commands from the repository root. Reuse an existing compiler and Conan environment when available.
For initial setup, [scripts/install_deps.sh](../../../scripts/install_deps.sh) is the maintained source for
platform dependencies and tool versions, including CMake, ccache, and Conan. The script also configures the
default Conan profile and remote; review it before using it in an existing environment.

Build options and output paths are defined in [Makefile](../../../Makefile) and
[conanfile.py](../../../conanfile.py). Use `make help` to inspect the available targets and flags.

## Build Commands

| Build Type | Command |
|------------|---------|
| CPU Release | `make` |
| CPU Debug | `make WITH_DEBUG=True` |
| CPU + UT | `make WITH_UT=True` |
| CPU + UT + ASAN | `make WITH_UT=True WITH_ASAN=True` |
| Cardinal GPU | `make WITH_CARDINAL=True WITH_GPU=True` |
| GPU (cuVS) | `make WITH_GPU=True` |
| GPU (cuVS) + UT | `make WITH_GPU=True WITH_UT=True` |
| macOS | `make` (auto-detects libc++) |

Build outputs default to `build/Release/` or `build/Debug/`. Run tests after building with `WITH_UT=True`:

```bash
make test

# Match the build configuration and any custom output directory
make test WITH_DEBUG=True
make test BUILD_DIR=/path/to/build
```

## Build Flags

| Flag | Description |
|------|-------------|
| `WITH_GPU=True` | Enable cuVS, or Cardinal GPU when `WITH_CARDINAL=True` |
| `WITH_CARDINAL=True` | Enable Cardinal |
| `WITH_UT=True` | Enable unit tests |
| `WITH_BENCHMARK=True` | Enable benchmarks |
| `WITH_ASAN=True` | Enable AddressSanitizer |
| `WITH_DISKANN=False` | Disable DiskANN (enabled by default by Makefile on Linux) |
| `WITH_SVS=True` | Enable SVS on x86 |
| `WITH_DEBUG=True` | Debug build (default: Release) |
| `BUILD_DIR=<path>` | Set the output root; binaries remain under its Release/Debug subdirectory |
| `CONAN_PROFILE=<p>` | Use one custom Conan profile for host and build |
| `LIBCXX=<lib>` | Override compiler.libcxx (auto-detected from OS) |

Most enable flags use Make's `ifdef`: leave them unset to disable them; `WITH_UT=False`, for example, still
enables tests. `WITH_DISKANN` explicitly accepts true/false values.

## Common Issues

- **libstdc++ errors**: Override with `make LIBCXX=libstdc++11`
- **macOS**: `libc++` is auto-detected; OpenMP uses Homebrew's libomp
- **CMake 4.x**: Makefile already exports `CMAKE_POLICY_VERSION_MINIMUM=3.5` for old dependency recipes
- **ASAN failures**: Check for memory leaks, use-after-free, buffer overflows

---

If the build changes a feature or interface, update the corresponding repository documentation.
