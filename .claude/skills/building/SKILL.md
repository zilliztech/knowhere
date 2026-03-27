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
export PATH=$PATH:$HOME/.local/bin
```

## Build Commands

```bash
conan remote add default-conan-local2 https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local2```

| Build Type | Command |
|------------|---------|
| CPU Release | `conan install . --build=missing -o with_ut=True -s compiler.libcxx=libstdc++11 -s compiler.cppstd=17 -s build_type=Release` |
| CPU Debug | `conan install . --build=missing -o with_ut=True -s compiler.libcxx=libstdc++11 -s compiler.cppstd=17 -s build_type=Debug` |
| GPU (CUVS) | `conan install . --build=missing -o with_ut=True -o with_cuvs=True -s compiler.libcxx=libstdc++11 -s compiler.cppstd=17 -s build_type=Release` |
| DISKANN | `conan install . --build=missing -o with_ut=True -o with_diskann=True -s compiler.libcxx=libstdc++11 -s compiler.cppstd=17 -s build_type=Release` |
| ASAN (CI default) | `conan install . --build=missing -o with_ut=True -o with_diskann=True -o with_asan=True -s compiler.libcxx=libstdc++11 -s compiler.cppstd=17 -s build_type=Release` |
| macOS | `conan install . --build=missing -o with_ut=True -s compiler.libcxx=libc++ -s compiler.cppstd=17 -s build_type=Release` |

Then run: `conan build .`

## Build Options

| Option | Description |
|--------|-------------|
| `with_ut` | Enable unit tests |
| `with_diskann` | Enable DISKANN index support |
| `with_cuvs` | Enable GPU (CUVS) support |
| `with_asan` | Enable AddressSanitizer for memory error detection |

## Common Issues

- **libstdc++ errors**: Ensure `-s compiler.libcxx=libstdc++11` matches your system
- **Missing dependencies**: Run `conan install` with `--build=missing`
- **macOS**: Use `libc++` instead of `libstdc++11`
- **ASAN failures**: Check for memory leaks, use-after-free, buffer overflows

---

> If this build relates to new features or interface changes, consider updating documentation. See `documenting` skill.
