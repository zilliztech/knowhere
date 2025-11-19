#!/bin/bash
# Portable Wheel Builder for PyKnowhere
# Builds manylinux-compatible wheels with bundled dependencies

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../build/Release"
PYTHON="${PYTHON:-python3}"

# Color output (optional)
if [ -t 1 ]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; NC=''
fi

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# Show usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
  -h, --help       Show this help
  -c, --clean      Clean build artifacts first
  -v, --verbose    Verbose output
  -p, --python BIN Python executable (default: python3)

Examples:
  $0                    # Build with defaults
  $0 -c -v              # Clean build with verbose output
  $0 -p python3.10      # Use specific Python version

EOF
    exit 0
}

# Parse arguments
CLEAN=false
VERBOSE=false

while [ $# -gt 0 ]; do
    case $1 in
        -h|--help) usage ;;
        -c|--clean) CLEAN=true; shift ;;
        -v|--verbose) VERBOSE=true; shift ;;
        -p|--python) PYTHON="$2"; shift 2 ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

# Check dependencies
check_deps() {
    log_info "Checking dependencies..."

    for cmd in readelf unzip patchelf; do
        if ! command -v $cmd >/dev/null; then
            log_error "$cmd not found"
            log_error "Install with: sudo apt install -y $cmd"
            exit 1
        fi
    done

    command -v auditwheel >/dev/null || {
        log_warn "auditwheel not found, installing..."
        pip3 install -q auditwheel
    }

    command -v "$PYTHON" >/dev/null || { log_error "Python not found: $PYTHON"; exit 1; }

    [ -f "$BUILD_DIR/libknowhere.so" ] || {
        log_error "libknowhere.so not found at $BUILD_DIR"
        log_error "Please build Knowhere C++ library first:"
        echo "  cd build && conan install .. && conan build .."
        exit 1
    }

    log_info "All dependencies OK"
}

# Clean build artifacts
clean_build() {
    log_info "Cleaning build artifacts..."
    cd "$SCRIPT_DIR"
    rm -rf build *.egg-info
    rm -f knowhere/knowhere_wrap.cpp knowhere/swigknowhere.py
    rm -f knowhere/libknowhere.so knowhere/libknowhere.dylib
}

# Detect manylinux platform
detect_platform() {
    # Detect architecture
    local arch=$(uname -m)
    case "$arch" in
        x86_64)  arch="x86_64" ;;
        aarch64) arch="aarch64" ;;
        arm64)   arch="aarch64" ;;  # macOS uses arm64
        *)       log_error "Unsupported architecture: $arch"; exit 1 ;;
    esac

    # Extract GLIBC version: match x.y pattern only
    local glibc_ver=$(ldd --version 2>/dev/null | awk 'NR==1 {for(i=1;i<=NF;i++) if($i ~ /^[0-9]+\.[0-9]+$/) {print $i; exit}}')
    [ -z "$glibc_ver" ] && glibc_ver="2.17"

    local major=$(echo "$glibc_ver" | cut -d. -f1)
    local minor=$(echo "$glibc_ver" | cut -d. -f2)

    # Map GLIBC version to manylinux tag with detected architecture
    if   [ "$major" -eq 2 ] && [ "$minor" -ge 35 ]; then echo "manylinux_2_35_${arch}"
    elif [ "$major" -eq 2 ] && [ "$minor" -ge 31 ]; then echo "manylinux_2_31_${arch}"
    elif [ "$major" -eq 2 ] && [ "$minor" -ge 28 ]; then echo "manylinux_2_28_${arch}"
    elif [ "$major" -eq 2 ] && [ "$minor" -ge 24 ]; then echo "manylinux_2_24_${arch}"
    elif [ "$major" -eq 2 ] && [ "$minor" -ge 17 ]; then echo "manylinux_2_17_${arch}"
    else echo "linux_${arch}"
    fi
}

# Build wheel
build_wheel() {
    log_info "Building wheel with $PYTHON..." >&2
    cd "$SCRIPT_DIR"

    if [ "$VERBOSE" = true ]; then
        $PYTHON setup.py bdist_wheel >&2
    else
        $PYTHON setup.py bdist_wheel >/dev/null 2>&1
    fi

    local wheel=$(ls -t dist/*-linux_*.whl 2>/dev/null | head -1)
    [ -n "$wheel" ] || { log_error "Wheel build failed" >&2; exit 1; }

    echo "$wheel"
}

# Repair wheel with auditwheel
repair_wheel() {
    local wheel="$1"
    local platform="$2"

    log_info "Repairing wheel for $platform..." >&2

    # Export library paths from libknowhere RUNPATH
    local lib_paths=$(readelf -d "$BUILD_DIR/libknowhere.so" | \
                      grep -E 'RUNPATH|RPATH' | \
                      sed 's/.*\[\(.*\)\]/\1/' | \
                      tr ':' '\n' | grep -v '^$' | tr '\n' ':')

    export LD_LIBRARY_PATH="${lib_paths}:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu"

    # Run auditwheel repair with full output
    if ! auditwheel repair "$wheel" -w dist/ --plat "$platform" 2>&1 | tee /dev/stderr; then
        log_error "auditwheel repair failed" >&2
        exit 1
    fi

    local repaired=$(ls -t dist/*-${platform}.whl 2>/dev/null | head -1)
    if [ -n "$repaired" ]; then
        rm -f "$wheel"  # Remove original non-portable wheel
        echo "$repaired"
    else
        log_error "No manylinux wheel generated. auditwheel repair failed." >&2
        log_error "This usually means:" >&2
        log_error "  1. Required libraries not found in LD_LIBRARY_PATH" >&2
        log_error "  2. Platform tag not supported" >&2
        log_error "  3. Dependencies have incompatible GLIBC versions" >&2
        exit 1
    fi
}

# Verify wheel
verify_wheel() {
    local wheel="$1"

    echo ""
    log_info "Verifying wheel: $(basename "$wheel")"

    # Disable exit on error for verification
    set +e

    # Check platform tag
    local tag=$(unzip -p "$wheel" '*.dist-info/WHEEL' 2>/dev/null | grep "Tag:" | cut -d: -f2 | tr -d ' ')
    [ -z "$tag" ] && tag="unknown"
    echo "  Platform: $tag"

    # Count bundled libraries
    local lib_count=$(unzip -l "$wheel" 2>/dev/null | grep "\.libs/.*\.so" | wc -l)
    echo "  Bundled libs: $lib_count"

    # Check RPATH
    local tmpdir=$(mktemp -d)
    if unzip -q "$wheel" -d "$tmpdir" 2>/dev/null; then
        local so_file=$(find "$tmpdir" -name "_swigknowhere*.so" 2>/dev/null | head -1)

        if [ -n "$so_file" ] && [ -f "$so_file" ]; then
            local rpath=$(readelf -d "$so_file" 2>/dev/null | grep -E "RPATH|RUNPATH" | sed 's/.*\[\(.*\)\]/\1/')
            if [ -n "$rpath" ]; then
                if [[ "$rpath" == *'$ORIGIN'* ]]; then
                    echo "  RPATH: $rpath (portable)"
                else
                    echo "  RPATH: $rpath (WARNING: not portable)"
                fi
            fi
        fi
    fi

    rm -rf "$tmpdir" 2>/dev/null || true

    # Show file size
    local size=$(du -h "$wheel" 2>/dev/null | cut -f1)
    [ -n "$size" ] && echo "  Size: $size"

    # Re-enable exit on error
    set -e
}

# Main
main() {
    local start=$(date +%s)

    echo "PyKnowhere Portable Wheel Builder"
    echo "=================================="
    echo ""

    # Check dependencies
    check_deps

    # Clean if requested
    [ "$CLEAN" = true ] && clean_build

    # Detect platform
    local platform=$(detect_platform)
    log_info "Target platform: $platform"

    # Build wheel
    local wheel=$(build_wheel)
    log_info "Built: $(basename "$wheel")"

    # Repair wheel
    local final_wheel=$(repair_wheel "$wheel" "$platform")
    log_info "Final: $(basename "$final_wheel")"

    # Verify
    verify_wheel "$final_wheel"

    # Summary
    local elapsed=$(($(date +%s) - start))
    echo ""
    log_info "Build completed in ${elapsed}s"
    echo ""
    echo "Install with:"
    echo "  pip install dist/$(basename "$final_wheel")"
    echo ""
}

cd "$SCRIPT_DIR"
main "$@"
