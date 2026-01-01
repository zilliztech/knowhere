#!/usr/bin/env bash

# Parse arguments
FULL_MODE=""

# Parse flags (support both in any order)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --full)
            if [[ -z "$2" || ("$2" != "Debug" && "$2" != "Release") ]]; then
                echo "Usage: $0 [--full <Debug|Release>]"
                exit 1
            fi
            FULL_MODE="$2"
            shift 2
            ;;
        *)
            echo "Usage: $0 [--full <Debug|Release>]"
            exit 1
            ;;
    esac
done

docker build -f Dockerfile.builder -t knowhere-builder:latest .

# Set entrypoint based on mode
ENTRYPOINT="/bin/bash"
if [[ -n "$FULL_MODE" ]]; then
    ENTRYPOINT="/workspace/builder_entrypoint.sh"
fi

docker run --rm -it \
    -v "$(pwd)":/workspace \
    -v "${HOME}/.conan":/root/.conan \
    -w /workspace \
    --entrypoint "$ENTRYPOINT" \
    knowhere-builder:latest ${FULL_MODE:+"$FULL_MODE"}
