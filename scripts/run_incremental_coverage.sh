#!/usr/bin/env bash

# Incremental and Global Code Coverage Script
# Inspired by milvus-io/milvus coverage implementation
#
# Usage:
#   ./scripts/run_incremental_coverage.sh [base_branch] [options]
#
# Options:
#   --skip-tests    Skip running tests (use existing .gcda files)
#   --html          Generate HTML reports
#   --fail-under N  Fail if incremental coverage < N%

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Parse arguments
BASE_BRANCH="origin/main"
SKIP_TESTS=false
GENERATE_HTML=false
FAIL_UNDER=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-tests) SKIP_TESTS=true; shift ;;
        --html) GENERATE_HTML=true; shift ;;
        --fail-under) FAIL_UNDER=$2; shift 2 ;;
        *) BASE_BRANCH="$1"; shift ;;
    esac
done

# Directories
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build/Release"
COVERAGE_DIR="${ROOT_DIR}/coverage"
UNITTEST_DIR="${BUILD_DIR}/tests/ut"

# Coverage files
FILE_INFO_BASE="${COVERAGE_DIR}/lcov_base.info"
FILE_INFO_UT="${COVERAGE_DIR}/lcov_ut.info"
FILE_INFO_COMBINE="${COVERAGE_DIR}/lcov_combine.info"
FILE_INFO_OUTPUT="${COVERAGE_DIR}/lcov_output.info"

print_header() {
    echo ""
    echo -e "${CYAN}================================================================${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}================================================================${NC}"
}

print_step() {
    echo -e "${BLUE}>>> $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check dependencies
check_dependencies() {
    print_step "Checking dependencies..."

    local missing=()
    command -v lcov >/dev/null 2>&1 || missing+=("lcov")
    command -v genhtml >/dev/null 2>&1 || missing+=("genhtml (part of lcov)")

    if [ ${#missing[@]} -gt 0 ]; then
        print_error "Missing dependencies: ${missing[*]}"
        echo "  Install with: sudo apt install lcov"
        exit 1
    fi

    if ! command -v diff-cover >/dev/null 2>&1; then
        print_warning "diff-cover not installed (optional, for detailed incremental report)"
        echo "  Install with: pip install diff-cover lcov_cobertura"
    fi

    print_success "Dependencies OK"
}

# Get changed files
get_changed_files() {
    print_step "Analyzing changes compared to ${BASE_BRANCH}..."

    CHANGED_FILES=$(git diff --name-only ${BASE_BRANCH} -- '*.cc' '*.h' '*.cpp' '*.hpp' 2>/dev/null | grep -E '^(src|include)/' || true)
    CHANGED_COUNT=$(echo "$CHANGED_FILES" | grep -c . || echo "0")

    if [ -z "$CHANGED_FILES" ]; then
        print_warning "No source files changed compared to ${BASE_BRANCH}"
        echo ""
        return 1
    fi

    echo -e "  Changed files: ${BOLD}${CHANGED_COUNT}${NC}"
    echo "$CHANGED_FILES" | head -10 | sed 's/^/    /'
    [ "$CHANGED_COUNT" -gt 10 ] && echo "    ... and $((CHANGED_COUNT - 10)) more"
    echo ""
    return 0
}

# Prepare coverage directory
prepare_coverage_dir() {
    print_step "Preparing coverage directory..."

    rm -rf ${COVERAGE_DIR}
    mkdir -p ${COVERAGE_DIR}

    # Clean old gcda files for fresh run
    if [ "$SKIP_TESTS" = false ]; then
        find ${BUILD_DIR} -name "*.gcda" -delete 2>/dev/null || true
    fi

    print_success "Coverage directory ready: ${COVERAGE_DIR}"
}

# Generate baseline coverage
generate_baseline() {
    print_step "Generating baseline coverage..."

    lcov -c -i -d ${BUILD_DIR} -o ${FILE_INFO_BASE} \
        --rc lcov_branch_coverage=1 \
        --ignore-errors gcov,source,graph 2>/dev/null

    print_success "Baseline generated"
}

# Run unit tests
run_tests() {
    if [ "$SKIP_TESTS" = true ]; then
        print_step "Skipping tests (--skip-tests)"
        return 0
    fi

    print_step "Running unit tests..."

    local start_time=$(date +%s)

    if [ -f "${UNITTEST_DIR}/knowhere_tests" ]; then
        ${UNITTEST_DIR}/knowhere_tests --durations=yes 2>&1 | tail -20
    else
        print_error "Test binary not found: ${UNITTEST_DIR}/knowhere_tests"
        echo "  Build with: conan install .. -o with_ut=True -o with_coverage=True && conan build .."
        exit 1
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    print_success "Tests completed in ${duration}s"
}

# Collect coverage data
collect_coverage() {
    print_step "Collecting coverage data..."

    lcov -c -d ${BUILD_DIR} -o ${FILE_INFO_UT} \
        --rc lcov_branch_coverage=1 \
        --ignore-errors gcov,source,graph 2>/dev/null

    print_success "Coverage data collected"
}

# Merge and filter coverage
merge_coverage() {
    print_step "Merging coverage data..."

    lcov -a ${FILE_INFO_BASE} -a ${FILE_INFO_UT} -o ${FILE_INFO_COMBINE} \
        --rc lcov_branch_coverage=1 2>/dev/null

    print_step "Filtering coverage data..."

    lcov -r ${FILE_INFO_COMBINE} -o ${FILE_INFO_OUTPUT} \
        '/usr/*' \
        '*/.conan/*' \
        '*/nlohmann/*' \
        '*/build/*' \
        '*/tests/*' \
        '*/thirdparty/*' \
        '*/benchmark/*' \
        --rc lcov_branch_coverage=1 2>/dev/null

    print_success "Coverage data filtered"
}

# Calculate coverage percentage from lcov info
calculate_coverage() {
    local info_file=$1
    local result=$(lcov --summary ${info_file} 2>&1)

    local lines_pct=$(echo "$result" | grep -E "lines\.*:" | sed 's/.*: //' | sed 's/%.*//')
    local functions_pct=$(echo "$result" | grep -E "functions\.*:" | sed 's/.*: //' | sed 's/%.*//')
    local branches_pct=$(echo "$result" | grep -E "branches\.*:" | sed 's/.*: //' | sed 's/%.*//' || echo "N/A")

    echo "${lines_pct}|${functions_pct}|${branches_pct}"
}

# Calculate incremental coverage
calculate_incremental_coverage() {
    print_step "Calculating incremental coverage..."

    if [ -z "$CHANGED_FILES" ]; then
        echo "0|0|0|0|0"
        return
    fi

    local total_lines=0
    local covered_lines=0

    # Create temporary filtered info for changed files only
    local temp_info="${COVERAGE_DIR}/incremental.info"

    # Build pattern for changed files
    local pattern=""
    for file in $CHANGED_FILES; do
        [ -n "$pattern" ] && pattern="${pattern}|"
        pattern="${pattern}${file}"
    done

    # Extract coverage for changed files only
    lcov -e ${FILE_INFO_OUTPUT} "*/${pattern}*" -o ${temp_info} 2>/dev/null || true

    if [ -f "${temp_info}" ] && [ -s "${temp_info}" ]; then
        local result=$(lcov --summary ${temp_info} 2>&1)
        local lines_info=$(echo "$result" | grep -E "lines\.*:" | head -1)

        # Parse: "lines......: 85.0% (170 of 200 lines)"
        local pct=$(echo "$lines_info" | sed 's/.*: //' | sed 's/%.*//')
        local covered=$(echo "$lines_info" | grep -oE '\([0-9]+ of' | grep -oE '[0-9]+')
        local total=$(echo "$lines_info" | grep -oE 'of [0-9]+' | grep -oE '[0-9]+')

        echo "${pct}|${covered}|${total}"
    else
        echo "N/A|0|0"
    fi
}

# Display coverage summary
display_summary() {
    print_header "Coverage Summary"

    # Global coverage
    local global_cov=$(calculate_coverage ${FILE_INFO_OUTPUT})
    IFS='|' read -r global_lines global_funcs global_branches <<< "$global_cov"

    echo ""
    echo -e "${BOLD}┌─────────────────────────────────────────────────────────────┐${NC}"
    echo -e "${BOLD}│                    GLOBAL COVERAGE                          │${NC}"
    echo -e "${BOLD}├─────────────────────────────────────────────────────────────┤${NC}"

    # Color based on coverage level
    local lines_color=$RED
    [ "$(echo "$global_lines >= 60" | bc -l 2>/dev/null || echo 0)" = "1" ] && lines_color=$YELLOW
    [ "$(echo "$global_lines >= 80" | bc -l 2>/dev/null || echo 0)" = "1" ] && lines_color=$GREEN

    printf "${BOLD}│${NC}  Lines:      ${lines_color}%6.1f%%${NC}                                       ${BOLD}│${NC}\n" "$global_lines"
    printf "${BOLD}│${NC}  Functions:  %6.1f%%                                       ${BOLD}│${NC}\n" "$global_funcs"
    [ "$global_branches" != "N/A" ] && printf "${BOLD}│${NC}  Branches:   %6.1f%%                                       ${BOLD}│${NC}\n" "$global_branches"
    echo -e "${BOLD}└─────────────────────────────────────────────────────────────┘${NC}"

    # Incremental coverage
    if [ -n "$CHANGED_FILES" ]; then
        local incr_cov=$(calculate_incremental_coverage)
        IFS='|' read -r incr_pct incr_covered incr_total <<< "$incr_cov"

        echo ""
        echo -e "${BOLD}┌─────────────────────────────────────────────────────────────┐${NC}"
        echo -e "${BOLD}│                  INCREMENTAL COVERAGE                       │${NC}"
        echo -e "${BOLD}│              (Changed files vs ${BASE_BRANCH})${NC}"
        echo -e "${BOLD}├─────────────────────────────────────────────────────────────┤${NC}"

        if [ "$incr_pct" != "N/A" ]; then
            local incr_color=$RED
            [ "$(echo "$incr_pct >= 60" | bc -l 2>/dev/null || echo 0)" = "1" ] && incr_color=$YELLOW
            [ "$(echo "$incr_pct >= 80" | bc -l 2>/dev/null || echo 0)" = "1" ] && incr_color=$GREEN

            printf "${BOLD}│${NC}  Lines:      ${incr_color}%6.1f%%${NC}  (%s of %s lines)                  ${BOLD}│${NC}\n" "$incr_pct" "$incr_covered" "$incr_total"
            printf "${BOLD}│${NC}  Files:      %6d                                        ${BOLD}│${NC}\n" "$CHANGED_COUNT"
        else
            echo -e "${BOLD}│${NC}  ${YELLOW}No coverage data for changed files${NC}                        ${BOLD}│${NC}"
        fi
        echo -e "${BOLD}└─────────────────────────────────────────────────────────────┘${NC}"

        # Check threshold
        if [ "$FAIL_UNDER" -gt 0 ] && [ "$incr_pct" != "N/A" ]; then
            if [ "$(echo "$incr_pct < $FAIL_UNDER" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
                echo ""
                print_error "Incremental coverage ${incr_pct}% is below threshold ${FAIL_UNDER}%"
                return 1
            fi
        fi
    fi

    return 0
}

# Generate HTML reports
generate_html_reports() {
    if [ "$GENERATE_HTML" = false ]; then
        return
    fi

    print_step "Generating HTML reports..."

    genhtml ${FILE_INFO_OUTPUT} \
        --output-directory ${COVERAGE_DIR}/html \
        --branch-coverage \
        --legend \
        --title "Knowhere Coverage Report" 2>/dev/null

    print_success "HTML report: ${COVERAGE_DIR}/html/index.html"

    # Generate diff-cover report if available
    if command -v diff-cover >/dev/null 2>&1 && command -v lcov_cobertura >/dev/null 2>&1; then
        lcov_cobertura ${FILE_INFO_OUTPUT} -o ${COVERAGE_DIR}/coverage.xml 2>/dev/null

        diff-cover ${COVERAGE_DIR}/coverage.xml \
            --compare-branch=${BASE_BRANCH} \
            --html-report ${COVERAGE_DIR}/diff-coverage.html \
            --fail-under=0 2>/dev/null || true

        print_success "Incremental report: ${COVERAGE_DIR}/diff-coverage.html"
    fi
}

# Main execution
main() {
    print_header "Knowhere Code Coverage Analysis"
    echo -e "  Base branch: ${BOLD}${BASE_BRANCH}${NC}"
    echo -e "  Build dir:   ${BUILD_DIR}"
    echo ""

    check_dependencies

    HAS_CHANGES=true
    get_changed_files || HAS_CHANGES=false

    prepare_coverage_dir
    generate_baseline
    run_tests
    collect_coverage
    merge_coverage

    if ! display_summary; then
        exit 1
    fi

    generate_html_reports

    print_header "Done"
    echo -e "  Coverage data: ${COVERAGE_DIR}/lcov_output.info"
    [ "$GENERATE_HTML" = true ] && echo -e "  HTML report:   ${COVERAGE_DIR}/html/index.html"
    echo ""
}

main "$@"
