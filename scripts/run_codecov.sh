#!/usr/bin/env bash

# Licensed to the LF AI & Data foundation under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Exit immediately for non zero status
set -e

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
ROOT_DIR="$( cd -P "$( dirname "$SOURCE" )/.." && pwd )"

KNOWHERE_CORE_DIR="${ROOT_DIR}/src"
KNOWHERE_UNITTEST_DIR="${ROOT_DIR}/build/Release/tests/ut"

echo "ROOT_DIR = ${ROOT_DIR}"
echo "KNOWHERE_CORE_DIR = ${KNOWHERE_CORE_DIR}"
echo "KNOWHERE_UNITTEST_DIR = ${KNOWHERE_UNITTEST_DIR}"

LCOV_CMD="lcov"
LCOV_GEN_CMD="genhtml"

FILE_INFO_BASE="${ROOT_DIR}/lcov_base.info"
FILE_INFO_UT="${ROOT_DIR}/lcov_ut.info"
FILE_INFO_COMBINE="${ROOT_DIR}/lcov_combine.info"
FILE_INFO_OUTPUT="${ROOT_DIR}/lcov_output.info"
DIR_LCOV_OUTPUT="${ROOT_DIR}/coverage"
DIR_GCNO="${ROOT_DIR}/build/"

# delete old code coverage info files
rm -f ${FILE_INFO_BASE}
rm -f ${FILE_INFO_UT}
rm -f ${FILE_INFO_COMBINE}
rm -f ${FILE_INFO_OUTPUT}
rm -rf ${DIR_LCOV_OUTPUT}

# generate baseline
${LCOV_CMD} -c -i -d ${DIR_GCNO} -o ${FILE_INFO_BASE}
if [ $? -ne 0 ]; then
    echo "generate ${FILE_INFO_BASE} failed"
    exit 1
fi

# run unittest
for test in `ls ${KNOWHERE_UNITTEST_DIR}/*test*`; do
    echo "Running unittest: $test"
    # run unittest
    ${test}
    if [ $? -ne 0 ]; then
        echo ${args}
        echo ${test} "run failed"
        exit 1
    fi
done

# generate ut file
${LCOV_CMD} -c -d ${DIR_GCNO} -o ${FILE_INFO_UT}

# merge baseline and ut file
${LCOV_CMD} -a ${FILE_INFO_BASE} -a ${FILE_INFO_UT} -o ${FILE_INFO_COMBINE}
if [ $? -ne 0 ]; then
    echo "generate ${FILE_INFO_COMBINE} failed"
    exit 1
fi

# remove unnecessary info
${LCOV_CMD} -r "${FILE_INFO_COMBINE}" -o "${FILE_INFO_OUTPUT}" \
    "/usr/*" \
    "*/.conan/*" \
    "*/nlohmann/*" \
    "*/build/*" \
    "*/tests/*" \
    "*/thirdparty/*"

if [ $? -ne 0 ]; then
    echo "generate ${FILE_INFO_OUTPUT} failed"
    exit 1
fi

# generate html report
${LCOV_GEN_CMD} ${FILE_INFO_OUTPUT} --output-directory ${DIR_LCOV_OUTPUT}
echo "Generate code coverage report to ${DIR_LCOV_OUTPUT}"
