#!/usr/bin/env bash

# Copyright (C) 2019-2023 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
# http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under the License.

# Exit immediately for non zero status
set -e

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
ROOT_DIR="$( cd -P "$( dirname "$SOURCE" )/.." && pwd )"

# recreate the symlink to the compile_commands.json file
if [ -f "$ROOT_DIR/build/compile_commands.json" ]; then
  rm "$ROOT_DIR/build/compile_commands.json"
fi
if [ -f "$ROOT_DIR/build/Release/compile_commands.json" ]; then
  ln -sf "$ROOT_DIR/build/Release/compile_commands.json" "$ROOT_DIR/build/compile_commands.json"
elif [ -f "$ROOT_DIR/build/Debug/compile_commands.json" ]; then
  ln -sf "$ROOT_DIR/build/Debug/compile_commands.json" "$ROOT_DIR/build/compile_commands.json"
else
  echo "Error: No compile_commands.json file found in either $ROOT_DIR/build/Release or $ROOT_DIR/build/Debug"
  echo "Please first build the project so precommit checks can run."
  exit 1
fi
