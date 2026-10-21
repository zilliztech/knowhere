#!/usr/bin/env python3
# Copyright (C) 2026 Zilliz. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Fail when HotSpot reports JNI misuse or overwritten VM signal handlers.

-Xcheck:jni can emit warnings while Java tests still exit successfully. Check
both process logs and Surefire reports, including captured native dumpstreams.
Pass reports from the current test run rather than stale failed-run reports.
"""

import argparse
from pathlib import Path
import re

DIAGNOSTIC = re.compile(
    r"(?:WARNING|FATAL ERROR) in native method|JNI local refs:|"
    r"Warning: SIG[A-Z0-9]+ handler|Handler was modified!"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    failures = []
    for path in args.paths:
        if not path.exists():
            parser.error("Missing JNI diagnostic input: " + str(path))
        files = sorted(path.rglob("*")) if path.is_dir() else [path]
        for file in files:
            if not file.is_file():
                continue
            for line_number, line in enumerate(file.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if DIAGNOSTIC.search(line):
                    failures.append("%s:%d: %s" % (file, line_number, line))
    if failures:
        raise SystemExit("HotSpot JNI diagnostics detected:\n" + "\n".join(failures))
    print("No HotSpot JNI misuse or overwritten signal handlers detected")


if __name__ == "__main__":
    main()
