/*
 * Copyright (C) 2026 Zilliz. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
package io.knowhere;

/** Element encodings in native byte order; binary dimensions count bits. */
public enum DType {
    FLOAT32(1), BINARY(2), FLOAT16(3), BFLOAT16(4), INT8(5);

    final int code;

    DType(int code) {
        this.code = code;
    }
}
