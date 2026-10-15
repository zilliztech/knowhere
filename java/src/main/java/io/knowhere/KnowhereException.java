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

/** A failure returned by the Knowhere C API, including its original status message. */
public final class KnowhereException extends RuntimeException {
    private static final long serialVersionUID = 1L;
    private final int code;

    public KnowhereException(int code, String message) {
        super(message);
        this.code = code;
    }

    /** C API status: 1 argument, 2 closed, 3 unsupported, 4 operation, 5 allocation. */
    public int code() {
        return code;
    }
}
