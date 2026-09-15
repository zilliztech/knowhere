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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

final class NativeBuffers {
    private NativeBuffers() {
    }

    static ByteBuffer region(ByteBuffer buffer, boolean writable, boolean numeric) {
        Objects.requireNonNull(buffer, "buffer");
        if (!buffer.isDirect()) {
            throw new IllegalArgumentException("A direct ByteBuffer is required");
        }
        if (writable && buffer.isReadOnly()) {
            throw new IllegalArgumentException("The output buffer is read-only");
        }
        if (numeric && buffer.order() != ByteOrder.nativeOrder()) {
            throw new IllegalArgumentException("Numeric buffers must use native byte order");
        }
        return buffer.slice().order(ByteOrder.nativeOrder());
    }

    static ByteBuffer mask(ByteBuffer buffer, long bits) {
        if (bits == 0 && buffer == null) {
            return null;
        }
        return region(buffer, false, false);
    }
}
