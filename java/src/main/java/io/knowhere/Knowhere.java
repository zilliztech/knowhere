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
import java.util.Objects;

/** Framework-independent entry points for synchronous Knowhere operations. */
public final class Knowhere {
    private Knowhere() {
    }

    public static int cAbiVersion() {
        return NativeBindings.cAbiVersion();
    }

    public static int minimumIndexVersion() {
        return NativeBindings.minimumIndexVersion();
    }

    public static int currentIndexVersion() {
        return NativeBindings.currentIndexVersion();
    }

    public static int maximumIndexVersion() {
        return NativeBindings.maximumIndexVersion();
    }

    /** Creates an owned index. Type, element encoding and index format version are separate. */
    public static KnowhereIndex createIndex(String type, DType dtype, int indexVersion) {
        return new KnowhereIndex(Objects.requireNonNull(type, "type"),
                Objects.requireNonNull(dtype, "dtype"), indexVersion);
    }

    /**
     * Searches base vectors without building an index. Buffers are borrowed until return;
     * positions and limits remain unchanged. The exclusion bitmap is low-bit-first, 1 excludes.
     * IDs are int64 and distances float32. Parameters are a Knowhere JSON object.
     */
    public static void bruteForce(DType dtype, ByteBuffer base, long baseRows,
            ByteBuffer queries, long queryRows, int dimension, int topK,
            ByteBuffer excludedRows, long excludedBitCount,
            ByteBuffer outputIds, ByteBuffer outputDistances, String parameters) {
        NativeBindings.bruteForce(Objects.requireNonNull(dtype, "dtype").code,
                NativeBuffers.region(base, false, true), baseRows,
                NativeBuffers.region(queries, false, true), queryRows, dimension, topK,
                NativeBuffers.mask(excludedRows, excludedBitCount), excludedBitCount,
                NativeBuffers.region(outputIds, true, true),
                NativeBuffers.region(outputDistances, true, true),
                Objects.requireNonNull(parameters, "parameters"));
    }
}
