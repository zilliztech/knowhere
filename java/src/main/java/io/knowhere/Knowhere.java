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

    /**
     * Creates the process-wide search pool with {@code threads} threads, or resizes it. Index
     * search and brute force run one single-threaded task per query row on this pool, so a pool
     * that was never sized is created on first use with the hardware thread count. Only a call
     * before the first search replaces that default; later calls resize the live pool. Valid
     * sizes are 1 to {@link Integer#MAX_VALUE}.
     */
    public static void resizeSearchThreadPool(int threads) {
        NativeBindings.resizeSearchThreadPool(threads);
    }

    /** Current search pool size, or 0 while the pool does not exist yet. */
    public static int searchThreadPoolSize() {
        return NativeBindings.searchThreadPoolSize();
    }

    /** Creates or resizes the process-wide build pool used by index builds; same rules as search. */
    public static void resizeBuildThreadPool(int threads) {
        NativeBindings.resizeBuildThreadPool(threads);
    }

    /** Current build pool size, or 0 while the pool does not exist yet. */
    public static int buildThreadPoolSize() {
        return NativeBindings.buildThreadPoolSize();
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

    /**
     * Searches FLOAT32 base vectors without an index or a filter, handing every query to the
     * bundled faiss in one call so that it takes its BLAS (SGEMM) path. The call runs on the
     * calling thread with OpenMP and OpenBLAS at one thread; callers provide the parallelism
     * by calling from several threads or processes. Buffers are borrowed until return;
     * positions and limits remain unchanged. IDs are int64 (-1 where fewer than topK rows
     * exist) and distances float32 in the metric's own semantics. Parameters are a Knowhere
     * JSON object whose metric_type is L2, IP or COSINE.
     */
    public static void bruteForceBatched(DType dtype, ByteBuffer base, long baseRows,
            ByteBuffer queries, long queryRows, int dimension, int topK,
            ByteBuffer outputIds, ByteBuffer outputDistances, String parameters) {
        NativeBindings.bruteForceBatched(Objects.requireNonNull(dtype, "dtype").code,
                NativeBuffers.region(base, false, true), baseRows,
                NativeBuffers.region(queries, false, true), queryRows, dimension, topK,
                NativeBuffers.region(outputIds, true, true),
                NativeBuffers.region(outputDistances, true, true),
                Objects.requireNonNull(parameters, "parameters"));
    }
}
