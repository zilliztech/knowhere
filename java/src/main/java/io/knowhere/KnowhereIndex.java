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

/** An owned native index. Operations on one instance are serialized; close is idempotent. */
public final class KnowhereIndex implements AutoCloseable {
    private final DType dtype;
    private long handle;

    KnowhereIndex(String type, DType dtype, int indexVersion) {
        this.dtype = dtype;
        handle = NativeBindings.indexCreate(type, dtype.code, indexVersion);
    }

    private long requireHandle() {
        if (handle == 0) {
            throw new IllegalStateException("The index is closed");
        }
        return handle;
    }

    /** Builds once. Native storage retains a copy, so the input may be reused after return. */
    public synchronized void build(ByteBuffer vectors, long rows, int dimension, String parameters) {
        NativeBindings.indexBuild(requireHandle(), NativeBuffers.region(vectors, false, true),
                rows, dimension, dtype.code, Objects.requireNonNull(parameters, "parameters"));
    }

    /**
     * Builds once from vectors at a native address in native byte order; {@code bytes} is the
     * length of that memory. It takes what a ByteBuffer cannot address: more than
     * Integer.MAX_VALUE bytes. Native storage retains a copy, so the memory may be reused after
     * return.
     */
    public synchronized void build(long address, long bytes, long rows, int dimension, String parameters) {
        if (address == 0L) {
            throw new IllegalArgumentException("A native address is required");
        }
        if (bytes < 0L) {
            throw new IllegalArgumentException("The byte length is negative");
        }
        NativeBindings.indexBuildAddress(requireHandle(), address, bytes, rows, dimension, dtype.code,
                Objects.requireNonNull(parameters, "parameters"));
    }

    /**
     * Builds a file-backed index such as DiskANN using Knowhere's data_path/index_prefix parameters.
     * Keep generated index files available until every loaded index using them is closed.
     */
    public synchronized void buildFromFile(String parameters) {
        NativeBindings.indexBuildFromFile(requireHandle(), Objects.requireNonNull(parameters, "parameters"));
    }

    /**
     * Searches borrowed query buffers and writes int64 IDs and float32 distances.
     * Numeric buffers must be direct and in native byte order. Only position..limit is used.
     */
    public synchronized void search(ByteBuffer queries, long queryRows, int dimension, int topK,
            ByteBuffer excludedRows, long excludedBitCount,
            ByteBuffer outputIds, ByteBuffer outputDistances, String parameters) {
        NativeBindings.indexSearch(requireHandle(), NativeBuffers.region(queries, false, true),
                queryRows, dimension, dtype.code, topK,
                NativeBuffers.mask(excludedRows, excludedBitCount), excludedBitCount,
                NativeBuffers.region(outputIds, true, true), NativeBuffers.region(outputDistances, true, true),
                Objects.requireNonNull(parameters, "parameters"));
    }

    public synchronized long rows() {
        return NativeBindings.indexInfo(requireHandle())[0];
    }

    public synchronized long dimensions() {
        return NativeBindings.indexInfo(requireHandle())[1];
    }

    /** Returns owned index blobs. File-backed indexes may instead require their existing local files. */
    public synchronized BinarySet serialize() {
        return BinarySet.fromIndex(requireHandle());
    }

    /**
     * Loads once, sharing the BinarySet's bytes with the index instead of copying them. The
     * BinarySet may be closed, and its entries re-allocated, after this call; writing into it is
     * refused from then on, because an index loaded from those bytes may still read them.
     */
    public synchronized void deserialize(BinarySet data, String parameters) {
        NativeBindings.indexDeserialize(requireHandle(), Objects.requireNonNull(data, "data").requireHandle(),
                Objects.requireNonNull(parameters, "parameters"));
    }

    @Override
    public synchronized void close() {
        if (handle != 0) {
            NativeBindings.indexDestroy(handle);
            handle = 0;
        }
    }
}
