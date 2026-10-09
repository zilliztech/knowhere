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

/** Owned named index blobs. Large entries are allocated once and transferred in bounded chunks. */
public final class BinarySet implements AutoCloseable {
    private long handle;

    private BinarySet() {
    }

    public static BinarySet create() {
        BinarySet result = new BinarySet();
        result.handle = NativeBindings.binarySetCreate();
        return result;
    }

    static BinarySet fromIndex(long index) {
        BinarySet result = new BinarySet();
        result.handle = NativeBindings.indexSerialize(index);
        return result;
    }

    synchronized long requireHandle() {
        if (handle == 0) {
            throw new IllegalStateException("The BinarySet is closed");
        }
        return handle;
    }

    public synchronized long count() {
        return NativeBindings.binarySetCount(requireHandle());
    }

    public synchronized String name(long index) {
        return NativeBindings.binarySetName(requireHandle(), index);
    }

    public synchronized String[] names() {
        int size = Math.toIntExact(count());
        String[] names = new String[size];
        for (int i = 0; i < size; i++) {
            names[i] = name(i);
        }
        return names;
    }

    public synchronized long length(String name) {
        return NativeBindings.binarySetLength(requireHandle(), Objects.requireNonNull(name, "name"));
    }

    /** Allocates a zero-filled blob; replaces an existing entry with the same name. */
    public synchronized void allocate(String name, long length) {
        NativeBindings.binarySetAllocate(requireHandle(), Objects.requireNonNull(name, "name"), length);
    }

    /**
     * Copies a chunk into an entry. Refused with {@link KnowhereException} once the set has been
     * passed to {@link KnowhereIndex#deserialize}: an index loaded from it may still read the bytes.
     */
    public synchronized void write(String name, long offset, ByteBuffer source) {
        NativeBindings.binarySetWrite(requireHandle(), Objects.requireNonNull(name, "name"), offset,
                NativeBuffers.region(source, false, false));
    }

    public synchronized void read(String name, long offset, ByteBuffer destination) {
        NativeBindings.binarySetRead(requireHandle(), Objects.requireNonNull(name, "name"), offset,
                NativeBuffers.region(destination, true, false));
    }

    @Override
    public synchronized void close() {
        if (handle != 0) {
            NativeBindings.binarySetDestroy(handle);
            handle = 0;
        }
    }
}
