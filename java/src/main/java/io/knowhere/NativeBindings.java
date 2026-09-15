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

final class NativeBindings {
    static {
        NativeLibraryLoader.load();
        if (cAbiVersion() != 1) {
            throw new UnsatisfiedLinkError("Knowhere Java bindings require C ABI version 1");
        }
    }

    private NativeBindings() {
    }

    static native int cAbiVersion();
    static native int minimumIndexVersion();
    static native int currentIndexVersion();
    static native int maximumIndexVersion();
    static native long indexCreate(String type, int dtype, int version);
    static native void indexDestroy(long handle);
    static native void indexBuild(long handle, ByteBuffer data, long rows, int dimension, int dtype, String parameters);
    static native void indexBuildFromFile(long handle, String parameters);
    static native void indexSearch(long handle, ByteBuffer queries, long rows, int dimension, int dtype, int topK,
            ByteBuffer excluded, long bits, ByteBuffer ids, ByteBuffer distances, String parameters);
    static native long[] indexInfo(long handle);
    static native long indexSerialize(long handle);
    static native void indexDeserialize(long handle, long binarySet, String parameters);
    static native void bruteForce(int dtype, ByteBuffer base, long baseRows, ByteBuffer queries, long rows,
            int dimension, int topK, ByteBuffer excluded, long bits,
            ByteBuffer ids, ByteBuffer distances, String parameters);
    static native long binarySetCreate();
    static native void binarySetDestroy(long handle);
    static native long binarySetCount(long handle);
    static native String binarySetName(long handle, long index);
    static native long binarySetLength(long handle, String name);
    static native void binarySetAllocate(long handle, String name, long length);
    static native void binarySetWrite(long handle, String name, long offset, ByteBuffer data);
    static native void binarySetRead(long handle, String name, long offset, ByteBuffer data);
}
