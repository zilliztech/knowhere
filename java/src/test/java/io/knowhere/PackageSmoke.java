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

/** Runs with only the API JAR, platform JAR and this class on a clean JRE. */
public final class PackageSmoke {
    private PackageSmoke() {
    }

    public static void main(String[] args) {
        if (Knowhere.cAbiVersion() != 1) {
            throw new AssertionError("Incorrect C ABI version");
        }
        ByteBuffer base = ByteBuffer.allocateDirect(24).order(ByteOrder.nativeOrder());
        base.putFloat(0).putFloat(0).putFloat(1).putFloat(0).putFloat(0).putFloat(2).flip();
        ByteBuffer query = ByteBuffer.allocateDirect(8).order(ByteOrder.nativeOrder());
        ByteBuffer ids = ByteBuffer.allocateDirect(16).order(ByteOrder.nativeOrder());
        ByteBuffer distances = ByteBuffer.allocateDirect(8).order(ByteOrder.nativeOrder());
        String parameters = "{\"metric_type\":\"L2\"}";
        try (KnowhereIndex index = Knowhere.createIndex("FLAT", DType.FLOAT32, Knowhere.currentIndexVersion())) {
            index.build(base, 3, 2, parameters);
            try (BinarySet data = index.serialize();
                    KnowhereIndex restored = Knowhere.createIndex("FLAT", DType.FLOAT32, Knowhere.currentIndexVersion())) {
                restored.deserialize(data, parameters);
                restored.search(query, 1, 2, 2, null, 0, ids, distances, parameters);
                check(ids, distances);
            }
        }
        Knowhere.bruteForce(DType.FLOAT32, base, 3, query, 1, 2, 2, null, 0, ids, distances, parameters);
        check(ids, distances);
        System.out.println("Knowhere packaged JNI smoke passed on Java " + System.getProperty("java.version"));
    }

    private static void check(ByteBuffer ids, ByteBuffer distances) {
        if (ids.getLong(0) != 0 || ids.getLong(8) != 1 || distances.getFloat(0) != 0f
                || distances.getFloat(4) != 1f) {
            throw new AssertionError("Incorrect native search result");
        }
    }
}
