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
import java.util.Random;
import org.junit.Test;
import org.junit.function.ThrowingRunnable;
import static org.junit.Assert.*;

public class KnowhereTest {
    private static final String L2 = "{\"metric_type\":\"L2\"}";

    static ByteBuffer floats(float... values) {
        ByteBuffer data = ByteBuffer.allocateDirect(values.length * Float.BYTES).order(ByteOrder.nativeOrder());
        for (float value : values) {
            data.putFloat(value);
        }
        return data.flip();
    }

    static ByteBuffer bytes(int size) {
        return ByteBuffer.allocateDirect(size).order(ByteOrder.nativeOrder());
    }

    @Test
    public void flatSearchAppliesExclusionAndPreservesBufferPositions() {
        try (KnowhereIndex index = Knowhere.createIndex("FLAT", DType.FLOAT32, Knowhere.currentIndexVersion())) {
            index.build(floats(0, 0, 1, 0, 0, 2), 3, 2, L2);
            ByteBuffer query = floats(999, 0, 0, 999);
            query.position(4).limit(12);
            ByteBuffer ids = bytes(32);
            ids.putLong(0, -99).putLong(24, -99);
            ids.position(8).limit(24);
            ByteBuffer distances = bytes(16);
            distances.putFloat(0, -99).putFloat(12, -99);
            distances.position(4).limit(12);
            ByteBuffer excluded = bytes(1).put(0, (byte) 1);
            index.search(query, 1, 2, 2, excluded, 3, ids, distances, L2);
            assertEquals(1, ids.getLong(8));
            assertEquals(2, ids.getLong(16));
            assertEquals(1f, distances.getFloat(4), 0f);
            assertEquals(4f, distances.getFloat(8), 0f);
            assertEquals(-99, ids.getLong(0));
            ids.limit(32);
            distances.limit(16);
            assertEquals(-99, ids.getLong(24));
            assertEquals(-99f, distances.getFloat(12), 0f);
            assertEquals(4, query.position());
            assertEquals(8, ids.position());
            assertEquals(4, distances.position());
        }
    }

    @Test
    public void serializationSurvivesSourceAndBinarySetClose() {
        KnowhereIndex first = Knowhere.createIndex("FLAT", DType.FLOAT32, Knowhere.currentIndexVersion());
        first.build(floats(0, 0, 1, 0, 0, 2), 3, 2, L2);
        BinarySet serialized = first.serialize();
        first.close();
        try (KnowhereIndex loaded = Knowhere.createIndex("FLAT", DType.FLOAT32, Knowhere.currentIndexVersion())) {
            loaded.deserialize(serialized, L2);
            serialized.close();
            ByteBuffer ids = bytes(16);
            ByteBuffer distances = bytes(8);
            loaded.search(floats(0, 0), 1, 2, 2, null, 0, ids, distances, L2);
            assertEquals(3, loaded.rows());
            assertEquals(2, loaded.dimensions());
            assertEquals(0, ids.getLong(0));
            assertEquals(1, ids.getLong(8));
            assertEquals(0f, distances.getFloat(0), 0f);
            assertEquals(1f, distances.getFloat(4), 0f);
        } finally {
            serialized.close();
            first.close();
        }
    }

    @Test
    public void binarySetSupportsNamedChunksAndRangeChecks() {
        try (final BinarySet data = BinarySet.create()) {
            data.allocate("向量-\uD83D\uDD0D", 8);
            ByteBuffer source = bytes(6).put(new byte[] {99, 1, 2, 3, 4, 99});
            source.position(1).limit(5);
            data.write("向量-\uD83D\uDD0D", 2, source);
            ByteBuffer output = bytes(8);
            data.read("向量-\uD83D\uDD0D", 0, output);
            byte[] actual = new byte[8];
            output.get(actual);
            assertArrayEquals(new byte[] {0, 0, 1, 2, 3, 4, 0, 0}, actual);
            assertEquals(1, data.count());
            assertEquals(8, data.length("向量-\uD83D\uDD0D"));
            assertArrayEquals(new String[] {"向量-\uD83D\uDD0D"}, data.names());
            assertEquals(1, source.position());
            assertThrows(KnowhereException.class, new ThrowingRunnable() {
                public void run() { data.read("向量-\uD83D\uDD0D", 7, bytes(2)); }
            });
            assertThrows(KnowhereException.class, new ThrowingRunnable() {
                public void run() { data.length("missing"); }
            });
        }
    }

    @Test
    public void bruteForceReturnsExactDistancesAndHonorsEmptyQueries() {
        ByteBuffer ids = bytes(16);
        ByteBuffer distances = bytes(8);
        Knowhere.bruteForce(DType.FLOAT32, floats(0, 0, 1, 0, 0, 2), 3,
                floats(0, 0), 1, 2, 2, null, 0, ids, distances, L2);
        assertEquals(0, ids.getLong(0));
        assertEquals(1, ids.getLong(8));
        assertEquals(0f, distances.getFloat(0), 0f);
        assertEquals(1f, distances.getFloat(4), 0f);
        Knowhere.bruteForce(DType.FLOAT32, floats(0, 0), 1, bytes(0), 0, 2, 2,
                null, 0, bytes(0), bytes(0), L2);
    }

    @Test
    public void invalidNativeInputsProduceExceptions() {
        assertThrows(KnowhereException.class, new ThrowingRunnable() {
            public void run() { Knowhere.createIndex("UNKNOWN", DType.FLOAT32, Knowhere.currentIndexVersion()); }
        });
        assertThrows(KnowhereException.class, new ThrowingRunnable() {
            public void run() { Knowhere.createIndex("FLAT", DType.FLOAT32, Integer.MAX_VALUE); }
        });
        try (final KnowhereIndex index = Knowhere.createIndex("FLAT", DType.FLOAT32, Knowhere.currentIndexVersion())) {
            assertThrows(KnowhereException.class, new ThrowingRunnable() {
                public void run() { index.buildFromFile(L2); }
            });
            assertThrows(KnowhereException.class, new ThrowingRunnable() {
                public void run() { index.build(floats(0, 0), 1, 2, "{"); }
            });
            assertThrows(KnowhereException.class, new ThrowingRunnable() {
                public void run() { index.build(floats(0, 0), Long.MAX_VALUE, 2, L2); }
            });
        }
    }

    @Test
    public void heapAndReadOnlyOutputsAreRejectedBeforeNativeWrites() {
        assertThrows(IllegalArgumentException.class, new ThrowingRunnable() {
            public void run() {
                Knowhere.bruteForce(DType.FLOAT32, ByteBuffer.allocate(8), 1,
                        floats(0, 0), 1, 2, 1, null, 0, bytes(8), bytes(4), L2);
            }
        });
        assertThrows(IllegalArgumentException.class, new ThrowingRunnable() {
            public void run() {
                Knowhere.bruteForce(DType.FLOAT32, floats(0, 0), 1, floats(0, 0), 1, 2, 1,
                        null, 0, bytes(8).asReadOnlyBuffer().order(ByteOrder.nativeOrder()), bytes(4), L2);
            }
        });
    }

    @Test
    public void closeIsIdempotentAndClosedObjectsCannotBeUsed() {
        final KnowhereIndex index = Knowhere.createIndex("FLAT", DType.FLOAT32, Knowhere.currentIndexVersion());
        index.close();
        index.close();
        assertThrows(IllegalStateException.class, new ThrowingRunnable() {
            public void run() { index.build(floats(0, 0), 1, 2, L2); }
        });
        final BinarySet data = BinarySet.create();
        data.close();
        data.close();
        assertThrows(IllegalStateException.class, new ThrowingRunnable() {
            public void run() { data.count(); }
        });
    }

    @Test
    public void typedFlatIndexesRoundTripThroughJni() {
        DType[] types = {DType.FLOAT32, DType.FLOAT16, DType.BFLOAT16, DType.INT8, DType.BINARY};
        ByteBuffer[] bases = {
            floats(0, 0, 1, 0, 0, 2),
            bytes(12).putShort((short) 0).putShort((short) 0).putShort((short) 0x3c00)
                    .putShort((short) 0).putShort((short) 0).putShort((short) 0x4000).flip(),
            bytes(12).putShort((short) 0).putShort((short) 0).putShort((short) 0x3f80)
                    .putShort((short) 0).putShort((short) 0).putShort((short) 0x4000).flip(),
            bytes(6).put(new byte[] {0, 0, 1, 0, 0, 2}).flip(),
            bytes(3).put(new byte[] {0, 1, 3}).flip()
        };
        for (int i = 0; i < types.length; i++) {
            boolean binary = types[i] == DType.BINARY;
            String indexType = binary ? "BIN_FLAT" : "FLAT";
            String parameters = binary ? "{\"metric_type\":\"HAMMING\"}" : L2;
            int dimension = binary ? 8 : 2;
            ByteBuffer query = bases[i].duplicate().order(ByteOrder.nativeOrder());
            query.limit(bases[i].remaining() / 3);
            try (KnowhereIndex index = Knowhere.createIndex(indexType, types[i], Knowhere.currentIndexVersion())) {
                index.build(bases[i], 3, dimension, parameters);
                try (BinarySet serialized = index.serialize();
                        KnowhereIndex loaded = Knowhere.createIndex(indexType, types[i], Knowhere.currentIndexVersion())) {
                    loaded.deserialize(serialized, parameters);
                    ByteBuffer ids = bytes(16);
                    ByteBuffer distances = bytes(8);
                    loaded.search(query, 1, dimension, 2, null, 0, ids, distances, parameters);
                    assertEquals(0, ids.getLong(0));
                    assertEquals(1, ids.getLong(8));
                    assertEquals(0f, distances.getFloat(0), 0f);
                    assertEquals(1f, distances.getFloat(4), 0f);
                    Knowhere.bruteForce(types[i], bases[i], 3, query, 1, dimension, 2,
                            null, 0, ids, distances, parameters);
                    assertEquals(0, ids.getLong(0));
                    assertEquals(1, ids.getLong(8));
                    assertEquals(1f, distances.getFloat(4), 0f);
                }
            }
        }
    }

    @Test
    public void nativeBufferCapacityAlignmentAndNamesAreValidated() {
        assertThrows(KnowhereException.class, new ThrowingRunnable() {
            public void run() {
                Knowhere.bruteForce(DType.FLOAT32, floats(0, 0), 1, floats(0, 0), 1, 2, 1,
                        null, 0, bytes(7), bytes(4), L2);
            }
        });
        assertThrows(KnowhereException.class, new ThrowingRunnable() {
            public void run() {
                ByteBuffer unaligned = bytes(9);
                unaligned.position(1);
                Knowhere.bruteForce(DType.FLOAT32, unaligned, 1, floats(0, 0), 1, 2, 1,
                        null, 0, bytes(8), bytes(4), L2);
            }
        });
        try (final BinarySet data = BinarySet.create()) {
            assertThrows(IllegalArgumentException.class, new ThrowingRunnable() {
                public void run() { data.allocate("bad\u0000name", 1); }
            });
            assertThrows(IllegalArgumentException.class, new ThrowingRunnable() {
                public void run() { data.allocate("bad\uD800", 1); }
            });
        }
    }

    @Test
    public void hnswAndIvfRoundTripsPreserveRecall() {
        int rows = 512;
        int dimensions = 8;
        int queries = 20;
        int topK = 5;
        ByteBuffer base = bytes(rows * dimensions * Float.BYTES);
        Random random = new Random(42);
        for (int i = 0; i < rows * dimensions; i++) {
            base.putFloat(random.nextFloat());
        }
        base.flip();
        ByteBuffer query = bytes(queries * dimensions * Float.BYTES);
        for (int row = 0; row < queries; row++) {
            for (int column = 0; column < dimensions; column++) {
                query.putFloat(base.getFloat((row * 17 * dimensions + column) * Float.BYTES) + 0.001f);
            }
        }
        query.flip();
        ByteBuffer expected = bytes(queries * topK * Long.BYTES);
        ByteBuffer distances = bytes(queries * topK * Float.BYTES);
        Knowhere.bruteForce(DType.FLOAT32, base, rows, query, queries, dimensions, topK,
                null, 0, expected, distances, L2);
        String[] types = {"HNSW", "IVF_FLAT"};
        String[] parameters = {
            "{\"metric_type\":\"L2\",\"M\":16,\"efConstruction\":128,\"ef\":128}",
            "{\"metric_type\":\"L2\",\"nlist\":8,\"nprobe\":8}"
        };
        for (int algorithm = 0; algorithm < types.length; algorithm++) {
            BinarySet data;
            try (KnowhereIndex index = Knowhere.createIndex(types[algorithm], DType.FLOAT32, Knowhere.currentIndexVersion())) {
                index.build(base, rows, dimensions, parameters[algorithm]);
                data = index.serialize();
            }
            try (BinarySet serialized = data;
                    KnowhereIndex loaded = Knowhere.createIndex(types[algorithm], DType.FLOAT32, Knowhere.currentIndexVersion())) {
                loaded.deserialize(serialized, parameters[algorithm]);
                serialized.close();
                ByteBuffer actual = bytes(expected.capacity());
                loaded.search(query, queries, dimensions, topK, null, 0, actual, distances, parameters[algorithm]);
                int hits = 0;
                for (int row = 0; row < queries; row++) {
                    for (int k = 0; k < topK; k++) {
                        long id = actual.getLong((row * topK + k) * Long.BYTES);
                        for (int truth = 0; truth < topK; truth++) {
                            if (id == expected.getLong((row * topK + truth) * Long.BYTES)) {
                                hits++;
                                break;
                            }
                        }
                    }
                }
                assertTrue(types[algorithm] + " recall: " + hits + "/" + queries * topK,
                        hits >= queries * topK * 0.95);
            }
        }
    }
}
