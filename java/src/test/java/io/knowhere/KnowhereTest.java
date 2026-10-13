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
    public void binarySetIsFrozenOnceAnIndexLoadedFromIt() {
        KnowhereIndex first = Knowhere.createIndex("FLAT", DType.FLOAT32, Knowhere.currentIndexVersion());
        first.build(floats(0, 0, 1, 0, 0, 2), 3, 2, L2);
        final BinarySet serialized = first.serialize();
        first.close();
        try (KnowhereIndex loaded = Knowhere.createIndex("FLAT", DType.FLOAT32, Knowhere.currentIndexVersion())) {
            loaded.deserialize(serialized, L2);
            final String name = serialized.name(0);
            final long length = serialized.length(name);
            // The index may still read these bytes, so writing in place is refused.
            assertThrows(KnowhereException.class, new ThrowingRunnable() {
                public void run() { serialized.write(name, 0, bytes(1)); }
            });
            // Replacing the entry with a fresh buffer and closing the set leave the index intact.
            serialized.allocate(name, length);
            serialized.close();
            ByteBuffer ids = bytes(16);
            ByteBuffer distances = bytes(8);
            loaded.search(floats(0, 0), 1, 2, 2, null, 0, ids, distances, L2);
            assertEquals(0, ids.getLong(0));
            assertEquals(1, ids.getLong(8));
            assertEquals(0f, distances.getFloat(0), 0f);
            assertEquals(1f, distances.getFloat(4), 0f);
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

    static ByteBuffer randomFloats(Random random, int count) {
        ByteBuffer data = bytes(count * Float.BYTES);
        for (int i = 0; i < count; i++) {
            data.putFloat(i * Float.BYTES, (float) random.nextGaussian());
        }
        return data;
    }

    @Test
    public void batchedBruteForceAnswersAsThePerQueryEntryDoes() {
        Random random = new Random(7);
        int rows = 200;
        int queries = 40;
        int dimension = 16;
        int topK = 4;
        ByteBuffer base = randomFloats(random, rows * dimension);
        ByteBuffer query = randomFloats(random, queries * dimension);
        for (String metric : new String[] {"L2", "IP", "COSINE"}) {
            String parameters = "{\"metric_type\":\"" + metric + "\"}";
            ByteBuffer ids = bytes(queries * topK * Long.BYTES);
            ByteBuffer distances = bytes(queries * topK * Float.BYTES);
            ByteBuffer batchedIds = bytes(queries * topK * Long.BYTES);
            ByteBuffer batchedDistances = bytes(queries * topK * Float.BYTES);
            Knowhere.bruteForce(DType.FLOAT32, base, rows, query, queries, dimension, topK,
                    null, 0, ids, distances, parameters);
            Knowhere.bruteForceBatched(DType.FLOAT32, base, rows, query, queries, dimension, topK,
                    batchedIds, batchedDistances, parameters);
            for (int q = 0; q < queries; q++) {
                for (int i = 0; i < topK; i++) {
                    long id = ids.getLong((q * topK + i) * Long.BYTES);
                    float expected = distances.getFloat((q * topK + i) * Float.BYTES);
                    boolean found = false;
                    for (int j = 0; j < topK; j++) {
                        if (batchedIds.getLong((q * topK + j) * Long.BYTES) == id) {
                            assertEquals(metric + " query " + q, expected,
                                    batchedDistances.getFloat((q * topK + j) * Float.BYTES),
                                    1e-3f * Math.max(1f, Math.abs(expected)));
                            found = true;
                        }
                    }
                    assertTrue(metric + " query " + q + " misses id " + id, found);
                }
            }
        }
        Knowhere.bruteForceBatched(DType.FLOAT32, floats(0, 0), 1, bytes(0), 0, 2, 2,
                bytes(0), bytes(0), L2);
    }

    @Test
    public void batchedBruteForceTakesFloat32AndTheFloatMetricsOnly() {
        assertThrows(KnowhereException.class, new ThrowingRunnable() {
            public void run() {
                Knowhere.bruteForceBatched(DType.FLOAT32, floats(0, 0), 1, floats(0, 0), 1, 2, 1,
                        bytes(8), bytes(4), "{\"metric_type\":\"HAMMING\"}");
            }
        });
        assertThrows(KnowhereException.class, new ThrowingRunnable() {
            public void run() {
                Knowhere.bruteForceBatched(DType.FLOAT16, bytes(4), 1, bytes(4), 1, 2, 1,
                        bytes(8), bytes(4), L2);
            }
        });
        assertThrows(IllegalArgumentException.class, new ThrowingRunnable() {
            public void run() {
                Knowhere.bruteForceBatched(DType.FLOAT32, ByteBuffer.allocate(8), 1,
                        floats(0, 0), 1, 2, 1, bytes(8), bytes(4), L2);
            }
        });
    }

    @Test
    public void batchedBruteForceWritesMinusOneForAnEmptyBase() {
        // Twenty queries take faiss's BLAS path, which writes nothing itself for an empty base.
        int queries = 20;
        int topK = 2;
        ByteBuffer query = randomFloats(new Random(3), queries * 2);
        for (String metric : new String[] {"L2", "IP", "COSINE"}) {
            ByteBuffer ids = bytes(queries * topK * Long.BYTES);
            for (int i = 0; i < queries * topK; i++) {
                ids.putLong(i * Long.BYTES, 777);
            }
            Knowhere.bruteForceBatched(DType.FLOAT32, bytes(0), 0, query, queries, 2, topK,
                    ids, bytes(queries * topK * Float.BYTES), "{\"metric_type\":\"" + metric + "\"}");
            for (int i = 0; i < queries * topK; i++) {
                assertEquals(metric + " slot " + i, -1, ids.getLong(i * Long.BYTES));
            }
        }
    }

    @Test
    public void batchedL2RescoringComputesTheCandidatesDirectly() {
        // Far from the origin and close together: rows at squared distances 25, 25 and 0.25 from
        // every query. Twenty queries take faiss's BLAS path, whose estimates cannot separate these
        // rows; whichever rows it returns, their direct distances are exact.
        int rows = 3;
        int queries = 20;
        int dimension = 16;
        int topK = 2;
        float[] offsets = {1.25f, -1.25f, 0.125f};
        ByteBuffer base = bytes(rows * dimension * Float.BYTES);
        for (int i = 0; i < rows * dimension; i++) {
            base.putFloat(i * Float.BYTES, 1e6f + offsets[i / dimension]);
        }
        ByteBuffer query = bytes(queries * dimension * Float.BYTES);
        for (int i = 0; i < queries * dimension; i++) {
            query.putFloat(i * Float.BYTES, 1e6f);
        }
        ByteBuffer ids = bytes(queries * topK * Long.BYTES);
        ByteBuffer distances = bytes(queries * topK * Float.BYTES);
        ByteBuffer direct = bytes(queries * topK * Float.BYTES);
        ByteBuffer counts = bytes(queries * Integer.BYTES);
        Knowhere.bruteForceBatchedL2Rescored(DType.FLOAT32, base, rows, query, queries, dimension, topK,
                null, ids, distances, direct, counts, L2);
        // The candidates and their estimates are the batched entry's.
        ByteBuffer batchedIds = bytes(queries * topK * Long.BYTES);
        ByteBuffer batchedDistances = bytes(queries * topK * Float.BYTES);
        Knowhere.bruteForceBatched(DType.FLOAT32, base, rows, query, queries, dimension, topK,
                batchedIds, batchedDistances, L2);
        assertEquals(batchedIds, ids);
        assertEquals(batchedDistances, distances);
        for (int q = 0; q < queries; q++) {
            assertEquals(topK, counts.getInt(q * Integer.BYTES));
            for (int i = 0; i < topK; i++) {
                long id = ids.getLong((q * topK + i) * Long.BYTES);
                assertTrue("query " + q + " id " + id, id >= 0 && id < rows);
                double expected = 0;
                for (int j = 0; j < dimension; j++) {
                    double difference = (double) query.getFloat((q * dimension + j) * Float.BYTES)
                            - base.getFloat(((int) id * dimension + j) * Float.BYTES);
                    expected += difference * difference;
                }
                assertEquals("query " + q, expected, direct.getFloat((q * topK + i) * Float.BYTES),
                        1e-6 * expected);
            }
        }
    }

    @Test
    public void batchedL2RescoringStopsAtTheCutoff() {
        // Small integers keep the estimates exact: rows at squared distances 9, 16, 1 and 4 from
        // every query, and topK 6 leaves two -1 slots. Each query's cutoff selects a different
        // prefix of the candidates 1, 4, 9, 16.
        float[] choices = {Float.NEGATIVE_INFINITY, 0, 1, 1.5f, 4, 4.5f, 9, 10, 16, 16.5f, Float.POSITIVE_INFINITY};
        int[] prefixes = {0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4};
        float[] exact = {1, 4, 9, 16};
        int queries = 22;
        int topK = 6;
        ByteBuffer base = floats(1, 5, 5, 2, 2, 2, 1, 4);
        ByteBuffer query = bytes(queries * 2 * Float.BYTES);
        ByteBuffer cutoffs = bytes(queries * Float.BYTES);
        for (int q = 0; q < queries; q++) {
            query.putFloat(q * 2 * Float.BYTES, 1).putFloat((q * 2 + 1) * Float.BYTES, 2);
            cutoffs.putFloat(q * Float.BYTES, choices[q % choices.length]);
        }
        ByteBuffer ids = bytes(queries * topK * Long.BYTES);
        ByteBuffer distances = bytes(queries * topK * Float.BYTES);
        ByteBuffer direct = bytes(queries * topK * Float.BYTES);
        for (int i = 0; i < queries * topK; i++) {
            direct.putFloat(i * Float.BYTES, -7);
        }
        ByteBuffer counts = bytes(queries * Integer.BYTES);
        Knowhere.bruteForceBatchedL2Rescored(DType.FLOAT32, base, 4, query, queries, 2, topK,
                cutoffs, ids, distances, direct, counts, L2);
        for (int q = 0; q < queries; q++) {
            int count = counts.getInt(q * Integer.BYTES);
            assertEquals("query " + q, prefixes[q % choices.length], count);
            for (int i = 0; i < topK; i++) {
                // Only the prefix is computed; the other slots keep their contents.
                assertEquals("query " + q + " slot " + i, i < count ? exact[i] : -7f,
                        direct.getFloat((q * topK + i) * Float.BYTES), 0f);
            }
            assertEquals(-1, ids.getLong((q * topK + 4) * Long.BYTES));
        }
        // Without cutoffs every candidate is computed, up to the -1 slots.
        Knowhere.bruteForceBatchedL2Rescored(DType.FLOAT32, base, 4, query, queries, 2, topK,
                null, ids, distances, direct, counts, L2);
        for (int q = 0; q < queries; q++) {
            assertEquals(4, counts.getInt(q * Integer.BYTES));
            for (int i = 0; i < topK; i++) {
                assertEquals(i < 4 ? exact[i] : -7f, direct.getFloat((q * topK + i) * Float.BYTES), 0f);
            }
        }
        // An empty base gives only -1 slots.
        Knowhere.bruteForceBatchedL2Rescored(DType.FLOAT32, base, 0, query, queries, 2, topK,
                null, ids, distances, direct, counts, L2);
        for (int q = 0; q < queries; q++) {
            assertEquals(0, counts.getInt(q * Integer.BYTES));
            assertEquals(-1, ids.getLong(q * topK * Long.BYTES));
        }
        assertEquals(0, cutoffs.position());
        Knowhere.bruteForceBatchedL2Rescored(DType.FLOAT32, base, 4, bytes(0), 0, 2, topK,
                bytes(0), bytes(0), bytes(0), bytes(0), bytes(0), L2);
    }

    @Test
    public void batchedL2RescoringTakesFloat32AndL2Only() {
        for (final String metric : new String[] {"IP", "COSINE"}) {
            assertThrows(KnowhereException.class, new ThrowingRunnable() {
                public void run() {
                    Knowhere.bruteForceBatchedL2Rescored(DType.FLOAT32, floats(0, 0), 1, floats(0, 0), 1, 2, 1,
                            null, bytes(8), bytes(4), bytes(4), bytes(4), "{\"metric_type\":\"" + metric + "\"}");
                }
            });
        }
        assertThrows(KnowhereException.class, new ThrowingRunnable() {
            public void run() {
                Knowhere.bruteForceBatchedL2Rescored(DType.FLOAT16, bytes(4), 1, bytes(4), 1, 2, 1,
                        null, bytes(8), bytes(4), bytes(4), bytes(4), L2);
            }
        });
        assertThrows(KnowhereException.class, new ThrowingRunnable() {
            public void run() {
                Knowhere.bruteForceBatchedL2Rescored(DType.FLOAT32, floats(0, 0), 1, floats(0, 0), 1, 2, 1,
                        floats(Float.NaN), bytes(8), bytes(4), bytes(4), bytes(4), L2);
            }
        });
        // One int32 count per query.
        assertThrows(KnowhereException.class, new ThrowingRunnable() {
            public void run() {
                Knowhere.bruteForceBatchedL2Rescored(DType.FLOAT32, floats(0, 0), 1, floats(0, 0), 1, 2, 1,
                        null, bytes(8), bytes(4), bytes(4), bytes(2), L2);
            }
        });
        assertThrows(IllegalArgumentException.class, new ThrowingRunnable() {
            public void run() {
                Knowhere.bruteForceBatchedL2Rescored(DType.FLOAT32, floats(0, 0), 1, floats(0, 0), 1, 2, 1,
                        null, bytes(8), bytes(4), ByteBuffer.allocate(4), bytes(4), L2);
            }
        });
        assertThrows(IllegalArgumentException.class, new ThrowingRunnable() {
            public void run() {
                Knowhere.bruteForceBatchedL2Rescored(DType.FLOAT32, floats(0, 0), 1, floats(0, 0), 1, 2, 1,
                        null, bytes(8), bytes(4), bytes(4), bytes(4).asReadOnlyBuffer(), L2);
            }
        });
    }

    @Test
    public void threadPoolsResizeBeforeAndAfterSearches() {
        assertThrows(KnowhereException.class, new ThrowingRunnable() {
            public void run() { Knowhere.resizeSearchThreadPool(0); }
        });
        assertThrows(KnowhereException.class, new ThrowingRunnable() {
            public void run() { Knowhere.resizeBuildThreadPool(-1); }
        });
        Knowhere.resizeSearchThreadPool(3);
        assertEquals(3, Knowhere.searchThreadPoolSize());
        Knowhere.resizeBuildThreadPool(2);
        assertEquals(2, Knowhere.buildThreadPoolSize());
        ByteBuffer ids = bytes(32);
        ByteBuffer distances = bytes(16);
        Knowhere.bruteForce(DType.FLOAT32, floats(0, 0, 1, 0, 0, 2), 3,
                floats(0, 0, 0, 2), 2, 2, 2, null, 0, ids, distances, L2);
        assertEquals(0, ids.getLong(0));
        assertEquals(1, ids.getLong(8));
        assertEquals(2, ids.getLong(16));
        assertEquals(0, ids.getLong(24));
        assertEquals(4f, distances.getFloat(12), 0f);
        Knowhere.resizeSearchThreadPool(5);
        assertEquals(5, Knowhere.searchThreadPoolSize());
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
    public void addressBuildRequiresAnAddressAndANonNegativeLength() {
        try (final KnowhereIndex index = Knowhere.createIndex("FLAT", DType.FLOAT32, Knowhere.currentIndexVersion())) {
            assertThrows(IllegalArgumentException.class, new ThrowingRunnable() {
                public void run() { index.build(0L, 24L, 3, 2, L2); }
            });
            assertThrows(IllegalArgumentException.class, new ThrowingRunnable() {
                public void run() { index.build(8L, -1L, 3, 2, L2); }
            });
            assertThrows(NullPointerException.class, new ThrowingRunnable() {
                public void run() { index.build(8L, 24L, 3, 2, null); }
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
