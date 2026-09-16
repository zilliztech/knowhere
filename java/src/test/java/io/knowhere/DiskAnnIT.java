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

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;
import org.junit.Test;
import static org.junit.Assert.*;

/** Explicitly run with -Dtest=KnowhereTest,DiskAnnIT against a DiskANN-enabled library. */
public class DiskAnnIT {
    @Test
    public void buildAndReloadLocalDiskIndexThroughJni() throws IOException {
        Path directory = Files.createTempDirectory("knowhere-java-diskann-");
        try {
            int rows = 1000;
            int dimensions = 32;
            int queries = 20;
            int topK = 5;
            Path source = directory.resolve("raw.bin");
            Path prefix = directory.resolve("index");
            ByteBuffer base = KnowhereTest.bytes(rows * dimensions * Float.BYTES);
            ByteBuffer fileData = ByteBuffer.allocate(8 + rows * dimensions * Float.BYTES)
                    .order(ByteOrder.nativeOrder());
            fileData.putInt(rows).putInt(dimensions);
            Random random = new Random(42);
            for (int row = 0; row < rows; row++) {
                for (int column = 0; column < dimensions; column++) {
                    float value = random.nextFloat();
                    base.putFloat(value);
                    fileData.putFloat(value);
                }
            }
            base.flip();
            fileData.flip();
            try (FileChannel file = FileChannel.open(source, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)) {
                while (fileData.hasRemaining()) {
                    file.write(fileData);
                }
            }
            String common = "\"dim\":32,\"metric_type\":\"L2\",\"index_prefix\":" + quoted(prefix);
            String build = "{" + common + ",\"data_path\":" + quoted(source)
                    + ",\"max_degree\":32,\"search_list_size\":128,\"pq_code_budget_gb\":0.000015"
                    + ",\"build_dram_budget_gb\":0.25,\"search_cache_budget_gb\":0,\"disk_pq_dims\":0} ";
            BinarySet serialized;
            try (KnowhereIndex builder = Knowhere.createIndex("DISKANN", DType.FLOAT32, Knowhere.currentIndexVersion())) {
                builder.buildFromFile(build);
                serialized = builder.serialize();
            }
            try (BinarySet data = serialized;
                    KnowhereIndex loaded = Knowhere.createIndex("DISKANN", DType.FLOAT32, Knowhere.currentIndexVersion())) {
                Files.delete(source);
                loaded.deserialize(data, "{" + common + ",\"search_cache_budget_gb\":0,\"warm_up\":false}");
                data.close();
                assertEquals(rows, loaded.rows());
                assertEquals(dimensions, loaded.dimensions());
                ByteBuffer query = KnowhereTest.bytes(queries * dimensions * Float.BYTES);
                for (int row = 0; row < queries; row++) {
                    for (int column = 0; column < dimensions; column++) {
                        query.putFloat(base.getFloat((row * 17 * dimensions + column) * Float.BYTES) + 0.001f);
                    }
                }
                query.flip();
                String search = "{" + common + ",\"search_list_size\":1000,\"beamwidth\":4}";
                assertSearch(loaded, base, rows, query, queries, dimensions, topK, null, search);
                ByteBuffer excluded = KnowhereTest.bytes((rows + 7) / 8);
                for (int row = 0; row < 100; row++) {
                    excluded.put(row / 8, (byte) (excluded.get(row / 8) | (1 << (row % 8))));
                }
                assertSearch(loaded, base, rows, query, queries, dimensions, topK, excluded, search);
            }
        } finally {
            List<Path> paths = new ArrayList<Path>();
            try (Stream<Path> walk = Files.walk(directory)) {
                Iterator<Path> iterator = walk.iterator();
                while (iterator.hasNext()) {
                    paths.add(iterator.next());
                }
            }
            Collections.sort(paths, Collections.reverseOrder());
            for (Path path : paths) {
                Files.delete(path);
            }
        }
    }

    private static void assertSearch(KnowhereIndex index, ByteBuffer base, int rows, ByteBuffer query,
            int queries, int dimensions, int topK, ByteBuffer excluded, String parameters) {
        ByteBuffer expectedIds = KnowhereTest.bytes(queries * topK * Long.BYTES);
        ByteBuffer expectedDistances = KnowhereTest.bytes(queries * topK * Float.BYTES);
        ByteBuffer ids = KnowhereTest.bytes(expectedIds.capacity());
        ByteBuffer distances = KnowhereTest.bytes(expectedDistances.capacity());
        long bits = excluded == null ? 0 : rows;
        Knowhere.bruteForce(DType.FLOAT32, base, rows, query, queries, dimensions, topK,
                excluded, bits, expectedIds, expectedDistances, "{\"metric_type\":\"L2\"}");
        index.search(query, queries, dimensions, topK, excluded, bits, ids, distances, parameters);
        int hits = 0;
        for (int row = 0; row < queries; row++) {
            float previousDistance = -1f;
            for (int k = 0; k < topK; k++) {
                int offset = row * topK + k;
                long id = ids.getLong(offset * Long.BYTES);
                assertTrue("Invalid row ID: " + id, id >= 0 && id < rows);
                if (excluded != null) {
                    assertEquals("Excluded row returned: " + id,
                            0, excluded.get((int) id / 8) & (1 << ((int) id % 8)));
                }
                for (int earlier = 0; earlier < k; earlier++) {
                    assertNotEquals("Duplicate row ID", id, ids.getLong((row * topK + earlier) * Long.BYTES));
                }
                float distance = distances.getFloat(offset * Float.BYTES);
                double squaredDistance = 0;
                for (int column = 0; column < dimensions; column++) {
                    double difference = (double) query.getFloat((row * dimensions + column) * Float.BYTES)
                            - base.getFloat(((int) id * dimensions + column) * Float.BYTES);
                    squaredDistance += difference * difference;
                }
                assertEquals("Distance does not match row " + id, squaredDistance, distance, 0.0001);
                assertTrue("Distances are not sorted", distance >= previousDistance);
                previousDistance = distance;
                for (int truth = 0; truth < topK; truth++) {
                    if (id == expectedIds.getLong((row * topK + truth) * Long.BYTES)) {
                        hits++;
                        break;
                    }
                }
            }
        }
        // DiskANN is approximate; isolated points need not be reachable from its graph entry point.
        // Follow the native DiskANN tests: compare recall with brute force on distributed samples.
        assertTrue("DiskANN recall: " + hits + "/" + queries * topK, hits >= queries * topK * 0.95);
        System.out.println("DiskANN " + (excluded == null ? "unfiltered" : "filtered")
                + " recall: " + hits + "/" + queries * topK);
    }

    private static String quoted(Path path) {
        return "\"" + path.toString().replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
    }
}
