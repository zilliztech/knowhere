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
            Path source = directory.resolve("raw.bin");
            Path prefix = directory.resolve("index");
            ByteBuffer fileData = ByteBuffer.allocate(8 + rows * dimensions * Float.BYTES)
                    .order(ByteOrder.nativeOrder());
            fileData.putInt(rows).putInt(dimensions);
            Random random = new Random(42);
            for (int row = 0; row < rows; row++) {
                for (int column = 0; column < dimensions; column++) {
                    float value = 0f;
                    if (row == 1 && column == 0) {
                        value = 1f;
                    } else if (row == 2 && column == 1) {
                        value = 2f;
                    } else if (row >= 3) {
                        value = 10f + random.nextFloat();
                    }
                    fileData.putFloat(value);
                }
            }
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
                ByteBuffer query = KnowhereTest.bytes(dimensions * Float.BYTES);
                ByteBuffer ids = KnowhereTest.bytes(16);
                ByteBuffer distances = KnowhereTest.bytes(8);
                String search = "{" + common + ",\"search_list_size\":1000,\"beamwidth\":4}";
                loaded.search(query, 1, dimensions, 2, null, 0, ids, distances, search);
                assertEquals(0, ids.getLong(0));
                assertEquals(1, ids.getLong(8));
                assertEquals(0f, distances.getFloat(0), 0f);
                assertEquals(1f, distances.getFloat(4), 0f);
                ByteBuffer excluded = KnowhereTest.bytes((rows + 7) / 8).put(0, (byte) 1);
                loaded.search(query, 1, dimensions, 2, excluded, rows, ids, distances, search);
                assertEquals(1, ids.getLong(0));
                assertEquals(2, ids.getLong(8));
                assertEquals(1f, distances.getFloat(0), 0f);
                assertEquals(4f, distances.getFloat(4), 0f);
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

    private static String quoted(Path path) {
        return "\"" + path.toString().replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
    }
}
