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

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Properties;

/** Loads a matching platform resource package, or an explicitly selected local JNI library. */
final class NativeLibraryLoader {
    private static boolean loaded;

    private NativeLibraryLoader() {
    }

    static synchronized void load() {
        if (loaded) {
            return;
        }
        String explicit = System.getProperty("knowhere.native.path");
        if (explicit != null) {
            Path path = Paths.get(explicit);
            if (!path.isAbsolute() || !Files.isRegularFile(path)) {
                throw new UnsatisfiedLinkError("knowhere.native.path must name an existing absolute library path");
            }
            System.load(path.toString());
        } else {
            // Class initialization cannot propagate checked I/O or digest exceptions.
            try {
                loadResources();
            } catch (IOException | NoSuchAlgorithmException failure) {
                UnsatisfiedLinkError error = new UnsatisfiedLinkError("Cannot load the Knowhere native package: " + failure);
                error.initCause(failure);
                throw error;
            }
        }
        loaded = true;
    }

    private static String platform() {
        String os = System.getProperty("os.name").toLowerCase(Locale.ROOT);
        String arch = System.getProperty("os.arch").toLowerCase(Locale.ROOT);
        if (!os.equals("linux")) {
            throw new UnsatisfiedLinkError("No bundled Knowhere native package for " + os);
        }
        if (arch.equals("amd64") || arch.equals("x86_64")) {
            return "linux-x86_64";
        }
        if (arch.equals("aarch64") || arch.equals("arm64")) {
            return "linux-aarch64";
        }
        throw new UnsatisfiedLinkError("Unsupported Knowhere architecture: " + arch);
    }

    private static InputStream resource(String name) throws IOException {
        InputStream input = NativeLibraryLoader.class.getResourceAsStream(name);
        if (input == null) {
            throw new IOException("Missing " + name + "; add the matching native platform JAR");
        }
        return input;
    }

    private static String hex(byte[] bytes) {
        StringBuilder result = new StringBuilder(bytes.length * 2);
        for (byte value : bytes) {
            result.append(Character.forDigit((value >>> 4) & 15, 16));
            result.append(Character.forDigit(value & 15, 16));
        }
        return result.toString();
    }

    private static void loadResources() throws IOException, NoSuchAlgorithmException {
        String resourceRoot = "/native/knowhere/1/" + platform() + "/";
        byte[] manifest;
        try (InputStream input = resource(resourceRoot + "manifest.properties")) {
            manifest = input.readAllBytes();
        }
        Properties properties = new Properties();
        properties.load(new ByteArrayInputStream(manifest));
        String libraries = properties.getProperty("libraries");
        if (!"1".equals(properties.getProperty("cAbiVersion")) || libraries == null || libraries.isEmpty()) {
            throw new IOException("Invalid Knowhere native manifest");
        }
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        String manifestHash = hex(digest.digest(manifest));
        Path cache = Paths.get(System.getProperty("java.io.tmpdir"), "knowhere-jni", "1", platform(), manifestHash);
        Files.createDirectories(cache);
        // Each JVM owns an extraction directory; no process can observe another's partial files.
        Path directory = Files.createTempDirectory(cache, "process-");
        directory.toFile().deleteOnExit();
        List<Path> extracted = new ArrayList<Path>();
        for (String library : libraries.split(",", -1)) {
            if (!library.matches("[A-Za-z0-9_+.-]+") || library.equals(".") || library.equals("..")) {
                throw new IOException("Invalid library name in native manifest");
            }
            String expected = properties.getProperty("sha256." + library);
            if (expected == null || !expected.matches("[a-f0-9]{64}")) {
                throw new IOException("Missing checksum for " + library);
            }
            Path temporary = Files.createTempFile(directory, "extract-", ".tmp");
            temporary.toFile().deleteOnExit();
            digest.reset();
            try (InputStream input = resource(resourceRoot + library);
                    java.io.OutputStream output = Files.newOutputStream(temporary)) {
                byte[] buffer = new byte[65536];
                int count;
                while ((count = input.read(buffer)) != -1) {
                    digest.update(buffer, 0, count);
                    output.write(buffer, 0, count);
                }
            }
            if (!expected.equals(hex(digest.digest()))) {
                throw new IOException("Checksum mismatch for " + library);
            }
            Path destination = directory.resolve(library);
            Files.move(temporary, destination, StandardCopyOption.ATOMIC_MOVE);
            destination.toFile().deleteOnExit();
            extracted.add(destination);
        }
        for (Path library : extracted) {
            System.load(library.toAbsolutePath().toString());
        }
    }
}
