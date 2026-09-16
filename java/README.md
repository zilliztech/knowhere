# Knowhere C and Java bindings

The optional bindings expose index creation, build, search, serialization,
deserialization and brute-force search. `libknowhere_jni` calls the standalone
`libknowhere_c` ABI, which calls Knowhere. The Java API targets Java 11 and has no
Spark, Scala, Arrow or Hadoop dependency. Both build options default to off.

The initial interface uses contiguous dense vectors: float32, binary, float16,
bfloat16 and int8. Index type, element encoding and index format version are
separate inputs. Available combinations depend on the compiled Knowhere backend.
Sparse layouts and the Aquila clustering, deduplication and Isolation Forest
operations are outside this interface.

## Build and test

Use the repository's normal C++20/Conan 2 prerequisites, plus a JDK 11 or later
and Maven. On Linux the platform packaging step also requires Python 3,
`readelf`, `ldd` and `patchelf`. Configure `JAVA_HOME` when CMake cannot find the
selected JDK. The example uses Conan 2.28.1 and CMake 3.28.1. Older dependency
recipes may fail with CMake 4.

From the repository root:

```sh
conan install . -of build --build=missing \
  -s compiler.cppstd=20 -s:b compiler.cppstd=20 \
  -o '&:with_c_api=True' -o '&:with_jni=True' -o '&:with_c_api_tests=True'
cmake -S . -B build/Release -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/build/Release/generators/conan_toolchain.cmake"
cmake --build build/Release --parallel 2
. build/Release/generators/conanrun.sh
ctest --test-dir build/Release --output-on-failure
LD_PRELOAD="$JAVA_HOME/lib/libjsig.so" mvn -f java/pom.xml test -DargLine=-Xcheck:jni \
  -Dknowhere.native.path="$PWD/build/Release/java/libknowhere_jni.so"
```

Activate the generated Conan run environment when testing or collecting libraries
from a build or installation tree. Some Conan shared libraries have transitive
dependencies without their own RUNPATH. The packaged-JAR smoke tests deliberately
run without that build environment.

For a C-only build, omit `with_jni`. The public header is
`include/knowhere/c_api.h`; link with `libknowhere_c`. Neither a JDK nor JNI
headers are needed by the C library. To include Linux DiskANN, add
`-o '&:with_diskann=True'` to the Conan command. DiskANN indexes receive a
`LocalFileManager`; use `buildFromFile(parameters)` with `data_path`,
`index_prefix` and the usual DiskANN configuration. Its serialized BinarySet
does not contain the generated files: loading uses the same local `index_prefix`.
Those files must remain available until their indexes close. A normal in-memory
index rejects `buildFromFile`; DiskANN rejects the vector-buffer `build` entry.
Run the DiskANN JNI test explicitly with
`mvn -f java/pom.xml test -Dtest=KnowhereTest,DiskAnnIT -DargLine=-Xcheck:jni`
and the same `knowhere.native.path` property and signal-chaining environment. On macOS, use the `.dylib`
extension for the development library.

The shared Folly dependency installs a SIGPIPE handler at library load time.
For HotSpot on Linux, preload the **same JRE's** `libjsig.so` before starting
Java, including test JVMs. This enables HotSpot's supported
[signal chaining](https://docs.oracle.com/en/java/javase/15/vm/signal-chaining.html)
and preserves the JVM's handler. It is a runtime requirement, not an optional
way to suppress JNI diagnostics. Do not bundle another JDK's `libjsig` into the
platform JAR. Consumers use:

```sh
LD_PRELOAD="$JAVA_HOME/lib/libjsig.so" java -cp 'knowhere-jni.jar:knowhere-jni-linux-aarch64.jar:app.jar' com.example.Main
```

For macOS development, use `DYLD_FORCE_FLAT_NAMESPACE=1` and
`DYLD_INSERT_LIBRARIES="$JAVA_HOME/lib/libjsig.dylib"` before starting Java.
Loading `libjsig` after JVM initialization does not establish signal chaining.
JDK 16 and later may report Folly's use of the deprecated `signal()` chaining
entry point; the runtime checks still verify that JVM handlers remain installed.

## Supported Java versions

The API JAR contains Java 11 bytecode (`javac --release 11`) and works unchanged
on later runtimes; the packaged JARs are verified on Temurin 11, 17, 21 and 25.
On JDK 24 and later, loading a JNI library is a restricted operation
([JEP 472](https://openjdk.org/jeps/472)). Start the JVM with
`--enable-native-access=ALL-UNNAMED` (or name the module that contains
`io.knowhere` when it is on the module path); otherwise the JVM prints a warning
at load time, and a future release will refuse the call.

For AddressSanitizer, use a separate output directory and add
`-o '&:with_asan=True'`. Do not package sanitizer libraries for release use.

CMake generates the JNI header directly from the Java declarations with
`javac --release 11 -h`, then compiles the implementation against that header.
After the tests above pass, build the ordinary Java JAR and platform JAR
separately without repeating those tests:

```sh
. build/Release/generators/conanrun.sh
python3 java/scripts/bundle_native.py \
  --library "$PWD/build/Release/java/libknowhere_jni.so" \
  --output "$PWD/java/target/native-resources" --platform linux-aarch64
mvn -f java/pom.xml package -Dnative.platform=linux-aarch64 -DskipTests
```

Use `linux-x86_64` on x86-64. Build on the target architecture. The packager
checks ELF architecture and dependency resolution, collects the dynamic library
closure, sets copied libraries' RPATH to `$ORIGIN`, and writes SHA-256 checksums.
It leaves the original build libraries untouched. The platform JAR contains
`native/knowhere/1/<platform>/manifest.properties` and the libraries. The API JAR
excludes those resources. Preserve dependencies' licenses with the package and
resolve any missing-license report before distributing it. The build also
collects licenses for statically linked and header-only dependencies; the ELF
dependency list alone cannot identify them.

The pinned `milvus-common/1.0.0-b589c5a` source currently omits the LICENSE and
NOTICE files referenced by its recipe and source headers. Its bundled
`flat_hash_map` additionally declares the Boost Software License. This missing
upstream material is reported in `missing-licenses.txt`; supplying only an
Apache license text does not resolve that report. Complete the upstream
license package before distributing native artifacts.

The API loader validates the manifest and all library checksums, extracts the
complete closure, and loads only the manifest's JNI entry library. The system
linker loads its dependencies as one group; preloading individual dependencies
is incorrect when a dependency references symbols exported by Knowhere itself. Each JVM extracts into its own temporary directory. There is
no silent fallback to `java.library.path`. An explicit absolute
`-Dknowhere.native.path=...` selects a local development library instead of the
packaged resources. It is not needed by consumers of the two JARs.

`io.knowhere.PackageSmoke` in the test sources uses only the API, platform JAR
and JRE, with the JRE's signal-chaining library preloaded. Run it in an environment without the source tree, Conan cache or build
library paths to verify the resulting package. Its checks perform real FLAT
build/serialize/deserialize/search and brute-force calls.

The [C and Java bindings workflow](../.github/workflows/jni.yaml) runs the
Linux x86-64 native tests, DiskANN round trip and clean JRE 11/17/21/25 package
checks. It tests the bindings independently of any artifact publication.

## Java usage

```java
ByteBuffer vectors = ByteBuffer.allocateDirect(24).order(ByteOrder.nativeOrder());
vectors.putFloat(0).putFloat(0).putFloat(1).putFloat(0).putFloat(0).putFloat(2).flip();
ByteBuffer query = ByteBuffer.allocateDirect(8).order(ByteOrder.nativeOrder());
ByteBuffer ids = ByteBuffer.allocateDirect(16).order(ByteOrder.nativeOrder());
ByteBuffer distances = ByteBuffer.allocateDirect(8).order(ByteOrder.nativeOrder());
String parameters = "{\"metric_type\":\"L2\"}";
try (KnowhereIndex index = Knowhere.createIndex(
        "FLAT", DType.FLOAT32, Knowhere.currentIndexVersion())) {
    index.build(vectors, 3, 2, parameters);
    index.search(query, 1, 2, 2, null, 0, ids, distances, parameters);
    // IDs [0, 1], squared L2 distances [0, 1].
    try (BinarySet serialized = index.serialize()) {
        // Persist each name and length, then transfer bounded direct-buffer chunks.
    }
}
```

Imports are `java.nio.ByteBuffer`, `java.nio.ByteOrder` and `io.knowhere.*`.
Search results use int64 row IDs and float32 distances. `topK` is authoritative;
a different `k` in the JSON is rejected. Other index/metric parameters retain
Knowhere semantics. For example, L2 returns squared distances. A set bit in the
optional bitmap excludes that public row ID; bit zero is the least significant
bit of byte zero. The bitmap must cover all base rows. Brute-force IDs start at
zero for the supplied base batch.

## Memory, lifecycle and errors

- Numeric buffers must be direct, naturally aligned and in native byte order.
  JNI uses only `position()` through `limit()` and leaves both unchanged. Output
  buffers must be writable. A zero-query search writes nothing. Shape, capacity,
  alignment and integer multiplication are validated before accessing data.
- Calls are synchronous. Keep borrowed memory alive and prevent concurrent
  buffer modification until return. Build retains an owned copy of its input;
  deserialization retains an independent copy of the BinarySet. The caller may
  reuse those inputs after return. Query buffers are borrowed, not retained.
- Indexes and BinarySets implement `AutoCloseable`; use try-with-resources.
  `close()` is idempotent and later calls fail. Native handles are typed registry
  identifiers, not exposed pointers, and are never reused. Calls on one resource
  are serialized. An already-started C call retains ownership across concurrent
  destruction.
- Initialize an index once using build or deserialize. If the engine fails
  during initialization, close and recreate it. A Java/C argument validation
  failure before the engine call does not initialize the resource.
- BinarySet preserves named blobs and 64-bit lengths. Allocate an entry, then
  write/read bounded chunks at 64-bit offsets. A single Java `ByteBuffer` still
  has Java's capacity limit; large blobs do not require a single Java array.
- C functions return status codes and a thread-local `knowhere_last_error()`.
  Read the message before another status-returning call on that thread. JNI
  translates these to `KnowhereException` with `code()` and a message; Java
  argument/state errors use the standard exceptions. C++ exceptions cannot
  cross either native boundary. An error does not produce an empty success.
- ABI version 1 describes the C interface. Maven/API version and Knowhere index
  format version are independent. Supply the original compatible index format
  version when restoring saved index blobs; the current version is not evidence
  that every historical Milvus index format is readable.

## Verification coverage

The tests exercise five dtypes with FLAT/BIN_FLAT and brute force, HNSW and
IVF_FLAT round trips, filtering, exact fixed examples, invalid inputs, Unicode
blob names, input retention, chunk bounds, double close and concurrent close.
DiskANN has a separate local-file test when enabled. It checks recall against
brute force on uniformly distributed samples, plus IDs, distances and exclusions;
its approximate graph does not guarantee exact neighbors for isolated points.
The packager tests compile
small real ELF libraries to exercise dependency resolution and relocation;
those tests do not substitute for the Knowhere native tests or packaged JNI
smoke test. Only combinations that pass the native tests on a target platform
should be advertised in a released artifact.
