// Copyright (C) 2026 Zilliz. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "io_knowhere_NativeBindings.h"
#include "knowhere/c_api.h"

namespace {
struct PendingException {};

void
CheckJava(JNIEnv* env) {
    if (env->ExceptionCheck()) {
        throw PendingException{};
    }
}

// JNI's modified UTF-8 is not the UTF-8 expected by JSON and filesystem paths.
class Utf8 {
 public:
    Utf8(JNIEnv* env, jstring input) {
        if (input == nullptr) {
            throw std::invalid_argument("A non-null string is required");
        }
        const jsize length = env->GetStringLength(input);
        const jchar* data = env->GetStringChars(input, nullptr);
        CheckJava(env);
        if (data == nullptr) {
            throw std::bad_alloc();
        }
        struct Release {
            JNIEnv* env = nullptr;
            jstring input = nullptr;
            const jchar* data = nullptr;
            ~Release() {
                env->ReleaseStringChars(input, data);
            }
        } release{env, input, data};
        for (jsize i = 0; i < length; ++i) {
            uint32_t value = data[i];
            if (value == 0) {
                throw std::invalid_argument("Embedded NUL is not allowed in native strings");
            }
            if (value >= 0xd800 && value <= 0xdbff) {
                if (i + 1 >= length || data[i + 1] < 0xdc00 || data[i + 1] > 0xdfff) {
                    throw std::invalid_argument("Unpaired UTF-16 surrogate");
                }
                value = 0x10000 + ((value - 0xd800) << 10) + (data[++i] - 0xdc00);
            } else if (value >= 0xdc00 && value <= 0xdfff) {
                throw std::invalid_argument("Unpaired UTF-16 surrogate");
            }
            if (value < 0x80) {
                value_.push_back(static_cast<char>(value));
            } else if (value < 0x800) {
                value_.push_back(static_cast<char>(0xc0 | (value >> 6)));
                value_.push_back(static_cast<char>(0x80 | (value & 0x3f)));
            } else if (value < 0x10000) {
                value_.push_back(static_cast<char>(0xe0 | (value >> 12)));
                value_.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3f)));
                value_.push_back(static_cast<char>(0x80 | (value & 0x3f)));
            } else {
                value_.push_back(static_cast<char>(0xf0 | (value >> 18)));
                value_.push_back(static_cast<char>(0x80 | ((value >> 12) & 0x3f)));
                value_.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3f)));
                value_.push_back(static_cast<char>(0x80 | (value & 0x3f)));
            }
        }
    }
    const char*
    get() const {
        return value_.c_str();
    }

 private:
    std::string value_;
};

jstring
JavaString(JNIEnv* env, const char* value) {
    const size_t length = std::strlen(value);
    if (length > static_cast<size_t>(std::numeric_limits<jsize>::max())) {
        throw std::length_error("Native string exceeds the Java array limit");
    }
    jbyteArray bytes = env->NewByteArray(static_cast<jsize>(length));
    CheckJava(env);
    env->SetByteArrayRegion(bytes, 0, static_cast<jsize>(length), reinterpret_cast<const jbyte*>(value));
    CheckJava(env);
    jclass cls = env->FindClass("java/lang/String");
    CheckJava(env);
    jmethodID constructor = env->GetMethodID(cls, "<init>", "([BLjava/lang/String;)V");
    CheckJava(env);
    jstring encoding = env->NewStringUTF("UTF-8");
    CheckJava(env);
    auto result = static_cast<jstring>(env->NewObject(cls, constructor, bytes, encoding));
    CheckJava(env);
    return result;
}

void
CheckStatus(JNIEnv* env, int32_t status) {
    if (status == KNOWHERE_SUCCESS) {
        return;
    }
    jstring message = JavaString(env, knowhere_last_error());
    jclass cls = env->FindClass("io/knowhere/KnowhereException");
    CheckJava(env);
    jmethodID constructor = env->GetMethodID(cls, "<init>", "(ILjava/lang/String;)V");
    CheckJava(env);
    auto error = static_cast<jthrowable>(env->NewObject(cls, constructor, static_cast<jint>(status), message));
    CheckJava(env);
    env->Throw(error);
    throw PendingException{};
}

void
Throw(JNIEnv* env, const char* type, const char* message) {
    if (!env->ExceptionCheck()) {
        jclass cls = env->FindClass(type);
        if (cls != nullptr) {
            env->ThrowNew(cls, message);
        }
    }
}

template <typename T, typename Function>
T
Call(JNIEnv* env, T failure, Function function) noexcept {
    try {
        return function();
    } catch (const PendingException&) {
        return failure;
    } catch (const std::bad_alloc&) {
        Throw(env, "java/lang/OutOfMemoryError", "Native JNI allocation failed");
    } catch (const std::invalid_argument& error) {
        Throw(env, "java/lang/IllegalArgumentException", error.what());
    } catch (const std::exception& error) {
        Throw(env, "java/lang/RuntimeException", error.what());
    } catch (...) {
        Throw(env, "java/lang/RuntimeException", "Unexpected native JNI exception");
    }
    return failure;
}

struct Region {
    void* data = nullptr;
    uint64_t bytes = 0;
    Region(JNIEnv* env, jobject buffer, bool writable = false) {
        if (buffer == nullptr) {
            throw std::invalid_argument("A direct ByteBuffer is required");
        }
        const jlong capacity = env->GetDirectBufferCapacity(buffer);
        data = env->GetDirectBufferAddress(buffer);
        CheckJava(env);
        if (capacity < 0 || (capacity > 0 && data == nullptr)) {
            throw std::invalid_argument("A direct ByteBuffer is required");
        }
        if (writable) {
            jclass cls = env->GetObjectClass(buffer);
            CheckJava(env);
            jmethodID method = env->GetMethodID(cls, "isReadOnly", "()Z");
            CheckJava(env);
            const jboolean read_only = env->CallBooleanMethod(buffer, method);
            CheckJava(env);
            if (read_only) {
                throw std::invalid_argument("The output buffer is read-only");
            }
        }
        bytes = static_cast<uint64_t>(capacity);
    }
};

knowhere_vectors
Vectors(JNIEnv* env, jobject buffer, jlong rows, jint dimension, jint dtype) {
    Region region(env, buffer);
    return {region.data, region.bytes, rows, dimension, dtype};
}

knowhere_bitset
Bitset(JNIEnv* env, jobject buffer, jlong bits) {
    if (buffer == nullptr && bits == 0) {
        return {nullptr, 0, 0};
    }
    Region region(env, buffer);
    return {static_cast<const uint8_t*>(region.data), region.bytes, bits};
}

knowhere_search_result
Result(JNIEnv* env, jobject ids, jobject distances, jint top_k) {
    Region id_region(env, ids, true);
    Region distance_region(env, distances, true);
    return {static_cast<int64_t*>(id_region.data), id_region.bytes, static_cast<float*>(distance_region.data),
            distance_region.bytes, top_k};
}

uint64_t
Unsigned(jlong value) {
    if (value < 0) {
        throw std::invalid_argument("Length, offset and entry index must be non-negative");
    }
    return static_cast<uint64_t>(value);
}
}  // namespace

extern "C" {
JNIEXPORT jint JNICALL
Java_io_knowhere_NativeBindings_cAbiVersion(JNIEnv*, jclass) {
    return knowhere_c_abi_version();
}
JNIEXPORT jint JNICALL
Java_io_knowhere_NativeBindings_minimumIndexVersion(JNIEnv*, jclass) {
    return knowhere_index_version_minimum();
}
JNIEXPORT jint JNICALL
Java_io_knowhere_NativeBindings_currentIndexVersion(JNIEnv*, jclass) {
    return knowhere_index_version_current();
}
JNIEXPORT jint JNICALL
Java_io_knowhere_NativeBindings_maximumIndexVersion(JNIEnv*, jclass) {
    return knowhere_index_version_maximum();
}
JNIEXPORT jlong JNICALL
Java_io_knowhere_NativeBindings_indexCreate(JNIEnv* env, jclass, jstring type, jint dtype, jint version) {
    return Call(env, jlong{0}, [&]() -> jlong {
        Utf8 name(env, type);
        knowhere_index_handle handle = 0;
        CheckStatus(env, knowhere_index_create(name.get(), dtype, version, &handle));
        return static_cast<jlong>(handle);
    });
}
JNIEXPORT void JNICALL
Java_io_knowhere_NativeBindings_indexDestroy(JNIEnv* env, jclass, jlong handle) {
    Call(env, 0, [&] {
        CheckStatus(env, knowhere_index_destroy(handle));
        return 0;
    });
}
JNIEXPORT void JNICALL
Java_io_knowhere_NativeBindings_indexBuild(JNIEnv* env, jclass, jlong handle, jobject data, jlong rows, jint dimension,
                                           jint dtype, jstring parameters) {
    Call(env, 0, [&] {
        auto vectors = Vectors(env, data, rows, dimension, dtype);
        Utf8 config(env, parameters);
        CheckStatus(env, knowhere_index_build(handle, &vectors, config.get()));
        return 0;
    });
}
JNIEXPORT void JNICALL
Java_io_knowhere_NativeBindings_indexBuildFromFile(JNIEnv* env, jclass, jlong handle, jstring parameters) {
    Call(env, 0, [&] {
        Utf8 config(env, parameters);
        CheckStatus(env, knowhere_index_build(handle, nullptr, config.get()));
        return 0;
    });
}
JNIEXPORT void JNICALL
Java_io_knowhere_NativeBindings_indexSearch(JNIEnv* env, jclass, jlong handle, jobject data, jlong rows, jint dimension,
                                            jint dtype, jint top_k, jobject excluded, jlong bits, jobject ids,
                                            jobject distances, jstring parameters) {
    Call(env, 0, [&] {
        auto vectors = Vectors(env, data, rows, dimension, dtype);
        auto bitset = Bitset(env, excluded, bits);
        auto result = Result(env, ids, distances, top_k);
        Utf8 config(env, parameters);
        CheckStatus(env, knowhere_index_search(handle, &vectors, &bitset, &result, config.get()));
        return 0;
    });
}
JNIEXPORT jlongArray JNICALL
Java_io_knowhere_NativeBindings_indexInfo(JNIEnv* env, jclass, jlong handle) {
    return Call(env, static_cast<jlongArray>(nullptr), [&] {
        int64_t rows = 0, dimensions = 0;
        CheckStatus(env, knowhere_index_info(handle, &rows, &dimensions));
        const jlong values[] = {rows, dimensions};
        jlongArray result = env->NewLongArray(2);
        CheckJava(env);
        env->SetLongArrayRegion(result, 0, 2, values);
        CheckJava(env);
        return result;
    });
}
JNIEXPORT jlong JNICALL
Java_io_knowhere_NativeBindings_indexSerialize(JNIEnv* env, jclass, jlong handle) {
    return Call(env, jlong{0}, [&]() -> jlong {
        knowhere_binary_set_handle output = 0;
        CheckStatus(env, knowhere_index_serialize(handle, &output));
        return static_cast<jlong>(output);
    });
}
JNIEXPORT void JNICALL
Java_io_knowhere_NativeBindings_indexDeserialize(JNIEnv* env, jclass, jlong handle, jlong data, jstring parameters) {
    Call(env, 0, [&] {
        Utf8 config(env, parameters);
        CheckStatus(env, knowhere_index_deserialize(handle, data, config.get()));
        return 0;
    });
}
JNIEXPORT void JNICALL
Java_io_knowhere_NativeBindings_bruteForce(JNIEnv* env, jclass, jint dtype, jobject base_data, jlong base_rows,
                                           jobject query_data, jlong query_rows, jint dimension, jint top_k,
                                           jobject excluded, jlong bits, jobject ids, jobject distances,
                                           jstring parameters) {
    Call(env, 0, [&] {
        auto base = Vectors(env, base_data, base_rows, dimension, dtype);
        auto queries = Vectors(env, query_data, query_rows, dimension, dtype);
        auto bitset = Bitset(env, excluded, bits);
        auto result = Result(env, ids, distances, top_k);
        Utf8 config(env, parameters);
        CheckStatus(env, knowhere_bruteforce(&base, &queries, &bitset, &result, config.get()));
        return 0;
    });
}
JNIEXPORT jlong JNICALL
Java_io_knowhere_NativeBindings_binarySetCreate(JNIEnv* env, jclass) {
    return Call(env, jlong{0}, [&]() -> jlong {
        knowhere_binary_set_handle handle = 0;
        CheckStatus(env, knowhere_binary_set_create(&handle));
        return static_cast<jlong>(handle);
    });
}
JNIEXPORT void JNICALL
Java_io_knowhere_NativeBindings_binarySetDestroy(JNIEnv* env, jclass, jlong handle) {
    Call(env, 0, [&] {
        CheckStatus(env, knowhere_binary_set_destroy(handle));
        return 0;
    });
}
JNIEXPORT jlong JNICALL
Java_io_knowhere_NativeBindings_binarySetCount(JNIEnv* env, jclass, jlong handle) {
    return Call(env, jlong{0}, [&]() -> jlong {
        uint64_t output = 0;
        CheckStatus(env, knowhere_binary_set_count(handle, &output));
        if (output > static_cast<uint64_t>(std::numeric_limits<jlong>::max())) {
            throw std::length_error("BinarySet count exceeds the Java long limit");
        }
        return static_cast<jlong>(output);
    });
}
JNIEXPORT jstring JNICALL
Java_io_knowhere_NativeBindings_binarySetName(JNIEnv* env, jclass, jlong handle, jlong index) {
    return Call(env, static_cast<jstring>(nullptr), [&] {
        const uint64_t ordinal = Unsigned(index);
        uint64_t required = 0;
        CheckStatus(env, knowhere_binary_set_name(handle, ordinal, nullptr, 0, &required));
        if (required > static_cast<uint64_t>(std::numeric_limits<jsize>::max())) {
            throw std::length_error("BinarySet name exceeds the Java string limit");
        }
        std::vector<char> name(static_cast<size_t>(required));
        CheckStatus(env, knowhere_binary_set_name(handle, ordinal, name.data(), name.size(), &required));
        return JavaString(env, name.data());
    });
}
JNIEXPORT jlong JNICALL
Java_io_knowhere_NativeBindings_binarySetLength(JNIEnv* env, jclass, jlong handle, jstring name) {
    return Call(env, jlong{0}, [&]() -> jlong {
        Utf8 key(env, name);
        uint64_t output = 0;
        CheckStatus(env, knowhere_binary_set_length(handle, key.get(), &output));
        if (output > static_cast<uint64_t>(std::numeric_limits<jlong>::max())) {
            throw std::length_error("BinarySet length exceeds the Java long limit");
        }
        return static_cast<jlong>(output);
    });
}
JNIEXPORT void JNICALL
Java_io_knowhere_NativeBindings_binarySetAllocate(JNIEnv* env, jclass, jlong handle, jstring name, jlong length) {
    Call(env, 0, [&] {
        Utf8 key(env, name);
        CheckStatus(env, knowhere_binary_set_allocate(handle, key.get(), Unsigned(length)));
        return 0;
    });
}
JNIEXPORT void JNICALL
Java_io_knowhere_NativeBindings_binarySetWrite(JNIEnv* env, jclass, jlong handle, jstring name, jlong offset,
                                               jobject data) {
    Call(env, 0, [&] {
        Utf8 key(env, name);
        Region region(env, data);
        CheckStatus(env, knowhere_binary_set_write(handle, key.get(), Unsigned(offset), region.data, region.bytes));
        return 0;
    });
}
JNIEXPORT void JNICALL
Java_io_knowhere_NativeBindings_binarySetRead(JNIEnv* env, jclass, jlong handle, jstring name, jlong offset,
                                              jobject data) {
    Call(env, 0, [&] {
        Utf8 key(env, name);
        Region region(env, data, true);
        CheckStatus(env, knowhere_binary_set_read(handle, key.get(), Unsigned(offset), region.data, region.bytes));
        return 0;
    });
}
}  // extern "C"
