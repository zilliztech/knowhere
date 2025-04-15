//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.
#ifndef KNOWHERE_KNOWHERE_H
#define KNOWHERE_KNOWHERE_H
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <vector>

#include "io/memory_io.h"
#include "knowhere/utils.h"
namespace knowhere {
template <typename T>
class BloomFilter {
 public:
    BloomFilter() {
    }
    BloomFilter(size_t expected_elements, double false_positive_prob) : n(expected_elements), p(false_positive_prob) {
        m = static_cast<size_t>(-(n * log(p)) / (log(2) * log(2)));
        k = static_cast<int>(m / n * log(2));

        m = std::max<size_t>(m, 1);
        k = std::max(k, 1);
        bits.resize(m, false);
    }

    void
    add(const T& element) {
        auto bytes = to_bytes(element);
        for (int i = 0; i < k; ++i) {
            size_t pos = hash(bytes, i);
            bits[pos] = true;
        }
    }

    bool
    contains(const T& element) const {
        auto bytes = to_bytes(element);
        for (int i = 0; i < k; ++i) {
            size_t pos = hash(bytes, i);
            if (!bits[pos])
                return false;
        }
        return true;
    }

    void
    save(const MemoryIOWriter& writer) const {
        writeBinaryPOD(writer, m);
        writeBinaryPOD(writer, k);
        writeBinaryPOD(writer, n);
        writeBinaryPOD(writer, p);

        for (bool bit : bits) {
            char byte = bit ? 1 : 0;
            writeBinaryPOD(writer, byte);
        }
    }

    void
    load(const MemoryIOReader& reader) {
        readBinaryPOD(reader, m);
        readBinaryPOD(reader, k);
        readBinaryPOD(reader, n);
        readBinaryPOD(reader, p);

        bits.clear();
        bits.resize(m);

        for (size_t i = 0; i < m; ++i) {
            char byte;
            readBinaryPOD(reader, byte);
            bits[i] = (byte != 0);
        }
    }
    size_t
    size() const {
        return n;
    }
    double
    false_positive_rate() const {
        return p;
    }
    size_t
    memory_usage() const {
        return m / 8;
    }

 private:
    std::vector<bool> bits;
    size_t m;
    int k;
    double p;
    size_t n;
    // todo: handle nullptr
    std::vector<unsigned char>
    to_bytes(const T& data) const {
        const unsigned char* byte_ptr = reinterpret_cast<const unsigned char*>(&data);
        return std::vector<unsigned char>(byte_ptr, byte_ptr + sizeof(T));
    }
    size_t
    hash(const std::vector<unsigned char>& data, size_t i) const {
        size_t hash = std::hash<std::string>{}(std::string(data.begin(), data.end())) + i;
        return hash % m;
    }
};
}  // namespace knowhere
#endif
