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
#include "knowhere/bitsetview.h"
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
        size_t glb_hash = hash((const char*)&element, sizeof(element), 0);
        for (int i = 0; i < k; ++i) {
            size_t pos = (glb_hash + i) % m;
            bits[pos] = true;
        }
    }

    bool
    contains(const T& element) const {
        size_t glb_hash = hash((const char*)&element, sizeof(element), 0);
        for (int i = 0; i < k; ++i) {
            size_t pos = (glb_hash + i) % m;
            if (!bits[pos])
                return false;
        }
        return true;
    }

    void
    save(MemoryIOWriter& writer) const {
        writeBinaryPOD(writer, m);
        writeBinaryPOD(writer, k);
        writeBinaryPOD(writer, n);
        writeBinaryPOD(writer, p);
        auto bytes_num = (m + 8 - 1) / 8;
        std::vector<char> buffer(bytes_num, 0);
        for (size_t i = 0; i < m; ++i) {
            if (bits[i]) {
                buffer[i / 8] |= (1 << (i % 8));
            }
        }
        writer.write(buffer.data(), buffer.size());
    }

    void
    load(MemoryIOReader& reader) {
        readBinaryPOD(reader, m);
        readBinaryPOD(reader, k);
        readBinaryPOD(reader, n);
        readBinaryPOD(reader, p);
        bits.clear();
        bits.resize(m);
        auto bytes_num = (m + 8 - 1) / 8;
        std::vector<char> buffer(bytes_num);
        reader.read(buffer.data(), bytes_num);
        for (size_t i = 0; i < m; ++i) {
            bool bit = (buffer[i / 8] >> (i % 8)) & 1;
            bits.push_back(bit);
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
    static constexpr size_t multiplier = 31;
    std::vector<bool> bits;
    size_t m = 0;
    int k = 0;
    double p = 0;
    size_t n = 0;

    size_t
    hash(const char* data, size_t length, size_t bucket_i) const {
        if (data == nullptr) {
            throw std::runtime_error("can't hash null data.");
        }
        size_t result = 0;
        for (size_t i = 0; i < length; ++i) {
            result = (result * multiplier) + static_cast<size_t>(data[i]);
        }
        return (result + bucket_i) % m;
    }
};
}  // namespace knowhere
#endif
