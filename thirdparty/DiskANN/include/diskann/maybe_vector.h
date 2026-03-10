// My old code from the bitset for Milvus

#pragma once

#include <array>
#include <memory>
#include <type_traits>

namespace diskann {

// A structure that allocates an array of elements.
// If the number of elements is small, 
//     then an allocation will be done on the stack.
// If the number of elements is large, 
//     then an allocation will be done on the heap.
// This struct is designed to avoid performing memory
//     allocations on small objects.
template<typename T>
struct MaybeVector {
public:
    static_assert(std::is_scalar_v<T>);
    
    MaybeVector(const size_t n_elements) {
        m_size = n_elements;

        if (n_elements < num_array_elements) {
            m_data = maybe_array.data();
        } else {
            maybe_memory = std::make_unique<T[]>(m_size);
            m_data = maybe_memory.get();
        }
    }

    MaybeVector(const size_t n_elements, const T default_value) {
        m_size = n_elements;

        if (n_elements < num_array_elements) {
            m_data = maybe_array.data();
        } else {
            maybe_memory = std::make_unique<T[]>(m_size);
            m_data = maybe_memory.get();
        }

        std::fill(m_data, m_data + m_size, default_value);
    }

    MaybeVector(const MaybeVector&) = delete;
    MaybeVector(MaybeVector&&) = delete;
    MaybeVector& operator =(const MaybeVector&) = delete;
    MaybeVector& operator =(MaybeVector&&) = delete;

    inline size_t size() const { return m_size; }
    inline T* data() { return m_data; }
    inline const T* data() const { return m_data; }

    inline T* begin() { return m_data; }
    inline T* end() { return m_data + m_size; }

    inline T& operator[](const size_t idx) { return m_data[idx]; }
    inline const T& operator[](const size_t idx) const { return m_data[idx]; }

private:
    size_t m_size = 0;

    T* m_data = nullptr;

    // we're not expecting to use anything, but small primitive types here
    static constexpr size_t num_array_elements = 4096;

    std::unique_ptr<T[]> maybe_memory;
    std::array<T, num_array_elements> maybe_array;
};

}
