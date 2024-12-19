#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <faiss/impl/FaissAssert.h>

namespace faiss {

struct MappingOwner {
    virtual ~MappingOwner() = default;
};

// a container that either works as std::vector<T> that owns its own memory, 
//    or as a mapped pointer owned by someone third-party owner
template<typename T>
struct MaybeOwnedVector {
    using value_type = T;
    using self_type = MaybeOwnedVector<T>;

    bool is_owned = true;

    // this one is used if is_owned == true
    std::vector<T> owned_data;

    // these three are used if is_owned == false
    T* mapped_data = nullptr;
    // the number of T elements
    size_t mapped_size = 0;
    std::shared_ptr<MappingOwner> mapping_owner;

    // points either to mapped_data, or to owned.data()
    T* c_ptr = nullptr;
    // uses either mapped_size, or owned.size();
    size_t c_size = 0; 

    MaybeOwnedVector() = default;
    MaybeOwnedVector(const size_t initial_size) {
        is_owned = true;
        
        owned_data.resize(initial_size);
        c_ptr = owned_data.data();
        c_size = owned_data.size();
    }

    MaybeOwnedVector(const MaybeOwnedVector& other) {
        is_owned = other.is_owned;
        owned_data = other.owned_data;

        mapped_data = other.mapped_data;
        mapped_size = other.mapped_size;
        mapping_owner = other.mapping_owner;

        if (is_owned) {
            c_ptr = owned_data.data();
            c_size = owned_data.size();
        } else {
            c_ptr = mapped_data;
            c_size = mapped_size;
        }
    }

    MaybeOwnedVector(MaybeOwnedVector&& other) {
        is_owned = other.is_owned;
        owned_data = std::move(other.owned_data);

        mapped_data = other.mapped_data;
        mapped_size = other.mapped_size;
        mapping_owner = std::move(other.mapping_owner);

        if (is_owned) {
            c_ptr = owned_data.data();
            c_size = owned_data.size();
        } else {
            c_ptr = mapped_data;
            c_size = mapped_size;
        }
    }

    MaybeOwnedVector& operator =(const MaybeOwnedVector& other) {
        if (this == &other) {
            return *this;
        }

        // create a copy
        MaybeOwnedVector cloned(other);
        // swap
        swap(*this, cloned);

        return *this;
    }

    MaybeOwnedVector& operator =(MaybeOwnedVector&& other) {
        if (this == &other) {
            return *this;
        }

        // moved
        MaybeOwnedVector moved(std::move(other));
        // swap
        swap(*this, moved);
        
        return *this;
    }

    MaybeOwnedVector(std::vector<T>&& other) {
        is_owned = true;

        owned_data = std::move(other);
        c_ptr = owned_data.data();
        c_size = owned_data.size();
    }

    static MaybeOwnedVector from_mmapped(
        void* address, 
        const size_t n_mapped_elements,
        const std::shared_ptr<MappingOwner>& owner
    ) {
        MaybeOwnedVector vec;
        vec.is_owned = false;
        vec.mapped_data = reinterpret_cast<T*>(address);
        vec.mapped_size = n_mapped_elements;
        vec.mapping_owner = owner;
        
        vec.c_ptr = vec.mapped_data;
        vec.c_size = vec.mapped_size;

        return vec;
    }

    const T* data() const {
        return c_ptr;
    }

    T* data() {
        return c_ptr;
    }

    size_t size() const {
        return c_size;
    }

    T& operator[](const size_t idx) {
        return c_ptr[idx];
    }

    const T& operator[](const size_t idx) const {
        return c_ptr[idx];
    }

    void clear() {
        FAISS_ASSERT_MSG(is_owned, "This operation cannot be performed on a memory-mapped vector");

        owned_data.clear();
        c_ptr = owned_data.data();
        c_size = owned_data.size();
    }

    void resize(const size_t new_size) {
        FAISS_ASSERT_MSG(is_owned, "This operation cannot be performed on a memory-mapped vector");

        owned_data.resize(new_size);
        c_ptr = owned_data.data();
        c_size = owned_data.size();
    }

    void resize(const size_t new_size, const value_type v) {
        FAISS_ASSERT_MSG(is_owned, "This operation cannot be performed on a memory-mapped vector");

        owned_data.resize(new_size, v);
        c_ptr = owned_data.data();
        c_size = owned_data.size();
    }

    friend void swap(self_type& a, self_type& b) {
        std::swap(a.is_owned, b.is_owned);
        std::swap(a.owned_data, b.owned_data);
        std::swap(a.mapped_data, b.mapped_data);
        std::swap(a.mapped_size, b.mapped_size);
        std::swap(a.mapping_owner, b.mapping_owner);
        std::swap(a.c_ptr, b.c_ptr);
        std::swap(a.c_size, b.c_size);
    }
};

}