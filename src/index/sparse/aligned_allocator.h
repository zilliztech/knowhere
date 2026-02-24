#pragma once
#include <cstddef>
#include <cstdlib>
#include <new>
#include <type_traits>

namespace knowhere::sparse::inverted {

// 64-byte aligned allocator suitable for SIMD-friendly containers
template <typename T, std::size_t Alignment = 64>
struct aligned_allocator {
    using value_type = T;

    aligned_allocator() noexcept = default;

    template <class U>
    aligned_allocator(const aligned_allocator<U, Alignment>& /*unused*/) noexcept {
    }

    [[nodiscard]] T*
    allocate(std::size_t n) {
        if (n == 0) {
            return nullptr;
        }
        void* ptr = nullptr;
        std::size_t bytes = n * sizeof(T);
#if defined(__linux__)
        if (posix_memalign(&ptr, Alignment, bytes) != 0) {
            ptr = nullptr;
        }
#else
        std::size_t aligned_bytes = (bytes + (Alignment - 1)) & ~static_cast<std::size_t>(Alignment - 1);
        ptr = std::aligned_alloc(Alignment, aligned_bytes);
#endif
        if (ptr == nullptr) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }

    void
    deallocate(T* p, std::size_t /*unused*/) noexcept {
        std::free(p);
    }

    template <class U>
    struct rebind {
        using other = aligned_allocator<U, Alignment>;
    };

    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;
};

template <class T, class U, std::size_t A>
constexpr bool
operator==(const aligned_allocator<T, A>&, const aligned_allocator<U, A>&) noexcept {
    return true;
}

template <class T, class U, std::size_t A>
constexpr bool
operator!=(const aligned_allocator<T, A>&, const aligned_allocator<U, A>&) noexcept {
    return false;
}
}  // namespace knowhere::sparse::inverted
