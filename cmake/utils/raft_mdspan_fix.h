// Workaround for RAFT's vendored mdspan (aligned_accessor.hpp) which uses
// std::has_single_bit (<bit>) and std::assume_aligned (<memory>) under C++20
// without including the required headers. Force-including this header ensures
// the declarations are available before any RAFT header is processed.
// Remove after upgrading to RAFT/cuVS 26.02+ (vendored mdspan deleted).
#if __cplusplus >= 202002L
#include <bit>
#include <memory>
#endif
