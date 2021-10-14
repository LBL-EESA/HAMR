#ifndef pmr_vector_h
#define pmr_vector_h

#include <memory>

// The clang on Mac OS still doesn't officially support C++17.
// When clang supports C++17 we can remove these and use
// std::pmr::vector
#if defined(__clang__)
#include <experimental/vector>
namespace hamr
{
template <typename data_t>
using pmr_vector = std::experimental::pmr::vector<data_t>;

template<typename data_t>
using pmr_polymorphic_allocator = std::experimental::pmr::polymorphic_allocator<data_t>;
}
#else
#include <vector>
namespace hamr
{
template <typename data_t>
using pmr_vector = std::pmr::vector<data_t>;

template<typename data_t>
using pmr_polymorphic_allocator = std::pmr::polymorphic_allocator<data_t>;
}
#endif

namespace hamr
{
/// a shared pointer to a std::vector that uses a polymorphic allocator
template <typename data_t>
using p_pmr_vector = std::shared_ptr<pmr_vector<data_t>>;
}

#endif
