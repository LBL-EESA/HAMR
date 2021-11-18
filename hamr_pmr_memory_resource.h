#ifndef hamr_pmr_memory_resource_h
#define hamr_pmr_memory_resource_h

// Fri Sep 24 09:58:56 AM PDT 2021
// The clang on Mac OS still doesn't officially support C++17.
// When clang supports C++17 we can remove these and use
// std::pmr::memory_resource
#if defined(__clang__)
#include <experimental/memory_resource>
namespace hamr
{
using pmr_memory_resource = std::experimental::pmr::memory_resource;
}
#else
#include <memory_resource>
namespace hamr
{
using pmr_memory_resource = std::pmr::memory_resource;
}
#endif

#endif
