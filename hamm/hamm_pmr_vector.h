#ifndef hamm_pmr_vector_h
#define hamm_pmr_vector_h

// Fri Sep 24 09:58:56 AM PDT 2021
// The clang on Mac OS still doesn't officially support C++17.
// When clang supports C++17 we can remove these and use
// std::pmr::vector
#if defined(__clang__)
#include <experimental/vector>
template <typename data_t>
using hamm_pmr_vector = std::experimental::pmr::vector<data_t>;
#else
#include <vector>
template <typename data_t>
using hamm_pmr_vector = std::pmr::vector<data_t>;
#endif

#endif
