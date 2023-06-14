#ifndef hamr_openmp_print_h
#define hamr_openmp_print_h

#include "hamr_config.h"
#include "hamr_env.h"
#if defined(HAMR_ENABLE_OPENMP)
#include "hamr_openmp_copy.h"
#include "hamr_malloc_allocator.h"
#endif

#include <iostream>

/// heterogeneous accelerator memory resource
namespace hamr
{
// ---------------------------------------------------------------------------
template <typename T>
int openmp_print(T *vals, size_t n_elem)
{
#if !defined(HAMR_ENABLE_OPENMP)
    (void) vals;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " print_openmp failed because OpenMP is not enabled." << std::endl;
    return -1;
#else

    // allocate a temporary on the host
    auto sptmp = hamr::malloc_allocator<T>::allocate(n_elem);
    T *ptmp = sptmp.get();

    // move to the host
    if (hamr::copy_to_host_from_openmp(ptmp, vals, n_elem))
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " failed to move data to the host" << std::endl;
        return -1;
    }

    // print
    if (n_elem)
    {
        std::cerr << ptmp[0];
        for (size_t i = 1; i < n_elem; ++i)
        {
            std::cerr << ", " << ptmp[i];
        }
    }
    std::cerr << std::endl;

    return 0;
#endif
}

}
#endif
