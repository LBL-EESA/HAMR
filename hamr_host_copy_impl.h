#ifndef hamr_host_copy_impl_h
#define hamr_host_copy_impl_h

#include "hamr_config.h"
#include "hamr_env.h"

#include <cstring>
#include <cstdlib>
#include <iostream>
#include <memory>

/// heterogeneous accelerator memory resource
namespace hamr
{

// --------------------------------------------------------------------------
template <typename T, typename U>
int copy_to_host_from_host(T *dest, const U *src, size_t n_elem)
{
    for (size_t i = 0; i < n_elem; ++i)
    {
        dest[i] = static_cast<T>(src[i]);
    }

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "hamr::copy_to_host_from_host " << n_elem
            << " from " << typeid(U).name() << sizeof(U) << " to "
            << typeid(T).name() << sizeof(T) << std::endl;
    }
#endif

    return 0;
}

// --------------------------------------------------------------------------
template <typename T>
int copy_to_host_from_host(T *dest, const T *src, size_t n_elem,
   typename std::enable_if<std::is_arithmetic<T>::value>::type *)
{
    // copy src to gpu
    size_t n_bytes = n_elem*sizeof(T);
    memcpy(dest, src, n_bytes);

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "hamr::copy_to_host_from_host same " << n_elem
            << " " << typeid(T).name() << sizeof(T) << std::endl;
    }
#endif

    return 0;
}

}

#endif
