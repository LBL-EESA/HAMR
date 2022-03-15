#ifndef hamr_cpu_copy_h
#define hamr_cpu_copy_h

#include "hamr_config.h"
#include "hamr_env.h"

#include <cstring>
#include <cstdlib>
#include <iostream>
#include <memory>

/// heterogeneous accelerator memory resource
namespace hamr
{

/** Copies an array on the CPU.
 *
 * @param[in] dest an array of n elements accessible on the CPU
 * @param[in] src an array of n elements accessible on the CPU
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
static int copy_to_cpu_from_cpu(T *dest, const U *src, size_t n_elem)
{
    for (size_t i = 0; i < n_elem; ++i)
    {
        dest[i] = static_cast<T>(src[i]);
    }

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "hamr::copy_to_cpu_from_cpu " << n_elem
            << " from " << typeid(U).name() << sizeof(U) << " to "
            << typeid(T).name() << sizeof(T) << std::endl;
    }
#endif

    return 0;
}

/** Copies an array on the CPU (fast path for arrays of arithmetic types of the
 * same type).
 *
 * @param[in] dest an array of n elements accessible in CUDA
 * @param[in] src an array of n elements accessible on the CPU
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T>
static int copy_to_cpu_from_cpu(T *dest, const T *src, size_t n_elem,
   typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr)
{
    // copy src to gpu
    size_t n_bytes = n_elem*sizeof(T);
    memcpy(dest, src, n_bytes);

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "hamr::copy_to_cpu_from_cpu same " << n_elem
            << " " << typeid(T).name() << sizeof(T) << std::endl;
    }
#endif

    return 0;
}

}

#endif

