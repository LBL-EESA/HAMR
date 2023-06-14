#ifndef hamr_host_copy_h
#define hamr_host_copy_h

#include "hamr_config.h"
#include <memory>
#include <type_traits>

/// heterogeneous accelerator memory resource
namespace hamr
{

/** Copies an array on the host.
 *
 * @param[in] dest an array of n elements accessible on the host
 * @param[in] src an array of n elements accessible on the host
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
int copy_to_host_from_host(T *dest, const U *src, size_t n_elem);

/** Copies an array on the host (fast path for arrays of arithmetic types of the
 * same type).
 *
 * @param[in] dest an array of n elements accessible in CUDA
 * @param[in] src an array of n elements accessible on the host
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T>
int copy_to_host_from_host(T *dest, const T *src, size_t n_elem,
   typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr);

}

#if !defined(HAMR_SEPARATE_IMPL)
#include "hamr_host_copy_impl.h"
#endif

#endif
