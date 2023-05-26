#ifndef hamr_cpu_copy_h
#define hamr_cpu_copy_h

#include "hamr_config.h"
#include <memory>
#include <type_traits>

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
int copy_to_cpu_from_cpu(T *dest, const U *src, size_t n_elem);

/** Copies an array on the CPU (fast path for arrays of arithmetic types of the
 * same type).
 *
 * @param[in] dest an array of n elements accessible in CUDA
 * @param[in] src an array of n elements accessible on the CPU
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T>
int copy_to_cpu_from_cpu(T *dest, const T *src, size_t n_elem,
   typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr);

}

#if !defined(HAMR_SEPARATE_IMPL)
#include "hamr_cpu_copy_impl.h"
#endif

#endif
