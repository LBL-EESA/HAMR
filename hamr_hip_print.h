#ifndef hamr_hip_print_h
#define hamr_hip_print_h

#include "hamr_config.h"

/// heterogeneous accelerator memory resource
namespace hamr
{

/** prints an array on the GPU
 * @param[in] vals an array of n elements accessible in HIP
 * @param[in] n_elem the length of the array
 * @returns 0 if there were no errors
 */
template <typename T>
int hip_print(T *vals, size_t n_elem);

}

#if !defined(HAMR_SEPARATE_IMPL)
#include "hamr_hip_print_impl.h"
#endif

#endif
