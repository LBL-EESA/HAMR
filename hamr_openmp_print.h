#ifndef hamr_openmp_print_impl_h
#define hamr_openmp_print_impl_h

#include "hamr_config.h"

/// heterogeneous accelerator memory resource
namespace hamr
{

/** prints an array on the host (note: OpenMP provides no way to print directly
 * from the device)
 * @param[in] vals an array of n elements accessible in OpenMP
 * @param[in] n_elem the length of the array
 * @returns 0 if there were no errors
 */
template <typename T>
HAMR_EXPORT
int openmp_print(T *vals, size_t n_elem);

}

#if !defined(HAMR_SEPARATE_IMPL)
#include "hamr_openmp_print_impl.h"
#endif
#endif
