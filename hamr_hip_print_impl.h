#ifndef hamr_hip_print_impl_h
#define hamr_hip_print_impl_h

#include "hamr_config.h"
#include "hamr_env.h"

#if defined(HAMR_ENABLE_HIP)
#include "hamr_hip_kernels.h"
#include "hamr_hip_launch.h"
#include <hip/hip_runtime.h>
#endif

#include <iostream>

/// heterogeneous accelerator memory resource
namespace hamr
{

/** prints an array on the GPU
 * @param[in] vals an array of n elements accessible in HIP
 * @param[in] n_elem the length of the array
 * @returns 0 if there were no errors
 */
template <typename T>
int hip_print(T *vals, size_t n_elem)
{
#if !defined(HAMR_ENABLE_HIP)
    (void) vals;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " print_hip failed because HIP is not enabled." << std::endl;
    return -1;
#else

    // get launch parameters
    int device_id = -1;
    dim3 block_grid;
    int n_blocks = 0;
    dim3 thread_grid = 0;
    if (hamr::partition_thread_blocks(device_id, n_elem, 8, block_grid,
        n_blocks, thread_grid))
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to determine launch properties." << std::endl;
        return -1;
    }

    // invoke the print kernel
    hipError_t ierr = hipSuccess;
    hamr::hip_kernels::print<<<block_grid, thread_grid>>>(vals, n_elem);
    if ((ierr = hipGetLastError()) != hipSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to launch the print kernel. "
            << hipGetErrorString(ierr) << std::endl;
        return -1;
    }

    return 0;
#endif
}

}
#endif
