#ifndef hamr_cuda_print_h
#define hamr_cuda_print_h

#include "hamr_env.h"
#if defined(HAMR_ENABLE_CUDA)
#include "hamr_cuda_kernels.h"
#include "hamr_cuda_launch.h"
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <iostream>

/// heterogeneous accelerator memory resource
namespace hamr
{

/** prints an array on the GPU
 * @param[in] vals an array of n elements accessible in CUDA
 * @param[in] n_elem the length of the array
 * @returns 0 if there were no errors
 */
template <typename T>
static int cuda_print(T *vals, size_t n_elem)
{
#if !defined(HAMR_ENABLE_CUDA)
    (void) vals;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " print_cuda failed because CUDA is not enabled." << std::endl;
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
    cudaError_t ierr = cudaSuccess;
    hamr::cuda_kernels::print<<<block_grid, thread_grid>>>(vals, n_elem);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to launch the print kernel. "
            << cudaGetErrorString(ierr) << std::endl;
        return -1;
    }

    return 0;
#endif
}

}
#endif
