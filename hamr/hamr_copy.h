#ifndef hamr_copy_h
#define hamr_copy_h

#include "hamr_config.h"
#include "hamr_env.h"
#if defined(HAMR_ENABLE_CUDA)
#include "hamr_cuda_kernels.h"
#include "hamr_cuda_launch.h"
#endif

#include <cstring>
#include <cstdlib>

/// heterogeneous accelerator memory resource
namespace hamr
{
#if !defined(HAMR_CUDA_OBJECTS)
/** copies an array
 * @param[in] dest an array of n elements accessible in CUDA
 * @param[in] src an array of n elements accessible on the CPU
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
static int copy_to_cuda_from_cpu(T *dest, const U *src, size_t n_elem,
   typename std::enable_if<!std::is_arithmetic<T>::value>::type * = nullptr)
{
#if !defined(HAMR_ENABLE_CUDA)
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "ERROR: copy_to_cuda_from_cpu CUDA is not enabled." << std::endl;
    return -1;
#else
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "ERROR: copy_to_cuda_from_cpu HAMR_CUDA_OBJECTS is not enabled." << std::endl;
    abort();
    return -1;
#endif
}
#else
/** copies an array (fast path for arrays of arithmetic types of the same type)
 * @param[in] dest an array of n elements accessible in CUDA
 * @param[in] src an array of n elements accessible on the CPU
 * @returns 0 if there were no errors
 */
template <typename T>
static int copy_to_cuda_from_cpu(T *dest, const T *src, size_t n_elem,
   typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr)
{
#if !defined(HAMR_ENABLE_CUDA)
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "ERROR: copy_to_cuda_from_cpu CUDA is not enabled." << std::endl;
    return -1;
#else
    // copy src to gpu
    size_t n_bytes = n_elem*sizeof(T);
    cudaError_t ierr = cudaSuccess;
    if ((ierr = cudaMemcpy(dest, src, n_bytes, cudaMemcpyHostToDevice)) != cudaSuccess)
    {
        std::cerr << "ERROR: Failed to copy " << n_bytes << ". "
            << cudaGetErrorString(ierr) << std::endl;
        return -1;
    }

    if (hamr::get_verbose())
    {
        std::cerr << "hamr::copy_to_cuda_from_cpu(f) " << n_elem << std::endl;
    }

    return 0;
#endif
}
#endif

/** copies an array
 * @param[in] dest an array of n elements accessible in CUDA
 * @param[in] src an array of n elements accessible on the CPU
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
static int copy_to_cuda_from_cpu(T *dest, const U *src, size_t n_elem
#if !defined(HAMR_CUDA_OBJECTS)
    ,typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr
#endif
    )
{
#if !defined(HAMR_ENABLE_CUDA)
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "ERROR: copy_to_cuda_from_cpu CUDA is not enabled." << std::endl;
    return -1;
#else
    // copy src to gpu
    // apply the copy on the gpu

    // allocate a temporary buffer on the GPU
    std::shared_ptr<U> pTmp = hamr::cuda_malloc_allocator<U>::allocate(n_elem);

    // copy the data
    size_t n_bytes = n_elem*sizeof(U);
    cudaError_t ierr = cudaSuccess;
    if ((ierr = cudaMemcpy(pTmp.get(), src, n_bytes, cudaMemcpyHostToDevice)) != cudaSuccess)
    {
        std::cerr << "ERROR: Failed to copy " << n_bytes << ". "
            << cudaGetErrorString(ierr) << std::endl;
        return -1;
    }

    // get launch parameters
    int device_id = -1;
    dim3 block_grid;
    int n_blocks = 0;
    dim3 thread_grid = 0;
    if (hamr::partition_thread_blocks(device_id, n_elem, 8, block_grid, n_blocks, thread_grid))
    {
        std::cerr << "ERROR: Failed to determine launch properties." << std::endl;
        return -1;
    }

    // invoke the casting copy kernel
    hamr::cuda_kernels::copy<<<block_grid, thread_grid>>>(dest, pTmp.get(), n_elem);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        std::cerr << "ERROR: Failed to launch the copy kernel. "
            << cudaGetErrorString(ierr) << std::endl;
        return -1;
    }

    if (hamr::get_verbose())
    {
        std::cerr << "hamr::copy_to_cuda_from_cpu " << n_elem << std::endl;
    }
    return 0;
#endif
}

#if !defined(HAMR_CUDA_OBJECTS)
/** copies an array
 * @param[in] dest an array of n elements accessible in CUDA
 * @param[in] src an array of n elements accessible in CUDA
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
static int copy_to_cuda_from_cuda(T *dest, const U *src, size_t n_elem,
   typename std::enable_if<!std::is_arithmetic<T>::value>::type * = nullptr)
{
#if !defined(HAMR_ENABLE_CUDA)
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "ERROR: copy_to_cuda_from_cuda CUDA is not enabled." << std::endl;
    return -1;
#else
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "ERROR: copy_to_cuda_from_cuda HAMR_CUDA_OBJECTS is not enabled." << std::endl;
    return -1;
#endif
}
#else
/** copies an array (fast path for arrays of arithmetic types of the same type)
 * @param[in] dest an array of n elements accessible in CUDA
 * @param[in] src an array of n elements accessible on the CPU
 * @returns 0 if there were no errors
 */
template <typename T>
static int copy_to_cuda_from_cuda(T *dest, const T *src, size_t n_elem,
    typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr)
{
#if !defined(HAMR_ENABLE_CUDA)
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "ERROR: copy_to_cuda_from_cuda CUDA is not enabled." << std::endl;
    return -1;
#else
    // copy src to gpu
    size_t n_bytes = n_elem*sizeof(T);
    cudaError_t ierr = cudaSuccess;
    if ((ierr = cudaMemcpy(dest, src, n_bytes, cudaMemcpyDeviceToDevice)) != cudaSuccess)
    {
        std::cerr << "ERROR: Failed to copy " << n_bytes << ". "
            << cudaGetErrorString(ierr) << std::endl;
        return -1;
    }

    if (hamr::get_verbose())
    {
        std::cerr << "hamr::copy_to_cuda_from_cuda(f) " << n_elem << std::endl;
    }

    return 0;
#endif
}
#endif

/** copies an array
 * @param[in] dest an array of n elements accessible in CUDA
 * @param[in] src an array of n elements accessible in CUDA
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
static int copy_to_cuda_from_cuda(T *dest, const U *src, size_t n_elem
#if !defined(HAMR_CUDA_OBJECTS)
    ,typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr
#endif
    )
{
#if !defined(HAMR_ENABLE_CUDA)
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "ERROR: copy_to_cuda_from_cuda CUDA is not enabled." << std::endl;
    return -1;
#else
    // copy on the gpu
    // get launch parameters
    int device_id = -1;
    dim3 block_grid;
    int n_blocks = 0;
    dim3 thread_grid = 0;
    if (hamr::partition_thread_blocks(device_id, n_elem, 8, block_grid, n_blocks, thread_grid))
    {
        std::cerr << "ERROR: Failed to determine launch properties." << std::endl;
        return -1;
    }

    // invoke the casting copy kernel
    cudaError_t ierr = cudaSuccess;
    hamr::cuda_kernels::copy<<<block_grid, thread_grid>>>(dest, src, n_elem);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        std::cerr << "ERROR: Failed to launch the copy kernel. "
            << cudaGetErrorString(ierr) << std::endl;
        return -1;
    }

    if (hamr::get_verbose())
    {
        std::cerr << "hamr::copy_to_cuda_from_cuda " << n_elem << std::endl;
    }

    return 0;
#endif
}

#if !defined(HAMR_CUDA_OBJECTS)
/** copies an array
 * @param[in] dest an array of n elements accessible on the CPU
 * @param[in] src an array of n elements accessible in CUDA
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
static int copy_to_cpu_from_cuda(T *dest, const U *src, size_t n_elem,
   typename std::enable_if<!std::is_arithmetic<T>::value>::type * = nullptr)
{
#if !defined(HAMR_ENABLE_CUDA)
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "ERROR: copy_to_cpu_from_cuda CUDA is not enabled." << std::endl;
    return -1;
#else
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "ERROR: copy_to_cpu_from_cuda HAMR_CUDA_OBJECTS is not enabled." << std::endl;
    abort();
    return -1;
#endif
}
#else
/** copies an array (fast path for arrays of arithmetic types of the same type)
 * @param[in] dest an array of n elements accessible in CUDA
 * @param[in] src an array of n elements accessible on the CPU
 * @returns 0 if there were no errors
 */
template <typename T>
static int copy_to_cpu_from_cuda(T *dest, const T *src, size_t n_elem,
   typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr)
{
#if !defined(HAMR_ENABLE_CUDA)
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "ERROR: copy_to_cpu_from_cuda CUDA is not enabled." << std::endl;
    return -1;
#else
    // copy src to gpu
    size_t n_bytes = n_elem*sizeof(T);
    cudaError_t ierr = cudaSuccess;
    if ((ierr = cudaMemcpy(dest, src, n_bytes, cudaMemcpyDeviceToHost)) != cudaSuccess)
    {
        std::cerr << "ERROR: Failed to copy " << n_bytes << ". "
            << cudaGetErrorString(ierr) << std::endl;
        return -1;
    }

    if (hamr::get_verbose())
    {
        std::cerr << "hamr::copy_to_cpu_from_cuda(f) " << n_elem << std::endl;
    }

    return 0;
#endif
}
#endif

/** copies an array
 * @param[in] dest an array of n elements accessible on the CPU
 * @param[in] src an array of n elements accessible in CUDA
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
static int copy_to_cpu_from_cuda(T *dest, const U *src, size_t n_elem
#if !defined(HAMR_CUDA_OBJECTS)
    ,typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr
#endif
    )
{
#if !defined(HAMR_ENABLE_CUDA)
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "ERROR: copy_to_cpu_from_cuda CUDA is not enabled." << std::endl;
    return -1;
#else
    // apply the copy on the gpu in a temporary buffer
    // copy the buffer to the cpu

    // allocate a temporary buffer on the GPU
    std::shared_ptr<T> pTmp = hamr::cuda_malloc_allocator<T>::allocate(n_elem);

    // get launch parameters
    int device_id = -1;
    dim3 block_grid;
    int n_blocks = 0;
    dim3 thread_grid = 0;
    if (hamr::partition_thread_blocks(device_id, n_elem, 8, block_grid, n_blocks, thread_grid))
    {
        std::cerr << "ERROR: Failed to determine launch properties." << std::endl;
        return -1;
    }

    // invoke the casting copy kernel on the GPU
    cudaError_t ierr = cudaSuccess;
    hamr::cuda_kernels::copy<<<block_grid, thread_grid>>>(pTmp.get(), src, n_elem);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        std::cerr << "ERROR: Failed to launch the copy kernel. "
            << cudaGetErrorString(ierr) << std::endl;
        return -1;
    }

    // copy the data to the CPU
    size_t n_bytes = n_elem*sizeof(T);
    if ((ierr = cudaMemcpy(dest, pTmp.get(), n_bytes, cudaMemcpyDeviceToHost)) != cudaSuccess)
    {
        std::cerr << "ERROR: Failed to copy " << n_bytes << ". "
            << cudaGetErrorString(ierr) << std::endl;
        return -1;
    }

    if (hamr::get_verbose())
    {
        std::cerr << "hamr::copy_to_cpu_from_cuda " << n_elem << std::endl;
    }

    return 0;
#endif
}


/** copies an array
 * @param[in] dest an array of n elements accessible on the CPU
 * @param[in] src an array of n elements accessible on the CPU
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
static int copy_to_cpu_from_cpu(T *dest, const U *src, size_t n_elem)
{
    for (size_t i = 0; i < n_elem; ++i)
    {
        dest[i] = static_cast<T>(src[i]);
    }

    if (hamr::get_verbose())
    {
        std::cerr << "hamr::copy_to_cpu_from_cpu " << n_elem << std::endl;
    }

    return 0;

}

/** copies an array (fast path for arrays of arithmetic types of the same type)
 * @param[in] dest an array of n elements accessible in CUDA
 * @param[in] src an array of n elements accessible on the CPU
 * @returns 0 if there were no errors
 */
template <typename T>
static int copy_to_cpu_from_cpu(T *dest, const T *src, size_t n_elem,
   typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr)
{
    // copy src to gpu
    size_t n_bytes = n_elem*sizeof(T);
    memcpy(dest, src, n_bytes);

    if (hamr::get_verbose())
    {
        std::cerr << "hamr::copy_to_cpu_from_cpu(f) " << n_elem << std::endl;
    }

    return 0;
}

}

#endif
