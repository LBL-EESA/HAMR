#ifndef hamr_hip_copy_h
#define hamr_hip_copy_h

#include "hamr_config.h"
#include "hamr_env.h"
#if defined(HAMR_ENABLE_HIP)
#include "hamr_hip_kernels.h"
#include "hamr_hip_launch.h"
#include "hamr_hip_malloc_allocator.h"
#endif
#include "hamr_malloc_allocator.h"

#include <cstring>
#include <cstdlib>
#include <iostream>
#include <memory>

/// heterogeneous accelerator memory resource
namespace hamr
{
#if !defined(HAMR_HIP_OBJECTS)
/** Copies an array to the active HIP device.
 *
 * @param[in] dest an array of n elements accessible in HIP
 * @param[in] src an array of n elements accessible on the CPU
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
static int copy_to_hip_from_cpu(T *dest, const U *src, size_t n_elem,
   typename std::enable_if<!std::is_arithmetic<T>::value>::type * = nullptr)
{
#if !defined(HAMR_ENABLE_HIP)
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_hip_from_cpu HIP is not enabled." << std::endl;
    return -1;
#else
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_hip_from_cpu HAMR_HIP_OBJECTS is not enabled." << std::endl;
    abort();
    return -1;
#endif
}
#else
/** Copies an array to the active HIP device (fast path for arrays of
 * arithmetic types of the same type).
 *
 * @param[in] dest an array of n elements accessible in HIP
 * @param[in] src an array of n elements accessible on the CPU
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T>
static int copy_to_hip_from_cpu(T *dest, const T *src, size_t n_elem,
   typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr)
{
#if !defined(HAMR_ENABLE_HIP)
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_hip_from_cpu HIP is not enabled." << std::endl;
    return -1;
#else
    // copy src to gpu
    size_t n_bytes = n_elem*sizeof(T);
    hipError_t ierr = hipSuccess;
    if ((ierr = hipMemcpy(dest, src, n_bytes, hipMemcpyHostToDevice)) != hipSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to copy " << n_bytes << ". "
            << hipGetErrorString(ierr) << std::endl;
        return -1;
    }

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "hamr::copy_to_hip_from_cpu same " << n_elem
            << " " << typeid(T).name() << sizeof(T) << std::endl;
    }
#endif

    return 0;
#endif
}
#endif

/** Copies an array to the active HIP device.
 *
 * @param[in] dest an array of n elements accessible in HIP
 * @param[in] src an array of n elements accessible on the CPU
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
static int copy_to_hip_from_cpu(T *dest, const U *src, size_t n_elem
#if !defined(HAMR_HIP_OBJECTS)
    ,typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr
#endif
    )
{
#if !defined(HAMR_ENABLE_HIP)
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_hip_from_cpu HIP is not enabled." << std::endl;
    return -1;
#else
    // copy src to gpu
    // apply the copy on the gpu

    // allocate a temporary buffer on the GPU
    std::shared_ptr<U> pTmp = hamr::hip_malloc_allocator<U>::allocate(n_elem);

    // copy the data
    size_t n_bytes = n_elem*sizeof(U);
    hipError_t ierr = hipSuccess;
    if ((ierr = hipMemcpy(pTmp.get(), src, n_bytes, hipMemcpyHostToDevice)) != hipSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to copy " << n_bytes << ". "
            << hipGetErrorString(ierr) << std::endl;
        return -1;
    }

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

    // invoke the casting copy kernel
    hamr::hip_kernels::copy<<<block_grid, thread_grid>>>(dest, pTmp.get(), n_elem);
    if ((ierr = hipGetLastError()) != hipSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to launch the copy kernel. "
            << hipGetErrorString(ierr) << std::endl;
        return -1;
    }

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "hamr::copy_to_hip_from_cpu " << n_elem
            << " from " << typeid(U).name() << sizeof(U) << " to "
            << typeid(T).name() << sizeof(T) << std::endl;
    }
#endif

    return 0;
#endif
}

#if !defined(HAMR_HIP_OBJECTS)
/** Copies an array on the active HIP device.
 *
 * @param[in] dest an array of n elements accessible in HIP
 * @param[in] src an array of n elements accessible in HIP
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
static int copy_to_hip_from_hip(T *dest, const U *src, size_t n_elem,
   typename std::enable_if<!std::is_arithmetic<T>::value>::type * = nullptr)
{
#if !defined(HAMR_ENABLE_HIP)
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_hip_from_hip HIP is not enabled." << std::endl;
    return -1;
#else
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_hip_from_hip HAMR_HIP_OBJECTS is not enabled." << std::endl;
    return -1;
#endif
}
#else
/** Ccopies an array on the active CUAD device (fast path for arrays of
 * arithmetic types of the same type).
 *
 * @param[in] dest an array of n elements accessible in HIP
 * @param[in] src an array of n elements accessible on the CPU
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T>
static int copy_to_hip_from_hip(T *dest, const T *src, size_t n_elem,
    typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr)
{
#if !defined(HAMR_ENABLE_HIP)
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_hip_from_hip HIP is not enabled." << std::endl;
    return -1;
#else
    // copy src to gpu
    size_t n_bytes = n_elem*sizeof(T);
    hipError_t ierr = hipSuccess;
    if ((ierr = hipMemcpy(dest, src, n_bytes, hipMemcpyDeviceToDevice)) != hipSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to copy " << n_bytes << ". "
            << hipGetErrorString(ierr) << std::endl;
        return -1;
    }

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "hamr::copy_to_hip_from_hip same " << n_elem
            << typeid(T).name() << sizeof(T) << std::endl;
    }
#endif

    return 0;
#endif
}
#endif

/** Copies an array on the active HIP device.
 *
 * @param[in] dest an array of n elements accessible in HIP
 * @param[in] src an array of n elements accessible in HIP
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
static int copy_to_hip_from_hip(T *dest, const U *src, size_t n_elem
#if !defined(HAMR_HIP_OBJECTS)
    ,typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr
#endif
    )
{
#if !defined(HAMR_ENABLE_HIP)
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_hip_from_hip HIP is not enabled." << std::endl;
    return -1;
#else
    // copy on the gpu
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

    // invoke the casting copy kernel
    hipError_t ierr = hipSuccess;
    hamr::hip_kernels::copy<<<block_grid, thread_grid>>>(dest, src, n_elem);
    if ((ierr = hipGetLastError()) != hipSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to launch the copy kernel. "
            << hipGetErrorString(ierr) << std::endl;
        return -1;
    }

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "hamr::copy_to_hip_from_hip " << n_elem
            << " from " << typeid(U).name() << sizeof(U) << " to "
            << typeid(T).name() << sizeof(T) << std::endl;
    }
#endif

    return 0;
#endif
}

#if !defined(HAMR_HIP_OBJECTS)
/** Copies an array to the active HIP device from the named HIP device,
 *
 * @param[in] dest an array of n elements accessible in HIP
 * @param[in] src an array of n elements accessible in HIP
 * @param[in] src_device the HIP device on which src is allocated
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
static int copy_to_hip_from_hip(T *dest, const U *src, int src_device, size_t n_elem,
   typename std::enable_if<!std::is_arithmetic<T>::value>::type * = nullptr)
{
#if !defined(HAMR_ENABLE_HIP)
    (void) dest;
    (void) src;
    (void) src_device;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_hip_from_hip HIP is not enabled." << std::endl;
    return -1;
#else
    (void) dest;
    (void) src;
    (void) src_device;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_hip_from_hip HAMR_HIP_OBJECTS is not enabled." << std::endl;
    return -1;
#endif
}
#else
/** Copies an array to the active HIP device from the named HIP device, (fast
 * path for arrays of arithmetic types of the same type).
 *
 * @param[in] dest an array of n elements accessible in HIP
 * @param[in] src an array of n elements accessible in HIP
 * @param[in] src_device the HIP device on which src is allocated
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T>
static int copy_to_hip_from_hip(T *dest, const T *src, int src_device, size_t n_elem,
    typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr)
{
#if !defined(HAMR_ENABLE_HIP)
    (void) dest;
    (void) src;
    (void) src_device;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_hip_from_hip HIP is not enabled." << std::endl;
    return -1;
#else
    // copy src to gpu
    size_t n_bytes = n_elem*sizeof(T);
    hipError_t ierr = hipSuccess;

    int dest_device = -1;
    if ((ierr = hipGetDevice(&dest_device)) != hipSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to get the current device id. "
            << hipGetErrorString(ierr) << std::endl;
        return -1;
    }

    if ((ierr = hipMemcpyPeer(dest, dest_device, src, src_device, n_bytes)) != hipSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to copy " << n_bytes
            << " from HIP device " << src_device << " to HIP device "
            << dest_device << ". " << hipGetErrorString(ierr) << std::endl;
        return -1;
    }

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "hamr::copy_to_hip_from_hip same " << n_elem
            << typeid(T).name() << sizeof(T) << " from device "
            << src_device << " to device " << dest_device << std::endl;
    }
#endif

    return 0;
#endif
}
#endif

/** Copies an array on the active HIP device.
 *
 * @param[in] dest an array of n elements accessible in HIP
 * @param[in] src an array of n elements accessible in HIP
 * @param[in] src_device the HIP device on which src is allocated
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
static int copy_to_hip_from_hip(T *dest, const U *src, int src_device, size_t n_elem
#if !defined(HAMR_HIP_OBJECTS)
    ,typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr
#endif
    )
{
#if !defined(HAMR_ENABLE_HIP)
    (void) dest;
    (void) src;
    (void) src_device;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_hip_from_hip HIP is not enabled." << std::endl;
    return -1;
#else
    // copy on the gpu
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

    hipError_t ierr = hipSuccess;

    // enable peer to peer access
    int dest_device = -1;
    if ((ierr = hipGetDevice(&dest_device)) != hipSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to get the current device id. "
            << hipGetErrorString(ierr) << std::endl;
        return -1;
    }

    int access = 0;
    if ((ierr = hipDeviceCanAccessPeer(&access, dest_device, src_device)) != hipSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to determine peer accessibility between "
            << dest_device << " and " << src_device << ". "
            << hipGetErrorString(ierr) << std::endl;
        return -1;
    }

    if (!access)
    {
        // NOTE: could fall back to cduaMemcpyPeer here
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Can't access device " << src_device
            << " from " << dest_device << std::endl;
        return -1;
    }

    if ((ierr = hipDeviceEnablePeerAccess(src_device, 0)) != hipSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to enable peer accessibility between "
            << dest_device << " and " << src_device << ". "
            << hipGetErrorString(ierr) << std::endl;
        return -1;
    }

    // invoke the casting copy kernel
    hamr::hip_kernels::copy<<<block_grid, thread_grid>>>(dest, src, n_elem);
    if ((ierr = hipGetLastError()) != hipSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to launch the copy kernel. "
            << hipGetErrorString(ierr) << std::endl;
        return -1;
    }

    // disable peer to peer memory map
    if ((ierr = hipDeviceDisablePeerAccess(src_device)) != hipSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to disable peer accessibility between "
            << dest_device << " and " << src_device << ". "
            << hipGetErrorString(ierr) << std::endl;
        return -1;
    }

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "hamr::copy_to_hip_from_hip " << n_elem
            << " from " << typeid(U).name() << sizeof(U) << " to "
            << typeid(T).name() << sizeof(T) << " from device "
            << src_device << " to device " << dest_device << std::endl;
    }
#endif

    return 0;
#endif
}

#if !defined(HAMR_HIP_OBJECTS)
/** Copies an array from the active HIP device.
 *
 * @param[in] dest an array of n elements accessible on the CPU
 * @param[in] src an array of n elements accessible in HIP
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
static int copy_to_cpu_from_hip(T *dest, const U *src, size_t n_elem,
   typename std::enable_if<!std::is_arithmetic<T>::value>::type * = nullptr)
{
#if !defined(HAMR_ENABLE_HIP)
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_cpu_from_hip HIP is not enabled." << std::endl;
    return -1;
#else
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_cpu_from_hip HAMR_HIP_OBJECTS is not enabled." << std::endl;
    abort();
    return -1;
#endif
}
#else
/** Copies an array from the active HIP device (fast path for arrays of
 * arithmetic types of the same type).
 *
 * @param[in] dest an array of n elements accessible in HIP
 * @param[in] src an array of n elements accessible on the CPU
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T>
static int copy_to_cpu_from_hip(T *dest, const T *src, size_t n_elem,
   typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr)
{
#if !defined(HAMR_ENABLE_HIP)
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_cpu_from_hip HIP is not enabled." << std::endl;
    return -1;
#else
    // copy src to cpu
    size_t n_bytes = n_elem*sizeof(T);
    hipError_t ierr = hipSuccess;
    if ((ierr = hipMemcpy(dest, src, n_bytes, hipMemcpyDeviceToHost)) != hipSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to copy " << n_bytes << ". "
            << hipGetErrorString(ierr) << std::endl;
        return -1;
    }

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "hamr::copy_to_cpu_from_hip same " << n_elem
            << " " << typeid(T).name() << sizeof(T) << std::endl;
    }
#endif

    return 0;
#endif
}
#endif

/** Copies an array from the active HIP device.
 *
 * @param[in] dest an array of n elements accessible on the CPU
 * @param[in] src an array of n elements accessible in HIP
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
static int copy_to_cpu_from_hip(T *dest, const U *src, size_t n_elem
#if !defined(HAMR_HIP_OBJECTS)
    ,typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr
#endif
    )
{
#if !defined(HAMR_ENABLE_HIP)
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_cpu_from_hip HIP is not enabled." << std::endl;
    return -1;
#else

    // apply the copy on the gpu in a temporary buffer
    // copy the buffer to the cpu

    // allocate a temporary buffer on the GPU
    auto sptmp = hamr::hip_malloc_allocator<T>::allocate(n_elem);
    T *ptmp = sptmp.get();

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

    // invoke the casting copy kernel on the GPU
    hipError_t ierr = hipSuccess;
    hamr::hip_kernels::copy<<<block_grid, thread_grid>>>(ptmp, src, n_elem);
    if ((ierr = hipGetLastError()) != hipSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to launch the copy kernel. "
            << hipGetErrorString(ierr) << std::endl;
        return -1;
    }

    // copy the data to the CPU
    size_t n_bytes = n_elem*sizeof(T);
    if ((ierr = hipMemcpy(dest, ptmp, n_bytes, hipMemcpyDeviceToHost)) != hipSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to copy " << n_bytes << ". "
            << hipGetErrorString(ierr) << std::endl;
        return -1;
    }

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "hamr::copy_to_cpu_from_hip " << n_elem
            << " from " << typeid(U).name() << sizeof(U) << " to "
            << typeid(T).name() << sizeof(T) << std::endl;
    }
#endif

    return 0;
#endif
}

}

#endif
