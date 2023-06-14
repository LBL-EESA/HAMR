#ifndef hamr_cuda_copy_async_impl_h
#define hamr_cuda_copy_async_impl_h

#include "hamr_config.h"
#include "hamr_env.h"
#if defined(HAMR_ENABLE_CUDA)
#include "hamr_cuda_kernels.h"
#include "hamr_cuda_launch.h"
#include "hamr_cuda_malloc_allocator.h"
#include <cuda.h>
#include <cuda_runtime.h>
#else
using cudaStream_t = void;
#endif
#include "hamr_malloc_allocator.h"

#include <cstring>
#include <cstdlib>
#include <iostream>
#include <memory>

/// heterogeneous accelerator memory resource
namespace hamr
{
#if !defined(HAMR_ENABLE_OBJECTS)
// ---------------------------------------------------------------------------
template <typename T, typename U>
int copy_to_cuda_from_host(cudaStream_t str, T *dest, const U *src, size_t n_elem,
   typename std::enable_if<!std::is_arithmetic<T>::value>::type *)
{
#if !defined(HAMR_ENABLE_CUDA)
    (void) str;
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_cuda_from_host CUDA is not enabled." << std::endl;
    return -1;
#else
    (void) str;
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_cuda_from_host HAMR_ENABLE_OBJECTS is not enabled." << std::endl;
    abort();
    return -1;
#endif
}
#else
// ---------------------------------------------------------------------------
template <typename T>
int copy_to_cuda_from_host(cudaStream_t str, T *dest, const T *src, size_t n_elem,
   typename std::enable_if<std::is_arithmetic<T>::value>::type *)
{
#if !defined(HAMR_ENABLE_CUDA)
    (void) str;
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_cuda_from_host CUDA is not enabled." << std::endl;
    return -1;
#else

    // copy src to gpu
    size_t n_bytes = n_elem*sizeof(T);
    cudaError_t ierr = cudaSuccess;
    if ((ierr = cudaMemcpyAsync(dest, src, n_bytes,
        cudaMemcpyHostToDevice, str)) != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to copy " << n_bytes << ". "
            << cudaGetErrorString(ierr) << std::endl;
        return -1;
    }

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "hamr::copy_to_cuda_from_host same " << n_elem
            << " " << typeid(T).name() << sizeof(T) << std::endl;
    }
#endif

    return 0;
#endif
}
#endif

// ---------------------------------------------------------------------------
template <typename T, typename U>
int copy_to_cuda_from_host(cudaStream_t str, T *dest, const U *src, size_t n_elem
#if !defined(HAMR_ENABLE_OBJECTS)
    ,typename std::enable_if<std::is_arithmetic<T>::value>::type *
#endif
    )
{
#if !defined(HAMR_ENABLE_CUDA)
    (void) str;
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_cuda_from_host CUDA is not enabled." << std::endl;
    return -1;
#else
    // copy src to gpu
    // apply the copy on the gpu

    // allocate a temporary buffer on the GPU
    std::shared_ptr<U> pTmp = hamr::cuda_malloc_allocator<U>::allocate(str, n_elem);

    // copy the data
    size_t n_bytes = n_elem*sizeof(U);
    cudaError_t ierr = cudaSuccess;
    if ((ierr = cudaMemcpyAsync(pTmp.get(), src, n_bytes, cudaMemcpyHostToDevice, str)) != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to copy " << n_bytes << ". "
            << cudaGetErrorString(ierr) << std::endl;
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
    hamr::cuda_kernels::copy<<<block_grid, thread_grid, 0, str>>>(dest, pTmp.get(), n_elem);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to launch the copy kernel. "
            << cudaGetErrorString(ierr) << std::endl;
        return -1;
    }

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "hamr::copy_to_cuda_from_host " << n_elem
            << " from " << typeid(U).name() << sizeof(U) << " to "
            << typeid(T).name() << sizeof(T) << std::endl;
    }
#endif

    return 0;
#endif
}

#if !defined(HAMR_ENABLE_OBJECTS)
// ---------------------------------------------------------------------------
template <typename T, typename U>
int copy_to_cuda_from_cuda(cudaStream_t str, T *dest, const U *src, size_t n_elem,
   typename std::enable_if<!std::is_arithmetic<T>::value>::type *)
{
#if !defined(HAMR_ENABLE_CUDA)
    (void) str;
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_cuda_from_cuda CUDA is not enabled." << std::endl;
    return -1;
#else
    (void) str;
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_cuda_from_cuda HAMR_ENABLE_OBJECTS is not enabled." << std::endl;
    return -1;
#endif
}
#else
// ---------------------------------------------------------------------------
template <typename T>
int copy_to_cuda_from_cuda(cudaStream_t str, T *dest, const T *src, size_t n_elem,
    typename std::enable_if<std::is_arithmetic<T>::value>::type *)
{
#if !defined(HAMR_ENABLE_CUDA)
    (void) str;
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_cuda_from_cuda CUDA is not enabled." << std::endl;
    return -1;
#else
    // copy src to gpu
    size_t n_bytes = n_elem*sizeof(T);
    cudaError_t ierr = cudaSuccess;
    if ((ierr = cudaMemcpyAsync(dest, src, n_bytes,
        cudaMemcpyDeviceToDevice, str)) != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to copy " << n_bytes << ". "
            << cudaGetErrorString(ierr) << std::endl;
        return -1;
    }

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "hamr::copy_to_cuda_from_cuda same " << n_elem
            << typeid(T).name() << sizeof(T) << std::endl;
    }
#endif

    return 0;
#endif
}
#endif

// ---------------------------------------------------------------------------
template <typename T, typename U>
int copy_to_cuda_from_cuda(cudaStream_t str, T *dest, const U *src, size_t n_elem
#if !defined(HAMR_ENABLE_OBJECTS)
    ,typename std::enable_if<std::is_arithmetic<T>::value>::type *
#endif
    )
{
#if !defined(HAMR_ENABLE_CUDA)
    (void) str;
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_cuda_from_cuda CUDA is not enabled." << std::endl;
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
    cudaError_t ierr = cudaSuccess;
    hamr::cuda_kernels::copy<<<block_grid, thread_grid, 0, str>>>(dest, src, n_elem);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to launch the copy kernel. "
            << cudaGetErrorString(ierr) << std::endl;
        return -1;
    }

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "hamr::copy_to_cuda_from_cuda " << n_elem
            << " from " << typeid(U).name() << sizeof(U) << " to "
            << typeid(T).name() << sizeof(T) << std::endl;
    }
#endif

    return 0;
#endif
}

#if !defined(HAMR_ENABLE_OBJECTS)
// ---------------------------------------------------------------------------
template <typename T, typename U>
int copy_to_cuda_from_cuda(cudaStream_t str, T *dest,
    const U *src, int src_device, size_t n_elem,
    typename std::enable_if<!std::is_arithmetic<T>::value>::type *)
{
#if !defined(HAMR_ENABLE_CUDA)
    (void) str;
    (void) dest;
    (void) src;
    (void) src_device;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_cuda_from_cuda CUDA is not enabled." << std::endl;
    return -1;
#else
    (void) str;
    (void) dest;
    (void) src;
    (void) src_device;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_cuda_from_cuda HAMR_ENABLE_OBJECTS is not enabled." << std::endl;
    return -1;
#endif
}
#else
// ---------------------------------------------------------------------------
template <typename T>
int copy_to_cuda_from_cuda(cudaStream_t str, T *dest,
    const T *src, int src_device, size_t n_elem,
    typename std::enable_if<std::is_arithmetic<T>::value>::type *)
{
#if !defined(HAMR_ENABLE_CUDA)
    (void) str;
    (void) dest;
    (void) src;
    (void) src_device;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_cuda_from_cuda CUDA is not enabled." << std::endl;
    return -1;
#else

    // copy src to gpu
    size_t n_bytes = n_elem*sizeof(T);
    cudaError_t ierr = cudaSuccess;

    int dest_device = -1;
    if ((ierr = cudaGetDevice(&dest_device)) != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to get the current device id. "
            << cudaGetErrorString(ierr) << std::endl;
        return -1;
    }

    if ((ierr = cudaMemcpyPeerAsync(dest, dest_device, src,
        src_device, n_bytes, str)) != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to copy " << n_bytes
            << " from CUDA device " << src_device << " to CUDA device "
            << dest_device << ". " << cudaGetErrorString(ierr) << std::endl;
        return -1;
    }

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "hamr::copy_to_cuda_from_cuda same " << n_elem
            << typeid(T).name() << sizeof(T) << " from device "
            << src_device << " to device " << dest_device << std::endl;
    }
#endif

    return 0;
#endif
}
#endif

// ---------------------------------------------------------------------------
template <typename T, typename U>
int copy_to_cuda_from_cuda(cudaStream_t str, T *dest,
    const U *src, int src_device, size_t n_elem
#if !defined(HAMR_ENABLE_OBJECTS)
    ,typename std::enable_if<std::is_arithmetic<T>::value>::type *
#endif
    )
{
#if !defined(HAMR_ENABLE_CUDA)
    (void) str;
    (void) dest;
    (void) src;
    (void) src_device;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_cuda_from_cuda CUDA is not enabled." << std::endl;
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

    cudaError_t ierr = cudaSuccess;

    // enable peer to peer access
    int dest_device = -1;
    if ((ierr = cudaGetDevice(&dest_device)) != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to get the current device id. "
            << cudaGetErrorString(ierr) << std::endl;
        return -1;
    }

    int access = 0;
    if ((ierr = cudaDeviceCanAccessPeer(&access, dest_device, src_device)) != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to determine peer accessibility between "
            << dest_device << " and " << src_device << ". "
            << cudaGetErrorString(ierr) << std::endl;
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

    if ((ierr = cudaDeviceEnablePeerAccess(src_device, 0)) != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to enable peer accessibility between "
            << dest_device << " and " << src_device << ". "
            << cudaGetErrorString(ierr) << std::endl;
        return -1;
    }

    // invoke the casting copy kernel
    hamr::cuda_kernels::copy<<<block_grid, thread_grid, 0, str>>>(dest, src, n_elem);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to launch the copy kernel. "
            << cudaGetErrorString(ierr) << std::endl;
        return -1;
    }

    // disable peer to peer memory map
    if ((ierr = cudaDeviceDisablePeerAccess(src_device)) != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to disable peer accessibility between "
            << dest_device << " and " << src_device << ". "
            << cudaGetErrorString(ierr) << std::endl;
        return -1;
    }

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "hamr::copy_to_cuda_from_cuda " << n_elem
            << " from " << typeid(U).name() << sizeof(U) << " to "
            << typeid(T).name() << sizeof(T) << " from device "
            << src_device << " to device " << dest_device << std::endl;
    }
#endif

    return 0;
#endif
}

#if !defined(HAMR_ENABLE_OBJECTS)
// ---------------------------------------------------------------------------
template <typename T, typename U>
int copy_to_host_from_cuda(cudaStream_t str, T *dest,
    const U *src, size_t n_elem,
    typename std::enable_if<!std::is_arithmetic<T>::value>::type *)
{
#if !defined(HAMR_ENABLE_CUDA)
    (void) str;
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_host_from_cuda CUDA is not enabled." << std::endl;
    return -1;
#else
    (void) str;
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_host_from_cuda HAMR_ENABLE_OBJECTS is not enabled." << std::endl;
    abort();
    return -1;
#endif
}
#else
// ---------------------------------------------------------------------------
template <typename T>
int copy_to_host_from_cuda(cudaStream_t str, T *dest,
    const T *src, size_t n_elem,
    typename std::enable_if<std::is_arithmetic<T>::value>::type *)
{
#if !defined(HAMR_ENABLE_CUDA)
    (void) str;
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_host_from_cuda CUDA is not enabled." << std::endl;
    return -1;
#else
    // copy src to host
    size_t n_bytes = n_elem*sizeof(T);
    cudaError_t ierr = cudaSuccess;
    if ((ierr = cudaMemcpyAsync(dest, src, n_bytes,
        cudaMemcpyDeviceToHost, str)) != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to copy " << n_bytes << ". "
            << cudaGetErrorString(ierr) << std::endl;
        return -1;
    }

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "hamr::copy_to_host_from_cuda same " << n_elem
            << " " << typeid(T).name() << sizeof(T) << std::endl;
    }
#endif

    return 0;
#endif
}
#endif

// ---------------------------------------------------------------------------
template <typename T, typename U>
int copy_to_host_from_cuda(cudaStream_t str, T *dest, const U *src, size_t n_elem
#if !defined(HAMR_ENABLE_OBJECTS)
    ,typename std::enable_if<std::is_arithmetic<T>::value>::type *
#endif
    )
{
#if !defined(HAMR_ENABLE_CUDA)
    (void) str;
    (void) dest;
    (void) src;
    (void) n_elem;/
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_host_from_cuda CUDA is not enabled." << std::endl;
    return -1;
#else
    // apply the copy on the gpu in a temporary buffer
    // copy the buffer to the host

    // allocate a temporary buffer on the GPU
    auto sptmp = hamr::cuda_malloc_allocator<T>::allocate(str, n_elem);
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
    cudaError_t ierr = cudaSuccess;
    hamr::cuda_kernels::copy<<<block_grid, thread_grid, 0, str>>>(ptmp, src, n_elem);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to launch the copy kernel. "
            << cudaGetErrorString(ierr) << std::endl;
        return -1;
    }

    // copy the data to the host
    size_t n_bytes = n_elem*sizeof(T);
    if ((ierr = cudaMemcpyAsync(dest, ptmp, n_bytes, cudaMemcpyDeviceToHost, str)) != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to copy " << n_bytes << ". "
            << cudaGetErrorString(ierr) << std::endl;
        return -1;
    }

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "hamr::copy_to_host_from_cuda " << n_elem
            << " from " << typeid(U).name() << sizeof(U) << " to "
            << typeid(T).name() << sizeof(T) << std::endl;
    }
#endif

    return 0;
#endif
}

}

#endif
