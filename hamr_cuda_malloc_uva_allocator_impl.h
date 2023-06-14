#ifndef hamr_cuda_malloc_uva_allocator_impl_h
#define hamr_cuda_malloc_uva_allocator_impl_h

#include "hamr_config.h"
#include "hamr_cuda_kernels.h"
#include "hamr_env.h"


#include <iostream>
#include <type_traits>
#include <memory>
#include <typeinfo>
#include <cassert>
#include <cstring>
#include <cstdlib>

#include <cuda.h>
#include <cuda_runtime.h>

namespace hamr
{

// --------------------------------------------------------------------------
template <typename T>
cuda_malloc_uva_deleter<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
    ::cuda_malloc_uva_deleter(cudaStream_t str, T *ptr, size_t n) :
        m_ptr(ptr), m_elem(n), m_str(str)
{
#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "created cuda_malloc_uva_deleter for array of " << n
            << " objects of type " << typeid(T).name() << std::endl;
    }
#endif
}

// --------------------------------------------------------------------------
template <typename T>
void
cuda_malloc_uva_deleter<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
    ::operator()(T *ptr)
{
#if !defined(HAMR_ENABLE_OBJECTS)
    (void) ptr;
     std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
         " cuda_malloc_uva_deleter dealllocate objects failed."
        " HAMR_ENABLE_OBJECTS is not enabled" << std::endl;
     abort();
#else
    assert(ptr == m_ptr);

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "cuda_malloc_uva_deleter deleting array of " << m_elem
            << " objects of type " << typeid(T).name() << sizeof(T)
            << " at " << m_ptr  << std::endl;
    }
#endif

    // get launch parameters
    int device_id = -1;
    dim3 block_grid;
    int n_blocks = 0;
    dim3 thread_grid = 0;
    if (hamr::partition_thread_blocks(device_id, m_elem, 8, block_grid,
        n_blocks, thread_grid))
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to determine launch properties." << std::endl;
        return;
    }

    // destruct
    cudaError_t ierr = cudaSuccess;
    cuda_kernels::destruct<T><<<block_grid, thread_grid, 0, m_str>>>(ptr, m_elem);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to launch the destruct kernel. "
            << cudaGetErrorString(ierr) << std::endl;
        return;
    }

    // free the array
    cudaFree(ptr);
#endif
}






// --------------------------------------------------------------------------
template <typename T>
cuda_malloc_uva_deleter<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
    ::cuda_malloc_uva_deleter(cudaStream_t str, T *ptr, size_t n) :
        m_ptr(ptr), m_elem(n)
{
    (void) str;
#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "created cuda_malloc_uva_deleter for array of " << n
            << " numbers of type " << typeid(T).name() << sizeof(T)
            << std::endl;
    }
#endif
}

// --------------------------------------------------------------------------
template <typename T>
void
cuda_malloc_uva_deleter<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
    ::operator()(T *ptr)
{
    assert(ptr == m_ptr);

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "cuda_malloc_uva_deleter deleting array of " << m_elem
            << " numbers of type " << typeid(T).name() << sizeof(T)
            << std::endl;
    }
#endif

    // free the array
    cudaFree(ptr);
}







// --------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T>
cuda_malloc_uva_allocator<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
    ::allocate(cudaStream_t str, size_t n_elem)
{
#if !defined(HAMR_ENABLE_OBJECTS)
    (void) str;
    (void) n_elem;
     std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
         " cuda_malloc_uva_allocator allocate objects failed."
        " HAMR_ENABLE_OBJECTS is not enabled" << std::endl;
     abort();
     return nullptr;
#else
    size_t n_bytes = n_elem*sizeof(T);

    // allocate
    T *ptr = nullptr;
    cudaError_t ierr = cudaSuccess;
    if ((ierr = cudaMallocManaged(&ptr, n_bytes)) != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to cudaMallocManaged " << n_elem << " of "
            << typeid(T).name() << sizeof(T) << " total " << n_bytes  << "bytes. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // attach the stream
    if (str)
    {
        if ((ierr = cudaStreamAttachMemAsync(str, ptr)) != cudaSuccess)
        {
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                " Failed to associate managed memory with the given stream. "
                << cudaGetErrorString(ierr) << std::endl;
            return nullptr;
        }
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
            " Failed to determine launch properties. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // construct
    cuda_kernels::construct<T><<<block_grid, thread_grid, 0, str>>>(ptr, n_elem);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to launch the construct kernel. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "cuda_malloc_uva_allocator allocating array of " << n_elem
            << " objects of type " << typeid(T).name() << sizeof(T)
            << " at " << ptr << std::endl;
    }
#endif

    // package
    return std::shared_ptr<T>(ptr, cuda_malloc_uva_deleter<T>(str, ptr, n_elem));
#endif
}

// --------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T>
cuda_malloc_uva_allocator<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
    ::allocate(cudaStream_t str, size_t n_elem, const T &val)
{
#if !defined(HAMR_ENABLE_OBJECTS)
    (void) str;
    (void) n_elem;
    (void) val;
     std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
         " cuda_malloc_uva_allocator allocate objects failed."
        " HAMR_ENABLE_OBJECTS is not enabled" << std::endl;
     abort();
     return nullptr;
#else

    // allocate
    T *ptr = nullptr;
    size_t n_bytes = n_elem*sizeof(T);

    cudaError_t ierr = cudaSuccess;
    if ((ierr = cudaMallocManaged(&ptr, n_bytes)) != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to cudaMallocManaged " << n_elem << " of "
            << typeid(T).name() << " total " << n_bytes  << "bytes. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // attach the stream
    if (str)
    {
        if ((ierr = cudaStreamAttachMemAsync(str, ptr)) != cudaSuccess)
        {
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                " Failed to associate managed memory with the given stream. "
                << cudaGetErrorString(ierr) << std::endl;
            return nullptr;
        }
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
            " Failed to determine launch properties. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // construct
    cuda_kernels::construct<T><<<block_grid, thread_grid, 0, str>>>(ptr, n_elem, val);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to launch the construct kernel. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "cuda_malloc_uva_allocator allocating array of " << n_elem
            << " objects of type " << typeid(T).name() << sizeof(T)
            << " at " << ptr << " initialized to " << val
            << std::endl;
    }
#endif

    // package
    return std::shared_ptr<T>(ptr, cuda_malloc_uva_deleter<T>(str, ptr, n_elem));
#endif
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
std::shared_ptr<T>
cuda_malloc_uva_allocator<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
    ::allocate(cudaStream_t str, size_t n_elem, const U *vals, bool cudaVals)
{
#if !defined(HAMR_ENABLE_OBJECTS)
    (void) str;
    (void) n_elem;
    (void) vals;
    (void) cudaVals;
     std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
         " cuda_malloc_uva_allocator allocate objects failed."
        " HAMR_ENABLE_OBJECTS is not enabled" << std::endl;
     abort();
     return nullptr;
#else

    // allocate
    T *ptr = nullptr;
    size_t n_bytes = n_elem*sizeof(T);

    cudaError_t ierr = cudaSuccess;
    if ((ierr = cudaMallocManaged(&ptr, n_bytes)) != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to cudaMallocManaged " << n_elem << " of "
            << typeid(T).name() << " total " << n_bytes  << "bytes. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // attach the stream
    if (str)
    {
        if ((ierr = cudaStreamAttachMemAsync(str, ptr)) != cudaSuccess)
        {
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                " Failed to associate managed memory with the given stream. "
                << cudaGetErrorString(ierr) << std::endl;
            return nullptr;
        }
    }

    // move the existing array to the GPU
    U *tmp = nullptr;
    if (!cudaVals)
    {
        size_t n_bytes_vals = n_elem*sizeof(U);
        if ((ierr = cudaMalloc(&tmp, n_bytes_vals)) != cudaSuccess)
        {
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                " Failed to cudaMalloc " << n_elem << " of "
                << typeid(U).name() << sizeof(U) << " total " << n_bytes_vals
                << " bytes. " << cudaGetErrorString(ierr) << std::endl;
            return nullptr;
        }

        if ((ierr = cudaMemcpyAsync(tmp, vals, n_bytes_vals,
            cudaMemcpyHostToDevice, str)) != cudaSuccess)
        {
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                " Failed to cudaMemcpy array of " << n_elem
                << " of " << typeid(T).name() << " total " << n_bytes_vals
                << " bytes. " << cudaGetErrorString(ierr) << std::endl;
            return nullptr;
        }

        vals = tmp;
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
            " Failed to determine launch properties. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // construct
    cuda_kernels::construct<T><<<block_grid, thread_grid, 0, str>>>(ptr, n_elem, vals);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to launch the construct kernel. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // free up temporary buffers
    if (!cudaVals)
    {
        cudaFree(tmp);
    }

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "cuda_malloc_uva_allocator allocating array of " << n_elem
            << " objects of type " << typeid(T).name() << sizeof(T)
            << " at " << ptr  << " initialized from " << (cudaVals ? "CUDA" : "host")
            << " array of objects of type " << typeid(U).name() << sizeof(U)
            << " at " << vals << std::endl;
    }
#endif

    // package
    return std::shared_ptr<T>(ptr, cuda_malloc_uva_deleter<T>(str, ptr, n_elem));
#endif
}






// --------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T>
cuda_malloc_uva_allocator<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
    ::allocate(cudaStream_t str, size_t n_elem)
{
    size_t n_bytes = n_elem*sizeof(T);

    // allocate
    T *ptr = nullptr;
    cudaError_t ierr = cudaSuccess;
    if ((ierr = cudaMallocManaged(&ptr, n_bytes)) != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to cudaMalloc " << n_elem << " of "
            << typeid(T).name() << " total " << n_bytes  << "bytes. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // attach the stream
    if (str)
    {
        if ((ierr = cudaStreamAttachMemAsync(str, ptr)) != cudaSuccess)
        {
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                " Failed to associate managed memory with the given stream. "
                << cudaGetErrorString(ierr) << std::endl;
            return nullptr;
        }
    }

    // construct
#if defined(HAMR_INIT_ALLOC)
    cudaMemset(ptr, 0, n_bytes);
#endif

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "cuda_malloc_uva_allocator allocating array of " << n_elem
            << " numbers of type " << typeid(T).name() << sizeof(T)
            << " at " << ptr << std::endl;
    }
#endif

    // package
    return std::shared_ptr<T>(ptr, cuda_malloc_uva_deleter<T>(str, ptr, n_elem));
}

// --------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T>
cuda_malloc_uva_allocator<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
    ::allocate(cudaStream_t str, size_t n_elem, const T &val)
{
    size_t n_bytes = n_elem*sizeof(T);

    // allocate
    T *ptr = nullptr;
    cudaError_t ierr = cudaSuccess;
    if ((ierr = cudaMallocManaged(&ptr, n_bytes)) != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to cudaMallocManaged " << n_elem << " of "
            << typeid(T).name() << " total " << n_bytes  << "bytes. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // attach the stream
    if (str)
    {
        if ((ierr = cudaStreamAttachMemAsync(str, ptr)) != cudaSuccess)
        {
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                " Failed to associate managed memory with the given stream. "
                << cudaGetErrorString(ierr) << std::endl;
            return nullptr;
        }
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
            " Failed to determine launch properties. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // construct
    cuda_kernels::fill<T><<<block_grid, thread_grid, 0, str>>>(ptr, n_elem, val);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to launch the construct kernel. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "cuda_malloc_uva_allocator allocating array of " << n_elem
            << " objects of type " << typeid(T).name() << sizeof(T)
            << " at " << ptr << " initialized to " << val << std::endl;
    }
#endif

    // package
    return std::shared_ptr<T>(ptr, cuda_malloc_uva_deleter<T>(str, ptr, n_elem));
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
std::shared_ptr<T>
cuda_malloc_uva_allocator<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
    ::allocate(cudaStream_t str, size_t n_elem, const U *vals, bool cudaVals)
{
    size_t n_bytes = n_elem*sizeof(T);

    // allocate
    T *ptr = nullptr;
    cudaError_t ierr = cudaSuccess;
    if ((ierr = cudaMallocManaged(&ptr, n_bytes)) != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to cudaMallocManaged " << n_elem << " of "
            << typeid(T).name() << " total " << n_bytes  << "bytes. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // attach the stream
    if (str)
    {
        if ((ierr = cudaStreamAttachMemAsync(str, ptr)) != cudaSuccess)
        {
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                " Failed to associate managed memory with the given stream. "
                << cudaGetErrorString(ierr) << std::endl;
            return nullptr;
        }
    }

    // move the existing array to the GPU
    U *tmp = nullptr;
    if (!cudaVals)
    {
        size_t n_bytes_vals = n_elem*sizeof(U);
        if ((ierr = cudaMalloc(&tmp, n_bytes_vals)) != cudaSuccess)
        {
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                " Failed to cudaMalloc " << n_elem << " of "
                << typeid(T).name() << " total " << n_bytes_vals  << "bytes. "
                << cudaGetErrorString(ierr) << std::endl;
            return nullptr;
        }

        if ((ierr = cudaMemcpyAsync(tmp, vals, n_bytes_vals,
            cudaMemcpyHostToDevice, str)) != cudaSuccess)
        {
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                " Failed to cudaMemcpy array of " << n_elem
                << " of " << typeid(T).name() << " total " << n_bytes_vals  << "bytes. "
                << cudaGetErrorString(ierr) << std::endl;
            return nullptr;
        }

        vals = tmp;
    }

    // get launch parameters
    int device_id = -1;
    dim3 block_grid;
    int n_blocks = 0;
    dim3 thread_grid = 0;
    if (hamr::partition_thread_blocks(device_id, n_elem, 8, block_grid, n_blocks, thread_grid))
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to determine launch properties. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // construct
    cuda_kernels::fill<T><<<block_grid, thread_grid, 0, str>>>(ptr, n_elem, vals);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to launch the construct kernel. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // free up temporary buffers
    if (!cudaVals)
    {
        cudaFree(tmp);
    }

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "cuda_malloc_uva_allocator allocating array of " << n_elem
            << " objects of type " << typeid(T).name() << sizeof(T)
            << " at " << ptr  << " initialized from " << (cudaVals ? "CUDA" : "host")
            << " array " << vals << " objects of type " << typeid(U).name() << sizeof(T)
            << std::endl;
    }
#endif


    // package
    return std::shared_ptr<T>(ptr, cuda_malloc_uva_deleter<T>(str, ptr, n_elem));
}

}

#endif
