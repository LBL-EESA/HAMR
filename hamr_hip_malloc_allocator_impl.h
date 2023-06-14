#ifndef hamr_hip_malloc_allocator_impl_h
#define hamr_hip_malloc_allocator_impl_h

#include <iostream>
#include <type_traits>
#include <memory>
#include <typeinfo>
#include <cassert>
#include <cstring>
#include <cstdlib>

#include <hip/hip_runtime.h>


#include "hamr_config.h"
#include "hamr_hip_kernels.h"
#include "hamr_env.h"

namespace hamr
{

// --------------------------------------------------------------------------
template <typename T>
hip_malloc_deleter<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
    ::hip_malloc_deleter(T *ptr, size_t n) : m_ptr(ptr), m_elem(n)
{
#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "created hip_malloc_deleter for array of " << n
            << " objects of type " << typeid(T).name() << sizeof(T)
            << " at " << m_ptr << std::endl;
    }
#endif
}

// --------------------------------------------------------------------------
template <typename T>
void
hip_malloc_deleter<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
    ::operator()(T *ptr)
{
#if !defined(HAMR_ENABLE_OBJECTS)
    (void) ptr;
     std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
         " hip_malloc_deleter dealllocate objects failed."
        " HAMR_ENABLE_OBJECTS is not enabled" << std::endl;
     abort();
#else
    assert(ptr == m_ptr);

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
    hipError_t ierr = hipSuccess;
    hip_kernels::destruct<T><<<block_grid, thread_grid>>>(ptr, m_elem);
    if ((ierr = hipGetLastError()) != hipSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to launch the construct kernel. "
            << hipGetErrorString(ierr) << std::endl;
        return;
    }

    // free the array
    ierr = hipFree(ptr);
    (void) ierr;

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "hip_malloc_deleter deleting array of " << m_elem
            << " objects of type " << typeid(T).name() << sizeof(T)
            << " at " << m_ptr << std::endl;
    }
#endif

#endif
}





// --------------------------------------------------------------------------
template <typename T>
hip_malloc_deleter<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
    ::hip_malloc_deleter(T *ptr, size_t n) : m_ptr(ptr), m_elem(n)
{
#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "created hip_malloc_deleter for array of " << n
            << " numbers of type " << typeid(T).name() << sizeof(T)
            << " at " << m_ptr << std::endl;
    }
#endif
}

// --------------------------------------------------------------------------
template <typename T>
void
hip_malloc_deleter<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
    ::operator()(T *ptr)
{
    assert(ptr == m_ptr);

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "hip_malloc_deleter deleting array of " << m_elem
            << " numbers of type " << typeid(T).name() << sizeof(T)
            << " at " << m_ptr << std::endl;
    }
#endif

    // free the array
    hipError_t ierr = hipSuccess;
    ierr = hipFree(ptr);
    (void) ierr;
}






// --------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T>
hip_malloc_allocator<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
    ::allocate(size_t n_elem)
{
#if !defined(HAMR_ENABLE_OBJECTS)
    (void) n_elem;
     std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
         " hip_malloc_allocator allocate objects failed."
        " HAMR_ENABLE_OBJECTS is not enabled" << std::endl;
     abort();
     return nullptr;
#else
    size_t n_bytes = n_elem*sizeof(T);

    // allocate
    T *ptr = nullptr;
    hipError_t ierr = hipSuccess;
    if ((ierr = hipMalloc(&ptr, n_bytes)) != hipSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to hipMalloc " << n_elem << " of "
            << typeid(T).name() << " total " << n_bytes  << " bytes. "
            << hipGetErrorString(ierr) << std::endl;
        return nullptr;
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
            << hipGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // construct
    hip_kernels::construct<T><<<block_grid, thread_grid>>>(ptr, n_elem);
    if ((ierr = hipGetLastError()) != hipSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to launch the construct kernel. "
            << hipGetErrorString(ierr) << std::endl;
        return nullptr;
    }

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "hip_malloc_allocator allocating array of " << n_elem
            << " objects of type " << typeid(T).name() << sizeof(T)
            << " at " << ptr << std::endl;
    }
#endif

    // package
    return std::shared_ptr<T>(ptr, hip_malloc_deleter<T>(ptr, n_elem));
#endif
}

// --------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T>
hip_malloc_allocator<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
    ::allocate(size_t n_elem, const T &val)
{
#if !defined(HAMR_ENABLE_OBJECTS)
    (void) n_elem;
    (void) val;
     std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
         " hip_malloc_allocator allocate objects failed."
        " HAMR_ENABLE_OBJECTS is not enabled" << std::endl;
     abort();
     return nullptr;
#else
    size_t n_bytes = n_elem*sizeof(T);

    // allocate
    T *ptr = nullptr;
    hipError_t ierr = hipSuccess;
    if ((ierr = hipMalloc(&ptr, n_bytes)) != hipSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to hipMalloc " << n_elem << " of "
            << typeid(T).name() << sizeof(T) << " total " << n_bytes
            << " bytes. " << hipGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // get launch parameters
    int device_id = -1;
    dim3 block_grid;
    int n_blocks = 0;
    dim3 thread_grid = 0;
    if (hamr::partition_thread_blocks(device_id, n_elem, 8, block_grid, n_blocks,
        thread_grid))
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to determine launch properties. "
            << hipGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // construct
    hip_kernels::construct<T><<<block_grid, thread_grid>>>(ptr, n_elem, val);
    if ((ierr = hipGetLastError()) != hipSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to launch the construct kernel. "
            << hipGetErrorString(ierr) << std::endl;
        return nullptr;
    }

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "hip_malloc_allocator allocating array of " << n_elem
            << " objects of type " << typeid(T).name() << sizeof(T)
            << " at " << ptr << " initialized to " << val << std::endl;
    }
#endif

    // package
    return std::shared_ptr<T>(ptr, hip_malloc_deleter<T>(ptr, n_elem));
#endif
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
std::shared_ptr<T>
hip_malloc_allocator<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
    ::allocate(size_t n_elem, const U *vals, bool hipVals)
{
#if !defined(HAMR_ENABLE_OBJECTS)
    (void) n_elem;
    (void) vals;
    (void) hipVals;
     std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
         " hip_malloc_allocator allocate objects failed."
        " HAMR_ENABLE_OBJECTS is not enabled" << std::endl;
     abort();
     return nullptr;
#else
    size_t n_bytes = n_elem*sizeof(T);

    // allocate
    T *ptr = nullptr;
    hipError_t ierr = hipSuccess;
    if ((ierr = hipMalloc(&ptr, n_bytes)) != hipSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to hipMalloc " << n_elem << " of "
            << typeid(T).name() << " total " << n_bytes  << " bytes. "
            << hipGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // move the existing array to the GPU
    U *tmp = nullptr;
    if (!hipVals)
    {
        size_t n_bytes_vals = n_elem*sizeof(U);
        if ((ierr = hipMalloc(&tmp, n_bytes_vals)) != hipSuccess)
        {
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                " Failed to hipMalloc " << n_elem << " of "
                << typeid(T).name() << " total " << n_bytes_vals  << "bytes. "
                << hipGetErrorString(ierr) << std::endl;
            return nullptr;
        }

        if ((ierr = hipMemcpy(tmp, vals, n_bytes_vals, hipMemcpyHostToDevice)) != hipSuccess)
        {
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                " Failed to hipMemcpy array of " << n_elem
                << " of " << typeid(T).name() << " total " << n_bytes_vals  << "bytes. "
                << hipGetErrorString(ierr) << std::endl;
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
            << hipGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // construct
    hip_kernels::construct<T><<<block_grid, thread_grid>>>(ptr, n_elem, vals);
    if ((ierr = hipGetLastError()) != hipSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to launch the construct kernel. "
            << hipGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // free up temporary buffers
    if (!hipVals)
    {
        ierr = hipFree(tmp);
        (void) ierr;
    }

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "hip_malloc_allocator allocating array of " << n_elem
            << " objects of type " << typeid(T).name() << sizeof(T)
            << " at " << ptr << " initialized from the "
            << (hipVals ? "HIP" : "host") << " array of objects of "
            << typeid(U).name() << sizeof(U) << " at " << vals
            << std::endl;
    }
#endif

    // package
    return std::shared_ptr<T>(ptr, hip_malloc_deleter<T>(ptr, n_elem));
#endif
}





// --------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T>
hip_malloc_allocator<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
    ::allocate(size_t n_elem)
{
    size_t n_bytes = n_elem*sizeof(T);

    // allocate
    T *ptr = nullptr;
    hipError_t ierr = hipSuccess;
    if ((ierr = hipMalloc(&ptr, n_bytes)) != hipSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to hipMalloc " << n_elem << " of "
            << typeid(T).name() << " total " << n_bytes  << "bytes. "
            << hipGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // construct
#if defined(HAMR_INIT_ALLOC)
    hipMemset(ptr, 0, n_bytes);
#endif

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "hip_malloc_allocator allocating array of " << n_elem
            << " numbers of type " << typeid(T).name() << sizeof(T)
            << " at " << ptr << std::endl;
    }
#endif

    // package
    return std::shared_ptr<T>(ptr, hip_malloc_deleter<T>(ptr, n_elem));
}

// --------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T>
hip_malloc_allocator<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
    ::allocate(size_t n_elem, const T &val)
{
    size_t n_bytes = n_elem*sizeof(T);

    // allocate
    T *ptr = nullptr;
    hipError_t ierr = hipSuccess;
    if ((ierr = hipMalloc(&ptr, n_bytes)) != hipSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to hipMalloc " << n_elem << " of "
            << typeid(T).name() << " total " << n_bytes  << "bytes. "
            << hipGetErrorString(ierr) << std::endl;
        return nullptr;
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
            << hipGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // construct
    hip_kernels::fill<T><<<block_grid, thread_grid>>>(ptr, n_elem, val);
    if ((ierr = hipGetLastError()) != hipSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to launch the construct kernel. "
            << hipGetErrorString(ierr) << std::endl;
        return nullptr;
    }

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "hip_malloc_allocator allocating array of " << n_elem
            << " numbers of type " << typeid(T).name() << sizeof(T)
            << " at " << ptr << " initialized to " << val << std::endl;
    }
#endif

    // package
    return std::shared_ptr<T>(ptr, hip_malloc_deleter<T>(ptr, n_elem));
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
std::shared_ptr<T>
hip_malloc_allocator<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
    ::allocate(size_t n_elem, const U *vals, bool hipVals)
{
    size_t n_bytes = n_elem*sizeof(T);

    // allocate
    T *ptr = nullptr;
    hipError_t ierr = hipSuccess;
    if ((ierr = hipMalloc(&ptr, n_bytes)) != hipSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to hipMalloc " << n_elem << " of "
            << typeid(T).name() << " total " << n_bytes  << "bytes. "
            << hipGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // move the existing array to the GPU
    U *tmp = nullptr;
    if (!hipVals)
    {
        size_t n_bytes_vals = n_elem*sizeof(U);

        if ((ierr = hipMalloc(&tmp, n_bytes_vals)) != hipSuccess)
        {
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                " Failed to hipMalloc " << n_elem << " of "
                << typeid(T).name() << " total " << n_bytes_vals  << "bytes. "
                << hipGetErrorString(ierr) << std::endl;
            return nullptr;
        }

        if ((ierr = hipMemcpy(tmp, vals, n_bytes_vals,
            hipMemcpyHostToDevice)) != hipSuccess)
        {
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                " Failed to hipMemcpy array of " << n_elem
                << " of " << typeid(T).name() << " total " << n_bytes_vals  << "bytes. "
                << hipGetErrorString(ierr) << std::endl;
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
            << hipGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // construct
    hip_kernels::fill<T><<<block_grid, thread_grid>>>(ptr, n_elem, vals);
    if ((ierr = hipGetLastError()) != hipSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to launch the construct kernel. "
            << hipGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // free up temporary buffers
    if (!hipVals)
    {
        ierr = hipFree(tmp);
        (void) ierr;
    }

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "hip_malloc_allocator allocating array of " << n_elem
            << " numbers of type " << typeid(T).name() << sizeof(T)
            << " at " << ptr << " initialized from " << (hipVals ? "HIP" : "host")
            <<  " array at " << vals << std::endl;
    }
#endif


    // package
    return std::shared_ptr<T>(ptr, hip_malloc_deleter<T>(ptr, n_elem));
}

}

#endif
