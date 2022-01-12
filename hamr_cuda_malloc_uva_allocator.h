#ifndef hamr_cuda_malloc_uva_allocator_h
#define hamr_cuda_malloc_uva_allocator_h

#include <iostream>
#include <type_traits>
#include <memory>
#include <typeinfo>
#include <cassert>
#include <cstring>
#include <cstdlib>

#include <cuda.h>
#include <cuda_runtime.h>

#include "hamr_config.h"
#include "hamr_cuda_kernels.h"
#include "hamr_env.h"

namespace hamr
{

/// a deleter for arrays allocated with cuda_malloc_uva
template <typename T, typename E = void>
class cuda_malloc_uva_deleter {};

/// a deleter for arrays allocated with cuda_malloc_uva, specialized for objects
template <typename T>
class cuda_malloc_uva_deleter<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
{
public:
    /** constructs the deleter
     * @param[in] ptr the pointer to the array to delete
     * @param[in] n the number of elements in the array
     */
    cuda_malloc_uva_deleter(T *ptr, size_t n);

    /** deletes the array
     * @param[in] ptr the pointer to the array to delete. must be the same as
     *                that passed during construction.
     */
    void operator()(T *ptr);

private:
    T *m_ptr;
    size_t m_elem;
};

// --------------------------------------------------------------------------
template <typename T>
cuda_malloc_uva_deleter<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
    ::cuda_malloc_uva_deleter(T *ptr, size_t n) : m_ptr(ptr), m_elem(n)
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
#if !defined(HAMR_CUDA_OBJECTS)
    (void) ptr;
     std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
         " cuda_malloc_uva_deleter dealllocate objects failed."
        " HAMR_CUDA_OBJECTS is not enabled" << std::endl;
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
    cuda_kernels::destruct<T><<<block_grid, thread_grid>>>(ptr, m_elem);
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





/// a deleter for arrays allocated with cuda_malloc_uva, specialized for numbers
template <typename T>
class cuda_malloc_uva_deleter<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
{
public:
    /** constructs the deleter
     * @param[in] ptr the pointer to the array to delete
     * @param[in] n the number of elements in the array
     */
    cuda_malloc_uva_deleter(T *ptr, size_t n);

    /** deletes the array
     * @param[in] ptr the pointer to the array to delete. must be the same as
     *                that passed during construction.
     */
    void operator()(T *ptr);

private:
    T *m_ptr;
    size_t m_elem;
};

// --------------------------------------------------------------------------
template <typename T>
cuda_malloc_uva_deleter<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
    ::cuda_malloc_uva_deleter(T *ptr, size_t n) : m_ptr(ptr), m_elem(n)
{
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





/// a class for allocating arrays with cuda_malloc_uva
template <typename T, typename E = void>
struct cuda_malloc_uva_allocator {};

/// a class for allocating arrays with cuda_malloc_uva, specialized for objects
template <typename T>
struct HAMR_EXPORT cuda_malloc_uva_allocator<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
{
    /** allocate an array of n elements.
     * @param[in] n the number of elements to allocate
     * @returns a shared point to the array that holds a deleter for the memory
     */
    static std::shared_ptr<T> allocate(size_t n);

    /** allocate an array of n elements.
     * @param[in] n the number of elements to allocate
     * @param[in] val a value to initialize the elements to
     * @returns a shared point to the array that holds a deleter for the memory
     */
    static std::shared_ptr<T> allocate(size_t n, const T &val);

    /** allocate an array of n elements.
     * @param[in] n the number of elements to allocate
     * @param[in] vals an array of values to initialize the elements with
     * @param[in] cudaVals a flag that is set to true if vals is accessible from codes running in CUDA
     * @returns a shared point to the array that holds a deleter for the memory
     */
    template <typename U>
    static std::shared_ptr<T> allocate(size_t n, const U *vals, bool cudaVals = false);
};

// --------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T>
cuda_malloc_uva_allocator<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
    ::allocate(size_t n_elem)
{
#if !defined(HAMR_CUDA_OBJECTS)
    (void) n_elem;
     std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
         " cuda_malloc_uva_allocator allocate objects failed."
        " HAMR_CUDA_OBJECTS is not enabled" << std::endl;
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
    cuda_kernels::construct<T><<<block_grid, thread_grid>>>(ptr, n_elem);
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
    return std::shared_ptr<T>(ptr, cuda_malloc_uva_deleter<T>(ptr, n_elem));
#endif
}

// --------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T>
cuda_malloc_uva_allocator<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
    ::allocate(size_t n_elem, const T &val)
{
#if !defined(HAMR_CUDA_OBJECTS)
    (void) n_elem;
    (void) val;
     std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
         " cuda_malloc_uva_allocator allocate objects failed."
        " HAMR_CUDA_OBJECTS is not enabled" << std::endl;
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
    cuda_kernels::construct<T><<<block_grid, thread_grid>>>(ptr, n_elem, val);
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
    return std::shared_ptr<T>(ptr, cuda_malloc_uva_deleter<T>(ptr, n_elem));
#endif
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
std::shared_ptr<T>
cuda_malloc_uva_allocator<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
    ::allocate(size_t n_elem, const U *vals, bool cudaVals)
{
#if !defined(HAMR_CUDA_OBJECTS)
    (void) n_elem;
    (void) vals;
    (void) cudaVals;
     std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
         " cuda_malloc_uva_allocator allocate objects failed."
        " HAMR_CUDA_OBJECTS is not enabled" << std::endl;
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

        if ((ierr = cudaMemcpy(tmp, vals, n_bytes_vals, cudaMemcpyHostToDevice)) != cudaSuccess)
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
    cuda_kernels::construct<T><<<block_grid, thread_grid>>>(ptr, n_elem, vals);
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
            << " at " << ptr  << " initialized from " << (cudaVals ? "CUDA" : "CPU")
            << " array of objects of type " << typeid(U).name() << sizeof(U)
            << " at " << vals << std::endl;
    }
#endif

    // package
    return std::shared_ptr<T>(ptr, cuda_malloc_uva_deleter<T>(ptr, n_elem));
#endif
}




/// a class for allocating arrays with cuda_malloc_uva, specialized for numbers
template <typename T>
struct HAMR_EXPORT cuda_malloc_uva_allocator<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
{
    /** allocate an array of n elements.
     * @param[in] n the number of elements to allocate
     * @returns a shared point to the array that holds a deleter for the memory
     */
    static std::shared_ptr<T> allocate(size_t n);

    /** allocate an array of n elements.
     * @param[in] n the number of elements to allocate
     * @param[in] val a value to initialize the elements to
     * @returns a shared point to the array that holds a deleter for the memory
     */
    static std::shared_ptr<T> allocate(size_t n, const T &val);

    /** allocate an array of n elements.
     * @param[in] n the number of elements to allocate
     * @param[in] vals an array of values to initialize the elements with
     * @param[in] cudaVals a flag set to true if vals is accessible from codes running in CUDA
     * @returns a shared point to the array that holds a deleter for the memory
     */
    template <typename U>
    static std::shared_ptr<T> allocate(size_t n, const U *vals, bool cudaVals = false);
};

// --------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T>
cuda_malloc_uva_allocator<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
    ::allocate(size_t n_elem)
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
    return std::shared_ptr<T>(ptr, cuda_malloc_uva_deleter<T>(ptr, n_elem));
}

// --------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T>
cuda_malloc_uva_allocator<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
    ::allocate(size_t n_elem, const T &val)
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
    cuda_kernels::fill<T><<<block_grid, thread_grid>>>(ptr, n_elem, val);
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
    return std::shared_ptr<T>(ptr, cuda_malloc_uva_deleter<T>(ptr, n_elem));
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
std::shared_ptr<T>
cuda_malloc_uva_allocator<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
    ::allocate(size_t n_elem, const U *vals, bool cudaVals)
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

        if ((ierr = cudaMemcpy(tmp, vals, n_bytes_vals, cudaMemcpyHostToDevice)) != cudaSuccess)
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
    cuda_kernels::fill<T><<<block_grid, thread_grid>>>(ptr, n_elem, vals);
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
            << " at " << ptr  << " initialized from " << (cudaVals ? "CUDA" : "CPU")
            << " array " << vals << " objects of type " << typeid(U).name() << sizeof(T)
            << std::endl;
    }
#endif


    // package
    return std::shared_ptr<T>(ptr, cuda_malloc_uva_deleter<T>(ptr, n_elem));
}

}

#endif
