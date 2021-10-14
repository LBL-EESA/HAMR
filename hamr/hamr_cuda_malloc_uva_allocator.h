#ifndef hamr_cuda_malloc_uva_allocator_h
#define hamr_cuda_malloc_uva_allocator_h

#include <iostream>
#include <type_traits>
#include <memory>
#include <typeinfo>
#include <cassert>
#include <cstring>

#include <cuda.h>
#include <cuda_runtime.h>

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
     * @param[in] the number of elements in the array
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
    if (hamr::get_verbose())
    {
        std::cerr << "created cuda_malloc_uva_deleter for array of " << n
            << " objects of type " << typeid(T).name() << std::endl;
    }
}

// --------------------------------------------------------------------------
template <typename T>
void
cuda_malloc_uva_deleter<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
    ::operator()(T *ptr)
{
    assert(ptr == m_ptr);

    if (hamr::get_verbose())
    {
        std::cerr << "cuda_malloc_uva_deleter deleting array of " << m_elem
            << " objects of type " << typeid(T).name() << std::endl;
    }

    // get launch parameters
    int device_id = -1;
    dim3 block_grid;
    int n_blocks = 0;
    dim3 thread_grid = 0;
    if (hamr::partition_thread_blocks(-1, m_elem, 8, block_grid, n_blocks, thread_grid))
    {
        std::cerr << "ERROR: Failed to determine launch properties." << std::endl;
        return;
    }

    // destruct
    cudaError_t ierr = cudaSuccess;
    cuda_kernels::destruct<T><<<block_grid, thread_grid>>>(ptr, m_elem);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        std::cerr << "ERROR: Failed to launch the construct kernel. "
            << cudaGetErrorString(ierr) << std::endl;
        return;
    }

    // free the array
    cudaFree(ptr);
}





/// a deleter for arrays allocated with cuda_malloc_uva, specialized for numbers
template <typename T>
class cuda_malloc_uva_deleter<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
{
public:
    /** constructs the deleter
     * @param[in] ptr the pointer to the array to delete
     * @param[in] the number of elements in the array
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
    if (hamr::get_verbose())
    {
        std::cerr << "created cuda_malloc_uva_deleter for array of " << n
            << " numbers of type " << typeid(T).name() << sizeof(T)
            << std::endl;
    }
}

// --------------------------------------------------------------------------
template <typename T>
void
cuda_malloc_uva_deleter<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
    ::operator()(T *ptr)
{
    assert(ptr == m_ptr);

    if (hamr::get_verbose())
    {
        std::cerr << "cuda_malloc_uva_deleter deleting array of " << m_elem
            << " numbers of type " << typeid(T).name() << sizeof(T)
            << std::endl;
    }

    // free the array
    cudaFree(ptr);
}





/// a class for allocating arrays with cuda_malloc_uva
template <typename T, typename E = void>
struct cuda_malloc_uva_allocator {};

/// a class for allocating arrays with cuda_malloc_uva, specialized for objects
template <typename T>
struct cuda_malloc_uva_allocator<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
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
};

// --------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T>
cuda_malloc_uva_allocator<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
    ::allocate(size_t n_elem)
{
    if (hamr::get_verbose())
    {
        std::cerr << "cuda_malloc_uva_allocator allocating array of " << n_elem
            << " objects of type " << typeid(T).name() << std::endl;
    }

    size_t n_bytes = n_elem*sizeof(T);

    // allocate
    T *ptr = nullptr;
    cudaError_t ierr = cudaSuccess;
    if ((ierr = cudaMallocManaged(&ptr, n_bytes)) != cudaSuccess)
    {
        std::cerr << "ERROR: Failed to cudaMallocManaged " << n_elem << " of "
            << typeid(T).name() << " total " << n_bytes  << "bytes. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // get launch parameters
    int device_id = -1;
    dim3 block_grid;
    int n_blocks = 0;
    dim3 thread_grid = 0;
    if (hamr::partition_thread_blocks(-1, n_elem, 8, block_grid, n_blocks, thread_grid))
    {
        std::cerr << "ERROR: Failed to determine launch properties. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // construct
    cuda_kernels::construct<T><<<block_grid, thread_grid>>>(ptr, n_elem);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        std::cerr << "ERROR: Failed to launch the construct kernel. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // package
    return std::shared_ptr<T>(ptr, cuda_malloc_uva_deleter<T>(ptr, n_elem));
}

// --------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T>
cuda_malloc_uva_allocator<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
    ::allocate(size_t n_elem, const T &val)
{
    if (hamr::get_verbose())
    {
        std::cerr << "cuda_malloc_uva_allocator allocating array of " << n_elem
            << " objects of type " << typeid(T).name() << " initialized"
            << std::endl;
    }

    // allocate
    T *ptr = nullptr;
    size_t n_bytes = n_elem*sizeof(T);

    cudaError_t ierr = cudaSuccess;
    if ((ierr = cudaMallocManaged(&ptr, n_bytes)) != cudaSuccess)
    {
        std::cerr << "ERROR: Failed to cudaMallocManaged " << n_elem << " of "
            << typeid(T).name() << " total " << n_bytes  << "bytes. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // get launch parameters
    int device_id = -1;
    dim3 block_grid;
    int n_blocks = 0;
    dim3 thread_grid = 0;
    if (hamr::partition_thread_blocks(-1, n_elem, 8, block_grid, n_blocks, thread_grid))
    {
        std::cerr << "ERROR: Failed to determine launch properties. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // construct
    cuda_kernels::construct<T><<<block_grid, thread_grid>>>(ptr, n_elem, val);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        std::cerr << "ERROR: Failed to launch the construct kernel. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // package
    return std::shared_ptr<T>(ptr, cuda_malloc_uva_deleter<T>(ptr, n_elem));
}




/// a class for allocating arrays with cuda_malloc_uva, specialized for numbers
template <typename T>
struct cuda_malloc_uva_allocator<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
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
};

// --------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T>
cuda_malloc_uva_allocator<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
    ::allocate(size_t n_elem)
{
    if (hamr::get_verbose())
    {
        std::cerr << "cuda_malloc_uva_allocator allocating array of " << n_elem
            << " numbers of type " << typeid(T).name() << sizeof(T)
            << std::endl;
    }

    size_t n_bytes = n_elem*sizeof(T);

    // allocate
    T *ptr = nullptr;
    cudaError_t ierr = cudaSuccess;
    if ((ierr = cudaMalloc(&ptr, n_bytes)) != cudaSuccess)
    {
        std::cerr << "ERROR: Failed to cudaMalloc " << n_elem << " of "
            << typeid(T).name() << " total " << n_bytes  << "bytes. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // construct
#if defined(HAMR_INIT_ALLOC)
    cudaMemset(ptr, 0, n_bytes);
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
    if (hamr::get_verbose())
    {
        std::cerr << "cuda_malloc_uva_allocator allocating array of " << n_elem
            << " objects of type " << typeid(T).name() << " initialized"
            << std::endl;
    }

    size_t n_bytes = n_elem*sizeof(T);

    // allocate
    T *ptr = nullptr;
    cudaError_t ierr = cudaSuccess;
    if ((ierr = cudaMalloc(&ptr, n_bytes)) != cudaSuccess)
    {
        std::cerr << "ERROR: Failed to cudaMalloc " << n_elem << " of "
            << typeid(T).name() << " total " << n_bytes  << "bytes. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // get launch parameters
    int device_id = -1;
    dim3 block_grid;
    int n_blocks = 0;
    dim3 thread_grid = 0;
    if (hamr::partition_thread_blocks(device_id, n_elem, 8, block_grid, n_blocks, thread_grid))
    {
        std::cerr << "ERROR: Failed to determine launch properties. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // construct
    cuda_kernels::fill<T><<<block_grid, thread_grid>>>(ptr, n_elem, val);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        std::cerr << "ERROR: Failed to launch the construct kernel. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // package
    return std::shared_ptr<T>(ptr, cuda_malloc_uva_deleter<T>(ptr, n_elem));
}

}

#endif
