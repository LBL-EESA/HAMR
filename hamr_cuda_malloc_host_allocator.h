#ifndef hamr_cuda_malloc_host_allocator_h
#define hamr_cuda_malloc_host_allocator_h

#include "hamr_config.h"
#include "hamr_env.h"

#include <iostream>
#include <type_traits>
#include <memory>
#include <typeinfo>
#include <cassert>
#include <cstring>

namespace hamr
{
/// a deleter for arrays allocated with malloc
template <typename T, typename E = void>
class cuda_malloc_host_deleter {};

/// a deleter for arrays allocated with malloc, specialized for objects
template <typename T>
class HAMR_EXPORT cuda_malloc_host_deleter<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
{
public:
    /** constructs the deleter
     * @param[in] ptr the pointer to the array to delete
     * @param[in] n the number of elements in the array
     */
    cuda_malloc_host_deleter(T *ptr, size_t n);

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
cuda_malloc_host_deleter<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
    ::cuda_malloc_host_deleter(T *ptr, size_t n) : m_ptr(ptr), m_elem(n)
{
#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "created cuda_malloc_host_deleter for array of " << n
            << " objects of type " << typeid(T).name() << sizeof(T)
            << " at " << m_ptr << std::endl;
    }
#endif
}

// --------------------------------------------------------------------------
template <typename T>
void
cuda_malloc_host_deleter<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
    ::operator()(T *ptr)
{
    assert(ptr == m_ptr);

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "cuda_malloc_host_deleter deleting array of " << m_elem
            << " objects of type " << typeid(T).name() << sizeof(T)
            << " at " << m_ptr << std::endl;
    }
#endif

    // invoke the destructor
    for (size_t i = 0; i < m_elem; ++i)
        ptr[i].~T();

    // free the array
    cudaFreeHost(ptr);
}





/// a deleter for arrays allocated with malloc, specialized for numbers
template <typename T>
class HAMR_EXPORT cuda_malloc_host_deleter<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
{
public:
    /** constructs the deleter
     * @param[in] ptr the pointer to the array to delete
     * @param[in] n the number of elements in the array
     */
    cuda_malloc_host_deleter(T *ptr, size_t n);

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
cuda_malloc_host_deleter<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
    ::cuda_malloc_host_deleter(T *ptr, size_t n) : m_ptr(ptr), m_elem(n)
{
#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "created cuda_malloc_host_deleter for array of " << n
            << " numbers of type " << typeid(T).name() << sizeof(T)
            << " at " << m_ptr << std::endl;
    }
#endif
}

// --------------------------------------------------------------------------
template <typename T>
void
cuda_malloc_host_deleter<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
    ::operator()(T *ptr)
{
    assert(ptr == m_ptr);

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "cuda_malloc_host_deleter deleting array of " << m_elem
            << " numbers of type " << typeid(T).name() << sizeof(T)
            << " at " << m_ptr << std::endl;
    }
#endif

    // free the array
    cudaFreeHost(ptr);
}





/// a class for allocating arrays with malloc
template <typename T, typename E = void>
struct cuda_malloc_host_allocator {};

/// a class for allocating arrays with malloc, specialized for objects
template <typename T>
struct HAMR_EXPORT cuda_malloc_host_allocator<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
{
    /** allocate an array of n elements.
     * @param[in] n the number of elements to allocate
     * @returns a shared pointer to the array that holds a deleter for the memory
     */
    static std::shared_ptr<T> allocate(size_t n);

    /** allocate an array of n elements.
     * @param[in] n the number of elements to allocate
     * @param[in] val a value to initialize the elements to
     * @returns a shared pointer to the array that holds a deleter for the memory
     */
    static std::shared_ptr<T> allocate(size_t n, const T &val);

    /** allocate an array of n elements.
     * @param[in] n the number of elements to allocate
     * @param[in] vals an array of n elements to initialize the elements with
     * @returns a shared pointer to the array that holds a deleter for the memory
     */
    template <typename U>
    static std::shared_ptr<T> allocate(size_t n, const U *vals);
};

// --------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T>
cuda_malloc_host_allocator<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
    ::allocate(size_t n)
{
    // allocate
    T *ptr = nullptr;
    cudaError_t ierr = cudaSuccess;
    if ((ierr = cudaMallocHost(&ptr, n*sizeof(T))) != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to cudaMallocHost " << n << " of "
            << typeid(T).name() << " total " << n*sizeof(T)  << " bytes. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // construct
    for (size_t i = 0; i < n; ++i)
        new (&ptr[i]) T();

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "cuda_malloc_host_allocator allocating array of " << n
            << " objects of type " << typeid(T).name() << sizeof(T)
            << " at " << ptr << std::endl;
    }
#endif

    // package
    return std::shared_ptr<T>(ptr, cuda_malloc_host_deleter<T>(ptr, n));
}

// --------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T>
cuda_malloc_host_allocator<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
    ::allocate(size_t n, const T &val)
{
    // allocate
    T *ptr = nullptr;
    cudaError_t ierr = cudaSuccess;
    if ((ierr = cudaMallocHost(&ptr, n*sizeof(T))) != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to cudaMallocHost " << n << " of "
            << typeid(T).name() << " total " << n*sizeof(T)  << " bytes. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // construct
    for (size_t i = 0; i < n; ++i)
        new (&ptr[i]) T(val);

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "cuda_malloc_host_allocator allocating array of " << n
            << " objects of type " << typeid(T).name() << sizeof(T)
            << " at " << ptr << " initialized to " << val << std::endl;
    }
#endif

    // package
    return std::shared_ptr<T>(ptr, cuda_malloc_host_deleter<T>(ptr, n));
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
std::shared_ptr<T>
cuda_malloc_host_allocator<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
    ::allocate(size_t n, const U *vals)
{
    // allocate
    T *ptr = nullptr;
    cudaError_t ierr = cudaSuccess;
    if ((ierr = cudaMallocHost(&ptr, n*sizeof(T))) != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to cudaMallocHost " << n << " of "
            << typeid(T).name() << " total " << n*sizeof(T)  << " bytes. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // construct
    for (size_t i = 0; i < n; ++i)
        new (&ptr[i]) T(vals[i]);

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "cuda_malloc_host_allocator allocating array of " << n
            << " objects of type " << typeid(T).name() << sizeof(T)
            << " initialized from array of objects of type "
            << typeid(U).name() << sizeof(U) << " at " << vals
            << std::endl;
    }
#endif

    // package
    return std::shared_ptr<T>(ptr, cuda_malloc_host_deleter<T>(ptr, n));
}




/// a class for allocating arrays with malloc, specialized for numbers
template <typename T>
struct HAMR_EXPORT cuda_malloc_host_allocator<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
{
    /** allocate an array of n elements.
     * @param[in] n the number of elements to allocate
     * @returns a shared pointer to the array that holds a deleter for the memory
     */
    static std::shared_ptr<T> allocate(size_t n);

    /** allocate an array of n elements.
     * @param[in] n the number of elements to allocate
     * @param[in] val a value to initialize the elements to
     * @returns a shared pointer to the array that holds a deleter for the memory
     */
    static std::shared_ptr<T> allocate(size_t n, const T &val);

    /** allocate an array of n elements.
     * @param[in] n the number of elements to allocate
     * @param[in] vals an array of n elements to initialize the elements with
     * @returns a shared pointer to the array that holds a deleter for the memory
     */
    template <typename U>
    static std::shared_ptr<T> allocate(size_t n, const U *vals);
};

// --------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T>
cuda_malloc_host_allocator<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
    ::allocate(size_t n)
{
    size_t n_bytes = n*sizeof(T);

    // allocate
    T *ptr = nullptr;
    cudaError_t ierr = cudaSuccess;
    if ((ierr = cudaMallocHost(&ptr, n_bytes)) != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to cudaMallocHost " << n << " of "
            << typeid(T).name() << " total " << n_bytes  << " bytes. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // construct
#if defined(HAMR_INIT_ALLOC)
    memset(ptr, 0, n_bytes);
#endif

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "cuda_malloc_host_allocator allocating array of " << n
            << " numbers of type " << typeid(T).name() << sizeof(T)
            << " at " << ptr << std::endl;
    }
#endif

    // package
    return std::shared_ptr<T>(ptr, cuda_malloc_host_deleter<T>(ptr, n));
}

// --------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T>
cuda_malloc_host_allocator<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
    ::allocate(size_t n, const T &val)
{
    size_t n_bytes = n*sizeof(T);

    // allocate
    T *ptr = nullptr;
    cudaError_t ierr = cudaSuccess;
    if ((ierr = cudaMallocHost(&ptr, n_bytes)) != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to cudaMallocHost " << n << " of "
            << typeid(T).name() << " total " << n_bytes  << " bytes. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // construct
    for (size_t i = 0; i < n; ++i)
        ptr[i] = val;

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "cuda_malloc_host_allocator allocating array of " << n
            << " numbers of type " << typeid(T).name() << sizeof(T)
            << " at " << ptr << " initialized to " << val << std::endl;
    }
#endif

    // package
    return std::shared_ptr<T>(ptr, cuda_malloc_host_deleter<T>(ptr, n));
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
std::shared_ptr<T>
cuda_malloc_host_allocator<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
    ::allocate(size_t n, const U *vals)
{
    size_t n_bytes = n*sizeof(T);

    // allocate
    T *ptr = nullptr;
    cudaError_t ierr = cudaSuccess;
    if ((ierr = cudaMallocHost(&ptr, n_bytes)) != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to cudaMallocHost " << n << " of "
            << typeid(T).name() << " total " << n_bytes  << " bytes. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    // construct
    for (size_t i = 0; i < n; ++i)
        ptr[i] = vals[i];

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "cuda_malloc_host_allocator allocating array of " << n
            << " numbers of type " << typeid(T).name() << sizeof(T)
            << " at " << ptr << " initialized from an array of numbers of type "
            << typeid(U).name() << sizeof(U) << " at " << vals << std::endl;
    }
#endif

    // package
    return std::shared_ptr<T>(ptr, cuda_malloc_host_deleter<T>(ptr, n));
}

};

#endif
