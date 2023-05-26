#ifndef hamr_malloc_allocator_impl_h
#define hamr_malloc_allocator_impl_h

//#include "hamr_malloc_allocator.h"

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

// --------------------------------------------------------------------------
template <typename T>
malloc_deleter<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
    ::malloc_deleter(T *ptr, size_t n) : m_ptr(ptr), m_elem(n)
{
#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "created malloc_deleter for array of " << n
            << " objects of type " << typeid(T).name() << sizeof(T)
            << " at " << m_ptr << std::endl;
    }
#endif
}

// --------------------------------------------------------------------------
template <typename T>
void
malloc_deleter<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
    ::operator()(T *ptr)
{
    assert(ptr == m_ptr);

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "malloc_deleter deleting array of " << m_elem
            << " objects of type " << typeid(T).name() << sizeof(T)
            << " at " << m_ptr << std::endl;
    }
#endif

    // invoke the destructor
    for (size_t i = 0; i < m_elem; ++i)
        ptr[i].~T();

    // free the array
    free(ptr);
}






// --------------------------------------------------------------------------
template <typename T>
malloc_deleter<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
    ::malloc_deleter(T *ptr, size_t n) : m_ptr(ptr), m_elem(n)
{
#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "created malloc_deleter for array of " << n
            << " numbers of type " << typeid(T).name() << sizeof(T)
            << " at " << m_ptr << std::endl;
    }
#endif
}

// --------------------------------------------------------------------------
template <typename T>
void
malloc_deleter<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
    ::operator()(T *ptr)
{
    assert(ptr == m_ptr);

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "malloc_deleter deleting array of " << m_elem
            << " numbers of type " << typeid(T).name() << sizeof(T)
            << " at " << m_ptr << std::endl;
    }
#endif

    // free the array
    free(ptr);
}






// --------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T>
malloc_allocator<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
    ::allocate(size_t n)
{
    // allocate
    T *ptr = (T*)malloc(n*sizeof(T));

    // construct
    for (size_t i = 0; i < n; ++i)
        new (&ptr[i]) T();

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "malloc_allocator allocating array of " << n
            << " objects of type " << typeid(T).name() << sizeof(T)
            << " at " << ptr << std::endl;
    }
#endif

    // package
    return std::shared_ptr<T>(ptr, malloc_deleter<T>(ptr, n));
}

// --------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T>
malloc_allocator<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
    ::allocate(size_t n, const T &val)
{
    // allocate
    T *ptr = (T*)malloc(n*sizeof(T));

    // construct
    for (size_t i = 0; i < n; ++i)
        new (&ptr[i]) T(val);

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "malloc_allocator allocating array of " << n
            << " objects of type " << typeid(T).name() << sizeof(T)
            << " at " << ptr << " initialized to " << val << std::endl;
    }
#endif

    // package
    return std::shared_ptr<T>(ptr, malloc_deleter<T>(ptr, n));
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
std::shared_ptr<T>
malloc_allocator<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
    ::allocate(size_t n, const U *vals)
{
    // allocate
    T *ptr = (T*)malloc(n*sizeof(T));

    // construct
    for (size_t i = 0; i < n; ++i)
        new (&ptr[i]) T(vals[i]);

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "malloc_allocator allocating array of " << n
            << " objects of type " << typeid(T).name() << sizeof(T)
            << " initialized from array of objects of type "
            << typeid(U).name() << sizeof(U) << " at " << vals
            << std::endl;
    }
#endif

    // package
    return std::shared_ptr<T>(ptr, malloc_deleter<T>(ptr, n));
}





// --------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T>
malloc_allocator<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
    ::allocate(size_t n)
{
    size_t n_bytes = n*sizeof(T);

    // allocate
    T *ptr = (T*)malloc(n_bytes);

    // construct
#if defined(HAMR_INIT_ALLOC)
    memset(ptr, 0, n_bytes);
#endif

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "malloc_allocator allocating array of " << n
            << " numbers of type " << typeid(T).name() << sizeof(T)
            << " at " << ptr << std::endl;
    }
#endif

    // package
    return std::shared_ptr<T>(ptr, malloc_deleter<T>(ptr, n));
}

// --------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T>
malloc_allocator<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
    ::allocate(size_t n, const T &val)
{
    size_t n_bytes = n*sizeof(T);

    // allocate
    T *ptr = (T*)malloc(n_bytes);

    // construct
    for (size_t i = 0; i < n; ++i)
        ptr[i] = val;

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "malloc_allocator allocating array of " << n
            << " numbers of type " << typeid(T).name() << sizeof(T)
            << " at " << ptr << " initialized to " << val << std::endl;
    }
#endif

    // package
    return std::shared_ptr<T>(ptr, malloc_deleter<T>(ptr, n));
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
std::shared_ptr<T>
malloc_allocator<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
    ::allocate(size_t n, const U *vals)
{
    size_t n_bytes = n*sizeof(T);

    // allocate
    T *ptr = (T*)malloc(n_bytes);

    // construct
    for (size_t i = 0; i < n; ++i)
        ptr[i] = vals[i];

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "malloc_allocator allocating array of " << n
            << " numbers of type " << typeid(T).name() << sizeof(T)
            << " at " << ptr << " initialized from an array of numbers of type "
            << typeid(U).name() << sizeof(U) << " at " << vals << std::endl;
    }
#endif

    // package
    return std::shared_ptr<T>(ptr, malloc_deleter<T>(ptr, n));
}

}

#endif
