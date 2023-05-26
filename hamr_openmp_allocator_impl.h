#ifndef hamr_openmp_allocator_impl_h
#define hamr_openmp_allocator_impl_h

#include "hamr_config.h"
#include "hamr_env.h"

#include <iostream>
#include <type_traits>
#include <memory>
#include <typeinfo>
#include <cassert>
#include <cstring>

#include <omp.h>

namespace hamr
{

// --------------------------------------------------------------------------
template <typename T>
openmp_deleter<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
    ::openmp_deleter(T *ptr, size_t n, int dev) : m_ptr(ptr), m_elem(n), m_dev(dev)
{
#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "created openmp_deleter for array of " << n
            << " objects of type " << typeid(T).name() << sizeof(T)
            << " at " << m_ptr << " on device " << m_dev << std::endl;
    }
#endif
}

// --------------------------------------------------------------------------
template <typename T>
void
openmp_deleter<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
    ::operator()(T *ptr)
{
    assert(ptr == m_ptr);

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "openmp_deleter deleting array of " << m_elem
            << " objects of type " << typeid(T).name() << sizeof(T)
            << " at " << m_ptr << " on device " << m_dev << std::endl;
    }
#endif

    // invoke the destructor
    for (size_t i = 0; i < m_elem; ++i)
        ptr[i].~T();

    // free the array
    omp_target_free(ptr, m_dev);
}






// --------------------------------------------------------------------------
template <typename T>
openmp_deleter<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
    ::openmp_deleter(T *ptr, size_t n, int dev) : m_ptr(ptr), m_elem(n), m_dev(dev)
{
#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "created openmp_deleter for array of " << n
            << " numbers of type " << typeid(T).name() << sizeof(T)
            << " at " << m_ptr << " on device " << m_dev << std::endl;
    }
#endif
}

// --------------------------------------------------------------------------
template <typename T>
void
openmp_deleter<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
    ::operator()(T *ptr)
{
    assert(ptr == m_ptr);

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "openmp_deleter deleting array of " << m_elem
            << " numbers of type " << typeid(T).name() << sizeof(T)
            << " at " << m_ptr << " on device " << m_dev << std::endl;
    }
#endif

    // free the array
    omp_target_free(ptr, m_dev);
}






// --------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T>
openmp_allocator<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
    ::allocate(size_t n)
{
    // allocate
    int dev = omp_get_default_device();
    T *ptr = (T*)omp_target_alloc(n*sizeof(T), dev);

    // construct
    #pragma omp target teams HAMR_OPENMP_LOOP is_device_ptr(ptr)
    for (size_t i = 0; i < n; ++i)
        new (&ptr[i]) T();

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "openmp_allocator allocating array of " << n
            << " objects of type " << typeid(T).name() << sizeof(T)
            << " at " << ptr << " on device " << dev << std::endl;
    }
#endif

    // package
    return std::shared_ptr<T>(ptr, openmp_deleter<T>(ptr, n, dev));
}

// --------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T>
openmp_allocator<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
    ::allocate(size_t n, const T &val)
{
    // allocate
    int dev = omp_get_default_device();
    T *ptr = (T*)omp_target_alloc(n*sizeof(T), dev);

    // construct
    #pragma omp target teams HAMR_OPENMP_LOOP is_device_ptr(ptr) map(to: val)
    for (size_t i = 0; i < n; ++i)
        new (&ptr[i]) T(val);

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "openmp_allocator allocating array of " << n
            << " objects of type " << typeid(T).name() << sizeof(T)
            << " at " << ptr << " initialized to " << val << std::endl;
    }
#endif

    // package
    return std::shared_ptr<T>(ptr, openmp_deleter<T>(ptr, n));
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
std::shared_ptr<T>
openmp_allocator<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
    ::allocate(size_t n, const U *vals)
{
    // allocate
    int dev = omp_get_default_device();
    T *ptr = (T*)omp_target_alloc(n*sizeof(T), dev);

    // construct
    #pragma omp target teams HAMR_OPENMP_LOOP is_device_ptr(ptr) map(to: vals[0:n])
    for (size_t i = 0; i < n; ++i)
        new (&ptr[i]) T(vals[i]);

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "openmp_allocator allocating array of " << n
            << " objects of type " << typeid(T).name() << sizeof(T)
            << " initialized from array of objects of type "
            << typeid(U).name() << sizeof(U) << " at " << vals
            << " on device " << dev << std::endl;
    }
#endif

    // package
    return std::shared_ptr<T>(ptr, openmp_deleter<T>(ptr, n));
}





// --------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T>
openmp_allocator<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
    ::allocate(size_t n)
{
    size_t n_bytes = n*sizeof(T);

    // allocate
    int dev = omp_get_default_device();
    T *ptr = (T*)omp_target_alloc(n_bytes, dev);

    // construct
#if defined(HAMR_INIT_ALLOC)
    #pragma omp target teams HAMR_OPENMP_LOOP is_device_ptr(ptr)
    for (size_t i = 0; i < n; ++i)
        ptr[i] = T(0);
#endif

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "openmp_allocator allocating array of " << n
            << " numbers of type " << typeid(T).name() << sizeof(T)
            << " at " << ptr << " on device " << dev << std::endl;
    }
#endif

    // package
    return std::shared_ptr<T>(ptr, openmp_deleter<T>(ptr, n, dev));
}

// --------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T>
openmp_allocator<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
    ::allocate(size_t n, const T &val)
{
    size_t n_bytes = n*sizeof(T);

    // allocate
    int dev = omp_get_default_device();
    T *ptr = (T*)omp_target_alloc(n_bytes, dev);

    // construct
    #pragma omp target teams HAMR_OPENMP_LOOP is_device_ptr(ptr) map(to: val)
    for (size_t i = 0; i < n; ++i)
        ptr[i] = val;

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "openmp_allocator allocating array of " << n
            << " numbers of type " << typeid(T).name() << sizeof(T)
            << " at " << ptr << " initialized to " << val << std::endl;
    }
#endif

    // package
    return std::shared_ptr<T>(ptr, openmp_deleter<T>(ptr, n, dev));
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
std::shared_ptr<T>
openmp_allocator<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
    ::allocate(size_t n, const U *vals)
{
    size_t n_bytes = n*sizeof(T);

    // allocate
    int dev = omp_get_default_device();
    T *ptr = (T*)omp_target_alloc(n_bytes, dev);

    // construct
    #pragma omp target teams HAMR_OPENMP_LOOP is_device_ptr(ptr) map(to: vals[0:n])
    for (size_t i = 0; i < n; ++i)
        ptr[i] = vals[i];

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "openmp_allocator allocating array of " << n
            << " numbers of type " << typeid(T).name() << sizeof(T)
            << " at " << ptr << " initialized from an array of numbers of type "
            << typeid(U).name() << sizeof(U) << " at " << vals << std::endl;
    }
#endif

    // package
    return std::shared_ptr<T>(ptr, openmp_deleter<T>(ptr, n, dev));
}

}

#endif
