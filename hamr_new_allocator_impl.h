#ifndef hamr_new_allocator_impl_h
#define hamr_new_allocator_impl_h

//#include "hamr_new_allocator.h"

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
new_deleter<T>::new_deleter(T *ptr, size_t n) : m_ptr(ptr), m_elem(n)
{
#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "created new_deleter for array of " << n
            << " objects of type " << typeid(T).name() << std::endl;
    }
#endif
}

// --------------------------------------------------------------------------
template <typename T>
void new_deleter<T>::operator()(T *ptr)
{
    assert(ptr == m_ptr);

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "new_deleter deleting array of " << m_elem
            << " objects of type " << typeid(T).name() << std::endl;
    }
#endif

    delete [] ptr;
}






// --------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T> new_allocator<T>::allocate(size_t n)
{
#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "new_allocator allocating array of " << n
            << " objects of type " << typeid(T).name() << std::endl;
    }
#endif

    // allocate
    T *ptr = new T[n];

    // package
    return std::shared_ptr<T>(ptr, new_deleter<T>(ptr, n));
}

// --------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T> new_allocator<T>::allocate(size_t n, const T &val)
{
#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "new_allocator allocating array of " << n
            << " objects of type " << typeid(T).name() << " initialized"
            << std::endl;
    }
#endif

    // allocate
    T *ptr = (T*)new unsigned char[n*sizeof(T)];

    // construct
    for (size_t i = 0; i < n; ++i)
        new (&ptr[i]) T(val);

    // package
    return std::shared_ptr<T>(ptr, new_deleter<T>(ptr, n));
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
std::shared_ptr<T> new_allocator<T>::allocate(size_t n, const U *vals)
{
#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "new_allocator allocating array of " << n
            << " objects of type " << typeid(T).name() << " initialized"
            << std::endl;
    }
#endif

    // allocate
    T *ptr = (T*)new unsigned char[n*sizeof(T)];

    // construct
    for (size_t i = 0; i < n; ++i)
        new (&ptr[i]) T(vals[i]);

    // package
    return std::shared_ptr<T>(ptr, new_deleter<T>(ptr, n));
}

};

#endif
