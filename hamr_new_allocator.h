#ifndef hamr_new_allocator_h
#define hamr_new_allocator_h

#include <iostream>
#include <type_traits>
#include <memory>
#include <typeinfo>
#include <cassert>
#include <cstring>

namespace hamr
{

/// a deleter for arrays allocated with new
template <typename T>
class HAMR_EXPORT new_deleter
{
public:
    /** constructs the deleter
     * @param[in] ptr the pointer to the array to delete
     * @param[in] n the number of elements in the array
     */
    new_deleter(T *ptr, size_t n);

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





/// a class for allocating arrays with new
template <typename T>
struct HAMR_EXPORT new_allocator
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
     * @param[in] vals an array of n values to initialize the elements with
     * @returns a shared pointer to the array that holds a deleter for the memory
     */
    template <typename U>
    static std::shared_ptr<T> allocate(size_t n, const U *vals);
};

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
