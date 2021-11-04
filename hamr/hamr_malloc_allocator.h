#ifndef hamr_malloc_allocator_h
#define hamr_malloc_allocator_h

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
class malloc_deleter {};

/// a deleter for arrays allocated with malloc, specialized for objects
template <typename T>
class malloc_deleter<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
{
public:
    /** constructs the deleter
     * @param[in] ptr the pointer to the array to delete
     * @param[in] the number of elements in the array
     */
    malloc_deleter(T *ptr, size_t n);

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
malloc_deleter<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
    ::malloc_deleter(T *ptr, size_t n) : m_ptr(ptr), m_elem(n)
{
    if (hamr::get_verbose())
    {
        std::cerr << "created malloc_deleter for array of " << n
            << " objects of type " << typeid(T).name() << sizeof(T)
            << " at " << m_ptr << std::endl;
    }
}

// --------------------------------------------------------------------------
template <typename T>
void
malloc_deleter<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
    ::operator()(T *ptr)
{
    assert(ptr == m_ptr);

    if (hamr::get_verbose())
    {
        std::cerr << "malloc_deleter deleting array of " << m_elem
            << " objects of type " << typeid(T).name() << sizeof(T)
            << " at " << m_ptr << std::endl;
    }

    // invoke the destructor
    for (size_t i = 0; i < m_elem; ++i)
        ptr[i].~T();

    // free the array
    free(ptr);
}





/// a deleter for arrays allocated with malloc, specialized for numbers
template <typename T>
class malloc_deleter<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
{
public:
    /** constructs the deleter
     * @param[in] ptr the pointer to the array to delete
     * @param[in] the number of elements in the array
     */
    malloc_deleter(T *ptr, size_t n);

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
malloc_deleter<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
    ::malloc_deleter(T *ptr, size_t n) : m_ptr(ptr), m_elem(n)
{
    if (hamr::get_verbose())
    {
        std::cerr << "created malloc_deleter for array of " << n
            << " numbers of type " << typeid(T).name() << sizeof(T)
            << " at " << m_ptr << std::endl;
    }
}

// --------------------------------------------------------------------------
template <typename T>
void
malloc_deleter<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
    ::operator()(T *ptr)
{
    assert(ptr == m_ptr);

    if (hamr::get_verbose())
    {
        std::cerr << "malloc_deleter deleting array of " << m_elem
            << " numbers of type " << typeid(T).name() << sizeof(T)
            << " at " << m_ptr << std::endl;
    }

    // free the array
    free(ptr);
}





/// a class for allocating arrays with malloc
template <typename T, typename E = void>
struct malloc_allocator {};

/// a class for allocating arrays with malloc, specialized for objects
template <typename T>
struct malloc_allocator<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
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
malloc_allocator<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
    ::allocate(size_t n)
{
    // allocate
    T *ptr = (T*)malloc(n*sizeof(T));

    // construct
    for (size_t i = 0; i < n; ++i)
        new (&ptr[i]) T();

    if (hamr::get_verbose())
    {
        std::cerr << "malloc_allocator allocating array of " << n
            << " objects of type " << typeid(T).name() << sizeof(T)
            << " at " << ptr << std::endl;
    }

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

    if (hamr::get_verbose())
    {
        std::cerr << "malloc_allocator allocating array of " << n
            << " objects of type " << typeid(T).name() << sizeof(T)
            << " at " << ptr << " initialized to " << val << std::endl;
    }

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

    if (hamr::get_verbose())
    {
        std::cerr << "malloc_allocator allocating array of " << n
            << " objects of type " << typeid(T).name() << sizeof(T)
            << " initialized from array of objects of type "
            << typeid(U).name() << sizeof(U) << " at " << vals
            << std::endl;
    }

    // package
    return std::shared_ptr<T>(ptr, malloc_deleter<T>(ptr, n));
}




/// a class for allocating arrays with malloc, specialized for numbers
template <typename T>
struct malloc_allocator<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
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

    if (hamr::get_verbose())
    {
        std::cerr << "malloc_allocator allocating array of " << n
            << " numbers of type " << typeid(T).name() << sizeof(T)
            << " at " << ptr << std::endl;
    }

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

    if (hamr::get_verbose())
    {
        std::cerr << "malloc_allocator allocating array of " << n
            << " numbers of type " << typeid(T).name() << sizeof(T)
            << " at " << ptr << " initialized to " << val << std::endl;
    }

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

    if (hamr::get_verbose())
    {
        std::cerr << "malloc_allocator allocating array of " << n
            << " numbers of type " << typeid(T).name() << sizeof(T)
            << " at " << ptr << " initialized from an array of numbers of type "
            << typeid(U).name() << sizeof(U) << " at " << vals << std::endl;
    }

    // package
    return std::shared_ptr<T>(ptr, malloc_deleter<T>(ptr, n));
}

};

#endif
