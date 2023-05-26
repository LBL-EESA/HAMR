#ifndef hamr_malloc_allocator_h
#define hamr_malloc_allocator_h

#include "hamr_config.h"

#include <memory>
#include <type_traits>

namespace hamr
{
/// a deleter for arrays allocated with malloc
template <typename T, typename E = void>
class malloc_deleter {};

/// a deleter for arrays allocated with malloc, specialized for objects
template <typename T>
class HAMR_EXPORT malloc_deleter<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
{
public:
    /** constructs the deleter
     * @param[in] ptr the pointer to the array to delete
     * @param[in] n the number of elements in the array
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






/// a deleter for arrays allocated with malloc, specialized for numbers
template <typename T>
class HAMR_EXPORT malloc_deleter<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
{
public:
    /** constructs the deleter
     * @param[in] ptr the pointer to the array to delete
     * @param[in] n the number of elements in the array
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






/// a class for allocating arrays with malloc
template <typename T, typename E = void>
struct malloc_allocator {};

/// a class for allocating arrays with malloc, specialized for objects
template <typename T>
struct HAMR_EXPORT malloc_allocator<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
{
    /** allocate an array of n elements.
     * @param[in] n the number of elements to allocate
     * @returns a shared pointer to the array that holds a deleter for the memory
     */
    static std::shared_ptr<T> allocate(size_t n) HAMR_EXPORT;

    /** allocate an array of n elements.
     * @param[in] n the number of elements to allocate
     * @param[in] val a value to initialize the elements to
     * @returns a shared pointer to the array that holds a deleter for the memory
     */

    static std::shared_ptr<T> allocate(size_t n, const T &val) HAMR_EXPORT;

    /** allocate an array of n elements.
     * @param[in] n the number of elements to allocate
     * @param[in] vals an array of n elements to initialize the elements with
     * @returns a shared pointer to the array that holds a deleter for the memory
     */
    template <typename U>
    static std::shared_ptr<T> allocate(size_t n, const U *vals) HAMR_EXPORT;
};





/// a class for allocating arrays with malloc, specialized for numbers
template <typename T>
struct HAMR_EXPORT malloc_allocator<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
{
    /** allocate an array of n elements.
     * @param[in] n the number of elements to allocate
     * @returns a shared pointer to the array that holds a deleter for the memory
     */
    static std::shared_ptr<T> allocate(size_t n) HAMR_EXPORT;

    /** allocate an array of n elements.
     * @param[in] n the number of elements to allocate
     * @param[in] val a value to initialize the elements to
     * @returns a shared pointer to the array that holds a deleter for the memory
     */
    static std::shared_ptr<T> allocate(size_t n, const T &val) HAMR_EXPORT;

    /** allocate an array of n elements.
     * @param[in] n the number of elements to allocate
     * @param[in] vals an array of n elements to initialize the elements with
     * @returns a shared pointer to the array that holds a deleter for the memory
     */
    template <typename U>
    static std::shared_ptr<T> allocate(size_t n, const U *vals) HAMR_EXPORT;
};

}

#if !defined(HAMR_SEPARATE_IMPL)
#include "hamr_malloc_allocator_impl.h"
#endif

#endif
