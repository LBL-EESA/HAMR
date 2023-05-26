#ifndef hamr_hip_malloc_allocator_h
#define hamr_hip_malloc_allocator_h

#include "hamr_config.h"
#include <type_traits>
#include <memory>

namespace hamr
{

/// a deleter for arrays allocated with hip_malloc
template <typename T, typename E = void>
class hip_malloc_deleter {};

/// a deleter for arrays allocated with hip_malloc, specialized for objects
template <typename T>
class HAMR_EXPORT hip_malloc_deleter<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
{
public:
    /** constructs the deleter
     * @param[in] ptr the pointer to the array to delete
     * @param[in] n   the number of elements in the array
     */
    hip_malloc_deleter(T *ptr, size_t n);

    /** deletes the array
     * @param[in] ptr the pointer to the array to delete. must be the same as
     *                that passed during construction.
     */
    void operator()(T *ptr);

private:
    T *m_ptr;
    size_t m_elem;
};






/// a deleter for arrays allocated with hip_malloc, specialized for numbers
template <typename T>
class HAMR_EXPORT hip_malloc_deleter<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
{
public:
    /** constructs the deleter
     * @param[in] ptr the pointer to the array to delete
     * @param[in] n the number of elements in the array
     */
    hip_malloc_deleter(T *ptr, size_t n);

    /** deletes the array
     * @param[in] ptr the pointer to the array to delete. must be the same as
     *                that passed during construction.
     */
    void operator()(T *ptr);

private:
    T *m_ptr;
    size_t m_elem;
};






/// a class for allocating arrays with hip_malloc
template <typename T, typename E = void>
struct hip_malloc_allocator {};

/// a class for allocating arrays with hip_malloc, specialized for objects
template <typename T>
struct HAMR_EXPORT hip_malloc_allocator<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
{
    /** allocate an array of n elements.
     * @param[in] n the number of elements to allocate
     * @returns a shared pointer to the array that holds a deleter for the
     * memory
     */
    static std::shared_ptr<T> allocate(size_t n);

    /** allocate an array of n elements.
     * @param[in] n the number of elements to allocate
     * @param[in] val a value to initialize the elements to
     * @returns a shared pointer to the array that holds a deleter for the
     * memory
     */
    static std::shared_ptr<T> allocate(size_t n, const T &val);

    /** allocate an array of n elements.
     * @param[in] n the number of elements to allocate
     * @param[in] vals an array of values to initialize the elements with
     * @param[in] hipVals a flag set to true if vals are accessible by codes
     *                     running in HIP
     * @returns a shared pointer to the array that holds a deleter for the
     * memory
     */
    template <typename U>
    static std::shared_ptr<T> allocate(size_t n, const U *vals, bool hipVals = false);
};





/// a class for allocating arrays with hip_malloc, specialized for numbers
template <typename T>
struct HAMR_EXPORT hip_malloc_allocator<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
{
    /** allocate an array of n elements.
     * @param[in] n the number of elements to allocate
     * @returns a shared pointer to the array that holds a deleter for the
     * memory
     */
    static std::shared_ptr<T> allocate(size_t n);

    /** allocate an array of n elements.
     * @param[in] n the number of elements to allocate
     * @param[in] val a value to initialize the elements to
     * @returns a shared pointer to the array that holds a deleter for the
     * memory
     */
    static std::shared_ptr<T> allocate(size_t n, const T &val);

    /** allocate an array of n elements.
     * @param[in] n the number of elements to allocate
     * @param[in] vals an array of values to initialize the elements with
     * @param[in] hipVals a flag set to true if vals are accessible by codes
     *                     running in HIP
     * @returns a shared pointer to the array that holds a
     * deleter for the memory
     */
    template <typename U>
    static std::shared_ptr<T> allocate(size_t n, const U *vals, bool hipVals = false);
};

}

#if !defined(HAMR_SEPARATE_IMPL)
#include "hamr_hip_malloc_allocator_impl.h"
#endif

#endif
