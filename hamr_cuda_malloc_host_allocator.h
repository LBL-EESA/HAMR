#ifndef hamr_cuda_malloc_host_allocator_h
#define hamr_cuda_malloc_host_allocator_h

#include "hamr_config.h"
#include <type_traits>
#include <memory>

namespace hamr
{
/// a deleter for arrays allocated with cudaMallocHost
template <typename T, typename E = void>
class cuda_malloc_host_deleter {};

/// a deleter for arrays allocated with cudaMallocHost, specialized for objects
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






/// a deleter for arrays allocated with cudaMallocHost, specialized for numbers
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






/** A class for allocating arrays with cudaMallocHost.  Use this allocator for
 * host accessible memory when you want to overlap data movement and computation
 * with CUDA.
 */
template <typename T, typename E = void>
struct cuda_malloc_host_allocator {};

/** a class for allocating arrays with cudaMallocHost, specialized for objects
 * Use this allocator for host accessible memory when you want to overlap data movement and computation
 * with CUDA
 */
template <typename T>
struct HAMR_EXPORT cuda_malloc_host_allocator<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
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





/** a class for allocating arrays with cudaMallocHost, specialized for numbers.
 * Use this allocator for host accessible memory when you want to overlap data
 * movement and computation with CUDA
 */
template <typename T>
struct HAMR_EXPORT cuda_malloc_host_allocator<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
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
#include "hamr_cuda_malloc_host_allocator_impl.h"
#endif

#endif
