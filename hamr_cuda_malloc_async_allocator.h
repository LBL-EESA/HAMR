#ifndef hamr_cuda_malloc_async_allocator_h
#define hamr_cuda_malloc_async_allocator_h

#include "hamr_config.h"

#include <type_traits>
#include <memory>

namespace hamr
{

/// a deleter for arrays allocated with the cuda_malloc_async_allocator
template <typename T, typename E = void>
class cuda_malloc_async_deleter {};

/// a deleter for arrays allocated with the cuda_malloc_async_allocator, specialized for objects
template <typename T>
class HAMR_EXPORT cuda_malloc_async_deleter<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
{
public:
    /** constructs the deleter
     * @param[in] str a pointer to a CUDA stream or null for the default stream
     * @param[in] ptr the pointer to the array to delete
     * @param[in] n   the number of elements in the array
     */
    cuda_malloc_async_deleter(cudaStream_t str, T *ptr, size_t n);

    /** deletes the array
     * @param[in] ptr the pointer to the array to delete. must be the same as
     *                that passed during construction.
     */
    void operator()(T *ptr);

private:
    T *m_ptr;
    size_t m_elem;
    const cudaStream_t m_str;
};







/// A deleter for arrays allocated with the cuda_malloc_async_allocator, specialized for numbers.
template <typename T>
class HAMR_EXPORT cuda_malloc_async_deleter<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
{
public:
    /** constructs the deleter
     * @param[in] str a CUDA stream or null for the default stream
     * @param[in] ptr the pointer to the array to delete
     * @param[in] n the number of elements in the array
     */
    cuda_malloc_async_deleter(cudaStream_t str, T *ptr, size_t n);

    /** deletes the array
     * @param[in] ptr the pointer to the array to delete. must be the same as
     *                that passed during construction.
     */
    void operator()(T *ptr);

private:
    T *m_ptr;
    size_t m_elem;
    cudaStream_t m_str;
};






/** A class for allocating arrays on the GPU in CUDA. This is the preferred
 * allocator for device memory in CUDA because it does not synchronize the
 * entire device.
 */
template <typename T, typename E = void>
struct cuda_malloc_async_allocator {};

/** A class for allocating arrays on the GPU in CUDA, specialized for objects.
 * This is the preferred allocator for device memory in CUDA because it does
 * not synchronize the entire device.
 */
template <typename T>
struct HAMR_EXPORT cuda_malloc_async_allocator<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
{
    /** allocate an array of n elements.
     * @param[in] str a stream used to order operations, or null for the
     *                default stream
     * @param[in] n the number of elements to allocate
     * @returns a shared pointer to the array that holds a deleter for the
     * memory
     */
    static std::shared_ptr<T> allocate(cudaStream_t str, size_t n) HAMR_EXPORT;

    /** allocate an array of n elements.
     * @param[in] str a stream used to order operations, or null for the
     *                default stream
     * @param[in] n the number of elements to allocate
     * @param[in] val a value to initialize the elements to
     * @returns a shared pointer to the array that holds a deleter for the
     * memory
     */
    static std::shared_ptr<T> allocate(cudaStream_t str, size_t n, const T &val) HAMR_EXPORT;

    /** allocate an array of n elements.
     * @param[in] str a stream used to order operations, or null for the
     *                default stream
     * @param[in] n the number of elements to allocate
     * @param[in] vals an array of values to initialize the elements with
     * @param[in] cudaVals a flag set to true if vals are accessible by codes
     *                     running in CUDA
     * @returns a shared pointer to the array that holds a deleter for the
     * memory
     */
    template <typename U>
    static std::shared_ptr<T> allocate(cudaStream_t str,
        size_t n, const U *vals, bool cudaVals = false) HAMR_EXPORT;
};





/** A class for allocating arrays on the GPU in CUDA, specialized for numeric
 * types.  This is the preferred allocator for device memory in CUDA because it
 * does not synchronize the entire device.
 */
template <typename T>
struct HAMR_EXPORT cuda_malloc_async_allocator<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
{
    /** allocate an array of n elements.
     * @param[in] str a stream used to order operations, or null for the
     *                default stream
     * @param[in] n the number of elements to allocate
     * @returns a shared pointer to the array that holds a deleter for the
     * memory
     */
    static std::shared_ptr<T> allocate(cudaStream_t str, size_t n) HAMR_EXPORT;

    /** allocate an array of n elements.
     * @param[in] str a stream used to order operations, or null for the
     *                default stream
     * @param[in] n the number of elements to allocate
     * @param[in] val a value to initialize the elements to
     * @returns a shared pointer to the array that holds a deleter for the
     * memory
     */
    static std::shared_ptr<T> allocate(cudaStream_t str, size_t n, const T &val) HAMR_EXPORT;

    /** allocate an array of n elements.
     * @param[in] str a stream used to order operations, or null for the
     *                default stream
     * @param[in] n the number of elements to allocate
     * @param[in] vals an array of values to initialize the elements with
     * @param[in] cudaVals a flag set to true if vals are accessible by codes
     *                     running in CUDA
     * @returns a shared pointer to the array that holds a
     * deleter for the memory
     */
    template <typename U>
    static std::shared_ptr<T> allocate(cudaStream_t str, size_t n, const U *vals, bool cudaVals = false) HAMR_EXPORT;
};

}

#if !defined(HAMR_SEPARATE_IMPL)
#include "hamr_cuda_malloc_async_allocator_impl.h"
#endif

#endif
