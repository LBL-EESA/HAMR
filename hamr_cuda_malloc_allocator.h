#ifndef hamr_cuda_malloc_allocator_h
#define hamr_cuda_malloc_allocator_h

#include "hamr_config.h"

#include <memory>
#include <type_traits>

namespace hamr
{

/// a deleter for arrays allocated with cudaMalloc
template <typename T, typename E = void>
class cuda_malloc_deleter {};

/// a deleter for arrays allocated with cudaMalloc, specialized for objects
template <typename T>
class HAMR_EXPORT cuda_malloc_deleter<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
{
public:
    /** constructs the deleter
     * @param[in] ptr the pointer to the array to delete
     * @param[in] n   the number of elements in the array
     */
    cuda_malloc_deleter(T *ptr, size_t n);

    /** deletes the array
     * @param[in] ptr the pointer to the array to delete. must be the same as
     *                that passed during construction.
     */
    void operator()(T *ptr);

private:
    T *m_ptr;
    size_t m_elem;
};






/// a deleter for arrays allocated with cudaMalloc, specialized for numbers
template <typename T>
class HAMR_EXPORT cuda_malloc_deleter<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
{
public:
    /** constructs the deleter
     * @param[in] ptr the pointer to the array to delete
     * @param[in] n the number of elements in the array
     */
    cuda_malloc_deleter(T *ptr, size_t n);

    /** deletes the array
     * @param[in] ptr the pointer to the array to delete. must be the same as
     *                that passed during construction.
     */
    void operator()(T *ptr);

private:
    T *m_ptr;
    size_t m_elem;
};





/** A class for allocating arrays with cudaMalloc. However, note that because
 * cudaMalloc synchronizes across the device the cuda_malloc_async_allocator
 * should be preferred.
 */
template <typename T, typename E = void>
struct cuda_malloc_allocator {};

/** A class for allocating arrays with cudaMalloc, specialized for objects.
 * However, note that because cudaMalloc synchronizes across the device the
 * cuda_malloc_async_allocator should be preferred.
 */
template <typename T>
struct HAMR_EXPORT cuda_malloc_allocator<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
{
    /// @name synchronous allocation on the default stream.
    /// @{
    /** allocate an array of n elements.
     * @param[in] n the number of elements to allocate
     * @returns a shared pointer to the array that holds a deleter for the
     *          memory
     */
    static std::shared_ptr<T> allocate(size_t n) HAMR_EXPORT;

    /** allocate an array of n elements.
     * @param[in] n the number of elements to allocate
     * @param[in] val a value to initialize the elements to
     * @returns a shared pointer to the array that holds a deleter for the
     *          memory
     */
    static std::shared_ptr<T> allocate(size_t n, const T &val) HAMR_EXPORT;

    /** allocate an array of n elements.
     * @param[in] n the number of elements to allocate
     * @param[in] vals an array of values to initialize the elements with
     * @param[in] cudaVals a flag set to true if vals are accessible by codes
     *                     running in CUDA
     * @returns a shared pointer to the array that holds a deleter for the
     *          memory
     */
    template <typename U>
    static std::shared_ptr<T> allocate(size_t n, const U *vals, bool cudaVals = false) HAMR_EXPORT;
    /// @}

    /// @name asynchronous allocation
    /** These calls are forwarded to the hamr::cuda_malloc_async_allocator.
     * The passed stream is used for both allocation and initialization. The
     * caller is expected to appy explicit synchronization when it is needed.
     */
    ///@{
    /** allocate an array of n elements.
     * @param[in] str a stream used to order operations, or null for the
     *                default stream
     * @param[in] n the number of elements to allocate
     * @returns a shared pointer to the array that holds a deleter for the
     *          memory
     */
    static std::shared_ptr<T> allocate(cudaStream_t str, size_t n) HAMR_EXPORT;

    /** allocate an array of n elements.
     * @param[in] str a stream used to order operations, or null for the
     *                default stream
     * @param[in] n the number of elements to allocate
     * @param[in] val a value to initialize the elements to
     * @returns a shared pointer to the array that holds a deleter for the
     *          memory
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
     *          memory
     */
    template <typename U>
    static std::shared_ptr<T> allocate(cudaStream_t str, size_t n, const U *vals, bool cudaVals = false) HAMR_EXPORT;
    ///@}
};





/** A class for allocating arrays with cudaMalloc, specialized for numeric
 * types. However, note that because cudaMalloc synchronizes across the device
 * the cuda_malloc_async_allocator should be preferred.
 */
template <typename T>
struct HAMR_EXPORT cuda_malloc_allocator<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
{
    /// @name synchronous allocation on the default stream.
    /// @{
    /** allocate an array of n elements.
     * @param[in] n the number of elements to allocate
     * @returns a shared pointer to the array that holds a deleter for the
     * memory
     */
    static std::shared_ptr<T> allocate(size_t n) HAMR_EXPORT;

    /** allocate an array of n elements.
     * @param[in] n the number of elements to allocate
     * @param[in] val a value to initialize the elements to
     * @returns a shared pointer to the array that holds a deleter for the
     * memory
     */
    static std::shared_ptr<T> allocate(size_t n, const T &val) HAMR_EXPORT;

    /** allocate an array of n elements.
     * @param[in] n the number of elements to allocate
     * @param[in] vals an array of values to initialize the elements with
     * @param[in] cudaVals a flag set to true if vals are accessible by codes
     *                     running in CUDA
     * @returns a shared pointer to the array that holds a
     * deleter for the memory
     */
    template <typename U>
    static std::shared_ptr<T> allocate(size_t n, const U *vals, bool cudaVals = false) HAMR_EXPORT;
    /// @}

    /// @name asynchronous allocation
    /** These calls are forwarded to the hamr::cuda_malloc_async_allocator.
     * The passed stream is used for both allocation and initialization. The
     * caller is expected to appy explicit synchronization when it is needed.
     */
    ///@{
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
#include "hamr_cuda_malloc_allocator_impl.h"
#endif

#endif
