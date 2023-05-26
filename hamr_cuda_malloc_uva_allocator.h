#ifndef hamr_cuda_malloc_uva_allocator_h
#define hamr_cuda_malloc_uva_allocator_h

#include "hamr_config.h"
#include <type_traits>
#include <memory>

namespace hamr
{

/// a deleter for arrays allocated with cuda_malloc_uva
template <typename T, typename E = void>
class cuda_malloc_uva_deleter {};

/// a deleter for arrays allocated with cuda_malloc_uva, specialized for objects
template <typename T>
class cuda_malloc_uva_deleter<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
{
public:
    /** constructs the deleter
     * @param[in] ptr the pointer to the array to delete
     * @param[in] n the number of elements in the array
     */
    cuda_malloc_uva_deleter(cudaStream_t str, T *ptr, size_t n);

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






/// a deleter for arrays allocated with cuda_malloc_uva, specialized for numbers
template <typename T>
class cuda_malloc_uva_deleter<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
{
public:
    /** constructs the deleter
     * @param[in] ptr the pointer to the array to delete
     * @param[in] n the number of elements in the array
     */
    cuda_malloc_uva_deleter(cudaStream_t str, T *ptr, size_t n);

    /** deletes the array
     * @param[in] ptr the pointer to the array to delete. must be the same as
     *                that passed during construction.
     */
    void operator()(T *ptr);

private:
    T *m_ptr;
    size_t m_elem;
};






/// a class for allocating arrays with cuda_malloc_uva
template <typename T, typename E = void>
struct cuda_malloc_uva_allocator {};

/// a class for allocating arrays with cuda_malloc_uva, specialized for objects
template <typename T>
struct HAMR_EXPORT cuda_malloc_uva_allocator<T, typename std::enable_if<!std::is_arithmetic<T>::value>::type>
{
    /** allocate an array of n elements.
     * @param[in] str a stream used to order operations, or null for the
     *                default stream
     * @param[in] n the number of elements to allocate
     * @returns a shared point to the array that holds a deleter for the memory
     */
    static std::shared_ptr<T> allocate(cudaStream_t str, size_t n);

    /** allocate an array of n elements.
     * @param[in] str a stream used to order operations, or null for the
     *                default stream
     * @param[in] n the number of elements to allocate
     * @param[in] val a value to initialize the elements to
     * @returns a shared point to the array that holds a deleter for the memory
     */
    static std::shared_ptr<T> allocate(cudaStream_t str, size_t n, const T &val);

    /** allocate an array of n elements.
     * @param[in] str a stream used to order operations, or null for the
     *                default stream
     * @param[in] n the number of elements to allocate
     * @param[in] vals an array of values to initialize the elements with
     * @param[in] cudaVals a flag that is set to true if vals is accessible from codes running in CUDA
     * @returns a shared point to the array that holds a deleter for the memory
     */
    template <typename U>
    static std::shared_ptr<T> allocate(cudaStream_t str, size_t n, const U *vals, bool cudaVals = false);
};





/// a class for allocating arrays with cuda_malloc_uva, specialized for numbers
template <typename T>
struct HAMR_EXPORT cuda_malloc_uva_allocator<T, typename std::enable_if<std::is_arithmetic<T>::value>::type>
{
    /** allocate an array of n elements.
     * @param[in] str a stream used to order operations, or null for the
     *                default stream
     * @param[in] n the number of elements to allocate
     * @returns a shared point to the array that holds a deleter for the memory
     */
    static std::shared_ptr<T> allocate(cudaStream_t str, size_t n);

    /** allocate an array of n elements.
     * @param[in] str a stream used to order operations, or null for the
     *                default stream
     * @param[in] n the number of elements to allocate
     * @param[in] val a value to initialize the elements to
     * @returns a shared point to the array that holds a deleter for the memory
     */
    static std::shared_ptr<T> allocate(cudaStream_t str, size_t n, const T &val);

    /** allocate an array of n elements.
     * @param[in] str a stream used to order operations, or null for the
     *                default stream
     * @param[in] n the number of elements to allocate
     * @param[in] vals an array of values to initialize the elements with
     * @param[in] cudaVals a flag set to true if vals is accessible from codes running in CUDA
     * @returns a shared point to the array that holds a deleter for the memory
     */
    template <typename U>
    static std::shared_ptr<T> allocate(cudaStream_t str, size_t n, const U *vals, bool cudaVals = false);
};

}

#if !defined(HAMR_SEPARATE_IMPL)
#include "hamr_cuda_malloc_uva_allocator_impl.h"
#endif

#endif
