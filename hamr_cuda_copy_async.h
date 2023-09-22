#ifndef hamr_cuda_copy_async_h
#define hamr_cuda_copy_async_h

#include "hamr_config.h"
#include "hamr_copier_traits.h"

#include <memory>
#include <type_traits>

/// heterogeneous accelerator memory resource
namespace hamr
{
#if !defined(HAMR_ENABLE_OBJECTS)
/** Copies an array to the active CUDA device.
 *
 * @param[in] str a CUDA stream or nullptr to use the default stream
 * @param[in] dest an array of n elements accessible in CUDA
 * @param[in] src an array of n elements accessible on the host
 * @param[in] n_elem the number of elements in the array
 *
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
int copy_to_cuda_from_host(cudaStream_t str, T *dest, const U *src, size_t n_elem,
    hamr::use_object_copier_t<T,U> * = nullptr);
#endif

/** Copies an array to the active CUDA device (fast path for arrays of
 * arithmetic types of the same type).
 *
 * @param[in] str a CUDA stream or nullptr to use the default stream
 * @param[in] dest an array of n elements accessible in CUDA
 * @param[in] src an array of n elements accessible on the host
 * @param[in] n_elem the number of elements in the array
 *
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
int copy_to_cuda_from_host(cudaStream_t str, T *dest, const U *src, size_t n_elem,
    hamr::use_bytes_copier_t<T,U> * = nullptr);

/** Copies an array to the active CUDA device.
 *
 * @param[in] str a CUDA stream or nullptr to use the default stream
 * @param[in] dest an array of n elements accessible in CUDA
 * @param[in] src an array of n elements accessible on the host
 * @param[in] n_elem the number of elements in the array
 *
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
int copy_to_cuda_from_host(cudaStream_t str, T *dest, const U *src, size_t n_elem,
    hamr::use_cons_copier_t<T,U> * = nullptr);


#if !defined(HAMR_ENABLE_OBJECTS)
/** Copies an array on the active CUDA device.
 *
 * @param[in] str a CUDA stream or nullptr to use the default stream
 * @param[in] dest an array of n elements accessible in CUDA
 * @param[in] src an array of n elements accessible in CUDA
 * @param[in] n_elem the number of elements in the array
 *
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
int copy_to_cuda_from_cuda(cudaStream_t str, T *dest, const U *src, size_t n_elem,
    hamr::use_object_copier_t<T,U> * = nullptr);
#endif

/** Ccopies an array on the active CUAD device (fast path for arrays of
 * arithmetic types of the same type).
 *
 * @param[in] str a CUDA stream or nullptr to use the default stream
 * @param[in] dest an array of n elements accessible in CUDA
 * @param[in] src an array of n elements accessible on the host
 * @param[in] n_elem the number of elements in the array
 *
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
int copy_to_cuda_from_cuda(cudaStream_t str, T *dest, const U *src, size_t n_elem,
    hamr::use_bytes_copier_t<T,U> * = nullptr);


/** Copies an array on the active CUDA device.
 *
 * @param[in] str a CUDA stream or nullptr to use the default stream
 * @param[in] dest an array of n elements accessible in CUDA
 * @param[in] src an array of n elements accessible in CUDA
 * @param[in] n_elem the number of elements in the array
 *
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
int copy_to_cuda_from_cuda(cudaStream_t str, T *dest, const U *src, size_t n_elem,
    hamr::use_cons_copier_t<T,U> * = nullptr);

#if !defined(HAMR_ENABLE_OBJECTS)
/** Copies an array to the active CUDA device from the named CUDA device,
 *
 * @param[in] str a CUDA stream or nullptr to use the default stream
 * @param[in] dest an array of n elements accessible in CUDA
 * @param[in] src an array of n elements accessible in CUDA
 * @param[in] src_device the CUDA device on which src is allocated
 * @param[in] n_elem the number of elements in the array
 *
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
int copy_to_cuda_from_cuda(cudaStream_t str, T *dest, const U *src,
    int src_device, size_t n_elem, hamr::use_object_copier_t<T,U> * = nullptr);
#endif

/** Copies an array to the active CUDA device from the named CUDA device, (fast
 * path for arrays of arithmetic types of the same type).
 *
 * @param[in] str a CUDA stream or nullptr to use the default stream
 * @param[in] dest an array of n elements accessible in CUDA
 * @param[in] src an array of n elements accessible in CUDA
 * @param[in] src_device the CUDA device on which src is allocated
 * @param[in] n_elem the number of elements in the array
 *
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
int copy_to_cuda_from_cuda(cudaStream_t str, T *dest, const U *src,
    int src_device, size_t n_elem, hamr::use_bytes_copier_t<T,U> * = nullptr);

/** Copies an array on the active CUDA device.
 *
 * @param[in] str a CUDA stream or nullptr to use the default stream
 * @param[in] dest an array of n elements accessible in CUDA
 * @param[in] src an array of n elements accessible in CUDA
 * @param[in] src_device the CUDA device on which src is allocated
 * @param[in] n_elem the number of elements in the array
 *
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
int copy_to_cuda_from_cuda(cudaStream_t str, T *dest, const U *src,
    int src_device, size_t n_elem, hamr::use_cons_copier_t<T,U> * = nullptr);

#if !defined(HAMR_ENABLE_OBJECTS)
/** Copies an array from the active CUDA device.
 *
 * @param[in] str a CUDA stream or nullptr to use the default stream
 * @param[in] dest an array of n elements accessible on the host
 * @param[in] src an array of n elements accessible in CUDA
 * @param[in] n_elem the number of elements in the array
 *
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
int copy_to_host_from_cuda(cudaStream_t str, T *dest, const U *src, size_t n_elem,
    hamr::use_object_copier_t<T,U> * = nullptr);
#endif

/** Copies an array from the active CUDA device (fast path for arrays of
 * arithmetic types of the same type).
 *
 * @param[in] str a CUDA stream or nullptr to use the default stream
 * @param[in] dest an array of n elements accessible in CUDA
 * @param[in] src an array of n elements accessible on the host
 * @param[in] n_elem the number of elements in the array
 *
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
int copy_to_host_from_cuda(cudaStream_t str, T *dest, const U *src, size_t n_elem,
    hamr::use_bytes_copier_t<T,U> * = nullptr);

/** Copies an array from the active CUDA device.
 *
 * @param[in] dest an array of n elements accessible on the host
 * @param[in] src an array of n elements accessible in CUDA
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
int copy_to_host_from_cuda(cudaStream_t str, T *dest, const U *src, size_t n_elem,
    hamr::use_cons_copier_t<T,U> * = nullptr);

}

#if !defined(HAMR_SEPARATE_IMPL)
#include "hamr_cuda_copy_async_impl.h"
#endif

#endif
