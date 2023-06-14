#ifndef hamr_hip_copy_h
#define hamr_hip_copy_h

#include "hamr_config.h"
#include "hamr_env.h"
#include <memory>

/// heterogeneous accelerator memory resource
namespace hamr
{
#if !defined(HAMR_ENABLE_OBJECTS)
/** Copies an array to the active HIP device.
 *
 * @param[in] dest an array of n elements accessible in HIP
 * @param[in] src an array of n elements accessible on the host
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
int copy_to_hip_from_host(T *dest, const U *src, size_t n_elem,
   typename std::enable_if<!std::is_arithmetic<T>::value>::type * = nullptr);
#else
/** Copies an array to the active HIP device (fast path for arrays of
 * arithmetic types of the same type).
 *
 * @param[in] dest an array of n elements accessible in HIP
 * @param[in] src an array of n elements accessible on the host
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T>
int copy_to_hip_from_host(T *dest, const T *src, size_t n_elem,
   typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr);
#endif

/** Copies an array to the active HIP device.
 *
 * @param[in] dest an array of n elements accessible in HIP
 * @param[in] src an array of n elements accessible on the host
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
int copy_to_hip_from_host(T *dest, const U *src, size_t n_elem
#if !defined(HAMR_ENABLE_OBJECTS)
    ,typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr
#endif
    );

#if !defined(HAMR_ENABLE_OBJECTS)
/** Copies an array on the active HIP device.
 *
 * @param[in] dest an array of n elements accessible in HIP
 * @param[in] src an array of n elements accessible in HIP
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
int copy_to_hip_from_hip(T *dest, const U *src, size_t n_elem,
   typename std::enable_if<!std::is_arithmetic<T>::value>::type * = nullptr);
#else
/** Ccopies an array on the active HIP device (fast path for arrays of
 * arithmetic types of the same type).
 *
 * @param[in] dest an array of n elements accessible in HIP
 * @param[in] src an array of n elements accessible on the host
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T>
int copy_to_hip_from_hip(T *dest, const T *src, size_t n_elem,
    typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr);
#endif

/** Copies an array on the active HIP device.
 *
 * @param[in] dest an array of n elements accessible in HIP
 * @param[in] src an array of n elements accessible in HIP
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
int copy_to_hip_from_hip(T *dest, const U *src, size_t n_elem
#if !defined(HAMR_ENABLE_OBJECTS)
    ,typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr
#endif
    );

#if !defined(HAMR_ENABLE_OBJECTS)
/** Copies an array to the active HIP device from the named HIP device,
 *
 * @param[in] dest an array of n elements accessible in HIP
 * @param[in] src an array of n elements accessible in HIP
 * @param[in] src_device the HIP device on which src is allocated
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
int copy_to_hip_from_hip(T *dest, const U *src, int src_device, size_t n_elem,
   typename std::enable_if<!std::is_arithmetic<T>::value>::type * = nullptr);
#else
/** Copies an array to the active HIP device from the named HIP device, (fast
 * path for arrays of arithmetic types of the same type).
 *
 * @param[in] dest an array of n elements accessible in HIP
 * @param[in] src an array of n elements accessible in HIP
 * @param[in] src_device the HIP device on which src is allocated
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T>
int copy_to_hip_from_hip(T *dest, const T *src, int src_device, size_t n_elem,
    typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr);
#endif

/** Copies an array on the active HIP device.
 *
 * @param[in] dest an array of n elements accessible in HIP
 * @param[in] src an array of n elements accessible in HIP
 * @param[in] src_device the HIP device on which src is allocated
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
int copy_to_hip_from_hip(T *dest, const U *src, int src_device, size_t n_elem
#if !defined(HAMR_ENABLE_OBJECTS)
    ,typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr
#endif
    );

#if !defined(HAMR_ENABLE_OBJECTS)
/** Copies an array from the active HIP device.
 *
 * @param[in] dest an array of n elements accessible on the host
 * @param[in] src an array of n elements accessible in HIP
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
int copy_to_host_from_hip(T *dest, const U *src, size_t n_elem,
   typename std::enable_if<!std::is_arithmetic<T>::value>::type * = nullptr);
#else
/** Copies an array from the active HIP device (fast path for arrays of
 * arithmetic types of the same type).
 *
 * @param[in] dest an array of n elements accessible in HIP
 * @param[in] src an array of n elements accessible on the host
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T>
int copy_to_host_from_hip(T *dest, const T *src, size_t n_elem,
   typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr);
#endif

/** Copies an array from the active HIP device.
 *
 * @param[in] dest an array of n elements accessible on the host
 * @param[in] src an array of n elements accessible in HIP
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
int copy_to_host_from_hip(T *dest, const U *src, size_t n_elem
#if !defined(HAMR_ENABLE_OBJECTS)
    ,typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr
#endif
    );

}

#if !defined(HAMR_SEPARATE_IMPL)
#include "hamr_hip_copy_impl.h"
#endif

#endif
