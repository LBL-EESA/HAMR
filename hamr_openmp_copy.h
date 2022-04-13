#ifndef hamr_openmp_copy_h
#define hamr_openmp_copy_h

#include "hamr_config.h"
#include "hamr_env.h"
#if defined(HAMR_ENABLE_OPENMP)
#include "hamr_openmp_allocator.h"
#include <omp.h>
#endif
#include "hamr_malloc_allocator.h"

#include <cstring>
#include <cstdlib>
#include <iostream>
#include <memory>

/// heterogeneous accelerator memory resource
namespace hamr
{
#if !defined(HAMR_OPENMP_OBJECTS)
/** Copies an array to the active OpenMP device.
 *
 * @param[in] dest an array of n elements accessible in OpenMP
 * @param[in] src an array of n elements accessible on the CPU
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
static int copy_to_openmp_from_cpu(T *dest, const U *src, size_t n_elem,
   typename std::enable_if<!std::is_arithmetic<T>::value>::type * = nullptr)
{
#if !defined(HAMR_ENABLE_OPENMP)
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_openmp_from_cpu OpenMP is not enabled." << std::endl;
    return -1;
#else
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_openmp_from_cpu HAMR_OpenMP_OBJECTS is not enabled." << std::endl;
    abort();
    return -1;
#endif
}
#else
/** Copies an array to the active OpenMP device (fast path for arrays of
 * arithmetic types of the same type).
 *
 * @param[in] dest an array of n elements accessible in OpenMP
 * @param[in] src an array of n elements accessible on the CPU
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T>
static int copy_to_openmp_from_cpu(T *dest, const T *src, size_t n_elem,
   typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr)
{
#if !defined(HAMR_ENABLE_OPENMP)
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_openmp_from_cpu OpenMP is not enabled." << std::endl;
    return -1;
#else
    // copy src to gpu
    size_t n_bytes = n_elem*sizeof(T);

    int dest_device = omp_get_default_device();
    int src_device = omp_get_initial_device();

    if (omp_target_memcpy(dest, src, n_bytes, 0, 0, dest_device, src_device))
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to copy " << n_bytes << " from host " << src_device
            << " to device " << dest_device << std::endl;
        return -1;
    }

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "hamr::copy_to_openmp_from_cpu same " << n_elem
            << " " << typeid(T).name() << sizeof(T) <<  " from host "
            << src_device << " to device " << dest_device << std::endl;
    }
#endif

    return 0;
#endif
}
#endif

/** Copies an array to the active OpenMP device.
 *
 * @param[in] dest an array of n elements accessible in OpenMP
 * @param[in] src an array of n elements accessible on the CPU
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
static int copy_to_openmp_from_cpu(T *dest, const U *src, size_t n_elem
#if !defined(HAMR_OPENMP_OBJECTS)
    ,typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr
#endif
    )
{
#if !defined(HAMR_ENABLE_OPENMP)
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_openmp_from_cpu OpenMP is not enabled." << std::endl;
    return -1;
#else
    // copy src to gpu
    // apply the copy on the gpu

    // allocate a temporary buffer on the GPU
    std::shared_ptr<U> tmp = hamr::openmp_allocator<U>::allocate(n_elem);
    U *ptmp = tmp.get();

    // copy the data
    size_t n_bytes = n_elem*sizeof(U);

    int dest_device = omp_get_default_device();
    int src_device = omp_get_initial_device();

    if (omp_target_memcpy(ptmp, src, n_bytes, 0, 0, dest_device, src_device))
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to copy " << n_elem << " " << typeid(T).name() << sizeof(T)
            <<  " from host " << src_device << " to device " << dest_device << std::endl;
        return -1;
    }

    // invoke the casting copy kernel
    # pragma omp target teams HAMR_OPENMP_LOOP is_device_ptr(dest, ptmp)
    for (size_t i = 0; i < n_elem; ++i)
    {
        dest[i] = static_cast<T>(ptmp[i]);
    }

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "hamr::copy_to_openmp_from_cpu " << n_elem
            << " from " << typeid(U).name() << sizeof(U) << " to "
            << typeid(T).name() << sizeof(T) <<  " from host "
            << src_device << " to device " << dest_device << std::endl;
    }
#endif

    return 0;
#endif
}

#if !defined(HAMR_OPENMP_OBJECTS)
/** Copies an array on the active OpenMP device.
 *
 * @param[in] dest an array of n elements accessible in OpenMP
 * @param[in] src an array of n elements accessible in OpenMP
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
static int copy_to_openmp_from_openmp(T *dest, const U *src, size_t n_elem,
   typename std::enable_if<!std::is_arithmetic<T>::value>::type * = nullptr)
{
#if !defined(HAMR_ENABLE_OPENMP)
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_openmp_from_openmp OpenMP is not enabled." << std::endl;
    return -1;
#else
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_openmp_from_openmp HAMR_OPENMP_OBJECTS is not enabled." << std::endl;
    return -1;
#endif
}
#else
/** Copies an array on the active OpenMP device (fast path for arrays of
 * arithmetic types of the same type).
 *
 * @param[in] dest an array of n elements accessible in OpenMP
 * @param[in] src an array of n elements accessible on the CPU
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T>
static int copy_to_openmp_from_openmp(T *dest, const T *src, size_t n_elem,
    typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr)
{
#if !defined(HAMR_ENABLE_OPENMP)
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_openmp_from_openmp OpenMP is not enabled." << std::endl;
    return -1;
#else
    // copy src to gpu
    size_t n_bytes = n_elem*sizeof(T);

    int device = omp_get_default_device();

    if (omp_target_memcpy(dest, src, n_bytes, 0, 0, device, device))
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to copy " << n_bytes <<  " on device " << device
            << std::endl;
        return -1;
    }

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "hamr::copy_to_openmp_from_openmp same " << n_elem
            << typeid(T).name() << sizeof(T) <<  " on device " << device
            << std::endl;
    }
#endif

    return 0;
#endif
}
#endif

/** Copies an array on the active OpenMP device.
 *
 * @param[in] dest an array of n elements accessible in OpenMP
 * @param[in] src an array of n elements accessible in OpenMP
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
static int copy_to_openmp_from_openmp(T *dest, const U *src, size_t n_elem
#if !defined(HAMR_OPENMP_OBJECTS)
    ,typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr
#endif
    )
{
#if !defined(HAMR_ENABLE_OPENMP)
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_openmp_from_openmp OpenMP is not enabled." << std::endl;
    return -1;
#else

    // invoke the casting copy kernel
    # pragma omp target teams HAMR_OPENMP_LOOP is_device_ptr(dest, src)
    for (size_t i = 0; i < n_elem; ++i)
    {
        dest[i] = static_cast<T>(src[i]);
    }

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "hamr::copy_to_openmp_from_openmp " << n_elem
            << " from " << typeid(U).name() << sizeof(U) << " to "
            << typeid(T).name() << sizeof(T) << " on device "
            << omp_get_default_device() << std::endl;
    }
#endif

    return 0;
#endif
}

#if !defined(HAMR_OPENMP_OBJECTS)
/** Copies an array to the active OpenMP device from the named OpenMP device,
 *
 * @param[in] dest an array of n elements accessible in OpenMP
 * @param[in] src an array of n elements accessible in OpenMP
 * @param[in] src_device the OpenMP device on which src is allocated
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
static int copy_to_openmp_from_openmp(T *dest, const U *src, int src_device, size_t n_elem,
   typename std::enable_if<!std::is_arithmetic<T>::value>::type * = nullptr)
{
#if !defined(HAMR_ENABLE_OPENMP)
    (void) dest;
    (void) src;
    (void) src_device;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_openmp_from_openmp OpenMP is not enabled." << std::endl;
    return -1;
#else
    (void) dest;
    (void) src;
    (void) src_device;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_openmp_from_openmp HAMR_OpenMP_OBJECTS is not enabled." << std::endl;
    return -1;
#endif
}
#else
/** Copies an array to the active OpenMP device from the named OpenMP device,
 * (fast path for arrays of arithmetic types of the same type).
 *
 * @param[in] dest an array of n elements accessible in OpenMP
 * @param[in] src an array of n elements accessible in OpenMP
 * @param[in] src_device the OpenMP device on which src is allocated
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T>
static int copy_to_openmp_from_openmp(T *dest, const T *src, int src_device, size_t n_elem,
    typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr)
{
#if !defined(HAMR_ENABLE_OPENMP)
    (void) dest;
    (void) src;
    (void) src_device;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_openmp_from_openmp OpenMP is not enabled." << std::endl;
    return -1;
#else
    // copy src to gpu
    size_t n_bytes = n_elem*sizeof(T);
    int dest_device = omp_get_default_device();
    if (omp_target_memcpy(dest, src, n_bytes, 0, 0, dest_device, src_device))
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to copy " << n_bytes <<  " from device " << src_device
            << " to device " << dest_device << std::endl;
        return -1;
    }

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "hamr::copy_to_openmp_from_openmp same " << n_elem
            << typeid(T).name() << sizeof(T) << " from device "
            << src_device << " to device " << dest_device << std::endl;
    }
#endif

    return 0;
#endif
}
#endif

/** Copies an array on the active OpenMP device.
 *
 * @param[in] dest an array of n elements accessible in OpenMP
 * @param[in] src an array of n elements accessible in OpenMP
 * @param[in] src_device the OpenMP device on which src is allocated
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
static int copy_to_openmp_from_openmp(T *dest, const U *src, int src_device, size_t n_elem
#if !defined(HAMR_OPENMP_OBJECTS)
    ,typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr
#endif
    )
{
#if !defined(HAMR_ENABLE_OPENMP)
    (void) dest;
    (void) src;
    (void) src_device;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_openmp_from_openmp OpenMP is not enabled." << std::endl;
    return -1;
#else

    // allocate a temporary buffer on the GPU
    std::shared_ptr<U> tmp = hamr::openmp_allocator<U>::allocate(n_elem);
    U *ptmp = tmp.get();

    // copy src to temporary buffer
    size_t n_bytes = n_elem*sizeof(T);
    int dest_device = omp_get_default_device();
    if (omp_target_memcpy(ptmp, src, n_bytes, 0, 0, dest_device, src_device))
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to copy " << n_bytes <<  " from device " << src_device
            << " to device " << dest_device << std::endl;
        return -1;
    }

    // invoke the casting copy kernel
    # pragma omp target teams HAMR_OPENMP_LOOP is_device_ptr(dest, ptmp)
    for (size_t i = 0; i < n_elem; ++i)
    {
        dest[i] = static_cast<T>(ptmp[i]);
    }

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "hamr::copy_to_openmp_from_openmp " << n_elem
            << " from " << typeid(U).name() << sizeof(U) << " to "
            << typeid(T).name() << sizeof(T) << " from device "
            << src_device << " to device " << dest_device << std::endl;
    }
#endif

    return 0;
#endif
}

#if !defined(HAMR_OPENMP_OBJECTS)
/** Copies an array from the active OpenMP device.
 *
 * @param[in] dest an array of n elements accessible on the CPU
 * @param[in] src an array of n elements accessible in OpenMP
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
static int copy_to_cpu_from_openmp(T *dest, const U *src, size_t n_elem,
   typename std::enable_if<!std::is_arithmetic<T>::value>::type * = nullptr)
{
#if !defined(HAMR_ENABLE_OPENMP)
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_cpu_from_openmp OpenMP is not enabled." << std::endl;
    return -1;
#else
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_cpu_from_openmp HAMR_OpenMP_OBJECTS is not enabled." << std::endl;
    abort();
    return -1;
#endif
}
#else
/** Copies an array from the active OpenMP device (fast path for arrays of
 * arithmetic types of the same type).
 *
 * @param[in] dest an array of n elements accessible in OpenMP
 * @param[in] src an array of n elements accessible on the CPU
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T>
static int copy_to_cpu_from_openmp(T *dest, const T *src, size_t n_elem,
   typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr)
{
#if !defined(HAMR_ENABLE_OPENMP)
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_cpu_from_openmp OpenMP is not enabled." << std::endl;
    return -1;
#else
    // copy src to cpu
    size_t n_bytes = n_elem*sizeof(T);

    int dest_device = omp_get_initial_device();
    int src_device = omp_get_default_device();

    if (omp_target_memcpy(dest, src, n_bytes, 0, 0, dest_device, src_device))
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to copy " << n_bytes << " from host " << src_device
            << " to device " << dest_device << std::endl;
        return -1;
    }

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "hamr::copy_to_cpu_from_openmp same " << n_elem
            << " " << typeid(T).name() << sizeof(T) <<  " from device "
            << src_device << " to host " << dest_device << std::endl;
    }
#endif

    return 0;
#endif
}
#endif

/** Copies an array from the active OpenMP device.
 *
 * @param[in] dest an array of n elements accessible on the CPU
 * @param[in] src an array of n elements accessible in OpenMP
 * @param[in] n_elem the number of elements in the array
 * @returns 0 if there were no errors
 */
template <typename T, typename U>
static int copy_to_cpu_from_openmp(T *dest, const U *src, size_t n_elem
#if !defined(HAMR_OPENMP_OBJECTS)
    ,typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr
#endif
    )
{
#if !defined(HAMR_ENABLE_OPENMP)
    (void) dest;
    (void) src;
    (void) n_elem;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " copy_to_cpu_from_openmp OpenMP is not enabled." << std::endl;
    return -1;
#else

    // allocate a buffer on the cpu, copy the data into the temporary
    // apply copy constructor on the cpu

    // allocate a temporary buffer on the CPU
    auto sptmp = hamr::malloc_allocator<T>::allocate(n_elem);
    T *ptmp = sptmp.get();

    // copy src to cpu
    size_t n_bytes = n_elem*sizeof(T);

    int dest_device = omp_get_initial_device();
    int src_device = omp_get_default_device();

    if (omp_target_memcpy(ptmp, src, n_bytes, 0, 0, dest_device, src_device))
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to copy " << n_bytes << " from device " << src_device
            << " to host " << dest_device << std::endl;
        return -1;
    }

    // invoke the casting copy kernel on the CPU
    for (size_t i = 0; i < n_elem; ++i)
    {
        dest[i] = static_cast<T>(ptmp[i]);
    }

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "hamr::copy_to_cpu_from_openmp " << n_elem
            << " from " << typeid(U).name() << sizeof(U) << " to "
            << typeid(T).name() << sizeof(T)  << " from device "
            << src_device << " to host " << dest_device << std::endl;
    }
#endif

    return 0;
#endif
}

}

#endif
