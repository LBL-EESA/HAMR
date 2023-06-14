#ifndef hamr_device_h
#define hamr_device_h

#include "hamr_config.h"
#if defined(HAMR_ENABLE_CUDA)
#include "hamr_cuda_device.h"
#elif defined(HAMR_ENABLE_HIP)
#include "hamr_hip_device.h"
#elif defined(HAMR_ENABLE_OPENMP)
#include "hamr_openmp_device.h"
#endif

///@file

namespace hamr
{
/// gets the device identifier for the first GPU. @returns zero if successful.
inline int HAMR_EXPORT get_device_identifier(int &dev_id)
{
#if defined(HAMR_ENABLE_CUDA)
    return get_cuda_device_identifier(dev_id);
#elif defined(HAMR_ENABLE_HIP)
    return get_hip_device_identifier(dev_id);
#elif defined(HAMR_ENABLE_OPENMP)
    return get_openmp_device_identifier(dev_id);
#else
    dev_id = -1;
    return 0;
#endif
}

/// gets the device identifier for the host. @returns zero if successful.
inline int HAMR_EXPORT get_host_identifier(int &dev_id)
{
#if defined(HAMR_ENABLE_CUDA)
    return get_cuda_host_identifier(dev_id);
#elif defined(HAMR_ENABLE_HIP)
    return get_hip_host_identifier(dev_id);
#elif defined(HAMR_ENABLE_OPENMP)
    return get_openmp_host_identifier(dev_id);
#else
    dev_id = -1;
    return 0;
#endif
}

/// gets the currently atcive device. @returns zero if successful.
inline int HAMR_EXPORT get_active_device(int &dev_id)
{
#if defined(HAMR_ENABLE_CUDA)
    return get_active_cuda_device(dev_id);
#elif defined(HAMR_ENABLE_HIP)
    return get_active_hip_device(dev_id);
#elif defined(HAMR_ENABLE_OPENMP)
    return get_active_openmp_device(dev_id);
#else
    dev_id = -1;
    return 0;
#endif
}

/// sets the active  device. returns zero if successful.
inline int HAMR_EXPORT set_active_device(int dev_id)
{
#if defined(HAMR_ENABLE_CUDA)
    return set_active_cuda_device(dev_id);
#elif defined(HAMR_ENABLE_HIP)
    return set_active_hip_device(dev_id);
#elif defined(HAMR_ENABLE_OPENMP)
    return set_active_openmp_device(dev_id);
#else
    return 0;
#endif
}

/// gets the device that owns the given pointer. @returns zero if successful.
inline int HAMR_EXPORT get_device(const void *ptr, int &device_id)
{
#if defined(HAMR_ENABLE_CUDA)
    return get_cuda_device(ptr, device_id);
#elif defined(HAMR_ENABLE_HIP)
    return get_hip_device(ptr, device_id);
#elif defined(HAMR_ENABLE_OPENMP)
    return get_openmp_device(ptr, device_id);
#else
    device_id = -1;
    return 0;
#endif
}

#if defined(HAMR_ENABLE_CUDA)
using activate_device = activate_cuda_device;
#elif defined(HAMR_ENABLE_HIP)
using activate_device = activate_hip_device;
#elif defined(HAMR_ENABLE_OPENMP)
using activate_device = activate_openmp_device;
#else
/** Activate the specified device, and restore the previously active
 * device when the object is destroyed.
 */
class HAMR_EXPORT activate_device
{
public:
    activate_device() = delete;
    activate_device(const activate_device &) = delete;
    void operator=(const activate_device &) = delete;
    activate_device(int) {}
    ~activate_device() {}
};
#endif

}
#endif
